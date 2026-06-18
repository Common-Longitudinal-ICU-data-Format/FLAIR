"""Synthetic-data tests for the continuous report paths (peak + landmark).

No CLIF data needed. We build a multi-row-per-stay frame that mimics a windowed
task (task4-style): each stay has several hourly windows; a positive stay is
labeled 1 only in the last `HORIZON` windows (an onset that ends the stay), with
y_prob rising toward it. This exercises:

  - peak    : collapse to stay-peak risk -> discrimination + fairness + NNE
  - landmark: 12 risk-set slices counting back from the last window
  - leadtime: per-landmark sensitivity + median lead-time
  - n<10 whole-landmark suppression on the deep (long-stay-only) landmarks
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flair_benchmark.report.landmark import N_LANDMARKS, build_landmark_report
from flair_benchmark.report.leadtime import build_leadtime
from flair_benchmark.report.peak import build_peak_report
from flair_benchmark.tasks._finalize import finalize_cohort

HORIZON = 6
META = {
    "name": "synthetic_continuous_task",
    "display_name": "Synthetic Continuous Task",
    "task_class": "intervention",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-window",
    "task_type": "binary",
    "label_column": "label",
    "clinical_threshold": 0.20,
    "clinical_direction": "above",
    "window_unit": "hour",
    "prediction_window": [0, HORIZON],
}


def make_synth_continuous(n_stays: int = 240, seed: int = 0) -> pl.DataFrame:
    """Multi-row-per-stay frame. A handful of stays are long (>=12 windows) so
    deep landmarks fall below the n<10 suppression floor."""
    rng = np.random.default_rng(seed)
    base = datetime(2021, 1, 1, 0, 0)
    rows = []
    for i in range(n_stays):
        # mostly short stays; ~6 long ones (>=12 windows) on purpose
        n_win = int(rng.integers(2, 9)) if i >= 6 else int(rng.integers(12, 15))
        positive = bool(rng.random() < 0.4)
        latent = float(rng.normal())
        for w in range(n_win):
            from_end = (n_win - 1) - w           # 0 == last window
            if positive and from_end < HORIZON:
                label = 1
                logit = 0.4 + 1.4 * latent + 0.25 * (HORIZON - from_end)
            else:
                label = 0
                logit = -2.6 + 0.8 * latent
            prob = 1.0 / (1.0 + np.exp(-logit))
            rows.append({
                "patient_id": f"p{i}",
                "hospitalization_id": f"h{i}",
                "prediction_dttm": base + timedelta(hours=w),
                "label": label,
                "split": "test",
                "age_at_admission": int(rng.integers(18, 95)),
                "sex_category": "Female" if rng.random() < 0.5 else "Male",
                "race_category": "White",
                "ethnicity_category": "Non-Hispanic",
                "y_prob": float(prob),
                "y_pred": int(prob >= 0.5),
            })
    df = pl.DataFrame(rows)
    cols = ["hospitalization_id", "prediction_dttm", "label", "split",
            "age_at_admission", "sex_category", "race_category",
            "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(df, cols, META)


def _fake_task():
    return SimpleNamespace(META=META, __file__=__file__)


# ----------------------------------------------------------------- peak


def test_peak_emits_discrimination_and_fairness_only():
    df = make_synth_continuous()
    rep = build_peak_report(df, META)
    assert set(rep) == {"discrimination", "fairness"}
    # peak deliberately drops calibration + dca (peak-calibration trap)
    assert "calibration" not in rep and "dca" not in rep
    assert rep["discrimination"]["metadata"]["peak_calibration_omitted"] is True
    # one row per stay after the collapse
    assert rep["discrimination"]["metadata"]["n_stays"] == df["hospitalization_id"].n_unique()


def test_peak_injects_nne():
    df = make_synth_continuous()
    op = build_peak_report(df, META)["discrimination"]["metrics"]["operating_point"]
    ppv = op["ppv"]
    assert op["nne"] == pytest.approx(round(1.0 / ppv, 1))
    # NNE CI is the inverted PPV CI (bounds swap under 1/x)
    if op.get("ppv_ci") and op["ppv_ci"][0]:
        assert op["nne_ci"][1] == pytest.approx(round(1.0 / op["ppv_ci"][0], 1))


# ----------------------------------------------------------------- landmark


def test_landmark_shape_and_one_row_per_stay():
    df = make_synth_continuous()
    rep = build_landmark_report(df, META)
    assert set(rep) == {"discrimination", "calibration", "dca", "fairness"}
    for pillar in rep.values():
        assert len(pillar) == N_LANDMARKS

    # landmark 0 = every stay's last window -> exactly one row per stay
    n_stays = df["hospitalization_id"].n_unique()
    lm0 = rep["discrimination"][0]
    assert lm0["windows_before_end"] == 0
    assert lm0["n"] == n_stays


def test_landmark_deep_landmarks_suppressed():
    df = make_synth_continuous()
    rep = build_landmark_report(df, META)
    # only ~6 stays are long enough to reach landmark 11 -> below n<10 floor
    deep = rep["discrimination"][N_LANDMARKS - 1]
    assert deep["suppressed"] is True
    assert deep["n"] == "<10"
    # a suppressed entry carries no metrics
    assert "metrics" not in deep


# ----------------------------------------------------------------- leadtime


def test_leadtime_curve_and_median():
    df = make_synth_continuous()
    lt = build_leadtime(df, META)
    assert len(lt["curve"]) == N_LANDMARKS
    m = lt["metrics"]
    assert 0.0 <= m["pct_caught"] <= 1.0
    assert m["n_caught"] <= m["n_positive_stays"]
    # caught stays should have a non-negative median warning, with a CI
    if m["median_lead_windows"] is not None:
        assert m["median_lead_windows"] >= 0
        assert len(m["median_lead_ci"]) == 2


# ----------------------------------------------------------------- end-to-end


def test_build_report_landmark_end_to_end(tmp_path):
    df = make_synth_continuous()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "lm"),
                         mode="landmark")
    assert {"discrimination", "calibration", "dca", "fairness", "leadtime"} <= set(paths)
    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["report_mode"] == "landmark"
    assert len(disc["landmarks"]) == N_LANDMARKS


def test_build_report_peak_end_to_end(tmp_path):
    df = make_synth_continuous()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "pk"),
                         mode="peak")
    assert set(k for k in paths if not k.endswith("_png")) == {"discrimination", "fairness"}
    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["report_mode"] == "peak"
    assert disc["metrics"]["operating_point"]["nne"] is not None


def test_landmark_viz_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    df = make_synth_continuous()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "lm"),
                         mode="landmark", viz=True)
    pngs = [p for k, p in paths.items() if k.endswith("_png")]
    assert pngs and all(p.exists() for p in pngs)
