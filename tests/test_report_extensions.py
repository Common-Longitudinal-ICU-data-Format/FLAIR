"""Synthetic-data tests for the extended report bundle (no CLIF data needed)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flair_benchmark.report._calib import calibration_extended
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.temporal import build_temporal
from flair_benchmark.report.threshold_analysis import build_threshold_analysis
from flair_benchmark.tasks._finalize import finalize_cohort

META = {
    "name": "synthetic_task",
    "display_name": "Synthetic Task",
    "task_class": "outcome",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-timestep",
    "task_type": "binary",
    "label_column": "label",
    "clinical_threshold": 0.20,
    "clinical_action": "intervene",
    "clinical_direction": "above",
}


def make_synth(n_stays: int = 400, max_days: int = 10, seed: int = 0,
               calibrated: bool = True) -> pl.DataFrame:
    """One row per (stay, day): demographics, y_prob correlated with label."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        n_days = int(rng.integers(1, max_days + 1))
        # admission in Feb-Nov to keep the min(prediction_dttm) year proxy exact
        adm = datetime(int(rng.integers(2017, 2024)), int(rng.integers(2, 12)),
                       int(rng.integers(1, 28)), 7, 0)
        latent = rng.normal()
        sex = "Female" if rng.random() < 0.5 else "Male"
        age = int(rng.integers(18, 95))
        for d in range(n_days):
            logit = -2.5 + 1.5 * latent + 0.1 * d + rng.normal(scale=0.3)
            prob = 1 / (1 + np.exp(-logit))
            label = int(rng.random() < prob) if calibrated else int(latent > 0.8)
            rows.append({
                "patient_id": f"p{i}",
                "hospitalization_id": f"h{i}",
                "prediction_dttm": adm + timedelta(days=d + 1),
                "label": label,
                "split": "test",
                "age_at_admission": age,
                "sex_category": sex,
                "race_category": "White",
                "ethnicity_category": "Non-Hispanic",
                "y_prob": float(prob),
                "y_pred": int(prob >= 0.5),
            })
    df = pl.DataFrame(rows)
    output_cols = ["hospitalization_id", "prediction_dttm",
                   "label", "split", "age_at_admission",
                   "sex_category", "race_category", "ethnicity_category",
                   "y_prob", "y_pred"]
    return finalize_cohort(df, output_cols, META)


# ---------------------------------------------------------------- finalize


def test_window_number_and_prediction_id():
    df = make_synth(n_stays=20)
    one = df.filter(pl.col("hospitalization_id") == "h0").sort("prediction_dttm")
    assert one["window_number"].to_list() == list(range(1, one.height + 1))
    assert one["prediction_id"][0] == "h0-h0-p0-1"


def test_prediction_id_one_per_stay():
    meta = {**META, "temporal_pattern": "episode", "prediction_unit": "one-per-stay"}
    df = pl.DataFrame({"patient_id": ["p1"], "hospitalization_id": ["h1"],
                       "label": [0], "split": ["test"]})
    out = finalize_cohort(df, ["hospitalization_id", "label", "split"], meta)
    assert out["prediction_id"][0] == "h1-h1-p1-1"
    assert out["window_number"][0] == 1


# ---------------------------------------------------------------- temporal


def test_temporal_window_bins():
    df = make_synth()
    t = build_temporal(df, META)
    bins = {b["bin"]: b for b in t["by_window"]["bins"]}
    assert "1" in bins and "7+" in bins
    n_test = df.height
    # bins with n<10 are dropped; here every bucket is large
    assert sum(b["n"] for b in bins.values()) == n_test
    assert bins["1"]["auroc"] is not None and 0.5 < bins["1"]["auroc"] <= 1.0
    assert bins["1"]["auroc_ci"] is None or bins["1"]["auroc_ci"][0] < bins["1"]["auroc_ci"][1]


def test_temporal_year_trend():
    df = make_synth()
    t = build_temporal(df, META)
    years = [b["bin"] for b in t["by_year"]["bins"]]
    assert years == sorted(years)
    assert set(years) <= set(range(2017, 2024))
    assert t["by_year"]["year_source"] == "derived_from_min_prediction_dttm"


def test_temporal_trajectory_two_series():
    df = make_synth()
    traj = build_temporal(df, META)["trajectory_by_outcome"]
    assert [s["outcome"] for s in traj["series"]] == ["positive", "negative"]
    pos, neg = traj["series"]
    # the model separates outcomes: positive-line above negative-line on day 1
    assert pos["mean_prob"][0] > neg["mean_prob"][0]


def test_temporal_one_per_stay_skips_windows():
    meta = {**META, "temporal_pattern": "episode", "prediction_unit": "one-per-stay"}
    df = make_synth().unique(subset="hospitalization_id").drop("window_number")
    t = build_temporal(df, meta)
    assert t["by_window"] == {"reason": "one_per_stay_task"}
    assert t["trajectory_by_outcome"] == {"reason": "one_per_stay_task"}
    assert t["by_year"] is not None


def test_temporal_suppression_small_year():
    df = make_synth(n_stays=200, max_days=1)
    # plant a tiny year: 5 stays in 1999
    tiny = df.head(5).with_columns(
        pl.col("prediction_dttm").dt.replace(year=1999),
        pl.Series("hospitalization_id", [f"tiny{i}" for i in range(5)]),
    )
    t = build_temporal(pl.concat([df.slice(5), tiny]), META)
    assert 1999 not in [b["bin"] for b in t["by_year"]["bins"]]


def test_temporal_auroc_suppressed_few_positives():
    rng = np.random.default_rng(1)
    n = 200
    labels = np.zeros(n, dtype=int)
    labels[:8] = 1  # only 8 positives
    df = pl.DataFrame({
        "hospitalization_id": [f"h{i}" for i in range(n)],
        "label": labels,
        "split": ["test"] * n,
        "y_prob": rng.random(n),
        "y_pred": np.zeros(n, dtype=int),
        "window_number": np.ones(n, dtype=np.int32),
        "prediction_dttm": [datetime(2020, 5, 1)] * n,
    })
    t = build_temporal(df, META)
    b = t["by_window"]["bins"][0]
    assert b["auroc"] is None
    assert b["n_pos"] == "<10"


# ---------------------------------------------------------------- calibration


def test_calibration_extended_well_calibrated():
    df = make_synth(n_stays=2000, calibrated=True)
    y = df["label"].to_numpy().astype(float)
    p = df["y_prob"].to_numpy()
    ext = calibration_extended(y, p)
    assert abs(ext["slope"] - 1.0) < 0.15
    assert abs(ext["intercept"]) < 0.15
    assert abs(ext["o_e_ratio"] - 1.0) < 0.1
    assert ext["smoothed"]["method"] == "logistic_recalibration"
    assert len(ext["smoothed"]["pred_grid"]) == 50


def test_calibration_extended_miscalibrated():
    df = make_synth(n_stays=2000, calibrated=True)
    y = df["label"].to_numpy().astype(float)
    p = df["y_prob"].to_numpy() ** 2  # systematic underestimation
    ext = calibration_extended(y, p)
    assert ext["o_e_ratio"] > 1.1  # observed exceeds expected


# ---------------------------------------------------------------- DCA


def test_dca_good_model_has_useful_range():
    df = make_synth(n_stays=1000)
    dca = build_dca(df, META)
    assert dca["useful_range"] is not None
    assert 0 < dca["useful_range"]["low"] < dca["useful_range"]["high"] < 1
    assert len(dca["thresholds"]) == 99
    # nb_all crosses zero near prevalence
    prev = dca["prevalence"]
    idx = int(round(prev * 100)) - 1
    assert abs(dca["net_benefit_all"][idx]) < 0.02


def test_dca_random_model_no_advantage():
    rng = np.random.default_rng(2)
    df = make_synth(n_stays=1000).with_columns(
        pl.Series("y_prob", rng.random(make_synth(n_stays=1000).height)))
    dca = build_dca(df, META)
    assert dca["useful_range"] is None or (
        dca["useful_range"]["high"] - dca["useful_range"]["low"] < 0.05)


def test_dca_suppresses_sparse_thresholds():
    df = make_synth(n_stays=100, max_days=1)
    dca = build_dca(df, META)
    for t, nb in zip(dca["thresholds"], dca["net_benefit_model"]):
        if t in dca["suppressed_thresholds"]:
            assert nb is None


# ------------------------------------------------------- threshold analysis


def test_threshold_label_and_action_metrics():
    df = make_synth(n_stays=1000)
    ta = build_threshold_analysis(df, META)
    assert ta["clinical_threshold"] == 0.20
    lm, am = ta["label_metrics"], ta["action_metrics"]
    assert 0 < lm["sensitivity"] <= 1 and 0 < lm["specificity"] <= 1
    assert lm["n_flagged"] == am["n_act"] or abs(lm["n_flagged"] - am["n_act"]) <= lm["n_flagged"]
    # acting on high-risk patients: event rate among acted-on must exceed rest
    assert am["event_rate_in_acted"] > am["event_rate_in_not_acted"]
    assert len(ta["sweep"]) == 4
    assert [s["threshold"] for s in ta["sweep"]] == [0.15, 0.18, 0.22, 0.25]


def test_threshold_direction_below_flips_action():
    meta = {**META, "clinical_direction": "below", "clinical_action": "extubate"}
    df = make_synth(n_stays=1000)
    ta = build_threshold_analysis(df, meta)
    am = ta["action_metrics"]
    # acting on LOW-risk patients: event rate among acted-on must be lower
    assert am["event_rate_in_acted"] < am["event_rate_in_not_acted"]
    # label metrics unchanged by direction (always label space)
    assert ta["label_metrics"] == build_threshold_analysis(df, META)["label_metrics"]


def test_partial_auroc_oracle_vs_random():
    rng = np.random.default_rng(3)
    n = 4000
    y = (rng.random(n) < 0.3).astype(int)
    base = {"hospitalization_id": [f"h{i}" for i in range(n)],
            "label": y, "split": ["test"] * n,
            "y_pred": np.zeros(n, dtype=int)}
    # oracle: every positive scores above the band, negatives spread below it
    oracle_prob = np.where(y == 1, 0.5, 0.19 * rng.random(n))
    ta = build_threshold_analysis(pl.DataFrame({**base, "y_prob": oracle_prob}), META)
    assert ta["partial_auroc"]["pauc_standardized"] is None or \
        ta["partial_auroc"]["pauc_standardized"] > 0.9
    # random: standardized pAUC near 0.5
    ta = build_threshold_analysis(
        pl.DataFrame({**base, "y_prob": rng.random(n)}), META)
    assert ta["partial_auroc"]["pauc_standardized"] == pytest.approx(0.5, abs=0.1)


def test_threshold_suppression_when_few_flagged():
    meta = {**META, "clinical_threshold": 0.99}
    df = make_synth(n_stays=300)
    ta = build_threshold_analysis(df, meta)
    assert ta["label_metrics"] is None  # almost nobody above 0.99
    assert ta["action_metrics"] is None


def test_fairness_parity_at_clinical_threshold():
    from flair_benchmark.report.fairness import build_fairness
    df = make_synth(n_stays=500)
    f = build_fairness(df, META)
    pc = f["parity_at_clinical_threshold"]["sex_category"]
    assert set(pc) >= {"demographic_parity_gap", "equalized_odds_tpr_gap",
                       "equalized_odds_fpr_gap"}
    assert f["metadata"]["clinical_threshold"] == 0.20


def test_dca_clinical_marker():
    df = make_synth(n_stays=1000)
    dca = build_dca(df, META)
    assert dca["clinical_threshold"] == 0.20
    nb = dca["net_benefit_at_clinical_threshold"]
    assert nb["model"] is not None and nb["model"] > nb["treat_all"]
    assert dca["clinically_useful"] is True


# ---------------------------------------------------------------- end-to-end


def test_build_report_end_to_end(tmp_path):
    fake_task = SimpleNamespace(META=META, __file__=__file__)
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), fake_task, str(tmp_path / "report"))
    assert set(paths) == {"table1", "tripod_ai", "fairness", "curves",
                          "temporal", "dca", "threshold_analysis"}
    for path in paths.values():
        payload = json.loads(path.read_text())
        assert "metadata" in payload

    curves = json.loads(paths["curves"].read_text())
    assert "calibration_extended" in curves
    fairness = json.loads(paths["fairness"].read_text())
    assert "reference_group" in fairness["parity"]["sex_category"]


def test_build_report_cohort_join(tmp_path):
    fake_task = SimpleNamespace(META=META, __file__=__file__)
    df = make_synth()
    cohort_path = tmp_path / "cohort.parquet"
    df.write_parquet(cohort_path)
    # preds stripped of temporal context, but carrying prediction_id
    preds = df.drop(["window_number", "prediction_dttm"])
    preds_path = tmp_path / "preds.parquet"
    preds.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), fake_task, str(tmp_path / "report"),
                         cohort_path=str(cohort_path))
    temporal = json.loads(paths["temporal"].read_text())
    assert temporal["by_window"]["bins"]


def test_viz_requires_matplotlib_or_renders(tmp_path):
    fake_task = SimpleNamespace(META=META, __file__=__file__)
    df = make_synth()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    try:
        import matplotlib  # noqa: F401
        has_mpl = True
    except ImportError:
        has_mpl = False

    if has_mpl:
        paths = build_report(str(preds_path), fake_task, str(tmp_path / "report"),
                             viz=True)
        pngs = [p for p in paths.values() if p.suffix == ".png"]
        assert pngs and all(p.exists() for p in pngs)
    else:
        with pytest.raises(ImportError, match="flair-benchmark\\[viz\\]"):
            build_report(str(preds_path), fake_task, str(tmp_path / "report"),
                         viz=True)
