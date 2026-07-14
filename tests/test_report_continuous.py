"""Synthetic-data tests for the continuous report (peak + landmark + leadtime
+ alert-vs-intervention timing in ONE bundle).

No CLIF data needed. We build a multi-row-per-stay frame that mimics a windowed
task (task4-style): each stay has several hourly windows; a positive stay is
labeled 1 only in the last `HORIZON` windows (an onset that ends the stay), with
y_prob rising toward it. This exercises:

  - peak section : collapse to stay-peak risk -> discrimination + fairness + NNE
  - landmark     : META['landmark_horizon'] FORWARD risk-set slices at
                   hospitalization time 1..horizon (window_number == t)
  - leadtime     : per-time sensitivity + median first-alert hospitalization time
  - operating_points : per-threshold (clinical ± 0.05) alert × label confusion
                   matrix + optional per-threshold sub-analyses (label_positive
                   hours-before-treatment, label_negative_treated Δ,
                   label_negative_untreated misfire count)
  - n<10 whole-landmark suppression on the deep (long-stay-only) landmarks
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from typer.testing import CliRunner

from flair_benchmark.cli import app
from flair_benchmark.report.continuous import build_continuous_report
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.landmark import build_landmark_report
from flair_benchmark.report.leadtime import build_leadtime
from flair_benchmark.report.operating_points import build_operating_points
from flair_benchmark.report.peak import build_peak_report
from flair_benchmark.tasks._finalize import finalize_cohort

runner = CliRunner()

HORIZON = 6
N_LANDMARKS = 12   # this synthetic task's horizon (stay geometry matches it)
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
    "report_mode": "continuous",
    "landmark_horizon": N_LANDMARKS,
    "window_unit": "hour",
    "prediction_window": [0, HORIZON],
}
META_IV = {**META,
           "intervention": {"column": "first_abx_dttm", "display": "treatment"}}

BASE = datetime(2021, 1, 1, 0, 0)


def make_synth_continuous(n_stays: int = 240, seed: int = 0) -> pl.DataFrame:
    """Multi-row-per-stay frame. A handful of stays are long (>=12 windows) so
    deep landmarks fall below the n<10 suppression floor."""
    rng = np.random.default_rng(seed)
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
                "hospitalization_join_id": f"h{i}",
                "hospitalization_id": f"h{i}",
                "prediction_dttm": BASE + timedelta(hours=w),
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
    cols = ["hospitalization_id", "hospitalization_join_id",
            "prediction_dttm", "label", "split",
            "age_at_admission", "sex_category", "race_category",
            "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(df, cols, META)


def make_alert_frame() -> pl.DataFrame:
    """Deterministic frame for alert-vs-intervention arms (threshold = 0.20).

    All stays have 10 hourly windows starting at BASE. Three groups of 20:
      positives         : first alert at hour 3, abx at hour 5 -> delta = -2h,
                          every positive fired before antibiotics.
      neg with abx      : abx at hour 4; half first-alert at hour 2 (before,
                          delta = -2h), half never fire.
      neg without abx   : half fire (hour 6), half never -> false alarms = 0.5.
    """
    rows = []
    n_win = 10

    def stay(i: int, label_last6: bool, alert_at: int | None, abx_at: int | None):
        abx = BASE + timedelta(hours=abx_at) if abx_at is not None else None
        for w in range(n_win):
            from_end = (n_win - 1) - w
            rows.append({
                "patient_id": f"p{i}", "hospitalization_join_id": f"h{i}",
                "hospitalization_id": f"h{i}",
                "prediction_dttm": BASE + timedelta(hours=w),
                "label": int(label_last6 and from_end < HORIZON),
                "split": "test",
                "age_at_admission": 60, "sex_category": "Female",
                "race_category": "White", "ethnicity_category": "Non-Hispanic",
                "y_prob": 0.9 if (alert_at is not None and w >= alert_at) else 0.05,
                "first_abx_dttm": abx,
            })

    i = 0
    for _ in range(20):                                    # positives
        stay(i, True, alert_at=3, abx_at=5); i += 1
    for k in range(20):                                    # negatives w/ abx
        stay(i, False, alert_at=2 if k < 10 else None, abx_at=4); i += 1
    for k in range(20):                                    # negatives w/o abx
        stay(i, False, alert_at=6 if k < 10 else None, abx_at=None); i += 1

    df = pl.DataFrame(rows)
    cols = ["hospitalization_id", "hospitalization_join_id",
            "prediction_dttm", "label", "split",
            "age_at_admission", "sex_category", "race_category",
            "ethnicity_category", "first_abx_dttm", "y_prob"]
    return finalize_cohort(df, cols, META_IV)


def _fake_task(meta: dict = META):
    return SimpleNamespace(META=meta, __file__=__file__)


# ----------------------------------------------------------------- peak section


def test_peak_emits_discrimination_and_fairness_only():
    df = make_synth_continuous()
    rep = build_peak_report(df, META)
    assert set(rep) == {"discrimination", "fairness"}
    # peak deliberately drops calibration + dca (peak-calibration trap)
    assert "calibration" not in rep and "dca" not in rep
    assert rep["discrimination"]["metadata"]["peak_calibration_omitted"] is True
    assert rep["discrimination"]["metadata"]["section"] == "peak"
    # one row per stay after the collapse
    assert rep["discrimination"]["metadata"]["n_stays"] == df["hospitalization_join_id"].n_unique()


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
        assert len(pillar) == META["landmark_horizon"]

    # forward landmark 1 = every stay's FIRST window (all stays still at risk)
    # -> exactly one row per stay; a true KM at-risk count
    n_stays = df["hospitalization_join_id"].n_unique()
    lm0 = rep["discrimination"][0]
    assert lm0["hospitalization_time"] == 1
    assert lm0["n"] == n_stays
    assert lm0["window_unit"] == "hour"
    assert rep["discrimination"][-1]["hospitalization_time"] == META["landmark_horizon"]


def test_landmark_horizon_from_meta_and_override():
    df = make_synth_continuous()
    # explicit override beats META
    rep = build_landmark_report(df, META, n_landmarks=3)
    assert len(rep["discrimination"]) == 3
    # META without landmark_horizon -> clear error
    meta = {k: v for k, v in META.items() if k != "landmark_horizon"}
    with pytest.raises(ValueError, match="landmark_horizon"):
        build_landmark_report(df, meta)
    with pytest.raises(ValueError, match="landmark_horizon"):
        build_leadtime(df, meta)


def test_landmark_day_unit_tagged():
    df = make_synth_continuous()
    meta = {**META, "window_unit": "day"}
    lm0 = build_landmark_report(df, meta, n_landmarks=2)["discrimination"][0]
    assert lm0["window_unit"] == "day"
    assert lm0["hospitalization_time"] == 1


def test_landmark_deep_landmarks_suppressed():
    df = make_synth_continuous()
    rep = build_landmark_report(df, META)
    # only ~6 stays are still at risk at hospitalization hour 12 -> below the
    # n<10 floor (the KM at-risk count shrinks as stays end)
    deep = rep["discrimination"][META["landmark_horizon"] - 1]
    assert deep["suppressed"] is True
    assert deep["n"] == "<10"
    # a suppressed entry carries no metrics
    assert "metrics" not in deep


def _converged_frame(n_stays: int = 400, seed: int = 3) -> pl.DataFrame:
    """Multi-window frame whose subgroup fits CONVERGE (labels overlap y_prob, no
    perfect separation), so the report is bit-reproducible across BLAS/OpenMP
    thread counts. (A non-converged logistic fit on a tiny perfectly-separable
    subgroup is platform-thread-sensitive — see report/landmark._build_deterministic
    and report/__init__._blas_single_thread; that is why this test avoids one.)"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        n_win = int(rng.integers(3, 6))
        for w in range(n_win):
            p = float(rng.uniform(0.12, 0.88))         # spread, overlapping
            rows.append({
                "patient_id": f"p{i}", "hospitalization_join_id": f"h{i}",
                "hospitalization_id": f"h{i}",
                "prediction_dttm": BASE + timedelta(hours=w),
                "label": int(rng.random() < p),        # noisy -> no separation
                "split": "test",
                "age_at_admission": int(rng.integers(30, 85)),
                "sex_category": "Female" if rng.random() < 0.5 else "Male",
                "race_category": "White" if rng.random() < 0.6 else "Black",
                "ethnicity_category": "Non-Hispanic",
                "y_prob": p, "y_pred": int(p >= 0.5),
            })
    cols = ["hospitalization_id", "hospitalization_join_id", "prediction_dttm",
            "label", "split", "age_at_admission", "sex_category",
            "race_category", "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(pl.DataFrame(rows), cols, META)


def test_landmark_parallel_matches_serial():
    # the ProcessPoolExecutor build must be byte-identical to the serial build
    # (fixed bootstrap seeds, order-preserving assembly, single-thread numerics).
    df = _converged_frame()
    serial = build_landmark_report(df, META, max_workers=1)
    parallel = build_landmark_report(df, META, max_workers=4)
    assert json.dumps(serial, default=str, sort_keys=True) == \
        json.dumps(parallel, default=str, sort_keys=True)


def test_landmark_forward_semantics():
    # make_alert_frame labels only the last 6 of 10 windows, so the FORWARD
    # risk sets at hospitalization hours 1-4 are all-negative (no event within
    # the horizon yet) -> undefined stubs; hours 5+ have contrast and score.
    rep = build_landmark_report(make_alert_frame(), META_IV)
    disc = rep["discrimination"]
    for t in range(4):
        assert disc[t]["undefined"] is True
    assert "metrics" in disc[5]
    # KM at-risk: all 60 stays have 10 windows -> n = 60 through hour 10,
    # then nobody is left at hour 11+
    assert disc[0]["n"] == 60 and disc[9]["n"] == 60
    assert disc[10]["suppressed"] is True


# ----------------------------------------------------------------- leadtime


def test_leadtime_curve_and_median():
    df = make_synth_continuous()
    lt = build_leadtime(df, META)
    assert len(lt["curve"]) == META["landmark_horizon"]
    assert lt["metadata"]["window_unit"] == "hour"
    m = lt["metrics"]
    assert 0.0 <= m["pct_caught"] <= 1.0
    assert m["n_caught"] <= m["n_positive_stays"]
    # forward view: first alert at hospitalization time >= 1 (window_number)
    if m["median_first_alert_hospitalization_time"] is not None:
        assert m["median_first_alert_hospitalization_time"] >= 1
        assert len(m["median_first_alert_hospitalization_time_ci"]) == 2
    # curve points are tagged 1..horizon
    assert lt["curve"][0]["hospitalization_time"] == 1
    assert lt["curve"][-1]["hospitalization_time"] == META["landmark_horizon"]


def test_leadtime_risk_trajectory_by_outcome():
    df = make_synth_continuous()
    traj = build_leadtime(df, META)["trajectory"]
    assert len(traj) == META["landmark_horizon"]
    shallow = [it for it in traj
               if not it.get("suppressed")
               and it["mean_prob_positive"] is not None
               and it["mean_prob_negative"] is not None]
    assert shallow  # at least some landmarks have both arms populated
    for it in shallow:
        # positive-outcome stays carry higher predicted risk than negatives
        assert it["mean_prob_positive"] > it["mean_prob_negative"]
        assert "n" in it and "n_positive" in it and "n_negative" in it


# ---------------------------------------------------- operating points


def _clinical_entry(op):
    """The by_threshold entry at the clinical threshold (0.20 here)."""
    return next(e for e in op["by_threshold"] if e["threshold"] == 0.20)


def test_operating_points():
    op = build_operating_points(make_alert_frame(), META_IV)

    # three thresholds analyzed: clinical (0.20) and +/- 0.05
    assert op["metadata"]["thresholds"] == [0.15, 0.20, 0.25]
    assert len(op["by_threshold"]) == 3

    e = _clinical_entry(op)
    c = e["confusion"]
    # positives (20) all fire; 20 of 40 negatives fire -> FP 20, TN 20
    assert c["sensitivity"] == pytest.approx(1.0)
    assert c["specificity"] == pytest.approx(0.5)
    assert c["fpr"] == pytest.approx(1 - c["specificity"])       # true FPR
    assert c["ppv"] == pytest.approx(0.5)
    assert c["nne"] == pytest.approx(round(1.0 / c["ppv"], 1))   # NNE = 1/PPV
    assert c["tp"] == 20 and c["fp"] == 20 and c["tn"] == 20     # >=10, not suppressed
    for k in ("sensitivity_ci", "specificity_ci", "fpr_ci", "ppv_ci"):
        assert len(c[k]) == 2

    sub = e["sub_analyses"]
    # Sub-1: 20 positives, all fired; fired hr3, abx hr5 -> 2h before treatment
    pos = sub["label_positive"]
    assert pos["n_stays"] == 20 and pos["n_fired"] == 20 and pos["pct_fired"] == 1.0
    assert pos["hours_before_treatment"]["median"] == pytest.approx(2.0)
    # Sub-2: 20 treated negatives, 10 fired at hr2 (abx hr4) -> 2h before tx.
    # Same sign convention as Sub-1 (positive = fired before treatment).
    negt = sub["label_negative_treated"]
    assert negt["n"] == 20 and negt["n_fired"] == 10 and negt["pct_fired"] == 0.5
    assert negt["hours_before_treatment"]["median"] == pytest.approx(2.0)
    # Sub-3: 20 untreated negatives, 10 misfire -> rate 0.5, NO timing keys
    negu = sub["label_negative_untreated"]
    assert negu["n"] == 20 and negu["n_misfired"] == 10
    assert negu["misfire_rate"] == pytest.approx(0.5)
    assert "delta_hours" not in negu and "hours_before_treatment" not in negu

    off = op["intervention_offsets"]
    assert off["n"] == 40 and off["window_unit"] == "hour"
    # anchored on stay START: abx hr5 (pos) -> time 6; hr4 (neg) -> 5; mean 5.5
    assert off["mean_hospitalization_time"] == pytest.approx(5.5)


def test_operating_points_per_threshold_differs():
    # firing is threshold-based, so the sub-analysis fire counts move with t.
    # all alert rows use y_prob 0.9, non-alert 0.05, so thresholds 0.15/0.20/0.25
    # give identical firing here -- but the confusion counts are still emitted
    # once per threshold (3 entries), proving the per-threshold structure.
    op = build_operating_points(make_alert_frame(), META_IV)
    assert [e["threshold"] for e in op["by_threshold"]] == [0.15, 0.20, 0.25]
    for e in op["by_threshold"]:
        assert set(e) >= {"threshold", "confusion", "sub_analyses"}


def test_operating_points_small_arm_suppressed():
    # drop most treated negatives -> that sub-arm falls under the n<10 floor
    df = make_alert_frame().filter(
        ~pl.col("hospitalization_join_id").is_in([f"h{i}" for i in range(21, 40)]))
    sub = _clinical_entry(build_operating_points(df, META_IV))["sub_analyses"]
    assert sub["label_negative_treated"]["suppressed"] is True
    assert sub["label_negative_treated"]["n"] == "<10"


def test_operating_points_no_intervention():
    # confusion matrix is MANDATORY; sub-analyses are omitted without a config.
    op = build_operating_points(make_synth_continuous(), META)
    assert len(op["by_threshold"]) == 3
    for e in op["by_threshold"]:
        assert "confusion" in e
        assert "sub_analyses" not in e
    assert "no intervention" in op["metadata"]["sub_analyses_reason"]
    assert op["intervention_offsets"] is None


def test_operating_points_generic_display():
    # the report layer is task-agnostic: user-facing text is the config `display`
    # ("treatment"); no sepsis/antibiotic/onset wording leaks into the output.
    # (The cohort column name first_abx_dttm is a task-supplied config value and
    # may appear in the metadata echo — that is not report-layer wording.)
    op = build_operating_points(make_alert_frame(), META_IV)
    blob = json.dumps(op, default=str).lower()
    for word in ("antibiotic", "sepsis", "onset"):
        assert word not in blob


def test_operating_points_direction_below():
    # with direction=below the alert fires on LOW risk: the never-fire stays
    # (y_prob 0.05 < 0.20) now alert -> untreated negatives all misfire.
    op = build_operating_points(make_alert_frame(),
                                {**META_IV, "clinical_direction": "below"})
    sub = _clinical_entry(op)["sub_analyses"]
    assert sub["label_negative_untreated"]["misfire_rate"] == pytest.approx(1.0)


# ----------------------------------------------------------------- end-to-end


def test_build_report_continuous_end_to_end(tmp_path):
    df = make_synth_continuous()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    # mode omitted -> inferred 'continuous' from temporal_pattern
    paths = build_report(str(preds_path), _fake_task(), str(tmp_path / "ct"),
                         viz=False)
    kinds = {k for k in paths if not k.endswith("_png")}
    assert kinds == {"discrimination", "calibration", "dca", "fairness",
                     "leadtime", "operating_points", "kpi"}

    disc = json.loads(paths["discrimination"].read_text())
    assert disc["metadata"]["report_mode"] == "continuous"
    assert disc["peak"]["metrics"]["operating_point"]["nne"] is not None
    assert len(disc["landmarks"]) == META["landmark_horizon"]

    cal = json.loads(paths["calibration"].read_text())
    assert "peak" not in cal
    assert "peak-calibration trap" in cal["metadata"]["peak_omitted_reason"]
    assert len(cal["landmarks"]) == META["landmark_horizon"]

    # operating points: confusion matrix present (mandatory), no sub-analyses
    op = json.loads(paths["operating_points"].read_text())
    assert len(op["by_threshold"]) == 3
    assert "confusion" in op["by_threshold"][0]
    assert "sub_analyses" not in op["by_threshold"][0]
    assert "no intervention" in op["metadata"]["sub_analyses_reason"]


def test_build_report_intervention_column_required(tmp_path):
    df = make_synth_continuous()   # has no first_abx_dttm column
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    with pytest.raises(ValueError, match="first_abx_dttm"):
        build_report(str(preds_path), _fake_task(META_IV),
                     str(tmp_path / "iv"), viz=False)


def test_build_report_intervention_end_to_end(tmp_path):
    df = make_alert_frame()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(META_IV),
                         str(tmp_path / "iv"), viz=False)
    op = json.loads(paths["operating_points"].read_text())
    clinical = next(e for e in op["by_threshold"] if e["threshold"] == 0.20)
    sub = clinical["sub_analyses"]
    assert sub["label_positive"]["pct_fired"] == 1.0
    assert op["intervention_offsets"]["mean_hospitalization_time"] == pytest.approx(5.5)
    kpi = json.loads(paths["kpi"].read_text())
    assert kpi["operating_points"]["sub_analyses"][
        "median_hours_before_treatment_treated_negative"] == pytest.approx(2.0)


def test_continuous_viz_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    df = make_alert_frame()
    preds_path = tmp_path / "preds.parquet"
    df.write_parquet(preds_path)

    from flair_benchmark.report import build_report
    paths = build_report(str(preds_path), _fake_task(META_IV),
                         str(tmp_path / "ct"), viz=True)
    pngs = {p.name for k, p in paths.items() if k.endswith("_png")}
    assert "peak_roc_pr.png" in pngs
    assert "peak_fairness_auroc.png" in pngs
    assert "landmark_auc_by_hospital_time.png" in pngs
    assert "landmark_risk_trajectory.png" in pngs
    assert "landmark_reliability.png" in pngs
    assert "sensitivity_by_hospital_time.png" in pngs
    assert "operating_points_confusion.png" in pngs
    assert "operating_points_subanalyses.png" in pngs
    # episodic-only PNGs must not leak into the continuous bundle
    assert "calibration.png" not in pngs and "dca.png" not in pngs
    assert all(p.exists() for k, p in paths.items() if k.endswith("_png"))


def test_continuous_composer_sections():
    rep = build_continuous_report(make_synth_continuous(), META)
    assert set(rep) == {"discrimination", "calibration", "dca", "fairness",
                        "leadtime", "operating_points", "kpi"}
    assert "peak" in rep["discrimination"] and "landmarks" in rep["discrimination"]
    assert "peak" in rep["fairness"] and "landmarks" in rep["fairness"]
    assert "peak" not in rep["calibration"] and "peak" not in rep["dca"]
    assert "by_threshold" in rep["operating_points"]


# ----------------------------------------------------------------- KPI


def test_kpi_scoreboard():
    rep = build_continuous_report(make_synth_continuous(), META)
    kpi = rep["kpi"]

    # pooled dynamic AUC (Heagerty-style) == hand-computed pair-weighted mean
    num = den = 0.0
    for e in rep["discrimination"]["landmarks"]:
        if e.get("suppressed") or e.get("undefined"):
            continue
        auc, prev, n = e["metrics"]["auroc"], e["prevalence"], e["n"]
        if auc is None:
            continue
        cases = n * prev
        w = cases * (n - cases)
        num += auc * w
        den += w
    assert kpi["pooled_dynamic_auroc"] == pytest.approx(num / den, abs=1e-4)
    assert 0.5 <= kpi["pooled_dynamic_auroc"] <= 1.0
    lo, hi = kpi["pooled_dynamic_auroc_ci"]
    assert lo <= kpi["pooled_dynamic_auroc"] <= hi

    # peak block mirrors the peak section (incl. NNE)
    peak_op = rep["discrimination"]["peak"]["metrics"]["operating_point"]
    assert kpi["peak"]["nne"] == peak_op["nne"]
    assert kpi["peak"]["auroc"] == rep["discrimination"]["peak"]["metrics"]["auroc"]

    # calibration at the earliest scored hospitalization time
    cal = kpi["calibration"]
    assert cal["at_hospitalization_time"] == 1
    assert cal["slope"] is not None and len(cal["slope_range"]) == 2

    # leadtime block, 1-based
    assert kpi["leadtime"]["pct_caught"] == rep["leadtime"]["metrics"]["pct_caught"]
    # operating points block is ALWAYS present (confusion is mandatory); with no
    # intervention config its sub-analyses summary is None.
    assert kpi["operating_points"] is not None
    assert kpi["operating_points"]["sub_analyses"] is None
    assert len(kpi["operating_points"]["fpr_by_threshold"]) == 3
    assert kpi["metadata"]["horizon"] == META["landmark_horizon"]


def test_kpi_operating_points_with_intervention():
    rep = build_continuous_report(make_alert_frame(), META_IV)
    op = rep["kpi"]["operating_points"]
    assert op["clinical_threshold"] == 0.20
    assert op["fpr"] == pytest.approx(0.5)          # clinical-threshold true FPR
    sub = op["sub_analyses"]
    assert sub["pct_positive_fired"] == 1.0
    assert sub["median_hours_before_treatment_positive"] == pytest.approx(2.0)
    assert sub["median_hours_before_treatment_treated_negative"] == pytest.approx(2.0)
    assert sub["misfire_rate_untreated"] == pytest.approx(0.5)
    # forward view over ALL windows of ever-positive stays: alerts start at
    # window 4 of 10 -> median first-alert hospitalization time = 4
    assert rep["kpi"]["leadtime"]["median_first_alert_hospitalization_time"] == pytest.approx(4)


# ----------------------------------------------------------------- CLI


def test_cli_peak_and_landmark_modes_removed(tmp_path):
    df = make_synth_continuous().rename({"label": "label_mortality"})
    preds = tmp_path / "preds.parquet"
    df.write_parquet(preds)
    for old_mode in ("peak", "landmark"):
        res = runner.invoke(app, ["build-report", str(preds),
                                  "--task", "task1_icu_daily_mortality",
                                  "--out", str(tmp_path / "rep"),
                                  "--mode", old_mode, "--no-viz"])
        assert res.exit_code != 0
        assert "continuous" in str(res.exception)


# ------------------------------------------------- single-class guard (P1.1)


def test_discrimination_single_class_guard():
    n = 60
    df = pl.DataFrame({
        "label": [0] * n,                       # all-negative cohort, no contrast
        "y_prob": [i / n for i in range(n)],
        "y_pred": [0] * n,
    })
    d = build_discrimination(df, META)
    assert "reason" in d["metadata"]
    assert "metrics" not in d and "curves" not in d
    assert d["prevalence"] == 0.0


# ------------------------------------------------- direction == below (P2)


def _balanced_binary(seed: int = 0, n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.45 * y + 0.3 * rng.random(n), 0, 1)
    return pl.DataFrame({"label": y, "y_prob": p,
                         "y_pred": (p >= 0.2).astype(int)})


def test_below_direction_operating_action_and_dca():
    meta = {**META, "clinical_direction": "below", "clinical_action": "extubate"}
    df = _balanced_binary()

    op = build_discrimination(df, meta)["metrics"]["operating_point"]
    # classifier metrics stay in p>=t convention (direction-agnostic)
    assert op["direction"] == "below"
    assert op["sensitivity"] is not None and op["specificity"] is not None
    # the action block is what flips: act on the LOW-risk side
    assert op["action"]["acts_on"] == "y_prob < threshold"

    dca = build_dca(df, meta)
    assert dca["direction"] == "below"
    assert "complement" in dca["basis"]
    assert dca["clinical_threshold_original"] == 0.20
    # evaluated on the complement -> marker threshold is 1 - 0.20 = 0.80
    assert dca["metrics"]["clinical_threshold"] == pytest.approx(0.80)


def test_above_direction_dca_standard():
    dca = build_dca(_balanced_binary(), META)   # META direction == above
    assert dca["direction"] == "above"
    assert "standard" in dca["basis"]
    assert dca["metrics"]["clinical_threshold"] == 0.20
