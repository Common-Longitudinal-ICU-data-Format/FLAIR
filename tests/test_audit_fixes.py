"""Regression tests for the evaluation-audit fixes (C1, C4, C5/C6, C9).

C2 (deterministic bootstrap CIs) is covered in test_report_continuous.py and
C3 (extubation-failure post-discharge-death censoring) in test_extubation_failure_label.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import _bin_age
from flair_benchmark.report.leadtime import build_leadtime
from flair_benchmark.report.operating_points import build_operating_points

META = {
    "name": "audit_fix_task",
    "task_class": "intervention",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-window",
    "task_type": "binary",
    "label_column": "label",
    "clinical_threshold": 0.15,
    "clinical_direction": "above",
    "report_mode": "continuous",
    "landmark_horizon": 3,
    "window_unit": "day",
}

BASE = datetime(2021, 1, 1)


def _strict_loads(blob: str):
    """json.loads that rejects Infinity/NaN like JS JSON.parse would."""
    def _reject(const):
        raise ValueError(f"non-finite JSON constant: {const}")
    return json.loads(blob, parse_constant=_reject)


# ------------------------------------------------------------------- C1


def test_discrimination_json_is_strict_json():
    """sklearn >= 1.3 puts inf first in roc_curve thresholds; the exported JSON
    must not contain the (RFC-8259-invalid) `Infinity` token."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.beta(2, 5, 400) + 0.3 * y, 0, 1)
    out = build_discrimination(pl.DataFrame({"label": y, "y_prob": p}),
                               {**META, "temporal_pattern": "episode"})
    blob = json.dumps(out, default=str)
    assert "Infinity" not in blob
    _strict_loads(blob)                                     # must not raise
    # the inf sentinel is exported as null, not dropped or clamped
    assert out["curves"]["roc"]["thresholds"][0] is None
    # and the rest of the array survives untouched
    assert all(isinstance(x, float) for x in out["curves"]["roc"]["thresholds"][1:])


# ------------------------------------------------------------------- C4


def test_bin_age_null_age_is_not_80_plus():
    df = pl.DataFrame({"age_at_admission": [30.0, None, 85.0, 79.9, 80.0]})
    bins = _bin_age(df)["age_bin"].to_list()
    assert bins == ["18-39", None, "80+", "65-79", "80+"]


# ------------------------------------------------------- C5 / C6 (below basis)


def _below_frame() -> pl.DataFrame:
    """12 ever-positive stays, 3 daily windows each. At window 2, stays 0-3 drop
    LOW (p=0.05, the action side of t=0.15) and stays 4-11 stay HIGH (p=0.9)."""
    rows = []
    for s in range(12):
        low_at_2 = s < 4
        probs = [0.9, 0.05 if low_at_2 else 0.9, 0.5]
        for w, pr in enumerate(probs, start=1):
            rows.append({
                "hospitalization_join_id": f"s{s}",
                "prediction_dttm": BASE + timedelta(days=w - 1),
                "y_prob": pr, "label": 1, "window_number": w,
            })
    return pl.DataFrame(rows)


BELOW = {**META, "clinical_direction": "below", "landmark_horizon": 3}


def test_continuous_confusion_is_alarm_basis_on_below_task():
    """Sensitivity must count stays that ever crossed ABOVE the threshold (all
    12 here), never 'the model said safe AND the event happened'."""
    op = build_operating_points(_below_frame(), BELOW)
    clinical = next(e for e in op["by_threshold"] if e["is_clinical"])
    assert clinical["metrics"]["sensitivity"] == 1.0     # 12/12 ever-alarmed
    # action-basis scoring would have booked 4/12 = 0.3333 here


def test_continuous_sub_analyses_stay_action_basis_on_below_task():
    """The timing arms describe the ACT (fire on LOW risk): 4 of 12 positive
    stays ever dropped below t."""
    df = _below_frame().with_columns(pl.lit(None, dtype=pl.Datetime)
                                     .alias("first_abx_dttm"))
    meta = {**BELOW, "intervention": {"column": "first_abx_dttm",
                                      "display": "treatment"}}
    op = build_operating_points(df, meta)
    clinical = next(e for e in op["by_threshold"] if e["is_clinical"])
    # sub_rates is now an alarm-COMPOSITION share: only positive stays exist, so of
    # the (4) stays that fire on the action basis, 100% are the positive arm.
    # (The 4-of-12 action-basis count itself is asserted in
    # test_leadtime_first_alert_is_action_basis on this same frame.)
    assert clinical["sub_rates"]["positive_share"] == pytest.approx(1.0)
    assert clinical["sub_rates"]["untreated_misfire_share"] is None   # empty arm


def test_leadtime_first_alert_is_action_basis():
    """The day-level sensitivity curve was removed; the first-alert NUMBER stays,
    and it is scored on the action basis (fire on LOW risk for a 'below' task)."""
    lt = build_leadtime(_below_frame(), BELOW)
    assert "curve" not in lt and "trajectory" not in lt
    # action basis: 4 of 12 stays ever fired (dropped below t)
    assert lt["metrics"]["pct_caught"] == pytest.approx(4 / 12, abs=1e-4)
    assert "action" in lt["metadata"]["first_alert_basis"]


# ------------------------------------------------------------------- C9


def test_every_live_task_owns_its_clinical_policy_inline():
    """Policy is no longer centralized in _constants.py — each task's META must
    self-contain the full clinical policy (threshold, action, direction,
    report_mode), value-validated by get_task."""
    from flair_benchmark.tasks import get_task, list_tasks

    for name in list_tasks():
        meta = get_task(name).META
        if meta.get("task_type") != "binary":
            continue
        for key in ("clinical_threshold", "clinical_action",
                    "clinical_direction", "report_mode"):
            assert key in meta, (name, key)
        assert 0 < meta["clinical_threshold"] < 1, name
        assert meta["clinical_direction"] in ("above", "below"), name


def test_finalize_cohort_always_stamps_a_feature_cutoff():
    """feature_cutoff_dttm is a cohort CONTRACT column, not a task option.

    Feature extraction reads a cohort parquet and never imports the task module
    (`flair fe-wide --task` is a path), so the cutoff has to travel as data. If it
    were optional, features/_scope.py could not tell a task that wants the default
    from a stale cohort that predates the column — and would have to guess.
    """
    from flair_benchmark.tasks._finalize import finalize_cohort

    df = pl.DataFrame({
        "hospitalization_id": ["h1", "h2"],
        "hospitalization_join_id": ["e1", "e2"],
        "patient_id": ["p1", "p2"],
        "prediction_dttm": [datetime(2020, 1, 1, 7), datetime(2020, 1, 2, 7)],
    })
    cols = ["hospitalization_id", "hospitalization_join_id", "prediction_dttm"]

    out = finalize_cohort(df, cols, {"prediction_unit": "one-per-stay"})
    assert "feature_cutoff_dttm" in out.columns
    # default is the inclusive cutoff — unchanged behaviour for every task but T3
    assert out["feature_cutoff_dttm"].to_list() == out["prediction_dttm"].to_list()

    # a task that sets its own keeps it, even without listing it in OUTPUT_COLS
    strict = df.with_columns(
        (pl.col("prediction_dttm") - pl.duration(microseconds=1))
        .alias("feature_cutoff_dttm"))
    kept = finalize_cohort(strict, cols, {"prediction_unit": "one-per-stay"})
    assert kept["feature_cutoff_dttm"].to_list() != kept["prediction_dttm"].to_list()


def test_at_least_one_live_task_acts_on_low_risk():
    """A shipped 'below' task is what keeps the below branches honest.

    The report layer carries a full act-on-LOW-risk path (report/_basis.py's
    action/decision bases, dca.py's complement frame, the dense _HIGH threshold
    tail, _viz.py's decision axis). It once had no live task, was exercised only
    by synthetic METAs, and quietly accumulated defects — a mirrored scorecard
    panel, a dead focus_range, and two files disagreeing about who was acted on.
    If this assert ever fails, those branches are unexercised again.
    """
    from flair_benchmark.tasks import get_task, list_tasks

    directions = {n: get_task(n).META.get("clinical_direction")
                  for n in list_tasks()
                  if get_task(n).META.get("task_type") == "binary"}
    assert "below" in directions.values(), directions
    assert "above" in directions.values(), directions


def _kpi_frame(n_stays: int, seed: int = 3) -> tuple[pl.DataFrame, list[dict]]:
    """Continuous frame + matching scored-landmark entries for kpi CI tests."""
    rng = np.random.default_rng(seed)
    rows, entries = [], []
    for w in (1, 2, 3):
        y = (rng.uniform(0, 1, n_stays) < 0.3).astype(int)
        p = np.clip(rng.beta(2, 5, n_stays) + 0.3 * y, 0, 1)
        for s in range(n_stays):
            rows.append({"hospitalization_join_id": f"s{s}", "window_number": w,
                         "label": int(y[s]), "y_prob": float(p[s])})
        from sklearn.metrics import roc_auc_score
        entries.append({"hospitalization_time": w, "n": n_stays,
                        "prevalence": float(y.mean()),
                        "metrics": {"auroc": float(roc_auc_score(y, p))}})
    return pl.DataFrame(rows), entries


def test_pooled_auroc_ci_is_stay_bootstrap():
    """C7: the pooled dynamic AUROC CI comes from a stay-level pooled bootstrap
    (no more [auc, auc] substitution), is null without the raw frame, brackets
    the point estimate, and narrows with cohort size."""
    from flair_benchmark.report.kpi import (
        _pooled_auroc_stay_bootstrap_ci,
        _pooled_dynamic_auroc,
        build_kpi,
    )

    meta = {**META, "label_column": "label"}
    df_small, e_small = _kpi_frame(80)
    df_big, e_big = _kpi_frame(640)

    point, _ = _pooled_dynamic_auroc(e_small)
    ci_small = _pooled_auroc_stay_bootstrap_ci(df_small, meta, e_small)
    ci_big = _pooled_auroc_stay_bootstrap_ci(df_big, meta, e_big)
    assert ci_small[0] <= point <= ci_small[1]
    assert (ci_big[1] - ci_big[0]) < (ci_small[1] - ci_small[0])

    # per-landmark CIs are NOT consulted — a missing one cannot shrink the pool
    e_noci = [dict(e, metrics={**e["metrics"], "auroc_ci": None}) for e in e_small]
    assert _pooled_auroc_stay_bootstrap_ci(df_small, meta, e_noci) == ci_small

    # pure-aggregation fallback (no raw frame) → null CI, never a fake interval
    kpi = build_kpi({}, {"discrimination": e_small}, {})
    assert kpi["pooled_dynamic_auroc"] == point
    assert kpi["pooled_dynamic_auroc_ci"] is None


def test_mimic_split_is_pinned_forever():
    """C8: the MIMIC random split must never move again.

    The deal is ordered by sha256 of the stay id (version-stable), replacing
    pl.Expr.hash whose values polars documents as unstable across releases —
    an upgrade would have silently re-dealt train/test (leakage for any model
    trained on the old deal). This fixture freezes the expected assignment; if
    it ever fails, the split changed and every existing MIMIC result is void.
    """
    from flair_benchmark._split import assign_split

    ids = [f"g{i:02d}" for i in range(20)]
    df = pl.DataFrame({
        "hospitalization_join_id": ids,
        "admission_dttm": [BASE] * 20,
    })
    out = assign_split(df, "mimic", None, None)
    split = dict(zip(out["hospitalization_join_id"].to_list(),
                     out["split"].to_list()))
    assert sorted(k for k, v in split.items() if v == "test") == [
        "g04", "g06", "g09", "g10", "g13"]
    assert sum(v == "train" for v in split.values()) == 15


def test_comorbidity_join_coalesces_keys():
    """D5: elix-only rows must keep their hospitalization_id (full-coalesce
    join), with no stray _right column entering the cohort_scores cache."""
    from flair_benchmark.report import _comorbidity as comorb

    cci = pl.DataFrame({"hospitalization_id": ["a", "b"],
                        "charlson_score": [1.0, 2.0]})
    elix = pl.DataFrame({"hospitalization_id": ["b", "c"],
                         "elix_score": [3.0, 4.0]})
    out = cci.join(elix, on="hospitalization_id", how="full", coalesce=True)
    assert set(out.columns) == {"hospitalization_id", "charlson_score",
                                "elix_score"}
    row_c = out.filter(pl.col("hospitalization_id") == "c")
    assert row_c.height == 1 and row_c["elix_score"][0] == 4.0
    # and the module itself uses the coalescing join
    import inspect
    src = inspect.getsource(comorb)
    assert 'how="full"' in src and "coalesce=True" in src


def test_read_clif_config_requires_site(tmp_path):
    """D6: a missing 'site' must fail loudly — a defaulted 'unknown' silently
    applies the DATA_CUTOFF year filter to MIMIC's shifted dates and empties
    the cohort."""
    from flair_benchmark._clif import read_clif_config

    cfg = tmp_path / "clif_config.json"
    cfg.write_text(json.dumps({"data_directory": "/tmp/x", "filetype": "parquet"}))
    with pytest.raises(ValueError, match="site"):
        read_clif_config(str(cfg))

    cfg.write_text(json.dumps({"site": "mimic", "data_directory": "/tmp/x",
                               "filetype": "parquet"}))
    assert read_clif_config(str(cfg))["site"] == "mimic"


def test_validate_taxonomy_rejects_bad_policy_values():
    """With policy inline in META, the guard validates VALUES (not drift vs a
    central dict): off-grid threshold, bad direction, and name/filename mismatch
    all raise."""
    from flair_benchmark.tasks import _validate_taxonomy

    name = "icu_daily_ltach"
    good = {"name": name, "task_type": "binary", "task_class": "outcome",
            "temporal_pattern": "continuous", "prediction_unit": "one-per-window",
            "clinical_threshold": 0.5, "clinical_action": "perform early tracheostomy",
            "clinical_direction": "above", "report_mode": "continuous",
            "landmark_horizon": 30}
    _validate_taxonomy(name, good)                          # passes

    off_grid = {**good, "clinical_threshold": 0.505}        # not a 0.01 multiple
    with pytest.raises(ValueError, match="multiple of 0.01"):
        _validate_taxonomy(name, off_grid)

    bad_dir = {**good, "clinical_direction": "sideways"}
    with pytest.raises(ValueError, match="clinical_direction"):
        _validate_taxonomy(name, bad_dir)

    wrong_name = {**good, "name": "icu_readmission"}
    with pytest.raises(ValueError, match="module filename"):
        _validate_taxonomy(name, wrong_name)
