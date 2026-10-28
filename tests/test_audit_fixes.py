"""Regression tests for the evaluation-audit fixes, outside the report layer.

Cohort splitting, comorbidity joins, config validation and the task-taxonomy
guard. The report-layer audit fixes moved to the per-file formula suites
(test_hospitalization_level, test_landmark, test_overall) and the contract suite
(test_report_validation) when the bundle collapsed to two JSONs.

C3 (extubation-failure post-discharge-death censoring) lives in
test_extubation_failure_label.py.
"""

from __future__ import annotations

import json
from datetime import datetime

import polars as pl
import pytest

META = {
    "name": "audit_fix_task",
    "task_class": "outcome",
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


def test_every_live_task_owns_its_clinical_policy_inline():
    """Each task's META self-contains the full clinical policy (threshold, action, direction,
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
    (`flair fe-meds --task` is a path), so the cutoff has to travel as data. If it
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
    action and decision bases, and the net-benefit column of the threshold
    sweep). It once had no live task, was exercised only by synthetic METAs, and
    quietly accumulated defects — including two files disagreeing about who was
    acted on at a tie. If this assert ever fails, those branches are
    unexercised again.
    """
    from flair_benchmark.tasks import get_task, list_tasks

    directions = {n: get_task(n).META.get("clinical_direction")
                  for n in list_tasks()
                  if get_task(n).META.get("task_type") == "binary"}
    assert "below" in directions.values(), directions
    assert "above" in directions.values(), directions


def test_mimic_split_is_pinned_forever():
    """The MIMIC split is fixed: encounters ordered by sha256 of the join id.

    This fixture freezes the exact assignment. It must never change; if it fails,
    the split changed and every MIMIC result is void.
    """
    from flair_benchmark.cohort.stitch import assign_split

    ids = [f"g{i:02d}" for i in range(20)]
    df = pl.DataFrame({
        "hospitalization_join_id": ids,
        "admission_dttm": [BASE] * 20,
    })
    out = assign_split(df, "mimic")
    split = dict(zip(out["hospitalization_join_id"].to_list(),
                     out["split"].to_list()))
    assert sorted(k for k, v in split.items() if v == "test") == [
        "g04", "g06", "g09", "g10", "g13"]
    assert sum(v == "train" for v in split.values()) == 15


def test_comorbidity_join_coalesces_keys():
    """D5: elix-only rows must keep their hospitalization_id (full-coalesce
    join), with no stray _right column entering the cohort_scores cache."""
    from flair_benchmark.cohort.table1 import compute_cci_elix

    cci = pl.DataFrame({"hospitalization_id": ["a", "b"],
                        "charlson_score": [1.0, 2.0]})
    elix = pl.DataFrame({"hospitalization_id": ["b", "c"],
                         "elix_score": [3.0, 4.0]})
    out = cci.join(elix, on="hospitalization_id", how="full", coalesce=True)
    assert set(out.columns) == {"hospitalization_id", "charlson_score",
                                "elix_score"}
    row_c = out.filter(pl.col("hospitalization_id") == "c")
    assert row_c.height == 1 and row_c["elix_score"][0] == 4.0
    # and the real implementation uses the coalescing join
    import inspect
    src = inspect.getsource(compute_cci_elix)
    assert 'how="full"' in src and "coalesce=True" in src


def test_read_clif_config_requires_site(tmp_path):
    """D6: a missing 'site' must fail loudly — a defaulted 'unknown' silently
    applies the benchmark-window filter to MIMIC's shifted dates and empties
    the cohort."""
    from flair_benchmark.cohort.clif import read_clif_config

    cfg = tmp_path / "clif_config.json"
    cfg.write_text(json.dumps({"data_directory": "/tmp/x", "filetype": "parquet"}))
    with pytest.raises(ValueError, match="site"):
        read_clif_config(str(cfg))

    cfg.write_text(json.dumps({"site": "mimic", "data_directory": "/tmp/x",
                               "filetype": "parquet", "timezone": "US/Eastern"}))
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


def test_task_digest_covers_constants_file(monkeypatch, tmp_path):
    """The whole of constants.py is hashed: changing a shared constant (here the
    stitch gap) re-cuts every cohort, so it must move the digest."""
    from pathlib import Path

    from flair_benchmark import constants
    from flair_benchmark.tasks import get_task

    task = get_task("icu_readmission")
    before = constants.task_script_sha256(task)
    edited = tmp_path / "constants.py"
    src = Path(constants.__file__).read_text()
    assert "STITCH_TIME_INTERVAL_HOURS = 6" in src
    edited.write_text(src.replace("STITCH_TIME_INTERVAL_HOURS = 6", "STITCH_TIME_INTERVAL_HOURS = 12"))
    monkeypatch.setattr(constants, "__file__", str(edited))
    assert constants.task_script_sha256(task) != before


def test_dependency_versions_recorded():
    from flair_benchmark.constants import dependency_versions

    v = dependency_versions()
    assert set(v) == {"clifpy_version", "polars_version"}
    assert all(isinstance(x, str) and x for x in v.values())
