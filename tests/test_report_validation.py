"""The preds-parquet contract and the orchestrator's guard rails.

These outlive any particular report schema: what a site is REQUIRED to hand in,
what the report refuses to score, and that the exported JSON is strict JSON.
The per-file formula tests live in test_hospitalization_level / test_landmark /
test_overall; the file list lives in test_report_bundle.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from flair_benchmark.report import build_report
from flair_benchmark.tasks._finalize import finalize_cohort

META = {
    "name": "synthetic_episodic_task",
    "display_name": "Synthetic Episodic Task",
    "task_class": "outcome",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    "label_column": "label",
    "clinical_threshold": 0.20,
    "clinical_action": "intervene",
    "clinical_direction": "above",
}


def make_synth(n_stays: int = 600, seed: int = 0) -> pl.DataFrame:
    """One row per stay: demographics + a calibrated y_prob."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stays):
        adm = datetime(int(rng.integers(2017, 2024)), int(rng.integers(2, 12)),
                       int(rng.integers(1, 28)), 7, 0)
        logit = -2.2 + 1.6 * rng.normal() + rng.normal(scale=0.3)
        prob = 1 / (1 + np.exp(-logit))
        rows.append({
            "patient_id": f"p{i}",
            "hospitalization_join_id": f"h{i}",
            "hospitalization_id": f"h{i}",
            "prediction_dttm": adm,
            "label": int(rng.random() < prob),
            "split": "test",
            "age_at_admission": int(rng.integers(18, 95)),
            "sex_category": "Female" if rng.random() < 0.5 else "Male",
            "race_category": "White",
            "ethnicity_category": "Non-Hispanic",
            "y_prob": float(prob),
            "y_pred": int(prob >= 0.5),
        })
    cols = ["hospitalization_id", "hospitalization_join_id", "prediction_dttm",
            "label", "split", "age_at_admission", "sex_category",
            "race_category", "ethnicity_category", "y_prob", "y_pred"]
    return finalize_cohort(pl.DataFrame(rows), cols, META)


def fake_task(**overrides):
    return SimpleNamespace(META={**META, **overrides}, __file__=__file__)


def write(tmp_path, df, name="preds.parquet"):
    path = tmp_path / name
    df.write_parquet(path)
    return str(path)


# ------------------------------------------------------------ the preds contract


def test_prediction_id_is_one_per_stay():
    df = make_synth(n_stays=20)
    assert df.height == df["hospitalization_join_id"].n_unique() == 20
    one = df.filter(pl.col("hospitalization_join_id") == "h0")
    assert one["window_number"][0] == 1
    assert one["prediction_id"][0] == "h0-h0-p0-1"


def test_prediction_id_is_required(tmp_path):
    path = write(tmp_path, make_synth(n_stays=50).drop("prediction_id"))
    with pytest.raises(ValueError, match="prediction_id"):
        build_report(path, fake_task(), str(tmp_path / "r"), mode="episodic")


def test_hospitalization_join_id_is_required(tmp_path):
    """The stitched encounter is the evaluation unit; the source
    hospitalization_id is audit-only and cannot stand in for it."""
    path = write(tmp_path, make_synth(n_stays=50).drop("hospitalization_join_id"))
    with pytest.raises(ValueError, match="hospitalization_join_id"):
        build_report(path, fake_task(), str(tmp_path / "r"), mode="episodic")


@pytest.mark.parametrize("bad,message", [
    (float("nan"), "NaN"),
    (1.5, r"\[0, 1\]"),
])
def test_y_prob_must_be_a_probability(tmp_path, bad, message):
    df = make_synth(n_stays=50)
    probs = df["y_prob"].to_list()
    probs[0] = bad
    df = df.with_columns(pl.Series("y_prob", probs))
    with pytest.raises(ValueError, match=message):
        build_report(write(tmp_path, df), fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_null_y_prob_is_rejected(tmp_path):
    df = make_synth(n_stays=50)
    probs = df["y_prob"].to_list()
    probs[0] = None
    df = df.with_columns(pl.Series("y_prob", probs, dtype=pl.Float64))
    with pytest.raises(ValueError, match="null"):
        build_report(write(tmp_path, df), fake_task(), str(tmp_path / "r"),
                     mode="episodic")


# --------------------------------------------------------------- split handling


def test_only_test_rows_are_ever_scored(tmp_path):
    """There is no split override: train rows are never scored, ever."""
    df = make_synth(n_stays=400).with_columns(
        pl.when(pl.int_range(pl.len()) < 200).then(pl.lit("train"))
        .otherwise(pl.lit("test")).alias("split"))
    paths = build_report(write(tmp_path, df), fake_task(), str(tmp_path / "r"),
                         mode="episodic")
    md = json.loads(paths["overall"].read_text())["metadata"]
    assert md["split"] == "test"
    assert md["n_stays"] == 200


def test_a_preds_file_with_no_test_rows_is_rejected(tmp_path):
    df = make_synth(n_stays=100).with_columns(pl.lit("train").alias("split"))
    with pytest.raises(ValueError, match="test"):
        build_report(write(tmp_path, df), fake_task(), str(tmp_path / "r"),
                     mode="episodic")


def test_one_encounter_in_both_splits_is_rejected(tmp_path):
    """The dangerous leakage case, caught before the test filter runs."""
    df = make_synth(n_stays=50)
    bad = pl.concat([df, df.head(1).with_columns(pl.lit("train").alias("split"))])
    with pytest.raises(ValueError, match="multiple splits"):
        build_report(write(tmp_path, bad), fake_task(), str(tmp_path / "r"),
                     mode="episodic")


# ------------------------------------------------------------------ mode guards


def test_episodic_mode_rejects_multi_row_per_stay(tmp_path):
    df = make_synth(n_stays=100)
    with pytest.raises(ValueError, match="one row per stay"):
        build_report(write(tmp_path, pl.concat([df, df])), fake_task(),
                     str(tmp_path / "r"), mode="episodic")


def test_continuous_mode_is_inferred_from_meta(tmp_path):
    path = write(tmp_path, make_synth())
    task = fake_task(temporal_pattern="continuous",
                     prediction_unit="one-per-window", landmark_horizon=2)
    paths = build_report(path, task, str(tmp_path / "r"))
    payload = json.loads(paths["landmark"].read_text())
    assert payload["metadata"]["report_mode"] == "continuous"
    assert len(payload["by_landmark"]) == 2


def test_continuous_mode_requires_a_landmark_horizon(tmp_path):
    path = write(tmp_path, make_synth())
    task = fake_task(temporal_pattern="continuous",
                     prediction_unit="one-per-window")
    with pytest.raises(ValueError, match="landmark_horizon"):
        build_report(path, task, str(tmp_path / "r"))


def test_the_retired_peak_and_landmark_modes_name_their_replacement(tmp_path):
    path = write(tmp_path, make_synth(n_stays=50))
    for mode in ("peak", "landmark"):
        with pytest.raises(ValueError, match="continuous"):
            build_report(path, fake_task(), str(tmp_path / "r"), mode=mode)


# ------------------------------------------------------------------ cohort join


def test_demographics_are_joined_from_the_cohort_when_absent_from_preds(tmp_path):
    df = make_synth()
    cohort = write(tmp_path, df, "cohort.parquet")
    stripped = df.drop(["age_at_admission", "sex_category", "race_category",
                        "ethnicity_category"])
    paths = build_report(write(tmp_path, stripped), fake_task(),
                         str(tmp_path / "r"), cohort_path=cohort,
                         mode="episodic")
    fair = json.loads(paths["hospitalization_level"].read_text())["fairness"]
    assert "sex_category" in fair["subgroups"]


def test_fairness_is_absent_rather_than_fatal_without_demographics(tmp_path):
    stripped = make_synth().drop(["age_at_admission", "sex_category",
                                  "race_category", "ethnicity_category"])
    paths = build_report(write(tmp_path, stripped), fake_task(),
                         str(tmp_path / "r"), mode="episodic")
    assert json.loads(
        paths["hospitalization_level"].read_text())["fairness"] is None


# ------------------------------------------------------------------ strict JSON


def test_exported_json_contains_no_infinity_or_nan(tmp_path):
    """`Infinity` and `NaN` are not valid JSON (RFC 8259) and break JS parsers.

    sklearn's roc_curve puts np.inf first in its threshold array, which is how
    this got shipped once already.
    """
    def reject(const):
        raise ValueError(f"non-finite JSON constant: {const}")

    paths = build_report(write(tmp_path, make_synth()), fake_task(),
                         str(tmp_path / "r"), mode="episodic")
    for name, path in paths.items():
        if name.endswith("_png"):
            continue
        json.loads(path.read_text(), parse_constant=reject)


def test_every_shipped_task_threshold_lands_on_the_sweep_grid():
    """The clinical policy must BE a grid row, or it cannot be flagged as one."""
    from flair_benchmark.report._sweep import SWEEP_GRID
    from flair_benchmark.tasks import get_task, list_tasks

    for name in list_tasks():
        t = get_task(name).META.get("clinical_threshold")
        if t is None:
            continue
        assert np.min(np.abs(SWEEP_GRID - round(float(t), 2))) < 1e-9, (
            f"task {name!r} ships clinical_threshold={t}, which is not a "
            f"multiple of 0.01 and so lands between sweep rows")
