"""Regression tests for the code-review findings fixed on 2026-07-26.

Every bug here was SILENT — wrong numbers, never an exception — so each test
asserts the specific wrong behaviour is gone rather than just that the code runs.
Numbered to match the findings list in plan.md's appendix.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from flair_benchmark.cohort.clif import _id_filter
from flair_benchmark.cohort.events import (
    compute_readmission_timing,
    detect_imv_episodes,
)
from flair_benchmark.cohort.stitch import _split_cutoff, assign_split
from flair_benchmark.meds_etl._emit import emit_events
from flair_benchmark.report._stats import suppress_confusion

T0 = datetime(2024, 1, 1, 0, 0)


def _h(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


# --------------------------------------------------------------------------
# 1 — string ids survive the MEDS emit
# --------------------------------------------------------------------------


def _emit_one(patient_ids: list[str], hosp_ids: list[str]) -> pl.DataFrame:
    df = pl.DataFrame({
        "patient_id": patient_ids,
        "hospitalization_id": hosp_ids,
        "recorded_dttm": [_h(i) for i in range(len(patient_ids))],
        "vital_value": [60.0] * len(patient_ids),
    })
    concepts = {"vital": {"code": ["VITAL", "col(vital_category)"],
                          "time": "col(recorded_dttm)",
                          "numeric_value": "vital_value"}}
    return emit_events(df, concepts, "patient_id")


def test_alphanumeric_ids_are_not_dropped():
    """The whole table used to vanish at any site with non-numeric ids."""
    out = _emit_one(["P101", "P102"], ["HOSP_1", "HOSP_2"])
    assert out.height == 2
    assert out["subject_id"].to_list() == ["P101", "P102"]
    assert out["hospitalization_id"].to_list() == ["HOSP_1", "HOSP_2"]


def test_leading_zero_ids_keep_their_zeros():
    """'007' cast to Int64 came back as 7 — a different patient."""
    out = _emit_one(["007"], ["0042"])
    assert out["subject_id"].to_list() == ["007"]
    assert out["hospitalization_id"].to_list() == ["0042"]


# --------------------------------------------------------------------------
# 2 — an empty id list means "nothing", not "everything"
# --------------------------------------------------------------------------


def test_empty_id_list_filters_to_nothing():
    assert _id_filter("hospitalization_id", None) is None          # load all
    empty = _id_filter("hospitalization_id", [])
    assert empty is not None                                       # NOT "load all"
    assert empty["hospitalization_id"] != []                       # matches nothing
    assert _id_filter("hospitalization_id", ["a"]) == {"hospitalization_id": ["a"]}


# --------------------------------------------------------------------------
# 3 — a multi-row OR trip is not an unplanned readmission
# --------------------------------------------------------------------------


def _adt(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"hospitalization_id": ["1"] * len(rows),
         "location_category": [r[0] for r in rows],
         "in_dttm": [_h(r[1]) for r in rows],
         "out_dttm": [_h(r[2]) for r in rows]},
        schema_overrides={"hospitalization_id": pl.Utf8},
    )


def test_multi_row_procedural_trip_is_planned():
    """ICU -> (OR, then a second procedural row) -> ICU is one ICU stay."""
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), ("procedural", 10, 12), ("procedural", 12, 14),
              ("icu", 14, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(30)]   # merged into one stay


def test_single_row_procedural_trip_still_planned():
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), ("procedural", 10, 12), ("icu", 12, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [None]


def test_ward_between_icu_stays_is_a_readmission():
    """The fix must not swallow a genuine unplanned return."""
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), ("ward", 10, 20), ("icu", 20, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [_h(20)]


def test_unmapped_location_becomes_other():
    """Pins the intended behaviour; does NOT reproduce a past bug.

    The old plain `.replace()` left an unmapped location as itself, but
    LOCATION_MAPPING happens to cover the mCIDE set exactly, so no reachable
    input distinguished the two. This guards the day a site charts something
    outside that set.
    """
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), ("some_site_local_ward", 10, 20), ("icu", 20, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [_h(20)]


# --------------------------------------------------------------------------
# 10 — a patient ventilated through an existing trach is not silently lost
# --------------------------------------------------------------------------


def test_trach_and_imv_on_the_same_row_emits_a_sentinel():
    resp = pl.DataFrame({
        "hospitalization_id": ["1", "1"],
        "recorded_dttm": [_h(0), _h(1)],
        "device_category": ["imv", "imv"],
        "tracheostomy": [True, True],
    })
    out = detect_imv_episodes(resp)
    assert out.height == 1                       # used to return zero rows
    assert out["is_pre_existing_trach"].to_list() == [True]
    assert out["episode_index"].to_list() == [0]  # the sentinel


def test_trach_strictly_after_imv_is_not_pre_existing():
    resp = pl.DataFrame({
        "hospitalization_id": ["1", "1", "1"],
        "recorded_dttm": [_h(0), _h(1), _h(5)],
        "device_category": ["imv", "imv", "trach collar"],
        "tracheostomy": [False, False, True],
    })
    out = detect_imv_episodes(resp)
    assert out["is_pre_existing_trach"].to_list() == [False]
    assert out["episode_start_dttm"].to_list() == [_h(0)]


# --------------------------------------------------------------------------
# 6 — cell suppression that cannot be undone by subtraction
# --------------------------------------------------------------------------


def test_small_cell_masks_its_partner():
    """tp alone is recoverable as n_flagged - fp, so fp goes too."""
    out = suppress_confusion(tp=3, fp=1000, fn=200, tn=3000)
    assert out["tp"] == "<10" and out["fp"] == "<10"
    assert out["fn"] == 200 and out["tn"] == 3000


def test_true_zero_is_published_and_spares_its_partner():
    out = suppress_confusion(tp=20, fp=20, fn=0, tn=20)
    assert out == {"tp": 20, "fp": 20, "fn": 0, "tn": 20}


def test_all_large_cells_pass_through():
    out = suppress_confusion(tp=10, fp=10, fn=10, tn=10)
    assert out == {"tp": 10, "fp": 10, "fn": 10, "tn": 10}


# --------------------------------------------------------------------------
# 11 — a patient never spans train and test
# --------------------------------------------------------------------------


def test_split_is_decided_per_patient_not_per_encounter():
    """Two encounters of one patient, far enough apart to have distinct join ids."""
    n_patients = 40
    rows = []
    for p in range(n_patients):
        for enc in range(2):
            rows.append({
                "patient_id": f"P{p}",
                "hospitalization_join_id": f"P{p}-enc{enc}",
                "admission_dttm": T0 + timedelta(days=365 * enc + p),
            })
    cohort = pl.DataFrame(rows)
    out = assign_split(cohort, "mimic", None, None)

    per_patient = out.group_by("patient_id").agg(
        pl.col("split").n_unique().alias("n_splits"))
    assert per_patient["n_splits"].max() == 1
    assert set(out["split"].to_list()) == {"train", "test"}


def test_split_falls_back_to_encounter_without_patient_id():
    cohort = pl.DataFrame({
        "hospitalization_join_id": [f"E{i}" for i in range(40)],
        "admission_dttm": [T0 + timedelta(days=i) for i in range(40)],
    })
    out = assign_split(cohort, "mimic", None, None)
    assert set(out["split"].to_list()) == {"train", "test"}


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 1), (3, 2), (4, 3), (100, 75)])
def test_split_cutoff_never_empties_an_arm(n, expected):
    """int(n * 0.75) was 0 for n == 1: a single-block cohort got no train rows."""
    assert _split_cutoff(n, 0.75) == expected


def test_two_block_cohort_carries_both_arms():
    cohort = pl.DataFrame({
        "hospitalization_join_id": ["A", "B"],
        "admission_dttm": [T0, T0 + timedelta(days=1)],
    })
    out = assign_split(cohort, "mimic", None, None)
    assert set(out["split"].to_list()) == {"train", "test"}
