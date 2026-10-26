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
# 3 — allowed ICU gaps are not unplanned readmissions
# --------------------------------------------------------------------------


def _adt(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"hospitalization_id": ["1"] * len(rows),
         "location_category": [r[0] for r in rows],
         "in_dttm": [_h(r[1]) for r in rows],
         "out_dttm": [_h(r[2]) for r in rows]},
        schema_overrides={"hospitalization_id": pl.Utf8},
    )


@pytest.mark.parametrize("location", ["procedural", "radiology", "dialysis"])
def test_multi_row_allowed_gap_is_planned(location):
    """Repeated rows from any allowed category remain part of one ICU stay."""
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), (location, 10, 12), (location, 12, 14),
              ("icu", 14, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(30)]   # merged into one stay


@pytest.mark.parametrize("location", ["procedural", "radiology", "dialysis"])
def test_single_row_allowed_gap_is_planned(location):
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), (location, 10, 12), ("icu", 12, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(30)]


def test_mixed_allowed_gap_is_planned():
    """Any contiguous combination of allowed categories is one ICU gap."""
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), ("radiology", 10, 12), ("dialysis", 12, 14),
              ("procedural", 14, 16), ("icu", 16, 30)])
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(30)]


def test_consecutive_icu_rows_are_one_stay():
    out = compute_readmission_timing(_adt([("icu", 0, 10), ("icu", 10, 30)]))
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(30)]


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


@pytest.mark.parametrize("location", ["other", "rehab", "stepdown"])
def test_ward_required_policy_ignores_other_return(location):
    out = compute_readmission_timing(
        _adt([("icu", 0, 10), (location, 10, 20), ("icu", 20, 30)]),
        require_ward_return=True,
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(10)]


def test_ward_required_policy_ignores_mixed_nonward_return():
    out = compute_readmission_timing(
        _adt([
            ("icu", 0, 10),
            ("other", 10, 12),
            ("procedural", 12, 14),
            ("radiology", 14, 16),
            ("icu", 16, 30),
        ]),
        require_ward_return=True,
    )
    assert out["second_icu_start_time"].to_list() == [None]
    assert out["first_icu_end_time"].to_list() == [_h(10)]


@pytest.mark.parametrize(
    ("rows", "second_icu_hour"),
    [
        (["ward"], 20),
        (["ward", "other"], 30),
        (["other", "ward", "procedural"], 40),
    ],
)
def test_ward_required_policy_keeps_any_ward_return(rows, second_icu_hour):
    locations = [("icu", 0, 10)]
    locations.extend(
        (location, 10 + index * 10, 20 + index * 10)
        for index, location in enumerate(rows)
    )
    locations.append(("icu", second_icu_hour, second_icu_hour + 10))

    out = compute_readmission_timing(
        _adt(locations),
        require_ward_return=True,
    )
    assert out["second_icu_start_time"].to_list() == [_h(second_icu_hour)]


def test_ward_required_policy_keeps_later_ward_return():
    out = compute_readmission_timing(
        _adt([
            ("icu", 0, 10),
            ("other", 10, 20),
            ("icu", 20, 30),
            ("ward", 30, 40),
            ("icu", 40, 50),
        ]),
        require_ward_return=True,
    )
    assert out["first_icu_end_time"].to_list() == [_h(10)]
    assert out["second_icu_start_time"].to_list() == [_h(40)]


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
# 11 — the split is dealt at ENCOUNTER grain, and train stays behind test in time
# --------------------------------------------------------------------------


def _two_encounters_per_patient(n_patients: int = 40) -> pl.DataFrame:
    """Each patient has two encounters a year apart, so they have distinct join ids."""
    rows = []
    for p in range(n_patients):
        for enc in range(2):
            rows.append({
                "patient_id": f"P{p}",
                "hospitalization_join_id": f"P{p}-enc{enc}",
                "admission_dttm": T0 + timedelta(days=365 * enc + p),
            })
    return pl.DataFrame(rows)


def test_split_is_decided_per_encounter():
    """No stitched encounter is torn across train and test — the invariant that holds.

    SPLIT_VERSION 2 dealt at patient grain instead. That guaranteed more (a patient
    never spanned arms) but destroyed the calendar separation, because a patient is
    unbounded in time: one first seen years before the boundary carries all their
    later encounters into train with them. See assign_split.
    """
    out = assign_split(_two_encounters_per_patient(), "mimic", None, None)

    per_encounter = out.group_by("hospitalization_join_id").agg(
        pl.col("split").n_unique().alias("n_splits"))
    assert per_encounter["n_splits"].max() == 1
    assert set(out["split"].to_list()) == {"train", "test"}


def test_a_patient_may_span_both_arms_at_encounter_grain():
    """The accepted price of the temporal cut, pinned so it is never a surprise.

    Encounters a year apart land on opposite sides of the boundary, so a returning
    patient appears in both arms. Measured at ~0.4% of patients on the real MIMIC
    mortality cohort. If this ever needs to be zero, drop test encounters whose
    patient_id also appears in train — do NOT go back to patient-grain dealing.
    """
    out = assign_split(_two_encounters_per_patient(), "mimic", None, None)

    per_patient = out.group_by("patient_id").agg(
        pl.col("split").n_unique().alias("n_splits"))
    assert per_patient["n_splits"].max() == 2


def test_temporal_regime_keeps_train_behind_test():
    """Non-mimic, no cutoff dates: every train encounter precedes every test one.

    Point-in-time encounters, so the separation is strict. Real cohorts overlap by
    at most one encounter's length (the stay open across the boundary).
    """
    cohort = pl.DataFrame({
        "patient_id": [f"P{i % 7}" for i in range(40)],   # patients recur; must not matter
        "hospitalization_join_id": [f"E{i}" for i in range(40)],
        "admission_dttm": [T0 + timedelta(days=i) for i in range(40)],
    })
    out = assign_split(cohort, "external", None, None)

    train_max = out.filter(pl.col("split") == "train")["admission_dttm"].max()
    test_min = out.filter(pl.col("split") == "test")["admission_dttm"].min()
    assert train_max < test_min


def test_split_works_without_a_patient_id_column():
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


def test_split_requires_encounter_key():
    """The split is always per stitched encounter; there is no patient/per-row fallback."""
    cohort = pl.DataFrame({
        "patient_id": [f"P{i}" for i in range(10)],
        "admission_dttm": [T0 + timedelta(days=i) for i in range(10)],
    })
    with pytest.raises(ValueError, match="hospitalization_join_id"):
        assign_split(cohort, "mimic", None, None)
