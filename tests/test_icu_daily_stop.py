"""Unit tests for the T1/T2 first-ICU-stay daily grid (flair_benchmark._encounter)
and the leakage fix in flair_benchmark._split.

All frame-level (no clifpy file IO), matching tests/test_stitch.py style. Covers:
  * pred_stop_expr  — the four boundary scenarios the audit flagged,
  * _encounter_timing — contiguous-ICU merge incl. across a stitch boundary,
  * _collapse_members — encounter final disposition,
  * assign_split — date-cutoff split is block-grain (no straddle leakage).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from flair_benchmark._encounter import (
    _encounter_timing,
    as_encounter_keyed,
    collapse_members,
    pred_stop_expr,
)
from flair_benchmark._split import assign_split


def _dt(y, m, d, h=0):
    return datetime(y, m, d, h)


# --------------------------------------------------------------------------- #
# pred_stop_expr — earliest of (ICU end | discharge), discharge, death.
# --------------------------------------------------------------------------- #
def _pred_stop(first_icu_end, discharge, death):
    df = pl.DataFrame(
        {
            "first_icu_end_time": [first_icu_end],
            "discharge_dttm": [discharge],
            "death_dttm": [death],
        },
        schema={
            "first_icu_end_time": pl.Datetime,
            "discharge_dttm": pl.Datetime,
            "death_dttm": pl.Datetime,
        },
    )
    return df.select(pred_stop_expr().alias("pred_stop"))["pred_stop"][0]


def test_pred_stop_survivor_who_dies_later_stops_at_icu_end():
    # ICU end present, patient dies 2 years later in a separate encounter.
    # Must NOT use the future death (the original coalesce bug).
    got = _pred_stop(_dt(2020, 1, 3), _dt(2020, 1, 5), _dt(2022, 1, 1))
    assert got == _dt(2020, 1, 3)


def test_pred_stop_null_icu_end_future_death_stops_at_discharge():
    # Terminal/open ICU stay (null out_dttm) + a future death -> discharge,
    # never the future death. This is the Defect-1 regression.
    got = _pred_stop(None, _dt(2020, 1, 5), _dt(2022, 1, 1))
    assert got == _dt(2020, 1, 5)


def test_pred_stop_in_stay_death_shortens_grid():
    # Death before ICU end (died in the first ICU stay) -> stop at death.
    got = _pred_stop(_dt(2020, 1, 10), _dt(2020, 1, 10), _dt(2020, 1, 4))
    assert got == _dt(2020, 1, 4)


def test_pred_stop_transfer_out_stops_at_first_icu_end():
    # Left the first ICU stay (ICU end < discharge), no death -> ICU end.
    got = _pred_stop(_dt(2020, 1, 3), _dt(2020, 1, 9), None)
    assert got == _dt(2020, 1, 3)


# --------------------------------------------------------------------------- #
# _encounter_timing — first ICU stay across the whole stitched encounter.
# --------------------------------------------------------------------------- #
def _adt(rows):
    """rows: list of (hosp_id, join_id, location, in_dttm, out_dttm)."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r[0] for r in rows],
            "hospitalization_join_id": [r[1] for r in rows],
            "location_category": [r[2] for r in rows],
            "in_dttm": [r[3] for r in rows],
            "out_dttm": [r[4] for r in rows],
        },
        schema={
            "hospitalization_id": pl.Utf8,
            "hospitalization_join_id": pl.Utf8,
            "location_category": pl.Utf8,
            "in_dttm": pl.Datetime,
            "out_dttm": pl.Datetime,
        },
    )


def _row(timing, join_id):
    return timing.filter(pl.col("hospitalization_join_id") == join_id).to_dicts()[0]


def test_timing_single_icu_row():
    adt = _adt([("h1", "e1", "icu", _dt(2020, 1, 1, 8), _dt(2020, 1, 3, 8))])
    r = _row(_encounter_timing(adt), "e1")
    assert r["first_icu_start_time"] == _dt(2020, 1, 1, 8)
    assert r["first_icu_end_time"] == _dt(2020, 1, 3, 8)
    assert r["hospitalization_id"] == "h1"


def test_timing_merges_contiguous_icu_rows():
    # Two back-to-back ICU ADT rows in one hospitalization -> one block, later out.
    adt = _adt([
        ("h1", "e1", "icu", _dt(2020, 1, 1, 8), _dt(2020, 1, 2, 8)),
        ("h1", "e1", "icu", _dt(2020, 1, 2, 8), _dt(2020, 1, 4, 8)),
    ])
    r = _row(_encounter_timing(adt), "e1")
    assert r["first_icu_start_time"] == _dt(2020, 1, 1, 8)
    assert r["first_icu_end_time"] == _dt(2020, 1, 4, 8)


def test_timing_merges_icu_across_stitch_boundary():
    # h1 -> h2 stitched into encounter e1; contiguous ICU spanning the boundary
    # collapses to one first ICU block. Representative id = earliest (h1).
    adt = _adt([
        ("h1", "e1", "icu", _dt(2020, 1, 1, 8), _dt(2020, 1, 2, 8)),
        ("h2", "e1", "icu", _dt(2020, 1, 2, 8), _dt(2020, 1, 5, 8)),
    ])
    r = _row(_encounter_timing(adt), "e1")
    assert r["first_icu_start_time"] == _dt(2020, 1, 1, 8)
    assert r["first_icu_end_time"] == _dt(2020, 1, 5, 8)
    assert r["hospitalization_id"] == "h1"


def test_timing_second_icu_stay_not_merged_into_first():
    # ICU -> Ward -> ICU within one encounter: first ICU block ends before the ward.
    adt = _adt([
        ("h1", "e1", "icu", _dt(2020, 1, 1, 8), _dt(2020, 1, 2, 8)),
        ("h1", "e1", "ward", _dt(2020, 1, 2, 8), _dt(2020, 1, 4, 8)),
        ("h1", "e1", "icu", _dt(2020, 1, 4, 8), _dt(2020, 1, 6, 8)),
    ])
    r = _row(_encounter_timing(adt), "e1")
    assert r["first_icu_end_time"] == _dt(2020, 1, 2, 8)


# --------------------------------------------------------------------------- #
# _collapse_members — encounter outcome is the latest-discharged member.
# --------------------------------------------------------------------------- #
def test_collapse_members_takes_final_disposition():
    # e1 = h1 (ED, transferred) -> h2 (final, Expired). Outcome must be Expired,
    # representative hospitalization_id = admitting member (h1).
    members = pl.DataFrame(
        {
            "hospitalization_join_id": ["e1", "e1"],
            "hospitalization_id": ["h1", "h2"],
            "admission_dttm": [_dt(2020, 1, 1), _dt(2020, 1, 2)],
            "discharge_dttm": [_dt(2020, 1, 2), _dt(2020, 1, 8)],
            "discharge_category": ["transfer", "Expired"],
            "age_at_admission": [70, 70],
        }
    )
    out = collapse_members(members).to_dicts()[0]
    assert out["discharge_category"] == "Expired"
    assert out["admission_dttm"] == _dt(2020, 1, 1)
    assert out["discharge_dttm"] == _dt(2020, 1, 8)
    assert out["hospitalization_id"] == "h1"


# --------------------------------------------------------------------------- #
# assign_split — date-cutoff split decided once per encounter block.
# --------------------------------------------------------------------------- #
def test_date_cutoff_split_is_block_grain_no_straddle():
    # Encounter e1's daily rows straddle the cutoff: first row in train window,
    # later rows past test_start. The whole block must land in ONE split.
    cohort = pl.DataFrame(
        {
            "hospitalization_join_id": ["e1", "e1", "e1", "e2"],
            "prediction_dttm": [
                _dt(2021, 12, 30, 7),   # <= train_end
                _dt(2022, 1, 2, 7),     # in the gap
                _dt(2022, 1, 5, 7),     # >= test_start
                _dt(2022, 6, 1, 7),     # e2 clearly test
            ],
        }
    )
    out = assign_split(
        cohort, site="external",
        train_end="2021-12-31", test_start="2022-01-01",
        admission_col="prediction_dttm",
    )
    by_block = {
        jid[0]: set(grp["split"].to_list())
        for jid, grp in out.group_by("hospitalization_join_id")
    }
    # e1 decided by its earliest prediction_dttm (2021-12-30) -> all 'train'.
    assert by_block["e1"] == {"train"}
    assert by_block["e2"] == {"test"}
    # No block appears in both splits.
    trains = set(out.filter(pl.col("split") == "train")["hospitalization_join_id"])
    tests = set(out.filter(pl.col("split") == "test")["hospitalization_join_id"])
    assert trains.isdisjoint(tests)


def test_no_dates_external_is_temporal_75_25():
    # External site, no cutoff dates -> temporal 75/25: oldest 75% train, newest 25%
    # test. Both labels always present regardless of inference-only intent.
    df = pl.DataFrame({
        "hospitalization_join_id": [str(i) for i in range(20)],
        "admission_dttm": [_dt(2020, 1, 1 + i) for i in range(20)],
    })
    out = assign_split(df, site="external_site", train_end=None, test_start=None)
    counts = dict(out.group_by("split").len().iter_rows())
    assert counts == {"train": 15, "test": 5}
    # newest admission -> test, oldest -> train (temporal direction).
    ordered = out.sort("admission_dttm")
    assert ordered["split"][-1] == "test"
    assert ordered["split"][0] == "train"


def test_mimic_no_dates_is_random_and_skips_cutoff():
    # MIMIC -> random (not temporal) 75/25, and the data cutoff is NOT applied
    # (de-identified/shifted dates), so a far-future row survives.
    df = pl.DataFrame({
        "hospitalization_join_id": [str(i) for i in range(20)] + ["future"],
        "admission_dttm": [_dt(2020, 1, 1 + i) for i in range(20)] + [_dt(2026, 5, 1)],
    })
    out = assign_split(df, site="mimic", train_end=None, test_start=None)
    assert set(out["split"].unique()) == {"train", "test"}
    assert "future" in out["hospitalization_join_id"].to_list()   # cutoff skipped


def test_data_cutoff_drops_post_2025_for_nonmimic():
    df = pl.DataFrame({
        "hospitalization_join_id": [str(i) for i in range(10)] + ["late"],
        "admission_dttm": [_dt(2024, 1, 1 + i) for i in range(10)] + [_dt(2026, 6, 1)],
    })
    out = assign_split(df, site="external_site", train_end=None, test_start=None)
    assert "late" not in out["hospitalization_join_id"].to_list()
    assert out["admission_dttm"].max() <= datetime(2025, 12, 31)


# --------------------------------------------------------------------------- #
# as_encounter_keyed — relabel hospitalization_id -> its join id.
# --------------------------------------------------------------------------- #
def test_as_encounter_keyed_replaces_key_when_join_id_present():
    df = pl.DataFrame({
        "hospitalization_id": ["h1", "h2"],
        "hospitalization_join_id": ["e1", "e1"],
        "v": [1, 2],
    })
    out = as_encounter_keyed(df)
    assert "hospitalization_join_id" not in out.columns
    assert out["hospitalization_id"].to_list() == ["e1", "e1"]  # join id now in the slot
    assert out["v"].to_list() == [1, 2]


def test_as_encounter_keyed_attaches_from_index_when_absent():
    from flair_benchmark._stitch import JOIN_ID
    df = pl.DataFrame({"hospitalization_id": ["h1", "h2", "h3"], "v": [1, 2, 3]})
    idx = pl.DataFrame({"hospitalization_id": ["h1", "h2"], JOIN_ID: ["e1", "e1"]})
    out = as_encounter_keyed(df, idx)
    # h1, h2 -> e1; h3 absent from index -> self-fallback to "h3".
    assert sorted(out["hospitalization_id"].to_list()) == ["e1", "e1", "h3"]


# --------------------------------------------------------------------------- #
# Task 4 sepsis pred_stop — same earliest-of as T1/T2 but anchored on t_zero.
# --------------------------------------------------------------------------- #
def _t4_pred_stop(t_zero, discharge, death):
    df = pl.DataFrame(
        {"t_zero": [t_zero], "discharge_dttm": [discharge], "death_dttm": [death]},
        schema={"t_zero": pl.Datetime, "discharge_dttm": pl.Datetime,
                "death_dttm": pl.Datetime},
    )
    expr = pl.min_horizontal(
        pl.coalesce(["t_zero", "discharge_dttm"]),
        pl.col("discharge_dttm"),
        pl.col("death_dttm"),
    )
    return df.select(expr.alias("pred_stop"))["pred_stop"][0]


def test_t4_pred_stop_never_septic_stops_at_discharge():
    # t_zero null (never septic), future death -> discharge, not the future death.
    assert _t4_pred_stop(None, _dt(2020, 1, 5), _dt(2022, 1, 1)) == _dt(2020, 1, 5)


def test_t4_pred_stop_septic_stops_at_t_zero():
    # ASE onset before discharge -> window ends at t_zero (leak-safe).
    assert _t4_pred_stop(_dt(2020, 1, 3), _dt(2020, 1, 9), None) == _dt(2020, 1, 3)


def test_t4_pred_stop_in_stay_death_before_t_zero():
    assert _t4_pred_stop(_dt(2020, 1, 6), _dt(2020, 1, 9), _dt(2020, 1, 4)) == _dt(2020, 1, 4)


# --------------------------------------------------------------------------- #
# Task 5 readmission at encounter grain — 2nd ICU stay across stitched members.
# --------------------------------------------------------------------------- #
def test_t5_readmission_detected_across_stitch_boundary():
    from flair_benchmark._readmission import compute_readmission_timing
    # Encounter e1: member A goes ICU -> Ward, member B (stitched) returns to ICU.
    # At encounter grain that 2nd ICU block is an unplanned readmission.
    adt = _adt([
        ("hA", "e1", "icu",  _dt(2020, 1, 1, 8), _dt(2020, 1, 2, 8)),
        ("hA", "e1", "ward", _dt(2020, 1, 2, 8), _dt(2020, 1, 3, 8)),
        ("hB", "e1", "icu",  _dt(2020, 1, 3, 8), _dt(2020, 1, 5, 8)),
    ])
    timing = (compute_readmission_timing(as_encounter_keyed(adt))
              .rename({"hospitalization_id": "hospitalization_join_id"}))
    r = _row(timing, "e1")
    assert r["first_icu_end_time"] == _dt(2020, 1, 2, 8)
    assert r["second_icu_start_time"] == _dt(2020, 1, 3, 8)  # readmission seen


def test_t5_two_separate_encounters_stay_separate():
    from flair_benchmark._readmission import compute_readmission_timing
    adt = _adt([
        ("hA", "e1", "icu", _dt(2020, 1, 1, 8), _dt(2020, 1, 2, 8)),
        ("hB", "e2", "icu", _dt(2020, 2, 1, 8), _dt(2020, 2, 2, 8)),
    ])
    timing = (compute_readmission_timing(as_encounter_keyed(adt))
              .rename({"hospitalization_id": "hospitalization_join_id"}))
    assert timing.height == 2
    assert _row(timing, "e1")["second_icu_start_time"] is None
    assert _row(timing, "e2")["second_icu_start_time"] is None


# --------------------------------------------------------------------------- #
# Task 3 extubation at encounter grain — episodes merge across stitched members.
# --------------------------------------------------------------------------- #
def _resp(rows):
    """rows: list of (hosp_id, join_id, device_category, recorded_dttm)."""
    return pl.DataFrame(
        {
            "hospitalization_id": [r[0] for r in rows],
            "hospitalization_join_id": [r[1] for r in rows],
            "device_category": [r[2] for r in rows],
            "recorded_dttm": [r[3] for r in rows],
        },
        schema={
            "hospitalization_id": pl.Utf8,
            "hospitalization_join_id": pl.Utf8,
            "device_category": pl.Utf8,
            "recorded_dttm": pl.Datetime,
        },
    )


def test_t3_reintubation_in_stitched_member_is_episode_two():
    from flair_benchmark._ventilation_episodes import detect_imv_episodes

    def at(h, mn):
        return datetime(2020, 1, 1, h, mn)

    # Member hA: IMV (>=15min) then extubated (non-IMV >=30min) = episode 1.
    # Member hB (stitched into e1): IMV again = a reintubation = episode 2 of
    # the same encounter once relabeled to join id.
    resp = _resp([
        ("hA", "e1", "imv", at(8, 0)),
        ("hA", "e1", "imv", at(8, 20)),
        ("hA", "e1", "nasal cannula", at(9, 0)),
        ("hA", "e1", "nasal cannula", at(9, 40)),
        ("hB", "e1", "imv", at(11, 0)),
        ("hB", "e1", "imv", at(11, 20)),
        ("hB", "e1", "nasal cannula", at(12, 0)),
        ("hB", "e1", "nasal cannula", at(12, 40)),
    ])
    episodes = detect_imv_episodes(as_encounter_keyed(resp))
    e1 = episodes.filter(pl.col("hospitalization_id") == "e1")
    assert set(e1["episode_index"].to_list()) >= {1, 2}  # reintubation captured
    assert not e1["is_pre_existing_trach"].any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
