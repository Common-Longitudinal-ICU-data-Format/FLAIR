"""Unit tests for encounter stitching → hospitalization_join_id.

Covers the two pieces flair_benchmark._stitch owns:
  * attach_join_id — pure polars mapping with self-fallback (no IO),
  * the clifpy stitch_encounters contract our index builder depends on
    (gap <= interval → one block; large gap → two blocks).
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from flair_benchmark._stitch import JOIN_ID, attach_join_id


def test_attach_join_id_maps_and_falls_back_to_self():
    idx = pl.DataFrame({
        "hospitalization_id": ["h1", "h2"],
        JOIN_ID: ["b1", "b1"],            # h1, h2 stitched into one block
    })
    df = pl.DataFrame({
        "hospitalization_id": ["h1", "h2", "h3"],  # h3 absent from index
        "x": [1, 2, 3],
    })
    out = attach_join_id(df, idx)

    got = {r["hospitalization_id"]: r[JOIN_ID] for r in out.iter_rows(named=True)}
    assert got["h1"] == "b1"
    assert got["h2"] == "b1"
    assert got["h3"] == "h3"          # never-stitched stay is its own block
    assert out[JOIN_ID].null_count() == 0


def _synth(gap_hours: int):
    """Two same-patient hospitalizations separated by gap_hours; minimal CLIF cols."""
    h1_admit = pd.Timestamp("2023-01-01 00:00")
    h1_discharge = h1_admit + pd.Timedelta(hours=10)
    h2_admit = h1_discharge + pd.Timedelta(hours=gap_hours)
    h2_discharge = h2_admit + pd.Timedelta(hours=14)
    hosp = pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "hospitalization_id": ["h1", "h2"],
        "admission_dttm": [h1_admit, h2_admit],
        "discharge_dttm": [h1_discharge, h2_discharge],
        "age_at_admission": [60, 60],
        "admission_type_category": ["inpatient", "inpatient"],
        "discharge_category": ["home", "home"],
    })
    adt = pd.DataFrame({
        "hospitalization_id": ["h1", "h2"],
        "in_dttm": hosp["admission_dttm"],
        "out_dttm": hosp["discharge_dttm"],
        "location_category": ["icu", "icu"],
        "hospital_id": ["A", "A"],
    })
    return hosp, adt


def test_stitch_links_within_interval():
    from clifpy.utils.stitching_encounters import stitch_encounters
    hosp, adt = _synth(gap_hours=3)            # 3h after the 10:00 discharge
    _, _, mapping = stitch_encounters(hospitalization=hosp, adt=adt, time_interval=6)
    blocks = mapping.set_index("hospitalization_id")["encounter_block"]
    assert blocks["h1"] == blocks["h2"]        # one stitched encounter


def test_stitch_separates_beyond_interval():
    from clifpy.utils.stitching_encounters import stitch_encounters
    hosp, adt = _synth(gap_hours=20)           # 20h gap > 6h
    _, _, mapping = stitch_encounters(hospitalization=hosp, adt=adt, time_interval=6)
    blocks = mapping.set_index("hospitalization_id")["encounter_block"]
    assert blocks["h1"] != blocks["h2"]        # two distinct encounters


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
