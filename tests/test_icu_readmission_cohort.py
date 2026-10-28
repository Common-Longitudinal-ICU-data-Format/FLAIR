from __future__ import annotations

from datetime import datetime

import polars as pl

from flair_benchmark.tasks import icu_readmission as t5


def _dt(day: int, hour: int) -> datetime:
    return datetime(2020, 1, day, hour)


def test_t5_drops_first_icu_stays_shorter_than_min_icu_stay_hours(monkeypatch):
    """The prediction is anchored at first-ICU discharge, so a stay too short to
    have produced a discharge decision does not belong in the cohort.

    e_short is a 3h pass-through that cleared the old 1h floor.
    """
    join_ids = ["e_short", "e_long"]
    hosp_ids = [f"h_{j}" for j in join_ids]

    cohort = pl.DataFrame(
        {"hospitalization_id": hosp_ids, "hospitalization_join_id": join_ids}
    )

    attrs = pl.DataFrame(
        {
            "hospitalization_join_id": join_ids,
            "hospitalization_id": hosp_ids,
            "patient_id": [f"p_{j}" for j in join_ids],
            "death_dttm": [None, None],
            "discharge_dttm": [_dt(5, 10), _dt(5, 10)],
            "age_at_admission": [65, 66],
            "sex_category": ["female", "male"],
            "race_category": ["unknown"] * 2,
            "ethnicity_category": ["unknown"] * 2,
        },
        schema={
            "hospitalization_join_id": pl.Utf8,
            "hospitalization_id": pl.Utf8,
            "patient_id": pl.Utf8,
            "death_dttm": pl.Datetime,
            "discharge_dttm": pl.Datetime,
            "age_at_admission": pl.Int64,
            "sex_category": pl.Utf8,
            "race_category": pl.Utf8,
            "ethnicity_category": pl.Utf8,
        },
    )

    # Identical except first-ICU-stay length, both ending at the same instant so
    # prediction_dttm (and therefore the split assignment) is unchanged:
    # e_short ran 3h, e_long ran 48h.
    timing = pl.DataFrame(
        {
            "hospitalization_id": join_ids,
            "first_icu_start_time": [_dt(1, 7), datetime(2019, 12, 30, 10)],
            "first_icu_end_time": [_dt(1, 10), _dt(1, 10)],
            "second_icu_start_time": [_dt(2, 6), _dt(2, 6)],
            "is_direct_icu_discharge": [False, False],
            "has_intermediate_er": [False, False],
        },
        schema={
            "hospitalization_id": pl.Utf8,
            "first_icu_start_time": pl.Datetime,
            "first_icu_end_time": pl.Datetime,
            "second_icu_start_time": pl.Datetime,
            "is_direct_icu_discharge": pl.Boolean,
            "has_intermediate_er": pl.Boolean,
        },
    )

    monkeypatch.setattr(t5, "read_clif_config", lambda _: {"site": "test"})
    monkeypatch.setattr(t5, "base_icu_cohort", lambda _: (cohort, pl.DataFrame()))
    monkeypatch.setattr(t5, "as_encounter_keyed", lambda df, *_a: df)
    def _readmission_timing(_, *, require_ward_return):
        assert require_ward_return is True
        return timing

    monkeypatch.setattr(t5, "compute_readmission_timing", _readmission_timing)
    monkeypatch.setattr(t5, "encounter_attributes", lambda *_: attrs)

    out = t5.build("unused.json")
    kept = set(out["hospitalization_join_id"].to_list())

    assert kept == {"e_long"}, (
        f"only the >= {t5.MIN_ICU_STAY_HOURS}h first ICU stay belongs in the "
        f"cohort, got {kept}"
    )
    assert out[t5.META["label_column"]].to_list() == [1]
