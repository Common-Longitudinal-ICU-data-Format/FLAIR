from __future__ import annotations

from datetime import datetime

import polars as pl

from flair_benchmark.tasks import task3_extubation_failure_24h as t3


def _dt(day: int, hour: int) -> datetime:
    return datetime(2020, 1, day, hour)


def test_t3_labels_reintubation_only_and_drops_death_only(monkeypatch):
    join_ids = ["e_reint", "e_death", "e_ok", "e_censor", "e_both"]
    hosp_ids = [f"h_{join_id}" for join_id in join_ids]

    cohort = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
        }
    )

    attrs = pl.DataFrame(
        {
            "hospitalization_join_id": join_ids,
            "hospitalization_id": hosp_ids,
            "patient_id": [f"p_{join_id}" for join_id in join_ids],
            "death_dttm": [None, _dt(1, 20), None, None, _dt(1, 21)],
            "discharge_dttm": [_dt(3, 10), _dt(1, 21), _dt(3, 10), _dt(1, 20), _dt(3, 10)],
            "age_at_admission": [65, 66, 67, 68, 69],
            "sex_category": ["female", "male", "female", "male", "female"],
            "race_category": ["unknown"] * 5,
            "ethnicity_category": ["unknown"] * 5,
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

    idx = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
        }
    )

    resp = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
            "device_category": ["nasal cannula"] * 5,
            "recorded_dttm": [_dt(1, 10)] * 5,
        }
    )

    episode_rows = []
    for join_id in join_ids:
        episode_rows.append(
            {
                "hospitalization_id": join_id,
                "episode_index": 1,
                "episode_start_dttm": _dt(1, 8),
                "episode_end_dttm": _dt(1, 10),
                "is_pre_existing_trach": False,
            }
        )
    for join_id in ["e_reint", "e_both"]:
        episode_rows.append(
            {
                "hospitalization_id": join_id,
                "episode_index": 2,
                "episode_start_dttm": _dt(1, 20),
                "episode_end_dttm": _dt(1, 23),
                "is_pre_existing_trach": False,
            }
        )
    episodes = pl.DataFrame(episode_rows)

    monkeypatch.setattr(t3, "read_clif_config", lambda _: {"site": "test"})
    monkeypatch.setattr(t3, "base_icu_cohort", lambda _: (cohort, pl.DataFrame()))
    monkeypatch.setattr(t3, "load_or_build_encounter_index", lambda *_: idx)
    monkeypatch.setattr(t3, "load_respiratory_support", lambda *_args, **_kwargs: resp)
    monkeypatch.setattr(t3, "detect_imv_episodes", lambda _: episodes)
    monkeypatch.setattr(t3, "encounter_attributes", lambda *_: attrs)

    out = t3.build("unused.json", train_end="2020-01-01", test_start="2020-01-02")
    labels = {
        row["hospitalization_join_id"]: row[t3.META["label_column"]]
        for row in out.to_dicts()
    }

    assert labels == {
        "e_reint": 1,
        "e_ok": 0,
        "e_both": 1,
    }
    assert "e_death" not in labels
    assert "e_censor" not in labels
