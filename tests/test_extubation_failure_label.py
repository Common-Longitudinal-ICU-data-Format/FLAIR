from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from flair_benchmark.tasks import extubation_failure_24h as t3


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
                # 50h of IMV — clears MIN_IMV_HOURS. Only the START moves; the end
                # stays at _dt(1, 10) so every downstream timestamp (deaths,
                # discharges, the split boundary) keeps its original meaning.
                "episode_start_dttm": datetime(2019, 12, 30, 8),
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

    out = t3.build("unused.json")
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

    # prediction_dttm leads the extubation by PREDICTION_LEAD_MINUTES, and the
    # feature cutoff sits on it. A "should we extubate?" model must not see the
    # respiratory_support row that announces the extubation; the lead is what
    # keeps it out. The cutoff rides in the cohort because `flair fe-meds` takes
    # a parquet path, never the task module — see tasks/_finalize.py.
    for row in out.to_dicts():
        assert (row["extubation_dttm"] - row["prediction_dttm"]
                == timedelta(minutes=t3.PREDICTION_LEAD_MINUTES))
        assert row["feature_cutoff_dttm"] == row["prediction_dttm"]
        assert row["feature_cutoff_dttm"] < row["extubation_dttm"]


def test_t3_label_window_is_anchored_at_the_extubation_not_the_prediction(monkeypatch):
    """The 30-min prediction lead must not shorten the outcome window.

    prediction_dttm sits PREDICTION_LEAD_MINUTES before extubation_dttm. If the
    24h horizon were measured from prediction_dttm instead, the label would
    quietly become "reintubation within 23.5h of extubation" and a reintubation
    in that last half hour would flip from positive to negative.
    """
    ext = _dt(1, 10)
    # e_edge reintubates at exactly extubation + 24h — inside the window from the
    # extubation, OUTSIDE it if the horizon were measured from prediction_dttm.
    # e_late reintubates 1 min past the extubation-anchored horizon: negative.
    join_ids = ["e_edge", "e_late"]
    hosp_ids = [f"h_{j}" for j in join_ids]
    reintub = {
        "e_edge": ext + timedelta(hours=24),
        "e_late": ext + timedelta(hours=24, minutes=1),
    }

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
    idx = pl.DataFrame(
        {"hospitalization_id": hosp_ids, "hospitalization_join_id": join_ids}
    )
    resp = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
            "device_category": ["nasal cannula"] * 2,
            "recorded_dttm": [ext] * 2,
        }
    )
    episodes = pl.DataFrame(
        [
            {
                "hospitalization_id": j,
                "episode_index": 1,
                "episode_start_dttm": datetime(2019, 12, 30, 8),
                "episode_end_dttm": ext,
                "is_pre_existing_trach": False,
            }
            for j in join_ids
        ]
        + [
            {
                "hospitalization_id": j,
                "episode_index": 2,
                "episode_start_dttm": reintub[j],
                "episode_end_dttm": reintub[j] + timedelta(hours=3),
                "is_pre_existing_trach": False,
            }
            for j in join_ids
        ]
    )

    monkeypatch.setattr(t3, "read_clif_config", lambda _: {"site": "test"})
    monkeypatch.setattr(t3, "base_icu_cohort", lambda _: (cohort, pl.DataFrame()))
    monkeypatch.setattr(t3, "load_or_build_encounter_index", lambda *_: idx)
    monkeypatch.setattr(t3, "load_respiratory_support", lambda *_a, **_k: resp)
    monkeypatch.setattr(t3, "detect_imv_episodes", lambda _: episodes)
    monkeypatch.setattr(t3, "encounter_attributes", lambda *_: attrs)

    out = t3.build("unused.json")
    labels = {
        row["hospitalization_join_id"]: row[t3.META["label_column"]]
        for row in out.to_dicts()
    }

    assert labels == {"e_edge": 1, "e_late": 0}


def test_t3_post_discharge_death_does_not_defeat_censoring(monkeypatch):
    """A registry death months AFTER discharge must not mark an early-discharge
    stay as fully observed.

    death_dttm is patient-level and can post-date discharge. With
    coalesce(death, discharge) as the observation end, a stay discharged alive
    6h post-extubation with a death two months later slipped past the
    ambiguous-censoring exclusion and was kept as a NEGATIVE, even though
    reintubation was unobservable after discharge. min_horizontal fixes it.
    """
    join_ids = ["e_latedeath", "e_observed"]
    hosp_ids = [f"h_{j}" for j in join_ids]

    cohort = pl.DataFrame(
        {"hospitalization_id": hosp_ids, "hospitalization_join_id": join_ids}
    )
    attrs = pl.DataFrame(
        {
            "hospitalization_join_id": join_ids,
            "hospitalization_id": hosp_ids,
            "patient_id": [f"p_{j}" for j in join_ids],
            # e_latedeath: discharged alive 6h post-extubation, dies 2 months on.
            # e_observed: same late death, but discharged well after the window.
            "death_dttm": [_dt(28, 0), _dt(28, 0)],
            "discharge_dttm": [_dt(1, 16), _dt(5, 10)],
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
    idx = pl.DataFrame(
        {"hospitalization_id": hosp_ids, "hospitalization_join_id": join_ids}
    )
    resp = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
            "device_category": ["nasal cannula"] * 2,
            "recorded_dttm": [_dt(1, 10)] * 2,
        }
    )
    episodes = pl.DataFrame(
        [
            {
                "hospitalization_id": j,
                "episode_index": 1,
                # 50h of IMV — clears MIN_IMV_HOURS (see note in the test above).
                "episode_start_dttm": datetime(2019, 12, 30, 8),
                "episode_end_dttm": _dt(1, 10),
                "is_pre_existing_trach": False,
            }
            for j in join_ids
        ]
    )

    monkeypatch.setattr(t3, "read_clif_config", lambda _: {"site": "test"})
    monkeypatch.setattr(t3, "base_icu_cohort", lambda _: (cohort, pl.DataFrame()))
    monkeypatch.setattr(t3, "load_or_build_encounter_index", lambda *_: idx)
    monkeypatch.setattr(t3, "load_respiratory_support", lambda *_a, **_k: resp)
    monkeypatch.setattr(t3, "detect_imv_episodes", lambda _: episodes)
    monkeypatch.setattr(t3, "encounter_attributes", lambda *_: attrs)

    out = t3.build("unused.json")
    kept = set(out["hospitalization_join_id"].to_list())

    assert "e_latedeath" not in kept, (
        "stay discharged alive <24h post-extubation must be dropped even when a "
        "post-discharge death_dttm exists"
    )
    assert "e_observed" in kept


def test_t3_drops_imv_episodes_shorter_than_min_imv_hours(monkeypatch):
    """A brief peri-procedural intubation is not the weaning decision modelled.

    The 15/30-min consensus windows in detect_imv_episodes are artifact filters,
    not a cohort criterion — without MIN_IMV_HOURS a 2h post-op intubation would
    qualify as an extubation-failure prediction.
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
    idx = pl.DataFrame(
        {"hospitalization_id": hosp_ids, "hospitalization_join_id": join_ids}
    )
    resp = pl.DataFrame(
        {
            "hospitalization_id": hosp_ids,
            "hospitalization_join_id": join_ids,
            "device_category": ["nasal cannula"] * 2,
            "recorded_dttm": [_dt(1, 10)] * 2,
        }
    )
    # Identical in every respect except episode-1 duration, both ending at the
    # same instant: e_short ventilated 2h, e_long ventilated 50h.
    episodes = pl.DataFrame(
        [
            {
                "hospitalization_id": "e_short",
                "episode_index": 1,
                "episode_start_dttm": _dt(1, 8),
                "episode_end_dttm": _dt(1, 10),
                "is_pre_existing_trach": False,
            },
            {
                "hospitalization_id": "e_long",
                "episode_index": 1,
                "episode_start_dttm": datetime(2019, 12, 30, 8),
                "episode_end_dttm": _dt(1, 10),
                "is_pre_existing_trach": False,
            },
        ]
    )

    monkeypatch.setattr(t3, "read_clif_config", lambda _: {"site": "test"})
    monkeypatch.setattr(t3, "base_icu_cohort", lambda _: (cohort, pl.DataFrame()))
    monkeypatch.setattr(t3, "load_or_build_encounter_index", lambda *_: idx)
    monkeypatch.setattr(t3, "load_respiratory_support", lambda *_a, **_k: resp)
    monkeypatch.setattr(t3, "detect_imv_episodes", lambda _: episodes)
    monkeypatch.setattr(t3, "encounter_attributes", lambda *_: attrs)

    out = t3.build("unused.json")
    kept = set(out["hospitalization_join_id"].to_list())

    assert kept == {"e_long"}, (
        f"only the >= {t3.MIN_IMV_HOURS}h ventilation course belongs in the "
        f"cohort, got {kept}"
    )
