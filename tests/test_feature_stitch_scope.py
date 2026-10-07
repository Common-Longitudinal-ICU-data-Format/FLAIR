"""Feature extraction must load all raw hospitalizations in a stitched encounter."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from flair_benchmark.cohort.stitch import JOIN_ID


def _dt(hour: int) -> datetime:
    return datetime(2020, 1, 1, hour)


def _idx() -> pl.DataFrame:
    return pl.DataFrame({
        "hospitalization_id": ["h1", "h2", "h3"],
        JOIN_ID: ["e1", "e1", "e2"],
    })


def _task_df(cutoff: datetime | None = None) -> pl.DataFrame:
    """A finalized cohort row.

    ``feature_cutoff_dttm`` is a cohort-contract column (tasks/_finalize.py) and
    defaults to prediction_dttm — the inclusive cutoff every task but one uses.
    Pass ``cutoff`` to model a task that cuts strictly before its own anchor.
    """
    return pl.DataFrame({
        "hospitalization_id": ["h1"],
        JOIN_ID: ["e1"],
        "prediction_dttm": [_dt(10)],
        "feature_cutoff_dttm": [cutoff if cutoff is not None else _dt(10)],
    })


def _patch_scope_index(monkeypatch):
    import flair_benchmark.features._scope as scope_mod

    monkeypatch.setattr(
        scope_mod,
        "load_or_build_encounter_index",
        lambda cfg, time_interval: _idx(),
    )


def test_feature_scope_expands_join_id_to_all_member_hospitalizations(monkeypatch):
    from flair_benchmark.features._scope import HORIZON_COL, feature_scope

    _patch_scope_index(monkeypatch)
    scope = feature_scope(_task_df(), {"stitch_time_interval_hours": 6})

    assert scope.hospitalization_ids == ["h1", "h2"]
    assert scope.horizons.to_dicts() == [{JOIN_ID: "e1", HORIZON_COL: _dt(10)}]


def test_meds_stream_includes_sibling_hospitalization_events(monkeypatch):
    import flair_benchmark.features.extractors as meds_mod

    _patch_scope_index(monkeypatch)
    seen_hosp_ids: list[str] = []

    def fake_loader(cfg, hospitalization_ids, med_categories=None, med_groups=None):
        seen_hosp_ids.extend(hospitalization_ids)
        return pl.DataFrame({
            "hospitalization_id": ["h1", "h2", "h2", "h3"],
            "admin_dttm": [_dt(9), _dt(9), _dt(11), _dt(9)],
            "med_category": ["norepinephrine"] * 4,
            "med_dose": [1.0, 2.0, 3.0, 4.0],
            "med_dose_unit": ["mg"] * 4,
        })

    def fake_attach_subject_id(df, cfg):
        return df.with_columns(pl.lit("p1").alias("subject_id"))

    monkeypatch.setattr(meds_mod, "load_medication_admin_continuous", fake_loader)
    monkeypatch.setitem(
        meds_mod._TABLES,
        "medication_admin_continuous",
        ("MED_CON", fake_loader),
    )
    monkeypatch.setattr(meds_mod, "attach_subject_id", fake_attach_subject_id)

    out = meds_mod.get_meds_stream(
        _task_df(),
        {"meds_stream": {"medication_admin_continuous": {"categories": ["norepinephrine"]}}},
        {"stitch_time_interval_hours": 6},
    )

    assert seen_hosp_ids == ["h1", "h2"]
    assert set(out["hospitalization_id"].to_list()) == {"h1", "h2"}
    assert set(out[JOIN_ID].to_list()) == {"e1"}
    assert out["time"].max() <= _dt(10)


def test_direct_meds_builder_expands_sibling_hospitalizations(monkeypatch, tmp_path):
    import flair_benchmark.features.extractors as fe_meds_mod

    _patch_scope_index(monkeypatch)
    seen_hosp_ids: list[str] = []

    def fake_build_meds_events(clif_cfg, elf_cfg, hosp_ids):
        seen_hosp_ids.extend(hosp_ids)
        return (
            pl.DataFrame({
                "subject_id": ["p1", "p1"],
                "time": [_dt(9), _dt(9)],
                "code": ["MED_CON//norepinephrine", "MED_CON//norepinephrine"],
                "numeric_value": [1.0, 2.0],
                "text_value": [None, None],
                "hospitalization_id": ["h1", "h2"],
            }),
            {"MED_CON": "test"},
        )

    monkeypatch.setattr(fe_meds_mod, "build_meds_events", fake_build_meds_events)
    monkeypatch.setattr(
        fe_meds_mod,
        "build_codes_table",
        lambda events, versions: pl.DataFrame({"code": ["MED_CON//norepinephrine"]}),
    )

    elf_config = tmp_path / "elf.yaml"
    elf_config.write_text("domains: {}\n")

    out = fe_meds_mod.build_meds_tables(
        _task_df(),
        {"stitch_time_interval_hours": 6},
        str(elf_config),
        str(tmp_path / "meds"),
    )

    assert seen_hosp_ids == ["h1", "h2"]
    assert set(out["hospitalization_id"].to_list()) == {"h1", "h2"}
    assert set(out[JOIN_ID].to_list()) == {"e1"}


# ==================================================== THE FEATURE-CUTOFF BOUNDARY
#
# The cutoff was untested in BOTH directions: every fixture above puts events at
# hours 9 and 11 against a horizon of 10, so nothing ever landed ON the boundary
# and neither an inclusive nor a strict filter could be distinguished. No task
# currently ships a cutoff earlier than its prediction_dttm (extubation_failure_24h
# used to, before it moved prediction_dttm itself 30 min ahead of the extubation),
# but the cohort contract still lets a task move the value rather than the
# comparison, so both directions stay under test here.

_STRICT = _dt(10) - timedelta(microseconds=1)


def _fake_meds(monkeypatch):
    """Monkeypatch the meds loader with an event exactly ON the horizon."""
    import flair_benchmark.features.extractors as meds_mod

    def fake_loader(cfg, hospitalization_ids, med_categories=None, med_groups=None):
        return pl.DataFrame({
            "hospitalization_id": ["h1", "h1", "h1"],
            "admin_dttm": [_dt(9), _dt(10), _dt(11)],       # before / ON / after
            "med_category": ["norepinephrine"] * 3,
            "med_dose": [1.0, 2.0, 3.0],
            "med_dose_unit": ["mg"] * 3,
        })

    monkeypatch.setattr(meds_mod, "load_medication_admin_continuous", fake_loader)
    monkeypatch.setitem(meds_mod._TABLES, "medication_admin_continuous",
                        ("MED_CON", fake_loader))
    monkeypatch.setattr(meds_mod, "attach_subject_id",
                        lambda df, cfg: df.with_columns(pl.lit("p1").alias("subject_id")))
    return meds_mod


def test_default_cutoff_keeps_the_boundary_sample(monkeypatch):
    """cutoff == prediction_dttm is inclusive — unchanged behaviour for 3 of 4 tasks."""
    _patch_scope_index(monkeypatch)
    meds_mod = _fake_meds(monkeypatch)
    out = meds_mod.get_meds_stream(
        _task_df(),
        {"meds_stream": {"medication_admin_continuous": {"categories": ["norepinephrine"]}}},
        {"stitch_time_interval_hours": 6})
    assert _dt(10) in out["time"].to_list()


def test_strict_cutoff_drops_the_boundary_sample(monkeypatch):
    """A cutoff 1us earlier excludes the anchor row itself.

    meds_stream.py still compares `time <= end_time`; only the value it compares
    against moved. That is the point of carrying the cutoff as a timestamp rather
    than a flag — the extractor never had to learn about strictness.
    """
    _patch_scope_index(monkeypatch)
    meds_mod = _fake_meds(monkeypatch)
    out = meds_mod.get_meds_stream(
        _task_df(_STRICT),
        {"meds_stream": {"medication_admin_continuous": {"categories": ["norepinephrine"]}}},
        {"stitch_time_interval_hours": 6})
    assert out["time"].to_list() == [_dt(9)]


def test_feature_scope_rejects_a_stale_cohort(monkeypatch):
    """A cohort predating the contract column must fail LOUDLY, not fall back.

    Silently defaulting to prediction_dttm is the exact failure this design exists
    to prevent: a site would get an inclusive cutoff on a task that needs a strict
    one and never know.
    """
    from flair_benchmark.features._scope import feature_scope

    _patch_scope_index(monkeypatch)
    stale = _task_df().drop("feature_cutoff_dttm")
    with pytest.raises(ValueError, match="feature_cutoff_dttm"):
        feature_scope(stale, {"stitch_time_interval_hours": 6})

    # ...but fe_meds deliberately takes no horizon at all, so it stays legal there
    scope = feature_scope(stale, {"stitch_time_interval_hours": 6},
                          require_horizon=False)
    assert scope.hospitalization_ids == ["h1", "h2"]


def test_feature_scope_coalesces_a_null_cutoff_per_row(monkeypatch):
    """A null cutoff falls back to THAT row's prediction time, not a sibling's.

    A bare .max() over the column would skip the null and let h2's later cutoff
    govern h1 — widening h1's window past its own prediction.
    """
    from flair_benchmark.features._scope import HORIZON_COL, feature_scope

    _patch_scope_index(monkeypatch)
    df = pl.DataFrame({
        "hospitalization_id": ["h1", "h2"],
        JOIN_ID: ["e1", "e1"],
        "prediction_dttm": [_dt(10), _dt(12)],
        "feature_cutoff_dttm": [None, _STRICT],
    }, schema_overrides={"feature_cutoff_dttm": pl.Datetime("us")})
    scope = feature_scope(df, {"stitch_time_interval_hours": 6})
    # max(coalesce) = max(10:00, 09:59:59.999999) = 10:00, not 09:59:59.999999
    assert scope.horizons.to_dicts() == [{JOIN_ID: "e1", HORIZON_COL: _dt(10)}]
