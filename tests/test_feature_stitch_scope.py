"""Feature extraction must load all raw hospitalizations in a stitched encounter."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl

from flair_benchmark._stitch import JOIN_ID


def _dt(hour: int) -> datetime:
    return datetime(2020, 1, 1, hour)


def _idx() -> pl.DataFrame:
    return pl.DataFrame({
        "hospitalization_id": ["h1", "h2", "h3"],
        JOIN_ID: ["e1", "e1", "e2"],
    })


def _task_df() -> pl.DataFrame:
    return pl.DataFrame({
        "hospitalization_id": ["h1"],
        JOIN_ID: ["e1"],
        "prediction_dttm": [_dt(10)],
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
    import flair_benchmark.features.meds_stream as meds_mod

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


def test_wide_includes_sibling_hospitalization_events(monkeypatch):
    import clifpy
    import flair_benchmark.features.wide as wide_mod

    _patch_scope_index(monkeypatch)

    class FakeClifOrchestrator:
        hospitalization_ids: list[str] = []

        def __init__(self, data_directory, filetype, timezone):
            self.wide_df = pd.DataFrame()

        def create_wide_dataset(
            self,
            tables_to_load,
            category_filters,
            hospitalization_ids,
            output_format,
            show_progress,
        ):
            FakeClifOrchestrator.hospitalization_ids = list(hospitalization_ids)
            self.wide_df = pd.DataFrame({
                "hospitalization_id": ["h1", "h2", "h2", "h3"],
                "patient_id": ["p1", "p1", "p1", "p2"],
                "event_time": [_dt(9), _dt(9), _dt(11), _dt(9)],
                "heart_rate": [80.0, 90.0, 95.0, 100.0],
            })

    monkeypatch.setattr(clifpy, "ClifOrchestrator", FakeClifOrchestrator)

    out = wide_mod.get_wide_feature_dataset(
        _task_df(),
        {"wide": {"vitals": {"categories": ["heart_rate"]}}},
        {
            "data_directory": "/tmp/unused",
            "filetype": "parquet",
            "timezone": "US/Eastern",
            "stitch_time_interval_hours": 6,
        },
    )

    assert FakeClifOrchestrator.hospitalization_ids == ["h1", "h2"]
    assert set(out["hospitalization_id"].to_list()) == {"h1", "h2"}
    assert set(out[JOIN_ID].to_list()) == {"e1"}
    assert out["time"].max() <= _dt(10)


def test_direct_meds_builder_expands_sibling_hospitalizations(monkeypatch, tmp_path):
    import flair_benchmark.features.fe_meds as fe_meds_mod

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
