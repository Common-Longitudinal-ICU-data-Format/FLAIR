"""Tests for the meds_etl performance work: predicate pushdown, TableCache,
and the parallel domain loop. All data is a tiny synthetic CLIF corpus written
to tmp_path — no site data required.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from flair_benchmark.meds_etl import _emit
from flair_benchmark.meds_etl._emit import (
    TableCache,
    hosp_filters,
    load_clif_pandas,
    load_clif_table,
)
from flair_benchmark.meds_etl.pipeline import build_meds_events

SORT = ["subject_id", "time", "code", "numeric_value", "text_value", "hospitalization_id"]


@pytest.fixture
def clif_dir(tmp_path):
    """Tiny CLIF corpus: hospitalization + vitals for 3 encounters."""
    data = tmp_path / "clif"
    data.mkdir()
    pl.DataFrame({
        "hospitalization_id": ["1", "2", "3"],
        "patient_id": ["P101", "P102", "P103"],  # CLIF ids are STRINGS, not numbers
        "admission_dttm": ["2024-01-01T00:00:00Z"] * 3,
    }).with_columns(
        pl.col("admission_dttm").str.to_datetime(time_zone="UTC")
    ).write_parquet(data / "clif_hospitalization.parquet")

    pl.DataFrame({
        "hospitalization_id": ["1", "1", "2", "2", "3", "3"],
        "recorded_dttm": [f"2024-01-01T0{i}:00:00Z" for i in range(6)],
        "vital_category": ["heart_rate", "weight_kg"] * 3,
        "vital_value": [60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
    }).with_columns(
        pl.col("recorded_dttm").str.to_datetime(time_zone="UTC")
    ).write_parquet(data / "clif_vitals.parquet")
    return data


@pytest.fixture
def clif_cfg(clif_dir):
    return {"data_directory": str(clif_dir), "filetype": "parquet",
            "timezone": "US/Central"}


@pytest.fixture
def elf_cfg(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "VITALTEST.yaml").write_text(
        "elf_version: '1.0'\n"
        "subject_id_col: patient_id\n"
        "vitals:\n"
        "  vital:\n"
        "    time: col(recorded_dttm)\n"
        "    code: [VITAL, col(vital_category)]\n"
        "    numeric_value: vital_value\n"
    )
    return {"concepts_dir": str(concepts), "domains": {"VITALTEST": {}}}


def test_hosp_filters_thresholds(monkeypatch):
    assert hosp_filters(None) is None
    assert hosp_filters(["1", "2"]) == {"hospitalization_id": ["1", "2"]}
    monkeypatch.setattr(_emit, "_PUSHDOWN_MAX_IDS", 1)
    assert hosp_filters(["1", "2"]) is None


def test_pushdown_equals_full_load(clif_cfg):
    ids = ["1", "3"]
    pushed = load_clif_table(clif_cfg, "vitals", hosp_ids=ids)
    full = load_clif_table(clif_cfg, "vitals").filter(
        pl.col("hospitalization_id").cast(pl.Utf8).is_in(ids))
    assert pushed.sort(pushed.columns).equals(full.sort(full.columns))
    assert set(pushed["hospitalization_id"].to_list()) == set(ids)


def test_pushdown_column_projection(clif_cfg):
    pdf = load_clif_pandas(clif_cfg, "vitals",
                           filters={"vital_category": ["weight_kg"]},
                           columns=["hospitalization_id", "vital_category", "vital_value"])
    assert list(pdf.columns) == ["hospitalization_id", "vital_category", "vital_value"]
    assert set(pdf["vital_category"]) == {"weight_kg"}


def test_pushdown_failure_falls_back(clif_cfg):
    with pytest.warns(UserWarning, match="pushdown failed"):
        pdf = load_clif_pandas(clif_cfg, "vitals",
                               filters={"no_such_column": ["x"]})
    assert len(pdf) == 6  # full load


def test_table_cache_loads_once_under_concurrency():
    cache = TableCache()
    calls = []

    def loader():
        calls.append(1)
        return object()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get_or_load(("k",), loader), range(32)))
    assert len(calls) == 1
    assert all(r is results[0] for r in results)


def test_table_cache_distinct_keys_load_separately():
    cache = TableCache()
    assert cache.get_or_load(("a",), lambda: 1) == 1
    assert cache.get_or_load(("b",), lambda: 2) == 2
    assert cache.get_or_load(("a",), lambda: 99) == 1


def test_serial_equals_parallel(clif_cfg, elf_cfg):
    ids = ["1", "2", "3"]
    serial, v1 = build_meds_events(clif_cfg, elf_cfg, ids, max_workers=1)
    parallel, v2 = build_meds_events(clif_cfg, elf_cfg, ids, max_workers=4)
    assert v1 == v2
    assert serial.sort(SORT).equals(parallel.sort(SORT))
    assert serial.height == 6  # every vitals row emits one event


def test_cohort_restriction(clif_cfg, elf_cfg):
    events, _ = build_meds_events(clif_cfg, elf_cfg, ["2"])
    assert set(events["hospitalization_id"].to_list()) == {"2"}
    assert events.height == 2
    # Alphanumeric ids survive: casting them to Int64 would null every row and
    # the null-filter in emit_events would then write an EMPTY table, silently.
    assert set(events["subject_id"].to_list()) == {"P102"}
