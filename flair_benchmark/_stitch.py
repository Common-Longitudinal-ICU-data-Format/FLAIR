"""Encounter stitching — the shared preprocessing step that runs before any task.

A single CLIF ``hospitalization_id`` is one admission→discharge record. When a
patient is discharged and re-admitted within a few hours (ED → inpatient transfer,
a facility switch), that one continuous encounter is split across *two*
hospitalization_ids. clifpy's ``stitch_encounters`` links such records (default
6h gap) into one ``encounter_block``; here we rename it ``hospitalization_join_id``
and expose it as the canonical encounter key threaded through every FLAIR stage.

The index is built **once per (config, time_interval)** and cached to parquet, so
the same stitched mapping is reused by every task / feature / metadata run rather
than recomputed each time.

Public API:
    build_encounter_index(cfg, time_interval)        -> pl.DataFrame  (always rebuilds)
    load_or_build_encounter_index(cfg, time_interval)-> pl.DataFrame  (cached)
    attach_join_id(df, idx)                          -> pl.DataFrame  (left-join + fallback)
    members_of_joins(idx, join_ids)                  -> list[str]     (join_id -> all member hosp_ids)

The mapping frame has columns: hospitalization_id, hospitalization_join_id, patient_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from flair_benchmark._clif import load_adt, load_hospitalization

JOIN_ID = "hospitalization_join_id"


def _cache_dir(cfg: dict) -> Path:
    """Where the encounter index parquet lives.

    Precedence:
      1. cfg["cache_directory"] — set this to the folder where you write task
         parquet so the index sits alongside your outputs.
      2. <data_directory>/_flair_cache (default; keeps the index next to source).
    """
    explicit = cfg.get("cache_directory")
    base = Path(explicit) if explicit else Path(cfg["data_directory"]) / "_flair_cache"
    return base


def _cache_path(cfg: dict, time_interval: int) -> Path:
    return _cache_dir(cfg) / f"encounter_index_{time_interval}h.parquet"


def build_encounter_index(cfg: dict, time_interval: int = 6) -> pl.DataFrame:
    """Run clifpy ``stitch_encounters`` on the FULL hospitalization+ADT set.

    Stitching shifts within ``patient_id``, so it must see every hospitalization
    for the relevant patients — not just the ICU subset a task ends up using.

    Returns one row per hospitalization_id:
        hospitalization_id, hospitalization_join_id (Utf8), patient_id
    """
    from clifpy.utils.stitching_encounters import stitch_encounters

    # stitch_encounters requires admission_type_category (absent from base_icu_cohort).
    hosp = load_hospitalization(
        cfg,
        columns=["hospitalization_id", "patient_id", "admission_dttm",
                 "discharge_dttm", "age_at_admission",
                 "admission_type_category", "discharge_category"],
    )
    adt = load_adt(cfg)

    # stitch_encounters is pandas; clifpy already gave us pandas upstream but we
    # converted to polars in the loaders — convert back for the call.
    _, _, mapping = stitch_encounters(
        hospitalization=hosp.to_pandas(),
        adt=adt.to_pandas(),
        time_interval=time_interval,
    )

    idx = (
        pl.from_pandas(mapping)
        .with_columns([
            pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col("encounter_block").cast(pl.Utf8).alias(JOIN_ID),
        ])
        .select(["hospitalization_id", JOIN_ID])
        .unique(subset=["hospitalization_id"])
    )

    # Carry patient_id for downstream block-level analysis / traceability.
    pat = hosp.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("patient_id").cast(pl.Utf8),
    ]).unique(subset=["hospitalization_id"])

    return idx.join(pat, on="hospitalization_id", how="left")


def load_or_build_encounter_index(cfg: dict, time_interval: int = 6,
                                  cache_dir: Optional[str] = None) -> pl.DataFrame:
    """Read the cached encounter index, building (and saving) it on first use.

    This is the "run before all tasks, saved" step: the first cohort build writes
    the parquet, every later run reads it.
    """
    if cache_dir is not None:
        cfg = {**cfg, "cache_directory": cache_dir}
    path = _cache_path(cfg, time_interval)
    if path.exists():
        # process-level memo: one task build calls this several times
        # (base cohort, encounter attributes, per-table attach), and the
        # index parquet can be large — read it once per (path, mtime).
        key = (str(path), path.stat().st_mtime_ns)
        cached = _INDEX_MEMO.get(key)
        if cached is None:
            cached = pl.read_parquet(path)
            _INDEX_MEMO.clear()          # one live index per process is plenty
            _INDEX_MEMO[key] = cached
        return cached

    idx = build_encounter_index(cfg, time_interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx.write_parquet(path)
    return idx


_INDEX_MEMO: dict[tuple[str, int], pl.DataFrame] = {}


def attach_join_id(df: pl.DataFrame, idx: pl.DataFrame) -> pl.DataFrame:
    """Left-join hospitalization_join_id onto any frame keyed on hospitalization_id.

    A hospitalization_id absent from the index (never stitched) maps to *itself*,
    so the join key is always non-null and a non-stitched stay is its own block.
    """
    mapping = idx.select(["hospitalization_id", JOIN_ID]).unique(subset=["hospitalization_id"])
    out = df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).join(
        mapping.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
        on="hospitalization_id", how="left",
    )
    return out.with_columns(
        pl.col(JOIN_ID).fill_null(pl.col("hospitalization_id"))
    )


def members_of_joins(idx: pl.DataFrame, join_ids: list[str]) -> list[str]:
    """All member hospitalization_ids of the given stitched encounters.

    A cohort knows its own hospitalization_ids, but a ``hospitalization_join_id``
    can stitch several admissions together — and a sibling member may be absent
    from the cohort (e.g. a non-ICU transfer). Pulling MEDS by cohort ids alone
    would silently drop that sibling's events. This resolves each join_id back to
    its FULL member set via the stitch index, so the ETL sees the whole encounter.

    join_ids absent from the index (never stitched) map to themselves, so the
    returned list always covers the input even for self-blocks. Matched as Utf8.
    """
    wanted = {str(j) for j in join_ids}
    sidx = idx.with_columns(
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col(JOIN_ID).cast(pl.Utf8),
    )
    matched = sidx.filter(pl.col(JOIN_ID).is_in(list(wanted)))
    members = set(matched["hospitalization_id"].to_list())
    # Self-fallback: a never-stitched join_id (== its own hospitalization_id) may be
    # absent from the index's join column; add those raw ids so none are lost. Only
    # the unmatched ones, so synthetic block ids never leak into the hosp_id list.
    present = set(matched[JOIN_ID].to_list())
    members |= (wanted - present)
    return sorted(members)
