"""The encounter key every task is built on, and the train/test split over it.

Two halves of one idea, which is why they share a file:

  1. STITCHING builds `hospitalization_join_id` — the canonical encounter key.
  2. SPLITTING assigns train/test *at that key's grain*.

A single CLIF `hospitalization_id` is one admission-to-discharge record. When a
patient is discharged and re-admitted within a few hours (an ED-to-inpatient
transfer, a facility switch), that one continuous encounter is split across *two*
hospitalization_ids. clifpy's `stitch_encounters` links such records into one
`encounter_block`; we rename it `hospitalization_join_id` and thread it through
every FLAIR stage.

The split is then decided once per join id and broadcast to all of that
encounter's rows, so a single encounter can never span train and test. That is
what prevents leakage in count-feature and time-series models — and it only works
because the split is keyed on the stitched id rather than the raw one.

The index is built ONCE per (config, time_interval) and cached to parquet, so the
same stitched mapping is reused by every task, feature and metadata run.

Depends on: cohort/clif.py (loaders), constants.py (DATA_CUTOFF), clifpy.
Used by: every task, features/_scope.py, cohort/table1.py, cohort/encounter.py.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl

from flair_benchmark.cohort.clif import load_adt, load_hospitalization
from flair_benchmark.constants import DATA_CUTOFF

JOIN_ID = "hospitalization_join_id"


# ===========================================================================
# Part 1 — Encounter stitching
#
# Public API:
#   build_encounter_index(cfg, time_interval)         -> always rebuilds
#   load_or_build_encounter_index(cfg, time_interval) -> cached
#   attach_join_id(df, idx)                           -> left-join + self-fallback
#   members_of_joins(idx, join_ids)                   -> join_id -> all member hosp_ids
#
# The mapping frame has columns:
#   hospitalization_id, hospitalization_join_id, patient_id
# ===========================================================================


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
    """Run clifpy `stitch_encounters` on the FULL hospitalization + ADT set.

    Stitching shifts within `patient_id`, so it must see every hospitalization for
    the relevant patients — not just the ICU subset a task ends up using. Passing
    a pre-filtered frame here would break encounters apart.

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

    # stitch_encounters is pandas-based. clifpy hands us pandas upstream but our
    # loaders convert to polars, so convert back just for this call.
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


# Process-level memo for the cached index. One task build calls
# load_or_build_encounter_index several times (base cohort, encounter attributes,
# per-table attach) and the parquet can be large, so we read it once per
# (path, mtime) rather than once per call.
_INDEX_MEMO: dict[tuple[str, int], pl.DataFrame] = {}


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

    A cohort knows its own hospitalization_ids, but one `hospitalization_join_id`
    can stitch several admissions together — and a sibling member may be absent
    from the cohort (e.g. a non-ICU transfer). Pulling data by cohort ids alone
    would silently drop that sibling's events. This resolves each join_id back to
    its FULL member set via the stitch index, so extraction sees the whole
    encounter.

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


# ===========================================================================
# Part 2 — Train/test split assignment
#
# Every cohort always carries BOTH 'train' and 'test', whether or not the site
# will fit a model, so the cohort and Table 1 schemas are identical across sites.
#
# Regimes, in priority order (all at block grain):
#   1. train_end AND test_start given -> date cutoff on the block's earliest
#      admission_col (the production temporal split).
#   2. site == 'mimic', no dates      -> RANDOM 75/25 over blocks. MIMIC admission
#      dates are de-identified/shifted, so temporal order is meaningless;
#      assignment is deterministic (hash-seeded), not date-driven.
#   3. other site, no dates           -> TEMPORAL 75/25: blocks ordered by earliest
#      admission_col, oldest 75% train, newest 25% test.
#
# Data cutoff: for non-mimic sites, rows after constants.DATA_CUTOFF are dropped
# before splitting (the current year is not fully ETL'd). MIMIC is exempt.
# ===========================================================================


def _stable_digest(group_id: str) -> str:
    """Version-stable shuffle key: sha256 of the group id string.

    Deliberately NOT `pl.Expr.hash` — polars documents its hash as unstable across
    releases, so a polars upgrade would silently re-deal the MIMIC train/test
    split, and a model trained before the upgrade would be scored on a test set
    containing its own training stays. sha256 of the id string is identical
    forever, on every machine and every library version.
    """
    return hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()


def assign_split(cohort: pl.DataFrame, site: str,
                 train_end: str | None, test_start: str | None,
                 *, admission_col: str = "admission_dttm",
                 split_col: str = "hospitalization_join_id",
                 train_ratio: float = 0.75) -> pl.DataFrame:
    """Add a 'split' column in {'train', 'test', null}; always produces train + test.

    `split_col` is the grain at which train/test is decided (default
    `hospitalization_join_id`); `admission_col` orders blocks for the temporal
    regime and drives the date cutoff for production sites.
    """
    is_mimic = bool(site) and site.lower() == "mimic"

    # Global data cutoff — drop rows after DATA_CUTOFF so partially-ETL'd
    # current-year data never enters the cohort. Skipped for mimic
    # (de-identified/shifted dates would make a real-date cap empty the cohort).
    if not is_mimic and admission_col in cohort.columns:
        cohort = cohort.filter(pl.col(admission_col).dt.date() <= DATA_CUTOFF)

    # 1. Explicit production cutoff dates take precedence at any site.
    if train_end is not None and test_start is not None:
        return _date_split(cohort, train_end, test_start, admission_col, split_col)

    # 2. MIMIC, no dates -> deterministic random split (dates de-identified).
    if is_mimic:
        return _random_split_by_group(cohort, split_col, train_ratio)

    # 3. Other site, no dates -> temporal split (oldest 75% train, newest 25% test).
    return _ratio_split_by_group(cohort, split_col, admission_col, train_ratio)


def _date_split(cohort: pl.DataFrame, train_end: str, test_start: str,
                admission_col: str, split_col: str) -> pl.DataFrame:
    """Production temporal split on explicit cutoff dates (block grain)."""
    train_end_d = date.fromisoformat(train_end)
    test_start_d = date.fromisoformat(test_start)
    # Decide the split once per encounter block (by its earliest admission_col),
    # then broadcast — so a continuous encounter whose per-row prediction_dttm
    # straddles train_end/test_start can never be torn across train and test
    # (the same guarantee the ratio path gives). Fall back to per-row only when
    # the block key is absent.
    block_date = (
        pl.col(admission_col).min().over(split_col).dt.date()
        if split_col in cohort.columns
        else pl.col(admission_col).dt.date()
    )
    return cohort.with_columns(
        pl.when(block_date <= train_end_d).then(pl.lit("train"))
        .when(block_date >= test_start_d).then(pl.lit("test"))
        .otherwise(pl.lit(None))
        .alias("split")
    )


def _random_split_by_group(cohort: pl.DataFrame, split_col: str,
                           train_ratio: float) -> pl.DataFrame:
    """Assign 'train'/'test' once per `split_col` group at RANDOM (block grain).

    Used for MIMIC, whose admission dates are de-identified/shifted so temporal
    order carries no meaning. Order is the sha256 digest of the group id (see
    `_stable_digest`), so the split is random-looking yet reproducible across
    runs, machines AND library versions; the first `train_ratio` of the
    digest-ordered groups are 'train'. Falls back to a per-row split only if
    `split_col` is absent.

    Split-version note: builds before 2026-07 ordered by `pl.Expr.hash(seed=0)`;
    this deal supersedes them.
    """
    if split_col not in cohort.columns:
        # No block key (rare): deterministic per-row split by row order.
        ranked = cohort.with_row_index("_row_num")
        cutoff = int(ranked.height * train_ratio)
        return ranked.with_columns(
            pl.when(pl.col("_row_num") < cutoff).then(pl.lit("train"))
            .otherwise(pl.lit("test")).alias("split")
        ).drop("_row_num")

    groups = (
        cohort.select(pl.col(split_col).unique())
        .with_columns(
            pl.col(split_col).cast(pl.Utf8)
            .map_elements(_stable_digest, return_dtype=pl.Utf8).alias("_h"))
        .sort(["_h", split_col])
        .with_row_index("_gi")
    )
    cutoff = int(groups.height * train_ratio)
    groups = groups.with_columns(
        pl.when(pl.col("_gi") < cutoff).then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).select([split_col, "split"])
    return cohort.join(groups, on=split_col, how="left")


def _ratio_split_by_group(cohort: pl.DataFrame, split_col: str, order_col: str,
                          train_ratio: float) -> pl.DataFrame:
    """Assign 'train'/'test' once per `split_col` group; broadcast to all its rows.

    Groups are ordered by their earliest `order_col` (then group id) so the split
    is deterministic; the first `train_ratio` of groups are 'train'. Falls back to
    a per-row split only if `split_col` is absent.
    """
    if split_col not in cohort.columns:
        return _ratio_split_per_row(cohort, order_col, train_ratio)

    groups = (
        cohort.group_by(split_col)
        .agg(pl.col(order_col).min().alias("_g_order"))
        .sort(["_g_order", split_col])
        .with_row_index("_gi")
    )
    cutoff = int(groups.height * train_ratio)
    groups = groups.with_columns(
        pl.when(pl.col("_gi") < cutoff).then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).select([split_col, "split"])
    return cohort.join(groups, on=split_col, how="left")


def _ratio_split_per_row(cohort: pl.DataFrame, order_col: str,
                         train_ratio: float) -> pl.DataFrame:
    sorted_df = cohort.sort(order_col).with_row_index("_row_num")
    cutoff = int(sorted_df.height * train_ratio)
    return (
        sorted_df.with_columns(
            pl.when(pl.col("_row_num") < cutoff).then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("split")
        ).drop("_row_num")
    )
