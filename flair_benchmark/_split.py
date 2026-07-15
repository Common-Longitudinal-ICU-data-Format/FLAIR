"""Train/test split assignment used by every task.

Splits are assigned at the **encounter-block** grain (``hospitalization_join_id``):
every prediction row of one block lands in the same split, so a single encounter
never spans train and test (prevents leakage in count-feature / time-series models).
Every cohort always carries BOTH 'train' and 'test' — irrespective of whether the
site will fit a model or only run inference — so the cohort/Table 1 schema is
identical across sites.

Regimes (priority order; all block-grain):
  1. train_end AND test_start given  → date cutoff on the block's earliest
                                       admission_col (production temporal split).
  2. site == 'mimic' (no dates)      → RANDOM 75/25 over blocks. MIMIC admission
                                       dates are de-identified/shifted, so temporal
                                       order is meaningless; assignment is
                                       deterministic (hash-seeded), not date-driven.
  3. other site (no dates)           → TEMPORAL 75/25: blocks ordered by earliest
                                       admission_col, oldest 75% train, newest 25% test.

Data cutoff: for non-mimic sites, rows after ``_constants.DATA_CUTOFF`` are dropped
before splitting (current year not fully ETL'd). MIMIC is exempt (de-identified dates).
"""

from __future__ import annotations

import hashlib
from datetime import date

import polars as pl

from flair_benchmark._constants import DATA_CUTOFF


def _stable_digest(group_id: str) -> str:
    """Version-stable shuffle key: sha256 of the group id string.

    NOT ``pl.Expr.hash`` — polars documents its hash as unstable across
    releases, so a polars upgrade would silently re-deal the MIMIC train/test
    split (a model trained pre-upgrade would be scored on a test set containing
    its own training stays). sha256 of the id string is identical forever, on
    every machine and every library version.
    """
    return hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()


def assign_split(cohort: pl.DataFrame, site: str,
                 train_end: str | None, test_start: str | None,
                 *, admission_col: str = "admission_dttm",
                 split_col: str = "hospitalization_join_id",
                 train_ratio: float = 0.75) -> pl.DataFrame:
    """Add a 'split' column ∈ {'train', 'test', null}; always produces train + test.

    ``split_col`` is the grain at which train/test is decided (default
    ``hospitalization_join_id``); ``admission_col`` orders blocks for the temporal
    regime and drives the date cutoff for production sites.
    """
    is_mimic = bool(site) and site.lower() == "mimic"

    # Global data cutoff — drop rows after DATA_CUTOFF so partially-ETL'd current-year
    # data never enters the cohort. Skipped for mimic (de-identified/shifted dates).
    if not is_mimic and admission_col in cohort.columns:
        cohort = cohort.filter(pl.col(admission_col).dt.date() <= DATA_CUTOFF)

    # 1. Explicit production cutoff dates take precedence at any site.
    if train_end is not None and test_start is not None:
        return _date_split(cohort, train_end, test_start, admission_col, split_col)

    # 2. MIMIC, no dates → deterministic random split (dates de-identified).
    if is_mimic:
        return _random_split_by_group(cohort, split_col, train_ratio)

    # 3. Other site, no dates → temporal split (oldest 75% train, newest 25% test).
    return _ratio_split_by_group(cohort, split_col, admission_col, train_ratio)


def _date_split(cohort: pl.DataFrame, train_end: str, test_start: str,
                admission_col: str, split_col: str) -> pl.DataFrame:
    """Production temporal split on explicit cutoff dates (block-grain)."""
    train_end_d = date.fromisoformat(train_end)
    test_start_d = date.fromisoformat(test_start)
    # Decide the split once per encounter block (by its earliest admission_col),
    # then broadcast — so a continuous encounter whose per-row prediction_dttm
    # straddles train_end/test_start can never be torn across train and test
    # (same guarantee the ratio path gives). Fall back to per-row only when the
    # block key is absent.
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
    """Assign 'train'/'test' once per ``split_col`` group at RANDOM (block-grain).

    Used for MIMIC, whose admission dates are de-identified/shifted so temporal order
    carries no meaning. Order is the sha256 digest of the group id (see
    ``_stable_digest`` — polars' own ``.hash`` is version-unstable), so the split
    is random-looking yet reproducible across runs, machines AND library
    versions; the first ``train_ratio`` of the digest-ordered groups are 'train'.
    Falls back to a per-row split only if ``split_col`` is absent.

    Split-version note: builds before 2026-07 ordered by ``pl.Expr.hash(seed=0)``;
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
    """Assign 'train'/'test' once per ``split_col`` group; broadcast to all its rows.

    Groups are ordered by their earliest ``order_col`` (then group id) so the split
    is deterministic; the first ``train_ratio`` of groups are 'train'. Falls back to a
    per-row split only if ``split_col`` is absent.
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
