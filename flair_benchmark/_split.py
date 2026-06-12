"""Train/test split assignment used by every task.

Splits are assigned at the **encounter-block** grain (``hospitalization_join_id``):
every prediction row of one block lands in the same split, so a single encounter
never spans train and test (prevents leakage in count-feature / time-series models).

Three regimes:
  - site == 'mimic'                  → 75/25 ratio split over blocks (dates de-identified).
  - other site, no train_end/test_start → all rows 'test' (external full-data validation).
  - other site, with cutoff dates    → date cutoff (production temporal split).
"""

from __future__ import annotations

from datetime import date

import polars as pl


def assign_split(cohort: pl.DataFrame, site: str,
                 train_end: str | None, test_start: str | None,
                 *, admission_col: str = "admission_dttm",
                 split_col: str = "hospitalization_join_id",
                 train_ratio: float = 0.75) -> pl.DataFrame:
    """Add a 'split' column ∈ {'train', 'test', null}.

    ``split_col`` is the grain at which train/test is decided (default
    ``hospitalization_join_id``); ``admission_col`` orders blocks deterministically
    for the ratio regime and drives the date cutoff for production sites.
    """
    is_mimic = bool(site) and site.lower() == "mimic"

    if is_mimic:
        return _ratio_split_by_group(cohort, split_col, admission_col, train_ratio)

    if train_end is None or test_start is None:
        # External site, no split dates → evaluate on the full cohort.
        return cohort.with_columns(pl.lit("test").alias("split"))

    train_end_d = date.fromisoformat(train_end)
    test_start_d = date.fromisoformat(test_start)
    return cohort.with_columns(
        pl.when(pl.col(admission_col).dt.date() <= train_end_d).then(pl.lit("train"))
        .when(pl.col(admission_col).dt.date() >= test_start_d).then(pl.lit("test"))
        .otherwise(pl.lit(None))
        .alias("split")
    )


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
