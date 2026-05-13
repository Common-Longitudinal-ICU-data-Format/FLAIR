"""Train/test split assignment used by every task."""

from __future__ import annotations

from datetime import date

import polars as pl


def assign_split(cohort: pl.DataFrame, site: str,
                 train_end: str | None, test_start: str | None,
                 *, admission_col: str = "admission_dttm",
                 train_ratio: float = 0.7) -> pl.DataFrame:
    """Add a 'split' column ∈ {'train', 'test', null}.

    Two regimes:
      - site == 'mimic' OR train_end is None OR test_start is None:
            sort by admission_col; first `train_ratio` of rows are 'train',
            rest are 'test'. Used for MIMIC where dates are de-identified.
      - Otherwise: date cutoff. admission <= train_end → 'train';
            admission >= test_start → 'test'; in between → null (washout,
            dropped by caller).
    """
    if site and site.lower() == "mimic" or train_end is None or test_start is None:
        return _mimic_70_30(cohort, admission_col, train_ratio)

    train_end_d = date.fromisoformat(train_end)
    test_start_d = date.fromisoformat(test_start)
    return cohort.with_columns(
        pl.when(pl.col(admission_col).dt.date() <= train_end_d).then(pl.lit("train"))
        .when(pl.col(admission_col).dt.date() >= test_start_d).then(pl.lit("test"))
        .otherwise(pl.lit(None))
        .alias("split")
    )


def _mimic_70_30(cohort: pl.DataFrame, admission_col: str,
                 train_ratio: float) -> pl.DataFrame:
    sorted_df = cohort.sort(admission_col).with_row_index("_row_num")
    cutoff = int(sorted_df.height * train_ratio)
    return (
        sorted_df.with_columns(
            pl.when(pl.col("_row_num") < cutoff).then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("split")
        ).drop("_row_num")
    )
