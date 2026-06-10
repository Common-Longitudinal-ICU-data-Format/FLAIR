"""Shared cohort finalization: stable row IDs + window numbering.

Every task's build() ends with `finalize_cohort(df, OUTPUT_COLS, META)`,
which selects the task's contract columns and appends:

  window_number  Int32   1-based k-th prediction timestep per stay, ordered
                         by prediction_dttm. Only for one-per-timestep tasks;
                         constant 1 for one-per-stay tasks.
  prediction_id  str     Stable per-row key sites carry through their
                         predictions parquet:
                           {patient_id}-{hospitalization_id}-{window_number}
                         (patient_id segment omitted when unavailable).

These columns let `flair build-report` bin discrimination by window and
join predictions back to the cohort without datetime arithmetic.
"""

from __future__ import annotations

import polars as pl


def finalize_cohort(df: pl.DataFrame, output_cols: list[str], meta: dict) -> pl.DataFrame:
    """Select contract columns and append window_number + prediction_id."""
    multi_row = meta["prediction_unit"] == "one-per-timestep"

    keep = list(output_cols)
    if "patient_id" in df.columns and "patient_id" not in keep:
        keep = ["patient_id", *keep]
    df = df.select(keep)

    if multi_row and "prediction_dttm" in df.columns:
        df = df.with_columns(
            pl.col("prediction_dttm")
            .rank("ordinal")
            .over("hospitalization_id")
            .cast(pl.Int32)
            .alias("window_number")
        )
    else:
        df = df.with_columns(pl.lit(1, dtype=pl.Int32).alias("window_number"))

    id_parts = []
    if "patient_id" in df.columns:
        id_parts.append(pl.col("patient_id").cast(pl.Utf8))
    id_parts.append(pl.col("hospitalization_id").cast(pl.Utf8))
    if multi_row:
        id_parts.append(pl.col("window_number").cast(pl.Utf8))
    df = df.with_columns(pl.concat_str(id_parts, separator="-").alias("prediction_id"))
    return df
