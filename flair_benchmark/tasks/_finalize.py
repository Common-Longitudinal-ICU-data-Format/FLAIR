"""Shared cohort finalization: stable row IDs + window numbering.

Every task's build() ends with `finalize_cohort(df, OUTPUT_COLS, META)`,
which selects the task's contract columns and appends:

  window_number  Int32   1-based k-th prediction timestep per stitched
                         encounter (hospitalization_join_id), ordered by
                         prediction_dttm. Only for one-per-timestep tasks;
                         constant 1 for one-per-stay tasks.
  prediction_id  str     Stable, auditable per-row key sites carry through
                         their predictions parquet:
                           {hospitalization_join_id}-{hospitalization_id}-{patient_id}-{window_number}
                         All four segments are always present (window_number = 1
                         when the task has no window), so any prediction row
                         traces back to its stitched stay, the specific
                         hospitalization inside that join block, the patient, and
                         the window. (For a non-stitched stay the join id equals
                         the hospitalization id, so those two segments coincide.)

Two levels of identity: hospitalization_join_id is the *stay-level* key (the
whole stitched encounter as one entity); prediction_id is the *prediction-level*
key (per-row prevalence and all per-row report metrics).

These columns let `flair build-report` bin discrimination by window and
join predictions back to the cohort without datetime arithmetic.
"""

from __future__ import annotations

import polars as pl


def finalize_cohort(df: pl.DataFrame, output_cols: list[str], meta: dict) -> pl.DataFrame:
    """Select contract columns and append window_number + prediction_id."""
    multi_row = meta["prediction_unit"] == "one-per-timestep"

    # The stitched-encounter key is the entity for window numbering + prediction_id;
    # fall back to hospitalization_id if a task somehow didn't carry it.
    block_key = ("hospitalization_join_id"
                 if "hospitalization_join_id" in df.columns
                 else "hospitalization_id")

    keep = list(output_cols)
    if "patient_id" in df.columns and "patient_id" not in keep:
        keep = ["patient_id", *keep]
    if block_key == "hospitalization_join_id" and block_key not in keep:
        keep = [*keep, "hospitalization_join_id"]
    df = df.select(keep)

    if multi_row and "prediction_dttm" in df.columns:
        df = df.with_columns(
            pl.col("prediction_dttm")
            .rank("ordinal")
            .over(block_key)
            .cast(pl.Int32)
            .alias("window_number")
        )
    else:
        df = df.with_columns(pl.lit(1, dtype=pl.Int32).alias("window_number"))

    # Auditable per-row key: join -> hospitalization -> patient -> window, all
    # always present (window_number = 1 when no window) so a prediction traces
    # back to its stitched stay, the specific hospitalization, patient, window.
    id_parts = [pl.col(block_key).cast(pl.Utf8)]            # hospitalization_join_id
    if "hospitalization_id" in df.columns:
        id_parts.append(pl.col("hospitalization_id").cast(pl.Utf8))
    if "patient_id" in df.columns:
        id_parts.append(pl.col("patient_id").cast(pl.Utf8))
    id_parts.append(pl.col("window_number").cast(pl.Utf8))
    df = df.with_columns(pl.concat_str(id_parts, separator="-").alias("prediction_id"))
    return df
