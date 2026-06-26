"""First ICU stay computation.

Algorithm: filter ADT to location_category='icu', rank by in_dttm within
each hospitalization_id, then pull rank-1 (first ICU stay) timestamps.
"""

from __future__ import annotations

import polars as pl


def compute_icu_timing(adt: pl.DataFrame) -> pl.DataFrame:
    """One row per hospitalization_id with first ICU stay times.

    Columns returned:
      hospitalization_id, first_icu_start_time, first_icu_end_time
    """
    icu = adt.filter(pl.col("location_category").str.to_lowercase() == "icu")

    if icu.height == 0:
        return pl.DataFrame({
            "hospitalization_id": [],
            "first_icu_start_time": [],
            "first_icu_end_time": [],
        })

    return (
        icu.with_columns(
            pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_icu_n")
        )
        .filter(pl.col("_icu_n") == 1)
        .select([
            "hospitalization_id",
            pl.col("in_dttm").alias("first_icu_start_time"),
            pl.col("out_dttm").alias("first_icu_end_time"),
        ])
    )
