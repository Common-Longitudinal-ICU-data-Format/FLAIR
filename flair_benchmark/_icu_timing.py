"""Rank-based first/second ICU stay computation.

Used by tasks 1-6. Task 7 has its own ADT collapse logic and does NOT
import this helper.

Algorithm: filter ADT to location_category='icu', rank by in_dttm within
each hospitalization_id, then pull rank-1 (first ICU stay) and rank-2
(second ICU stay) timestamps.
"""

from __future__ import annotations

import polars as pl


def compute_icu_timing(adt: pl.DataFrame) -> pl.DataFrame:
    """One row per hospitalization_id with first/second ICU stay times.

    Columns returned:
      hospitalization_id, first_icu_start_time, first_icu_end_time,
      second_icu_start_time (nullable)
    """
    icu = adt.filter(pl.col("location_category").str.to_lowercase() == "icu")

    if icu.height == 0:
        return pl.DataFrame({
            "hospitalization_id": [],
            "first_icu_start_time": [],
            "first_icu_end_time": [],
            "second_icu_start_time": [],
        })

    ranked = icu.with_columns(
        pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_icu_n")
    )

    first = ranked.filter(pl.col("_icu_n") == 1).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("first_icu_start_time"),
        pl.col("out_dttm").alias("first_icu_end_time"),
    ])
    second = ranked.filter(pl.col("_icu_n") == 2).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("second_icu_start_time"),
    ])
    return first.join(second, on="hospitalization_id", how="left")
