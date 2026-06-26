"""Unplanned ICU readmission timing from the CLIF ADT table.

Computes, per hospitalization, the first ICU stay window and whether the
patient returns to the ICU during the SAME hospitalization after an
intervening NON-procedural location. A planned readmission --- i.e.
ICU -> Procedural -> ICU (a trip to the cath lab / OR and back) --- is
NOT a readmission: the procedural row is removed and the two ICU stays
are merged into one.

Algorithm:
  1. Map raw location_category to a coarse std_location
     (ICU / Ward / ER / Procedural / Other) via LOCATION_MAPPING.
  2. Drop procedural rows sandwiched between two ICU rows
     (ICU -> Procedural -> ICU = planned, collapse to a single ICU stay).
  3. Merge consecutive same-location rows into one location_group
     (ICU -> ICU back-to-back ADT rows become a single stay).
  4. The 1st merged ICU group gives first_icu_start/end_time; the 2nd
     merged ICU group (if any) is the unplanned readmission.
  5. Flag direct ICU discharge (last location is ICU; no chance to
     readmit) and intermediate ER (ER after the first location).
"""

from __future__ import annotations

import polars as pl

# Coarse location buckets. Anything not explicitly relevant becomes "Other";
# an ICU -> Other -> ICU pattern still counts as an unplanned readmission.
LOCATION_MAPPING = {
    "icu": "ICU",
    "ward": "Ward",
    "ed": "ER",
    "procedural": "Procedural",
    "stepdown": "Other",
    "l&d": "Other",
    "hospice": "Other",
    "psych": "Other",
    "rehab": "Other",
    "radiology": "Other",
    "dialysis": "Other",
    "other": "Other",
}

_EMPTY_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "first_icu_start_time": pl.Datetime,
    "first_icu_end_time": pl.Datetime,
    "second_icu_start_time": pl.Datetime,
    "is_direct_icu_discharge": pl.Boolean,
    "has_intermediate_er": pl.Boolean,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_SCHEMA)


def compute_readmission_timing(adt: pl.DataFrame) -> pl.DataFrame:
    """One row per hospitalization_id with unplanned-readmission timing.

    Columns returned:
      hospitalization_id, first_icu_start_time, first_icu_end_time,
      second_icu_start_time (null if no unplanned readmission),
      is_direct_icu_discharge (bool), has_intermediate_er (bool)
    """
    if adt.height == 0:
        return _empty()

    # 1. Standardize location categories.
    adt = adt.with_columns(
        pl.col("location_category")
        .str.to_lowercase()
        .replace(LOCATION_MAPPING)
        .alias("std_location")
    ).sort(["hospitalization_id", "in_dttm"])

    # 2. Remove procedural rows sandwiched between two ICU rows (planned).
    adt = adt.with_columns([
        pl.col("std_location").shift(1).over("hospitalization_id").alias("_prev"),
        pl.col("std_location").shift(-1).over("hospitalization_id").alias("_next"),
    ])
    adt = adt.filter(
        ~(
            (pl.col("std_location") == "Procedural")
            & (pl.col("_prev") == "ICU")
            & (pl.col("_next") == "ICU")
        )
    ).sort(["hospitalization_id", "in_dttm"])

    if adt.height == 0:
        return _empty()

    # 3. Merge consecutive same-location rows into location groups.
    adt = adt.with_columns(
        (
            pl.col("std_location")
            != pl.col("std_location").shift(1).over("hospitalization_id")
        )
        .fill_null(True)
        .alias("_changed")
    )
    adt = adt.with_columns(
        pl.col("_changed").cum_sum().over("hospitalization_id").alias("_group")
    )

    merged = (
        adt.group_by(["hospitalization_id", "_group"])
        .agg([
            pl.col("in_dttm").min().alias("in_dttm"),
            pl.col("out_dttm").max().alias("out_dttm"),
            pl.col("std_location").first().alias("std_location"),
        ])
        .sort(["hospitalization_id", "in_dttm"])
    )

    merged = merged.with_columns(
        pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_loc_rank")
    )

    # 4. First and second ICU groups (after merging).
    icu = merged.filter(pl.col("std_location") == "ICU").with_columns(
        pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_icu_rank")
    )

    first_icu = icu.filter(pl.col("_icu_rank") == 1).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("first_icu_start_time"),
        pl.col("out_dttm").alias("first_icu_end_time"),
    ])
    second_icu = icu.filter(pl.col("_icu_rank") == 2).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("second_icu_start_time"),
    ])

    # 5. Direct ICU discharge (last merged location is ICU).
    last_loc = merged.group_by("hospitalization_id").agg(
        pl.col("std_location").sort_by("in_dttm").last().alias("_last")
    ).select([
        "hospitalization_id",
        (pl.col("_last") == "ICU").alias("is_direct_icu_discharge"),
    ])

    # Intermediate ER (ER appears after the first location).
    inter_er = (
        merged.filter((pl.col("_loc_rank") > 1) & (pl.col("std_location") == "ER"))
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("has_intermediate_er"))
    )

    result = (
        first_icu.join(second_icu, on="hospitalization_id", how="left")
        .join(last_loc, on="hospitalization_id", how="left")
        .join(inter_er, on="hospitalization_id", how="left")
        .with_columns([
            pl.col("is_direct_icu_discharge").fill_null(False),
            pl.col("has_intermediate_er").fill_null(False),
        ])
    )
    return result
