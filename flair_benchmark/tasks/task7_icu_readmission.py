"""
Task 7: Unplanned ICU Readmission (binary classification)

Predict whether the patient will have an UNPLANNED return to the ICU
during the same hospitalization.

Pipeline (reads top-to-bottom):
  1. base_icu_cohort()   — adult ICU hospitalizations + ADT (see _clif.py)
  2. _collapse_adt()     — map locations, remove ICU→Procedural→ICU sandwiches,
                           merge consecutive same-location rows, re-rank.
                           (custom logic; does NOT use _icu_timing.compute_icu_timing)
  3. exclusions:
       a. death during first ICU stay
       b. direct ICU discharge (last location is ICU)
       c. first ICU stay < 1 hour
       d. intermediate ER visit after first location
  4. label = 1 if any second ICU stay exists after collapse, else 0
  5. window = entire first ICU stay (variable length)
  6. assign_split()      — MIMIC 70/30 or date cutoffs (see _split.py)

Location classification:
  Keep as-is: icu, ward, ed, procedural
  Collapse to 'Other': stepdown, l&d, hospice, psych, rehab, radiology,
                       dialysis, other (and any unknown values)

Rationale for ICU→Procedural→ICU = planned (no readmission):
  e.g. patient leaves ICU briefly for OR / cath lab and returns. This
  is a logistical move, not an unplanned deterioration.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._split import assign_split

LOCATION_MAPPING = {
    "icu": "ICU", "ward": "Ward", "ed": "ER", "procedural": "Procedural",
    "stepdown": "Other", "l&d": "Other", "hospice": "Other", "psych": "Other",
    "rehab": "Other", "radiology": "Other", "dialysis": "Other", "other": "Other",
}

META = {
    "name": "task7_icu_readmission",
    "display_name": "Unplanned ICU Readmission",
    "task_type": "binary",
    "label_column": "label_icu_readmission",
    "positive_class": "readmitted",
    "input_window_hours": None,
    "prediction_window": None,
    "cohort_description": "Adults with first ICU stay ≥ 1h, no in-ICU death, "
                          "no direct ICU discharge, no intermediate ER",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "at least one ICU ADT record",
        "first ICU stay duration >= 1 hour",
    ],
    "exclusion_criteria": [
        "death during first ICU stay",
        "direct ICU discharge (last location = ICU)",
        "intermediate ER visit",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
}

OUTPUT_COLS = [
    "hospitalization_id", "window_start", "window_end",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)
    cohort, adt = base_icu_cohort(cfg)
    timing = _collapse_adt(adt)

    df = cohort.join(timing, on="hospitalization_id", how="inner")

    # Exclusion: death during first ICU stay
    if "death_dttm" in df.columns:
        df = df.filter(~(
            pl.col("death_dttm").is_not_null()
            & (pl.col("death_dttm") >= pl.col("first_icu_start_time"))
            & (pl.col("death_dttm") <= pl.col("first_icu_end_time"))
        ))

    # Exclusion: direct ICU discharge, short first ICU stay, intermediate ER
    df = df.filter(~pl.col("is_direct_icu_discharge"))
    df = df.filter(
        (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
        >= pl.duration(hours=1)
    )
    df = df.filter(~pl.col("has_intermediate_er"))

    df = df.with_columns(
        pl.col("second_icu_start_time").is_not_null()
        .cast(pl.Int32).alias("label_icu_readmission")
    )

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        pl.col("first_icu_end_time").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)


def _collapse_adt(adt: pl.DataFrame) -> pl.DataFrame:
    """ICU readmission-aware ADT collapse.

    Returns one row per hospitalization with:
      first_icu_start_time, first_icu_end_time, second_icu_start_time (nullable),
      is_direct_icu_discharge (bool), has_intermediate_er (bool)
    """
    if adt.height == 0:
        return pl.DataFrame({
            "hospitalization_id": [], "first_icu_start_time": [],
            "first_icu_end_time": [], "second_icu_start_time": [],
            "is_direct_icu_discharge": [], "has_intermediate_er": [],
        })

    a = adt.with_columns(
        pl.col("location_category").str.to_lowercase()
        .replace(LOCATION_MAPPING, default="Other")
        .alias("std_location")
    ).sort(["hospitalization_id", "in_dttm"])

    # Drop procedural rows sandwiched between two ICU rows (planned readmission).
    a = a.with_columns([
        pl.col("std_location").shift(1).over("hospitalization_id").alias("_prev"),
        pl.col("std_location").shift(-1).over("hospitalization_id").alias("_next"),
    ])
    a = a.filter(~(
        (pl.col("std_location") == "Procedural")
        & (pl.col("_prev") == "ICU") & (pl.col("_next") == "ICU")
    ))

    # Merge consecutive same-location rows: assign a group id whenever the
    # location changes within a hospitalization.
    a = a.sort(["hospitalization_id", "in_dttm"]).with_columns(
        (pl.col("std_location")
         != pl.col("std_location").shift(1).over("hospitalization_id"))
        .fill_null(True)
        .alias("_changed")
    )
    a = a.with_columns(
        pl.col("_changed").cum_sum().over("hospitalization_id").alias("_grp")
    )
    merged = (
        a.group_by(["hospitalization_id", "_grp"])
        .agg([pl.col("in_dttm").min().alias("in_dttm"),
              pl.col("out_dttm").max().alias("out_dttm"),
              pl.col("std_location").first().alias("std_location")])
        .sort(["hospitalization_id", "in_dttm"])
        .with_columns(
            pl.col("in_dttm").rank("ordinal").over("hospitalization_id")
            .alias("_loc_rank")
        )
    )

    # First/second ICU after merging.
    icu = (
        merged.filter(pl.col("std_location") == "ICU")
        .with_columns(
            pl.col("in_dttm").rank("ordinal").over("hospitalization_id")
            .alias("_icu_rank")
        )
    )
    first = icu.filter(pl.col("_icu_rank") == 1).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("first_icu_start_time"),
        pl.col("out_dttm").alias("first_icu_end_time"),
    ])
    second = icu.filter(pl.col("_icu_rank") == 2).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("second_icu_start_time"),
    ])

    # Direct ICU discharge = last merged location is ICU.
    last_loc = (
        merged.sort(["hospitalization_id", "in_dttm"])
        .group_by("hospitalization_id")
        .agg(pl.col("std_location").last().alias("_last_loc"))
        .select([
            "hospitalization_id",
            (pl.col("_last_loc") == "ICU").alias("is_direct_icu_discharge"),
        ])
    )

    # Intermediate ER = any ER row whose rank > 1.
    inter_er = (
        merged.filter((pl.col("_loc_rank") > 1) & (pl.col("std_location") == "ER"))
        .select("hospitalization_id").unique()
        .with_columns(pl.lit(True).alias("has_intermediate_er"))
    )

    return (
        first.join(second, on="hospitalization_id", how="left")
        .join(last_loc, on="hospitalization_id", how="left")
        .join(inter_er, on="hospitalization_id", how="left")
        .with_columns([
            pl.col("is_direct_icu_discharge").fill_null(False),
            pl.col("has_intermediate_er").fill_null(False),
        ])
    )
