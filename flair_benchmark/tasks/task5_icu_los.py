"""
Task 5: Remaining ICU Length of Stay (regression)

Pipeline (reads top-to-bottom):
  1. base_icu_cohort()    — adult ICU hospitalizations + ADT (see _clif.py)
  2. compute_icu_timing() — rank-based first ICU stay (see _icu_timing.py)
  3. compute remaining LOS = (first_icu_end - first_icu_start) - 24h
  4. filter: 2 < remaining_los < 1000  (outlier bounds in hours)
  5. window = first 24h of first ICU stay
  6. assign_split()       — MIMIC 70/30 or date cutoffs (see _split.py)
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

MIN_REMAINING_LOS_HOURS = 2
MAX_REMAINING_LOS_HOURS = 1000

META = {
    "name": "task5_icu_los",
    "display_name": "ICU Length of Stay (remaining hours)",
    "task_class": "operational",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "regression",
    "label_column": "icu_los_hours",
    "input_window_hours": 24,
    "prediction_window": None,
    "cohort_description": "Adults with first ICU stay > 26 hours",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        f"first ICU stay - 24h in ({MIN_REMAINING_LOS_HOURS}, {MAX_REMAINING_LOS_HOURS}) hours",
    ],
    "exclusion_criteria": [
        f"remaining LOS <= {MIN_REMAINING_LOS_HOURS} hours",
        f"remaining LOS >= {MAX_REMAINING_LOS_HOURS} hours",
    ],
    "evaluation_metrics": ["mse", "rmse", "mae", "r2", "explained_variance"],
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
    timing = compute_icu_timing(adt)

    df = cohort.join(timing, on="hospitalization_id", how="inner").with_columns(
        ((pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
         .dt.total_seconds() / 3600.0 - 24.0)
        .alias("icu_los_hours")
    ).filter(
        (pl.col("icu_los_hours") > MIN_REMAINING_LOS_HOURS)
        & (pl.col("icu_los_hours") < MAX_REMAINING_LOS_HOURS)
    )

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
