"""
Task 6: In-Hospital Mortality (binary classification)

Pipeline (reads top-to-bottom):
  1. base_icu_cohort()    — adult ICU hospitalizations + ADT (see _clif.py)
  2. compute_icu_timing() — rank-based first/second ICU stay (see _icu_timing.py)
  3. filter ≥ 24h first ICU stay
  4. label = discharge_category in {expired, hospice}
  5. window = first 24h of first ICU stay
  6. assign_split()       — MIMIC 70/30 or date cutoffs (see _split.py)

Expected positive prevalence: ~5-15%.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

META = {
    "name": "task6_hospital_mortality",
    "display_name": "In-Hospital Mortality",
    "task_class": "outcome",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    "label_column": "label_mortality",
    "positive_class": "expired_or_hospice",
    "input_window_hours": 24,
    "prediction_window": None,
    "cohort_description": "Adults with first ICU stay >= 24 hours",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "first ICU stay duration >= 24 hours",
    ],
    "exclusion_criteria": [
        "missing discharge_category",
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
    """End-to-end build. Returns a parquet-ready polars DataFrame."""
    cfg = read_clif_config(clif_config)
    cohort, adt = base_icu_cohort(cfg)
    timing = compute_icu_timing(adt)

    df = cohort.join(timing, on="hospitalization_id", how="inner").filter(
        ((pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
         .dt.total_seconds() / 3600.0) >= 24
    )

    df = df.with_columns(
        pl.col("discharge_category").str.to_lowercase()
        .is_in(["expired", "hospice"])
        .cast(pl.Int32)
        .alias("label_mortality")
    ).filter(pl.col("label_mortality").is_not_null())

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
