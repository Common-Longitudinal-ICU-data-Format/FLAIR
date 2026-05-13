"""
Task 2: Discharged to LTACH (binary classification, highly imbalanced)

Pipeline (reads top-to-bottom):
  1. base_icu_cohort()    — adult ICU hospitalizations + ADT (see _clif.py)
  2. compute_icu_timing() — rank-based first ICU stay (see _icu_timing.py)
  3. label = 'ltach' or 'long-term' substring in discharge_category
  4. window = first 24h of first ICU stay
  5. assign_split()       — MIMIC 70/30 or date cutoffs (see _split.py)

Expected positive prevalence: ~3-5%. AUPRC is more informative than AUROC.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

META = {
    "name": "task2_discharged_ltach",
    "display_name": "Discharged to LTACH",
    "task_type": "binary",
    "label_column": "label_ltach",
    "positive_class": "ltach",
    "input_window_hours": 24,
    "prediction_window": None,
    "cohort_description": "All adult ICU hospitalizations",
    "inclusion_criteria": ["age_at_admission >= 18",
                           "valid admission_dttm and discharge_dttm",
                           "at least one ICU ADT record"],
    "exclusion_criteria": ["missing discharge_category"],
    "evaluation_metrics": ["auprc", "auroc", "accuracy", "precision", "recall",
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
    timing = compute_icu_timing(adt)

    df = cohort.join(timing, on="hospitalization_id", how="inner")

    disp = pl.col("discharge_category").str.to_lowercase()
    df = df.with_columns(
        (disp.str.contains("ltach") | disp.str.contains("long.?term"))
        .cast(pl.Int32)
        .alias("label_ltach")
    ).filter(pl.col("label_ltach").is_not_null())

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
