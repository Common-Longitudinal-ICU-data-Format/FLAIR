"""
Task 3: 72-Hour Respiratory Outcome (multiclass)

Cohort: patients on IMV at the 24-hour mark of their first ICU stay.
Input window: first 24h of ICU.
Prediction window: 24h → 72h.

Classes (label = task3_label, integer-encoded):
  0 = imv_on   : still on IMV at the 72h mark
  1 = imv_off  : off IMV at the 72h mark, alive
  2 = expired  : died (death_dttm) on or before the 72h mark

Pipeline:
  1. base_icu_cohort()        — adult ICU hospitalizations + ADT (see _clif.py)
  2. compute_icu_timing()     — rank-based first ICU stay (see _icu_timing.py)
  3. load_respiratory_support() filtered to IMV
  4. cohort = hospitalizations with an IMV row covering t24
  5. label  = check IMV coverage at t72 vs death_dttm
  6. window = first 24h of first ICU stay
  7. assign_split()           — MIMIC 70/30 or date cutoffs (see _split.py)
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_respiratory_support,
                                    read_clif_config)
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

META = {
    "name": "task3_outcome_72hr",
    "display_name": "72-Hour Respiratory Outcome",
    "task_class": "deterioration",
    "temporal_pattern": "fixed_horizon",
    "prediction_unit": "one-per-stay",
    "task_type": "multiclass",
    "label_column": "task3_label",
    "class_names": ["imv_on", "imv_off", "expired"],
    "input_window_hours": 24,
    "prediction_window": [24, 72],
    "cohort_description": "Adults on IMV at the 24-hour mark of first ICU stay",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "at least one respiratory_support IMV row covering first_icu_start + 24h",
    ],
    "exclusion_criteria": [
        "no IMV at 24-hour mark",
    ],
    "evaluation_metrics": ["accuracy", "macro_f1", "weighted_f1",
                           "per_class_precision", "per_class_recall",
                           "per_class_f1", "confusion_matrix"],
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

    cohort = cohort.join(timing, on="hospitalization_id", how="inner").with_columns([
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("t24"),
        (pl.col("first_icu_start_time") + pl.duration(hours=72)).alias("t72"),
    ])

    hosp_ids = cohort["hospitalization_id"].to_list()
    resp = load_respiratory_support(cfg, hospitalization_ids=hosp_ids,
                                    device_categories=["IMV"])

    # An IMV row "covers" t if recorded_dttm <= t AND (next IMV record start > t OR null).
    # CLIF respiratory_support is a status stream — a row sets the device at recorded_dttm
    # and is active until the next row for the same hospitalization. We approximate
    # "on IMV at time t" by: any IMV row with recorded_dttm <= t exists, AND the most
    # recent respiratory_support row (any device) at or before t has device_category == 'IMV'.
    imv_at_t24 = _on_imv_at(resp, cohort.select(["hospitalization_id", "t24"]), "t24")
    imv_at_t72 = _on_imv_at(resp, cohort.select(["hospitalization_id", "t72"]), "t72")

    df = cohort.join(imv_at_t24, on="hospitalization_id", how="inner")  # cohort filter
    df = df.join(imv_at_t72, on="hospitalization_id", how="left")

    expired_by_72h = (
        pl.col("death_dttm").is_not_null() & (pl.col("death_dttm") <= pl.col("t72"))
    )
    df = df.with_columns(
        pl.when(expired_by_72h).then(pl.lit(2))      # expired
        .when(pl.col("on_imv_at_t72").fill_null(False)).then(pl.lit(0))  # imv_on
        .otherwise(pl.lit(1))                         # imv_off
        .alias("task3_label")
    )

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        pl.col("t24").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)


def _on_imv_at(resp: pl.DataFrame, targets: pl.DataFrame,
               at_col: str) -> pl.DataFrame:
    """For each (hospitalization_id, at_col) target, return whether the
    last respiratory_support row at or before `at_col` is an IMV row.

    Returns a DataFrame with [hospitalization_id, on_imv_at_<at_col>].
    """
    # Find most-recent respiratory_support row at or before each target time
    joined = (
        targets.join(resp, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm") <= pl.col(at_col))
        .sort([pl.col("hospitalization_id"), pl.col("recorded_dttm")])
        .group_by("hospitalization_id")
        .agg(pl.col("device_category").last().alias("last_device"))
    )
    out_name = f"on_imv_at_{at_col}"
    return joined.select([
        "hospitalization_id",
        (pl.col("last_device").str.to_uppercase() == "IMV").alias(out_name),
    ])
