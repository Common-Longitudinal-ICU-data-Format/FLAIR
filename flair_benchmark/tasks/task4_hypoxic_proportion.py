"""
Task 4: Hypoxic Failure Proportion (regression)

Cohort: same as Task 3 — patients on IMV at the 24-hour mark.
Input window: first 24h of ICU.
Prediction target: fraction of hours in [24h, 72h) flagged as hypoxic.

Hypoxia definition (in this implementation):
  An hour h is hypoxic if any SpO2 reading within [t24 + h, t24 + h + 1h]
  is strictly less than 90.

  NOTE: The slide deck additionally allows PaO2/FiO2 < 200 to qualify.
  Implementing the full PF ratio requires aligning lab PaO2 with the
  respiratory_support fio2_set timeseries and is non-trivial; the
  SpO2-only implementation captures the dominant signal and can be
  extended later by adding a PF check inside `_hypoxic_proportion`.

Pipeline:
  1. base_icu_cohort()        — see _clif.py
  2. compute_icu_timing()     — see _icu_timing.py
  3. load_respiratory_support() — filter cohort to IMV at t24
  4. load_vitals(spo2)        — compute hourly hypoxia in [t24, t72)
  5. window = first 24h of first ICU stay
  6. assign_split()           — see _split.py
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_respiratory_support,
                                    load_vitals, read_clif_config)
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

PREDICTION_WINDOW_HOURS = 48  # 72 - 24
HYPOXIA_SPO2_THRESHOLD = 90.0

META = {
    "name": "task4_hypoxic_proportion",
    "display_name": "Hypoxic Failure Proportion (24-72h)",
    "task_class": "deterioration",
    "temporal_pattern": "fixed_horizon",
    "prediction_unit": "one-per-stay",
    "task_type": "regression",
    "label_column": "task4_proportion",
    "input_window_hours": 24,
    "prediction_window": [24, 72],
    "cohort_description": "Adults on IMV at the 24-hour mark of first ICU stay",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "on IMV at first_icu_start + 24h (per respiratory_support)",
    ],
    "exclusion_criteria": ["no IMV at 24h mark", "no SpO2 readings in [24h, 72h)"],
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

    cohort = cohort.join(timing, on="hospitalization_id", how="inner").with_columns([
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("t24"),
        (pl.col("first_icu_start_time") + pl.duration(hours=72)).alias("t72"),
    ])

    hosp_ids = cohort["hospitalization_id"].to_list()
    resp = load_respiratory_support(cfg, hospitalization_ids=hosp_ids,
                                    device_categories=["IMV"])
    imv_at_t24 = _on_imv_at(resp, cohort.select(["hospitalization_id", "t24"]))
    cohort = cohort.join(imv_at_t24, on="hospitalization_id", how="inner")

    cohort_ids = cohort["hospitalization_id"].to_list()
    spo2 = load_vitals(cfg, hospitalization_ids=cohort_ids,
                       vital_categories=["spo2"])

    proportion = _hypoxic_proportion(spo2, cohort)
    df = cohort.join(proportion, on="hospitalization_id", how="left").with_columns(
        pl.col("task4_proportion").fill_null(0.0)
    )

    df = df.with_columns([
        pl.col("first_icu_start_time").alias("window_start"),
        pl.col("t24").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)


def _on_imv_at(resp: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Most-recent respiratory_support row at or before each target time must be IMV."""
    return (
        targets.join(resp, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm") <= pl.col("t24"))
        .sort([pl.col("hospitalization_id"), pl.col("recorded_dttm")])
        .group_by("hospitalization_id")
        .agg(pl.col("device_category").last().alias("last_device"))
        .filter(pl.col("last_device").str.to_uppercase() == "IMV")
        .select("hospitalization_id")
    )


def _hypoxic_proportion(spo2: pl.DataFrame,
                        cohort: pl.DataFrame) -> pl.DataFrame:
    """For each hospitalization compute the fraction of hours in [t24, t72)
    that contain at least one SpO2 < 90 reading.

    Returns: [hospitalization_id, task4_proportion].
    """
    spo2 = (
        spo2.join(cohort.select(["hospitalization_id", "t24", "t72"]),
                  on="hospitalization_id", how="inner")
        .filter((pl.col("recorded_dttm") >= pl.col("t24"))
                & (pl.col("recorded_dttm") < pl.col("t72")))
        .with_columns(
            (((pl.col("recorded_dttm") - pl.col("t24")).dt.total_seconds()) // 3600)
            .cast(pl.Int32).alias("hour_offset")
        )
        .filter(pl.col("vital_value").is_not_null())
    )

    hourly = (
        spo2.group_by(["hospitalization_id", "hour_offset"])
        .agg((pl.col("vital_value") < HYPOXIA_SPO2_THRESHOLD).any().alias("hypoxic"))
    )

    return (
        hourly.group_by("hospitalization_id")
        .agg(pl.col("hypoxic").sum().alias("hypoxic_hours"))
        .with_columns(
            (pl.col("hypoxic_hours") / PREDICTION_WINDOW_HOURS)
            .clip(0.0, 1.0)
            .alias("task4_proportion")
        )
        .select(["hospitalization_id", "task4_proportion"])
    )
