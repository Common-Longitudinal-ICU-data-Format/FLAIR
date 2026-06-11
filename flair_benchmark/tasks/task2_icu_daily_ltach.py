"""
Task 2: ICU Daily LTACH Discharge (binary, one-per-timestep at 07:00 local).

Clinical question:
  Given everything we know about this ICU patient up to today's 07:00 rounds,
  will the eventual disposition of this hospitalization be transfer to a
  Long-Term Acute Care Hospital (LTACH)?

Cohort (unconditional all-comers ICU):
  All adult hospitalizations with at least one ICU ADT record. NO filter
  on discharge_category. The cohort is identical to Task 1.

Label (label_ltach):
  1 iff discharge_category (lowercased, stripped) matches regex
    r"ltach|long.?term"
  0 otherwise — including Expired, Hospice, Home, SNF, Rehab, AMA, Other,
  Still Admitted, and null. Null and Still Admitted are intentionally
  retained as negatives. T1 and T2 are NOT a partition: a Home discharge
  is 0 in both; an Expired discharge is 1 in T1 and 0 in T2.

Row shape (one-per-timestep):
  prediction_dttm = 07:00 on day-of-admission + 1, +2, +3, ... while the
  timestamp is < min(first_icu_end_time, discharge_dttm, death_dttm).
  The label is the eventual hospitalization outcome and is constant across
  all rows for a given hospitalization_id.

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort + ADT (see _clif.py)
  2. compute_icu_timing()     — first_icu_start_time, first_icu_end_time
  3. pred_start = (first_icu_start_time truncated to date) + 1 day + 7 hours
  4. pred_stop  = min(first_icu_end_time, coalesce(death_dttm, discharge_dttm))
  5. Daily 07:00 grid via pl.datetime_ranges(..., interval="24h", closed="left")
  6. Broadcast label from hospitalization-level discharge_category
     (null-safe — null and Still Admitted become 0)
  7. assign_split() per-row by prediction_dttm
     (prediction_dttm is the no-data-after cutoff; all prior history is usable)

Timezone note:
  clifpy returns tz-aware datetimes typed to cfg["timezone"]; flair_benchmark._clif
  strips tz info via pd_to_pl, leaving naive datetimes whose wall-clock values are
  already local to the site. The literal "07:00" anchor IS local hospital time.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split
from flair_benchmark.tasks._finalize import finalize_cohort

PRED_HOUR = 7
GRID_STEP = "24h"

META = {
    "name": "task2_icu_daily_ltach",
    "display_name": "ICU Daily LTACH Discharge (7AM, binary)",
    "task_class": "outcome",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-timestep",
    "task_type": "binary",
    "clinical_threshold": 0.5,
    "clinical_action": "early tracheostomy",
    "clinical_direction": "above",  # act when risk of LTACH discharge with trach > 50%
    "label_column": "label_ltach",
    "positive_class": "ltach_discharge",
    "input_window_hours": None,
    "prediction_window": None,
    "window_unit": "day",  # window_number increments one row per local-07:00 day
    "cohort_description": (
        "All adult ICU hospitalizations (age >= 18, valid admission/discharge "
        "times, >= 1 ICU ADT record). NO discharge_category filter — every "
        "retained stay contributes rows. One row per local-07:00 timestamp "
        "during the first ICU stay, starting the calendar day after first "
        "ICU admission. discharge_category null or 'Still Admitted' is kept "
        "and yields label-0."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record (first_icu_start_time present)",
        "first ICU stay long enough to contain >= one 07:00 grid point after admission day",
    ],
    "exclusion_criteria": [
        "age_at_admission < 18",
        "missing admission_dttm or discharge_dttm",
        "no ICU ADT record",
        "first ICU stay too short for any 07:00 grid point",
    ],
    "evaluation_metrics": [
        "auroc", "auprc", "accuracy", "precision",
        "recall", "f1", "specificity", "npv",
    ],
}

OUTPUT_COLS = [
    "hospitalization_id", "hospitalization_join_id", "prediction_dttm",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)

    cohort, adt = base_icu_cohort(cfg)

    timing = compute_icu_timing(adt)
    cohort = cohort.join(timing, on="hospitalization_id", how="inner")

    cohort = cohort.with_columns([
        (
            pl.col("first_icu_start_time").dt.truncate("1d")
            + pl.duration(days=1, hours=PRED_HOUR)
        ).alias("pred_start"),
        pl.min_horizontal(
            pl.col("first_icu_end_time"),
            pl.coalesce(["death_dttm", "discharge_dttm"]),
        ).alias("pred_stop"),
    ])

    df = cohort.with_columns(
        pl.datetime_ranges(
            start=pl.col("pred_start"),
            end=pl.col("pred_stop"),
            interval=GRID_STEP,
            closed="left",
        ).alias("prediction_dttm")
    ).explode("prediction_dttm").filter(pl.col("prediction_dttm").is_not_null())

    disp = pl.col("discharge_category").str.to_lowercase().str.strip_chars()
    df = df.with_columns(
        disp.str.contains(r"ltach|long.?term")
            .fill_null(False)
            .cast(pl.Int32)
            .alias(META["label_column"])
    )

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return finalize_cohort(df, OUTPUT_COLS, META)
