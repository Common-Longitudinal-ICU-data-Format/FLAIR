"""
Task 13: ICU Daily Disposition (multiclass, one-per-timestep at 07:00 local).

Clinical question:
  Given everything we know about this ICU patient up to today's 07:00 rounds,
  what is the most likely eventual hospital disposition? This prediction
  informs daily goals-of-care discussions: continue vs. withdraw life support.

Classes (label = label_disposition, integer-encoded):
  0 = ltach        : discharged to a Long Term Acute Care Hospital
  1 = alive_other  : any other living disposition (home, SNF, rehab, AMA,
                     assisted living, group home, ... and also "Still Admitted"
                     and "Other" — see label rule below)
  2 = dead_hospice : died in hospital (Expired) or discharged to Hospice

Cohort:
  Adult hospitalizations with at least one ICU ADT record. One row per
  local-07:00 timestamp that falls inside the first ICU stay, starting the
  calendar day AFTER first ICU admission. Hospitalizations with a null
  discharge_category are excluded (un-labelable).

Row shape (one-per-timestep):
  prediction_dttm = 07:00 on day-of-admission + 1, +2, +3, ... while the
  timestamp is < min(first_icu_end_time, discharge_dttm, death_dttm).
  The label is the eventual hospitalization outcome and is constant across
  all rows for a given hospitalization_id.

Label rule (case-insensitive on discharge_category):
  2 (dead_hospice) iff discharge_category in {"expired", "hospice"}
  0 (ltach)        iff discharge_category matches r"ltach|long.?term"
  1 (alive_other)  otherwise

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort + ADT (see _clif.py)
  2. Drop null discharge_category
  3. compute_icu_timing()     — first_icu_start_time, first_icu_end_time
  4. pred_start = (first_icu_start_time truncated to date) + 1 day + 7 hours
  5. pred_stop  = min(first_icu_end_time, discharge_dttm, death_dttm)
  6. Daily 07:00 grid via pl.datetime_ranges(..., interval="24h", closed="left")
  7. Broadcast label from hospitalization-level discharge_category
  8. window_end = prediction_dttm; window_start = prediction_dttm - 24h
  9. assign_split() per-row by prediction_dttm (matches task11)

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

PRED_HOUR = 7
INPUT_WINDOW_HOURS = 24
GRID_STEP = "24h"

META = {
    "name": "task13_icu_daily_disposition",
    "display_name": "ICU Daily Disposition (7AM, multiclass)",
    "task_class": "outcome",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-timestep",
    "task_type": "multiclass",
    "label_column": "label_disposition",
    "class_names": ["ltach", "alive_other", "dead_hospice"],
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": None,
    "cohort_description": (
        "Adult ICU hospitalizations; one row per local-07:00 timestamp during "
        "the first ICU stay, starting the calendar day after first ICU admission."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record (first_icu_start_time present)",
        "first ICU stay long enough to contain >= one 07:00 grid point after admission day",
        "discharge_category is not null",
    ],
    "exclusion_criteria": [
        "discharge_category is null",
        "first ICU stay ends before any 07:00 grid point falls inside it (drops naturally)",
    ],
    "evaluation_metrics": [
        "accuracy", "macro_f1", "weighted_f1",
        "per_class_precision", "per_class_recall", "per_class_f1",
        "confusion_matrix",
    ],
}

OUTPUT_COLS = [
    "hospitalization_id", "prediction_dttm",
    "window_start", "window_end",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)

    cohort, adt = base_icu_cohort(cfg)
    cohort = cohort.filter(pl.col("discharge_category").is_not_null())

    timing = compute_icu_timing(adt)
    cohort = cohort.join(timing, on="hospitalization_id", how="inner")

    cohort = cohort.with_columns([
        (
            pl.col("first_icu_start_time").dt.truncate("1d")
            + pl.duration(days=1, hours=PRED_HOUR)
        ).alias("pred_start"),
        pl.min_horizontal(
            pl.col("first_icu_end_time"),
            pl.col("discharge_dttm"),
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
        pl.when(disp.is_in(["expired", "hospice"])).then(pl.lit(2))
        .when(disp.str.contains(r"ltach|long.?term")).then(pl.lit(0))
        .otherwise(pl.lit(1))
        .cast(pl.Int32)
        .alias(META["label_column"])
    )

    df = df.with_columns([
        (pl.col("prediction_dttm") - pl.duration(hours=INPUT_WINDOW_HOURS))
            .alias("window_start"),
        pl.col("prediction_dttm").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
