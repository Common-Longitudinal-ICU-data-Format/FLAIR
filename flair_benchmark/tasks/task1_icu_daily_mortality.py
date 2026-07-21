"""
Task 1: ICU Daily In-Hospital Mortality (binary, one-per-window at 07:00 local).

Clinical question:
  Given everything we know about this ICU patient up to today's 07:00 rounds,
  will this encounter end with the patient dying in hospital
  (CLIF discharge_category in {Expired, Hospice}) rather than any other
  disposition?

Cohort (unconditional all-comers ICU, stitched-encounter grain):
  All adult stitched encounters with at least one ICU ADT record. NO filter
  on discharge_category — every retained encounter contributes daily rows,
  including those discharged Home, SNF, Rehab, LTACH, AMA, Other,
  Still Admitted, or with a null discharge_category.

Label (label_mortality):
  1 iff the encounter's FINAL discharge_category (lowercased, stripped) in
  {"expired", "hospice"}; 0 otherwise — including LTACH, Home, SNF, Rehab,
  AMA, Other, Still Admitted, and null. Null and Still Admitted are
  intentionally retained as negatives.

Row shape (one-per-window):
  prediction_dttm = 07:00 on day-of-admission + 1, +2, +3, ... while the
  timestamp is < min(coalesce(first_icu_end_time, discharge_dttm),
  discharge_dttm, death_dttm). Predictions occur only during the encounter's
  first ICU stay. The label is the eventual encounter outcome and is constant
  across all rows for a given hospitalization_join_id.

Pipeline (see flair_benchmark/_encounter.py):
  1. build_icu_encounter_cohort() — one row per 07:00 day of each stitched
     encounter's first ICU stay, carrying the encounter's final
     discharge_category. First ICU stay comes from compute_readmission_timing
     (contiguous-ICU-block merge); pred_stop is a true earliest-of with a
     discharge fallback for a null/terminal ICU end.
  2. Broadcast label from the encounter's final discharge_category
     (null-safe — null and Still Admitted become 0)
  3. assign_split() per encounter block by prediction_dttm
     (prediction_dttm is the no-data-after cutoff; all prior history is usable)

Timezone note:
  clifpy returns tz-aware datetimes typed to cfg["timezone"]; flair_benchmark._clif
  strips tz info via pd_to_pl, leaving naive datetimes whose wall-clock values are
  already local to the site. The literal "07:00" anchor IS local hospital time.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import read_clif_config
from flair_benchmark._encounter import build_icu_encounter_cohort
from flair_benchmark._split import assign_split
from flair_benchmark.tasks._finalize import finalize_cohort

META = {
    "name": "task1_icu_daily_mortality",
    "display_name": "ICU Daily In-Hospital Mortality (7AM, binary)",
    "task_class": "outcome",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-window",
    "task_type": "binary",
    # Clinical policy — this task owns it; value-validated in tasks/__init__.py.
    "clinical_threshold": 0.95,
    "clinical_action": "order a palliative care consult",
    "clinical_direction": "above",
    "report_mode": "continuous",
    "landmark_horizon": 30,
    "dca_axis": {"event": "in-hospital mortality",
                 "action": "order a palliative care consult"},
    "label_column": "label_mortality",
    "positive_class": "in_hospital_mortality",
    "input_window_hours": None,
    "prediction_window": None,
    "window_unit": "day",  # window_number increments one row per local-07:00 day
    "cohort_description": (
        "All adult ICU stitched encounters (age >= 18, valid admission/discharge "
        "times, >= 1 ICU ADT record). NO discharge_category filter — every "
        "retained encounter contributes rows. One row per local-07:00 timestamp "
        "during the encounter's first ICU stay, starting the calendar day after "
        "first ICU admission. The encounter's final discharge_category null or "
        "'Still Admitted' is kept and yields label-0."
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

    df = build_icu_encounter_cohort(cfg)

    disp = pl.col("discharge_category").str.to_lowercase().str.strip_chars()
    df = df.with_columns(
        disp.is_in(["expired", "hospice"])
            .fill_null(False)
            .cast(pl.Int32)
            .alias(META["label_column"])
    )

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return finalize_cohort(df, OUTPUT_COLS, META)
