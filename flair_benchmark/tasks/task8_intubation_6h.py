"""
Task 8: Intubation prediction within next 6 hours (binary classification, decision-point).

Clinical question:
  Will this ICU patient require initiation of invasive mechanical ventilation
  (IMV) within the next 6 hours?

Row shape: one row per hourly decision point during the first ICU stay
at which the patient is NOT yet on IMV. A patient who is never intubated
contributes negative rows for every hour of their ICU stay; a patient
intubated at hour 12 contributes 11 rows, with rows at hours 6-11
labeled `1` (intubation falls within the 6h forward window).

Intubation detection uses the consensus-window state machine in
_ventilation_episodes.detect_imv_episodes: an IMV episode is only
counted when IMV persists for >= 15 min (filters out transient
mis-classified status rows). Patients with a tracheostomy sentinel
preceding any IMV row are excluded as pre-existing trach.

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort + ADT
  2. compute_icu_timing()     — rank-based first ICU stay
  3. load_respiratory_support() — all device_categories
  4. detect_imv_episodes()    — confirmed IMV episodes per hospitalization
  5. first_imv_dttm = first episode_start_dttm within the first ICU stay
  6. exclude: pre-existing trach; on IMV at first_icu_start_time
  7. generate hourly decision_point grid in
     (first_icu_start + 1h, end_of_decisions] where
     end_of_decisions = min(first_imv_dttm, first_icu_end_time)
  8. label = 1 iff first_imv_dttm in (decision_point, decision_point + 6h]
  9. assign_split()
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_respiratory_support,
                                    read_clif_config)
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split
from flair_benchmark._ventilation_episodes import detect_imv_episodes

DECISION_HORIZON_HOURS = 6
DECISION_GRID_STEP_HOURS = 1
EARLIEST_DECISION_OFFSET_HOURS = 1
INPUT_WINDOW_HOURS = 24

META = {
    "name": "task8_intubation_6h",
    "display_name": "Intubation within 6h (decision-point)",
    "task_class": "intervention",
    "temporal_pattern": "fixed_horizon",
    "prediction_unit": "one-per-decision-point",
    "task_type": "binary",
    "label_column": "label_intubation_6h",
    "positive_class": "intubated_within_6h",
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": [0, DECISION_HORIZON_HOURS],
    "cohort_description": "Adults in ICU, not on confirmed IMV at first_icu_start, "
                          "no pre-existing tracheostomy; one row per hourly decision "
                          "point until first confirmed IMV episode or ICU discharge",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record",
    ],
    "exclusion_criteria": [
        "pre-existing tracheostomy (trach sentinel before any IMV row)",
        "on IMV at first_icu_start_time (first confirmed IMV episode "
        "begins <= first_icu_start)",
        "no decision points (IMV starts within 1h of ICU entry)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
}

OUTPUT_COLS = [
    "hospitalization_id", "decision_point_dttm",
    "window_start", "window_end",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)
    cohort, adt = base_icu_cohort(cfg)
    timing = compute_icu_timing(adt)
    cohort = cohort.join(timing, on="hospitalization_id", how="inner")

    hosp_ids = cohort["hospitalization_id"].to_list()
    resp = load_respiratory_support(cfg, hospitalization_ids=hosp_ids)

    episodes = detect_imv_episodes(resp)

    # Drop pre-existing trach patients
    pre_trach_ids = (
        episodes.filter(pl.col("is_pre_existing_trach"))
        .select("hospitalization_id").unique()["hospitalization_id"].to_list()
    )
    cohort = cohort.filter(~pl.col("hospitalization_id").is_in(pre_trach_ids))

    # First confirmed IMV episode whose start falls within the first ICU stay
    first_imv = (
        episodes.filter(pl.col("episode_start_dttm").is_not_null())
        .join(cohort.select(["hospitalization_id", "first_icu_start_time",
                             "first_icu_end_time"]),
              on="hospitalization_id", how="inner")
        .filter(
            (pl.col("episode_start_dttm") >= pl.col("first_icu_start_time"))
            & (pl.col("episode_start_dttm") <= pl.col("first_icu_end_time"))
        )
        .group_by("hospitalization_id")
        .agg(pl.col("episode_start_dttm").min().alias("first_imv_dttm"))
    )
    df = cohort.join(first_imv, on="hospitalization_id", how="left")

    # Exclude patients on IMV at ICU admission
    df = df.filter(
        pl.col("first_imv_dttm").is_null()
        | (pl.col("first_imv_dttm") > pl.col("first_icu_start_time"))
    )

    df = df.with_columns(
        pl.min_horizontal(
            pl.coalesce(["first_imv_dttm", "first_icu_end_time"]),
            "first_icu_end_time",
        ).alias("end_of_decisions")
    )

    df = df.with_columns(
        pl.datetime_ranges(
            start=pl.col("first_icu_start_time")
                  + pl.duration(hours=EARLIEST_DECISION_OFFSET_HOURS),
            end=pl.col("end_of_decisions"),
            interval=f"{DECISION_GRID_STEP_HOURS}h",
            closed="left",
        ).alias("decision_point_dttm")
    ).explode("decision_point_dttm").filter(
        pl.col("decision_point_dttm").is_not_null()
    )

    df = df.with_columns(
        (
            pl.col("first_imv_dttm").is_not_null()
            & (pl.col("first_imv_dttm") > pl.col("decision_point_dttm"))
            & (pl.col("first_imv_dttm")
               <= pl.col("decision_point_dttm")
               + pl.duration(hours=DECISION_HORIZON_HOURS))
        ).cast(pl.Int32).alias("label_intubation_6h")
    )

    df = df.with_columns([
        (pl.col("decision_point_dttm")
         - pl.duration(hours=INPUT_WINDOW_HOURS)).alias("window_start"),
        pl.col("decision_point_dttm").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start)
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
