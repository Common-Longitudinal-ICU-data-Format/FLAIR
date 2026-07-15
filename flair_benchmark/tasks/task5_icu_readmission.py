"""
Task 5: Unplanned ICU Readmission (binary, one-per-stay).

Clinical question:
  At the end of a patient's first ICU stay, will this patient be
  readmitted to the ICU later in the SAME hospitalization through an
  UNPLANNED path (i.e. not a planned trip to a procedural suite and back)?

Definition:
  Working only from the CLIF ADT location history, ICU stays are collapsed
  to coarse locations (ICU / Ward / ER / Procedural / Other). A planned
  readmission --- ICU -> Procedural -> ICU --- is removed: the procedural
  row is dropped and the two ICU stays merge into one. Consecutive
  same-location ADT rows also merge. After this normalization, the
  existence of a SECOND ICU stay is an unplanned readmission (label = 1).
  See flair_benchmark/_readmission.py for the merge/sandwich algorithm.

Cohort exclusions (faithful port of the original task7 logic):
  1. No ICU data (no first ICU stay) — dropped by the inner join.
  2. Death during the first ICU stay (death_dttm in [start, end]).
  3. Direct ICU discharge (last location is ICU; no opportunity to readmit).
  4. Short first ICU stay (< MIN_ICU_STAY_HOURS).
  5. Intermediate ER (ER appears after the first location; complicates the
     planned/unplanned distinction).

Row shape (one-per-stay):
  Exactly one row per surviving hospitalization. The prediction is anchored
  at the end of the first ICU stay; all data up to prediction_dttm may be
  used.

Pipeline:
  1. base_icu_cohort()             — adult ICU cohort + ADT (see _clif.py)
  2. compute_readmission_timing()  — first ICU stay + 2nd-ICU / flags
  3. apply the five exclusions
  4. label = 1 iff second_icu_start_time is not null
  5. prediction_dttm = first_icu_end_time (the no-data-after cutoff)
  6. assign_split() keyed on prediction_dttm
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, read_clif_config
from flair_benchmark._encounter import as_encounter_keyed, encounter_attributes
from flair_benchmark._readmission import compute_readmission_timing
from flair_benchmark._split import assign_split
from flair_benchmark.tasks._finalize import finalize_cohort

MIN_ICU_STAY_HOURS = 1

from flair_benchmark._constants import TASK_POLICY

META = {
    "name": "task5_icu_readmission",
    "display_name": "Unplanned ICU Readmission (one-per-stay, binary)",
    "task_class": "outcome",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    # clinical_threshold, clinical_direction, report_mode — single-sourced
    **TASK_POLICY["task5_icu_readmission"],
    "label_column": "label_icu_readmission",
    "positive_class": "readmitted",
    "input_window_hours": None,
    "prediction_window": None,
    "cohort_description": (
        "Adults (age >= 18) with a first ICU stay, predicted at first-ICU "
        "discharge. Planned ICU -> Procedural -> ICU returns are merged (not "
        "readmissions). Excludes death during the first ICU stay, direct ICU "
        "discharge, first ICU stay < 1h, and intermediate ER visits."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record (first ICU stay defined)",
        "first ICU stay >= 1 hour",
    ],
    "exclusion_criteria": [
        "no ICU ADT record",
        "death during the first ICU stay",
        "direct ICU discharge (last location is ICU)",
        "first ICU stay < 1 hour",
        "intermediate ER visit (ER after the first location)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
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

    # Encounter grain: run the readmission timing across the WHOLE stitched
    # encounter (a 2nd ICU stay in a later member hospitalization is a
    # readmission of the same encounter, not a separate stay).
    timing = (
        compute_readmission_timing(as_encounter_keyed(adt))
        .rename({"hospitalization_id": "hospitalization_join_id"})
    )

    # Exclusion 1: no ICU data (inner join drops encounters with no first ICU
    # stay). encounter_attributes supplies patient-level death + final discharge.
    cohort = encounter_attributes(cfg, cohort).join(
        timing, on="hospitalization_join_id", how="inner"
    )

    # Exclusion 2: death during the first ICU stay.
    cohort = cohort.filter(
        ~(
            pl.col("death_dttm").is_not_null()
            & (pl.col("death_dttm") >= pl.col("first_icu_start_time"))
            & (pl.col("death_dttm") <= pl.col("first_icu_end_time"))
        )
    )

    # Exclusion 3: direct ICU discharge (no opportunity to readmit).
    cohort = cohort.filter(~pl.col("is_direct_icu_discharge"))

    # Exclusion 4: short first ICU stay.
    cohort = cohort.filter(
        (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
        >= pl.duration(hours=MIN_ICU_STAY_HOURS)
    )

    # Exclusion 5: intermediate ER visit.
    cohort = cohort.filter(~pl.col("has_intermediate_er"))

    df = cohort.with_columns([
        pl.col("second_icu_start_time").is_not_null().cast(pl.Int32)
            .alias(META["label_column"]),
        pl.col("first_icu_end_time").alias("prediction_dttm"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return finalize_cohort(df, OUTPUT_COLS, META)
