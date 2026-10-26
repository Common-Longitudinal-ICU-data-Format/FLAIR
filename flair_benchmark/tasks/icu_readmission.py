"""
Unplanned ICU Readmission (binary, one-per-stay).

Clinical question:
  At the end of a patient's first ICU stay, will this patient be
  readmitted to the ICU later in the SAME hospitalization through an
  UNPLANNED path (i.e. not a trip through an allowed location and back)?

Definition:
  Working only from the CLIF ADT location history, ICU stays are collapsed
  to coarse locations. A return must contain an intervening Ward location to
  count as an unplanned readmission. Returns through Other without a Ward do not
  set the label, but they do not change the first ICU stay or prediction time.
  Procedural, Radiology, and Dialysis gaps retain the standard merge behavior.
  See flair_benchmark/cohort/events.py for the merge/sandwich algorithm.

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
  1. base_icu_cohort()             — adult ICU cohort + ADT (see cohort/clif.py)
  2. compute_readmission_timing()  — first ICU stay + 2nd-ICU / flags
  3. apply the five exclusions
  4. label = 1 iff second_icu_start_time is not null
  5. prediction_dttm = first_icu_end_time (the no-data-after cutoff)
  6. assign_split() keyed on prediction_dttm
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.cohort.clif import base_icu_cohort, read_clif_config
from flair_benchmark.cohort.encounter import as_encounter_keyed, encounter_attributes
from flair_benchmark.cohort.events import compute_readmission_timing
from flair_benchmark.cohort.stitch import assign_split
from flair_benchmark.cohort.strobe import record
from flair_benchmark.tasks._finalize import finalize_cohort

MIN_ICU_STAY_HOURS = 24

META = {
    "name": "icu_readmission",
    "display_name": "Unplanned ICU Readmission (one-per-stay, binary)",
    "task_class": "outcome",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    # Clinical policy — this task owns it; value-validated in tasks/__init__.py.
    "clinical_threshold": 0.2,
    "clinical_action": "keep in ICU",
    "clinical_direction": "above",
    "report_mode": "episodic",
    "dca_axis": {"event": "unplanned ICU readmission",
                 "action": "keep in ICU"},
    "label_column": "label_icu_readmission",
    "positive_class": "readmitted",
    "input_window_hours": None,
    "prediction_window": None,
    "cohort_description": (
        "Adults (age >= 18) with a first ICU stay, predicted at first-ICU "
        "discharge. An ICU return must contain an intervening Ward location; returns "
        "through Other without a Ward do not set the readmission label. "
        "Excludes death during the first "
        "ICU stay, direct ICU discharge, first ICU stay < 24h, and intermediate "
        "ER visits."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record (first ICU stay defined)",
        "first ICU stay >= 24 hours",
    ],
    "exclusion_criteria": [
        "no ICU ADT record",
        "death during the first ICU stay",
        "direct ICU discharge (last location is ICU)",
        "first ICU stay < 24 hours",
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
        compute_readmission_timing(as_encounter_keyed(adt), require_ward_return=True)
        .rename({"hospitalization_id": "hospitalization_join_id"})
    )

    # Exclusion 1: no ICU data (inner join drops encounters with no first ICU
    # stay). encounter_attributes supplies patient-level death + final discharge.
    cohort = encounter_attributes(cfg, cohort).join(
        timing, on="hospitalization_join_id", how="inner"
    ).pipe(record, "first ICU stay defined")

    # Exclusion 2: death during the first ICU stay.
    cohort = cohort.filter(
        ~(
            pl.col("death_dttm").is_not_null()
            & (pl.col("death_dttm") >= pl.col("first_icu_start_time"))
            & (pl.col("death_dttm") <= pl.col("first_icu_end_time"))
        )
    ).pipe(record, "no death during the first ICU stay")

    # Exclusion 3: direct ICU discharge (no opportunity to readmit).
    cohort = cohort.filter(~pl.col("is_direct_icu_discharge")).pipe(
        record, "not discharged directly from the ICU")

    # Exclusion 4: short first ICU stay.
    cohort = cohort.filter(
        (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
        >= pl.duration(hours=MIN_ICU_STAY_HOURS)
    ).pipe(record, f"first ICU stay >= {MIN_ICU_STAY_HOURS}h")

    # Exclusion 5: intermediate ER visit.
    cohort = cohort.filter(~pl.col("has_intermediate_er")).pipe(
        record, "no intermediate ER visit")

    df = cohort.with_columns([
        pl.col("second_icu_start_time").is_not_null().cast(pl.Int32)
            .alias(META["label_column"]),
        pl.col("first_icu_end_time").alias("prediction_dttm"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null()).pipe(record, "assigned to train/test split")
    return finalize_cohort(df, OUTPUT_COLS, META)
