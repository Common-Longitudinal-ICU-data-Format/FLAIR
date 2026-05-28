"""
Task 5: Continuous AKI prediction — KDIGO stage 2 within next 24h
(binary classification, one-per-timestep).

Clinical question:
  Will this ICU patient develop KDIGO stage 2 AKI within the next 24 hours?

Reference:
  Koyner JL, Carey KA, Edelson DP, Churpek MM.
  The Development of a Machine Learning Inpatient Acute Kidney Injury
  Prediction Model. Crit Care Med 46(7):1070-1077, 2018.
  DOI: 10.1097/CCM.0000000000003123

Definition:
  KDIGO stage 2 = serum creatinine (SCr) >= 2.0x the patient's baseline,
  where baseline = min SCr in the prior 7 days (or the index SCr if no
  prior SCr exists in that window). Urine-output criteria are NOT used
  (matches Koyner; SCr-only).

Cohort (Koyner exclusions):
  1. Adult ICU hospitalizations (age >= 18)
  2. EXCLUDE first SCr >= 3.0 mg/dL
  3. EXCLUDE RRT within 48h of first SCr (skipped if RRT data absent)
  4. EXCLUDE ICD CKD stage 4+ — NOT applied in this build because CLIF
     Diagnoses is not yet exposed via FLAIR loaders (documented deviation;
     see paper/supplement.md §S2).

Row shape:
  One row per hourly timestep across the first ICU stay, truncated at first
  KDIGO-2 onset. Hospitalizations with no SCr measurement during the first
  ICU stay are dropped (no baseline computable).

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort + ADT
  2. compute_icu_timing()     — first ICU stay timestamps
  3. load_labs(['creatinine'])
  4. apply_koyner_exclusions  — see _aki.py
  5. compute_kdigo_stage2     — see _aki.py
  6. find first_kdigo2_dttm per hospitalization
  7. generate hourly prediction_dttm grid in [first_icu_start, end_of_predictions)
     where end_of_predictions = min(first_icu_end_time, first_kdigo2_dttm)
  8. label = 1 iff first_kdigo2_dttm in (prediction_dttm, prediction_dttm + 24h]
  9. assign_split()
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._aki import apply_koyner_exclusions, compute_kdigo_stage2
from flair_benchmark._clif import (base_icu_cohort, load_labs,
                                    read_clif_config)
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark._split import assign_split

AKI_HORIZON_HOURS = 24
GRID_STEP_HOURS = 1
INPUT_WINDOW_HOURS = 24

META = {
    "name": "task5_aki_kdigo2_24h",
    "display_name": "AKI KDIGO stage 2 within 24h (continuous)",
    "task_class": "deterioration",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-timestep",
    "task_type": "binary",
    "label_column": "label_aki_kdigo2_24h",
    "positive_class": "kdigo2_within_24h",
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": [0, AKI_HORIZON_HOURS],
    "cohort_description": "Adults with first ICU stay and >= 1 SCr measurement; "
                          "Koyner 2018 exclusions applied (first SCr < 3.0 mg/dL; "
                          "no RRT within 48h of first SCr)",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one ICU ADT record",
        "at least one creatinine measurement during first ICU stay",
    ],
    "exclusion_criteria": [
        "first SCr >= 3.0 mg/dL (Koyner rule 1)",
        "RRT within 48h of first SCr (Koyner rule 3; skipped if RRT data absent)",
        "ICD CKD stage 4+ — NOT APPLIED (documented deviation)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
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
    timing = compute_icu_timing(adt)
    cohort = cohort.join(timing, on="hospitalization_id", how="inner")

    hosp_ids = cohort["hospitalization_id"].to_list()
    creat_raw = load_labs(cfg, hospitalization_ids=hosp_ids,
                          lab_categories=["creatinine"])

    # Normalize CLIF Labs schema → (hospitalization_id, recorded_dttm, lab_value)
    # Use lab_result_dttm (result availability time — correct for label timing)
    # and lab_value_numeric (typed float, falls back to lab_value if absent).
    value_col = ("lab_value_numeric" if "lab_value_numeric" in creat_raw.columns
                 else "lab_value")
    creat = creat_raw.select([
        "hospitalization_id",
        pl.col("lab_result_dttm").alias("recorded_dttm"),
        pl.col(value_col).cast(pl.Float64).alias("lab_value"),
    ]).filter(
        pl.col("recorded_dttm").is_not_null()
        & pl.col("lab_value").is_not_null()
    )

    # Restrict creatinine to the first ICU stay window (baseline + onset detection
    # operate only on labs collected during the stay)
    creat = creat.join(
        cohort.select(["hospitalization_id", "first_icu_start_time",
                       "first_icu_end_time"]),
        on="hospitalization_id", how="inner",
    ).filter(
        (pl.col("recorded_dttm") >= pl.col("first_icu_start_time"))
        & (pl.col("recorded_dttm") <= pl.col("first_icu_end_time"))
    ).select(["hospitalization_id", "recorded_dttm", "lab_value"])

    # Drop hospitalizations with no in-stay creatinine
    have_scr = creat.select("hospitalization_id").unique()
    cohort = cohort.join(have_scr, on="hospitalization_id", how="inner")

    # Koyner exclusions (rule 1 always, rule 3 skipped — no RRT loader yet)
    cohort = apply_koyner_exclusions(cohort, creat, rrt_df=None)
    keep_ids = cohort["hospitalization_id"].to_list()
    creat = creat.filter(pl.col("hospitalization_id").is_in(keep_ids))

    # Compute KDIGO-2 flags and the first onset per hospitalization
    flagged = compute_kdigo_stage2(creat)
    first_onset = (
        flagged.filter(pl.col("kdigo_stage2_flag"))
        .group_by("hospitalization_id")
        .agg(pl.col("recorded_dttm").min().alias("first_kdigo2_dttm"))
    )
    cohort = cohort.join(first_onset, on="hospitalization_id", how="left")

    cohort = cohort.with_columns(
        pl.min_horizontal(
            pl.coalesce(["first_kdigo2_dttm", "first_icu_end_time"]),
            "first_icu_end_time",
        ).alias("end_of_predictions")
    )

    df = cohort.with_columns(
        pl.datetime_ranges(
            start=pl.col("first_icu_start_time"),
            end=pl.col("end_of_predictions"),
            interval=f"{GRID_STEP_HOURS}h",
            closed="left",
        ).alias("prediction_dttm")
    ).explode("prediction_dttm").filter(pl.col("prediction_dttm").is_not_null())

    df = df.with_columns(
        (
            pl.col("first_kdigo2_dttm").is_not_null()
            & (pl.col("first_kdigo2_dttm") > pl.col("prediction_dttm"))
            & (pl.col("first_kdigo2_dttm")
               <= pl.col("prediction_dttm")
               + pl.duration(hours=AKI_HORIZON_HOURS))
        ).cast(pl.Int32).alias("label_aki_kdigo2_24h")
    )

    df = df.with_columns([
        (pl.col("prediction_dttm")
         - pl.duration(hours=INPUT_WINDOW_HOURS)).alias("window_start"),
        pl.col("prediction_dttm").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)
