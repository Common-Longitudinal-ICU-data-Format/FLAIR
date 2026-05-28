"""
Task 4: Sepsis prediction with antibiotic-anchored time-zero
(binary classification, one-per-timestep).

Clinical question:
  Will this hospitalized patient receive their first qualifying systemic-IV
  antibiotic within the next 6 hours?

Reference:
  Kamran F, Tjandra D, Heiler A, Virzi J, Singh K, King JE, Valley TS, Wiens J.
  Evaluation of sepsis prediction models before onset of treatment.
  NEJM AI 1(3), 2024. DOI: 10.1056/AIoa2300032.

Rationale:
  The qualifying-antibiotic order is used as an operational proxy for clinician
  recognition of sepsis (per Kamran 2024). Because the label IS the clinician
  action, predictions made at or after the action would trivially correlate
  with the outcome ("label leakage"). The prediction window therefore
  terminates strictly BEFORE t_zero so the model never sees the action it is
  trying to forecast.

Anchors (per docs/task_schematics/sepsis_antibiotic_timezero.pdf §3.1):
  t_zero      = min(admin_dttm | med_group == cms_sepsis_qualifying_antibiotics,
                                  med_route_category == iv,
                                  mar_action_group == administered)
  pred_start  = max(min(vitals.recorded_dttm), admission_dttm + 1h)
  pred_stop   = min(t_zero, discharge_dttm, death_dttm)   # t_zero = +inf if none

Cohort (hospital-wide, Kamran 2024):
  1. Adult hospitalizations (age >= 18) with valid admission/discharge dttm.
  2. EXCLUDE hospitalizations with any ADT stay on psychiatric or
     rehabilitation unit.
  3. EXCLUDE admitted-on-treatment (t_zero <= pred_start).
  4. EXCLUDE hospitalizations with no vital sign measurements
     (pred_start is undefined).

Row shape:
  One row per hourly decision point in [pred_start, pred_stop), closed-open.
  Hospitalizations with no qualifying antibiotic ever produce all-negative
  rows over [pred_start, discharge or death].

Label rule (label_sepsis_abx_6h):
  y(d) = 1 iff t_zero is not null AND prediction_dttm < t_zero <= prediction_dttm + 6h.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (
    base_icu_cohort,  # noqa: F401 — re-exported for parity with other tasks
    load_adt,
    load_hospitalization,
    load_medication_admin_continuous,
    load_medication_admin_intermittent,
    load_patient,
    load_vitals,
    read_clif_config,
)
from flair_benchmark._split import assign_split

LABEL_HORIZON_HOURS = 6
INPUT_WINDOW_HOURS = 24
GRID_STEP_HOURS = 1

QUALIFYING_MED_GROUP = "cms_sepsis_qualifying_antibiotics"
QUALIFYING_ROUTE = "iv"
ADMINISTERED_GROUP = "administered"
EXCLUDE_LOCATION_CATEGORIES = ["psychiatric", "rehabilitation"]

META = {
    "name": "task4_sepsis_abx_6h",
    "display_name": "Sepsis (antibiotic-anchored) within 6h",
    "task_class": "intervention",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-timestep",
    "task_type": "binary",
    "label_column": "label_sepsis_abx_6h",
    "positive_class": "qualifying_iv_abx_within_6h",
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": [0, LABEL_HORIZON_HOURS],
    "cohort_description": (
        "Hospital-wide adult cohort (Kamran 2024); psych/rehab units excluded; "
        "admitted-on-treatment excluded; must have at least one vital sign."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one vital sign recorded during admission",
    ],
    "exclusion_criteria": [
        "any ADT stay on psychiatric or rehabilitation unit",
        "first qualifying antibiotic at or before first vital sign (admitted on treatment)",
        "no vital signs recorded",
    ],
    "evaluation_metrics": [
        "auroc", "auprc", "accuracy", "precision",
        "recall", "f1", "specificity", "npv",
    ],
}

OUTPUT_COLS = [
    "hospitalization_id", "prediction_dttm",
    "window_start", "window_end",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]

def _hospital_wide_cohort(cfg: dict, min_age: int = 18) -> pl.DataFrame:
    """Adult hospital-wide cohort (no ICU filter). Mirrors base_icu_cohort
    but drops the ADT-based ICU narrowing — Kamran 2024 evaluates the model
    hospital-wide, since ward-onset sepsis is where prediction lead time
    has the most clinical value."""
    hosp = load_hospitalization(
        cfg,
        columns=["hospitalization_id", "patient_id", "admission_dttm",
                 "discharge_dttm", "age_at_admission", "discharge_category"],
    )
    patient_ids = hosp.select("patient_id").unique()["patient_id"].to_list()
    patient = load_patient(
        cfg,
        patient_ids=patient_ids,
        columns=["patient_id", "sex_category", "race_category",
                 "ethnicity_category", "death_dttm"],
    )
    return (
        hosp.join(patient, on="patient_id", how="inner")
        .filter(pl.col("admission_dttm").is_not_null()
                & pl.col("discharge_dttm").is_not_null())
        .filter(pl.col("age_at_admission") >= min_age)
        .select([
            "hospitalization_id", "patient_id",
            "admission_dttm", "discharge_dttm", "death_dttm",
            "age_at_admission", "sex_category", "race_category",
            "ethnicity_category", "discharge_category",
        ])
    )


def _exclude_psych_rehab(cohort: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    hosp_ids = cohort["hospitalization_id"].to_list()
    adt = load_adt(cfg, hospitalization_ids=hosp_ids)
    excluded = (
        adt.filter(pl.col("location_category").str.to_lowercase()
                   .is_in(EXCLUDE_LOCATION_CATEGORIES))
        .select("hospitalization_id").unique()
    )
    return cohort.join(excluded, on="hospitalization_id", how="anti")


def _first_vital_dttm(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    vitals = load_vitals(cfg, hospitalization_ids=hosp_ids, vital_categories=None)
    return (
        vitals.filter(pl.col("recorded_dttm").is_not_null())
        .group_by("hospitalization_id")
        .agg(pl.col("recorded_dttm").min().alias("first_vital_dttm"))
    )


def _first_qualifying_abx_dttm(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """Union intermittent + continuous med tables, filter to qualifying
    systemic-IV antibiotics that were actually administered, return earliest
    admin_dttm per hospitalization.

    Intermittent and continuous tables have different CLIF semantics:
      - Intermittent: each row is a discrete MAR action; must filter
        mar_action_group == 'administered' to drop held/refused/etc.
      - Continuous: each row is an infusion-rate timeseries point; presence
        of the row implies administration (no mar_action_group column).
    """
    abx_int = load_medication_admin_intermittent(
        cfg, hosp_ids, med_groups=[QUALIFYING_MED_GROUP],
    )
    abx_int = abx_int.filter(
        pl.col("admin_dttm").is_not_null()
        & (pl.col("med_route_category").str.to_lowercase() == QUALIFYING_ROUTE)
        & (pl.col("mar_action_group").str.to_lowercase() == ADMINISTERED_GROUP)
    ).select(["hospitalization_id", "admin_dttm"])

    abx_cont = load_medication_admin_continuous(
        cfg, hosp_ids, med_groups=[QUALIFYING_MED_GROUP],
    )
    abx_cont = abx_cont.filter(
        pl.col("admin_dttm").is_not_null()
        & (pl.col("med_route_category").str.to_lowercase() == QUALIFYING_ROUTE)
    ).select(["hospitalization_id", "admin_dttm"])

    abx = pl.concat([abx_int, abx_cont], how="vertical")
    return (
        abx.group_by("hospitalization_id")
        .agg(pl.col("admin_dttm").min().alias("t_zero"))
    )


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)

    cohort = _hospital_wide_cohort(cfg)
    cohort = _exclude_psych_rehab(cohort, cfg)
    if cohort.is_empty():
        return cohort.select([])

    hosp_ids = cohort["hospitalization_id"].to_list()

    first_vital = _first_vital_dttm(cfg, hosp_ids)
    cohort = cohort.join(first_vital, on="hospitalization_id", how="inner")

    cohort = cohort.with_columns(
        pl.max_horizontal(
            pl.col("first_vital_dttm"),
            pl.col("admission_dttm") + pl.duration(hours=1),
        ).alias("pred_start")
    )

    t_zero = _first_qualifying_abx_dttm(cfg, hosp_ids)
    cohort = cohort.join(t_zero, on="hospitalization_id", how="left")

    cohort = cohort.filter(
        pl.col("t_zero").is_null() | (pl.col("t_zero") > pl.col("pred_start"))
    )

    cohort = cohort.with_columns(
        pl.min_horizontal(
            pl.coalesce(["t_zero", "discharge_dttm"]),
            pl.coalesce(["death_dttm", "discharge_dttm"]),
        ).alias("pred_stop")
    )

    df = cohort.with_columns(
        pl.datetime_ranges(
            start=pl.col("pred_start"),
            end=pl.col("pred_stop"),
            interval=f"{GRID_STEP_HOURS}h",
            closed="left",
        ).alias("prediction_dttm")
    ).explode("prediction_dttm").filter(pl.col("prediction_dttm").is_not_null())

    df = df.with_columns(
        (
            pl.col("t_zero").is_not_null()
            & (pl.col("t_zero") > pl.col("prediction_dttm"))
            & (pl.col("t_zero")
               <= pl.col("prediction_dttm")
               + pl.duration(hours=LABEL_HORIZON_HOURS))
        ).cast(pl.Int32).alias(META["label_column"])
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
