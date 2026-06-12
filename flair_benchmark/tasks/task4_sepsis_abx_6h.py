"""
Task 4: Sepsis prediction with CDC Adult Sepsis Event (ASE) time-zero
(binary classification, one-per-window).

Clinical question:
  Will this hospitalized patient meet the CDC Adult Sepsis Event (ASE)
  surveillance definition within the next 6 hours?

Reference:
  Kamran F, Tjandra D, Heiler A, Virzi J, Singh K, King JE, Valley TS, Wiens J.
  Evaluation of sepsis prediction models before onset of treatment.
  NEJM AI 1(3), 2024. DOI: 10.1056/AIoa2300032.

Rationale:
  Kamran 2024 labels sepsis with a composite of the CDC surveillance
  (Adult Sepsis Event) and CMS SEP-1 definitions, and defines onset as the
  first time at which that definition is met. We adopt the CDC ASE arm via
  clifpy's reference implementation (clifpy.utils.ase.compute_ase) and take
  t_zero as the ASE onset timestamp: the earliest contributing ASE criterion
  --- the "min/first timestamp" --- which equals
  LEAST(blood_culture, first_qualifying_antibiotic_day, vasopressor, IMV,
  AKI, hyperbilirubinemia, thrombocytopenia, lactate). Because ASE onset
  folds in treatment/diagnostic actions (blood culture, qualifying
  antibiotics, vasopressors), predictions made at or after it would trivially
  correlate with the outcome ("label leakage"); the prediction window
  therefore terminates strictly BEFORE t_zero, so the model only sees data
  predating clinical recognition.

  Caveat: ASE onset here is the *earliest* contributing criterion (clifpy's
  LEAST over component timestamps), not the instant both Component A
  (presumed infection) and Component B (organ dysfunction) are jointly
  satisfied. This matches the requested "1st min timestamp" and is the
  conservative (leak-safe) anchor.

Anchors:
  t_zero      = min(ase_onset_w_lactate_dttm | sepsis == 1)  per hospitalization
  pred_start  = max(min(vitals.recorded_dttm), admission_dttm + 1h)
  pred_stop   = min(t_zero, discharge_dttm, death_dttm)   # t_zero = +inf if never septic

Cohort (hospital-wide, Kamran 2024):
  1. Adult hospitalizations (age >= 18) with valid admission/discharge dttm.
  2. EXCLUDE hospitalizations with any ADT stay on psychiatric or
     rehabilitation unit.
  3. EXCLUDE already-septic-at-entry (t_zero <= pred_start).
  4. EXCLUDE hospitalizations with no vital sign measurements
     (pred_start is undefined).

Data requirement:
  compute_ase needs the CLIF microbiology_culture table (blood cultures,
  fluid_category == 'blood_buffy'). Exports lacking it cannot establish
  Component A and will not yield a usable label.

Row shape:
  One row per hourly decision point in [pred_start, pred_stop), closed-open.
  Hospitalizations that never meet ASE produce all-negative rows over
  [pred_start, discharge or death].

Label rule (label_sepsis_6h):
  y(d) = 1 iff t_zero is not null AND prediction_dttm < t_zero <= prediction_dttm + 6h.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from clifpy.utils.ase import compute_ase

from flair_benchmark._clif import (
    base_icu_cohort,  # noqa: F401 — re-exported for parity with other tasks
    load_adt,
    load_hospitalization,
    load_patient,
    load_vitals,
    pd_to_pl,
    read_clif_config,
)
from flair_benchmark._split import assign_split
from flair_benchmark._stitch import attach_join_id, load_or_build_encounter_index
from flair_benchmark.tasks._finalize import finalize_cohort

LABEL_HORIZON_HOURS = 6
GRID_STEP_HOURS = 1

# ASE onset column from clifpy.utils.ase.compute_ase. The "_w_lactate" arm
# includes lactate >= 2.0 mmol/L as an organ-dysfunction criterion, bringing
# the CDC ASE definition closer to the CMS SEP-1 arm Kamran 2024 also uses.
ASE_ONSET_COL = "ase_onset_w_lactate_dttm"
ASE_INCLUDE_LACTATE = True
EXCLUDE_LOCATION_CATEGORIES = ["psychiatric", "rehabilitation"]

META = {
    "name": "task4_sepsis_abx_6h",
    "display_name": "Sepsis (CDC ASE onset) within 6h",
    "task_class": "intervention",
    "temporal_pattern": "continuous",
    "prediction_unit": "one-per-window",
    "task_type": "binary",
    "clinical_threshold": 0.1,
    "clinical_action": "treat with antibiotics",
    "clinical_direction": "above",  # act when risk of sepsis > 10%
    "label_column": "label_sepsis_6h",
    "positive_class": "ase_onset_within_6h",
    "input_window_hours": None,
    "prediction_window": [0, LABEL_HORIZON_HOURS],
    "window_unit": "hour",  # window_number increments one row per hour
    "cohort_description": (
        "Hospital-wide adult cohort (Kamran 2024); psych/rehab units excluded; "
        "already-septic-at-entry excluded; must have at least one vital sign. "
        "Label anchored on CDC Adult Sepsis Event onset (clifpy compute_ase)."
    ),
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "valid admission_dttm and discharge_dttm",
        "at least one vital sign recorded during admission",
    ],
    "exclusion_criteria": [
        "any ADT stay on psychiatric or rehabilitation unit",
        "ASE onset at or before first vital sign (already septic at entry)",
        "no vital signs recorded",
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
    cohort = (
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
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    return attach_join_id(cohort, idx)


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


def _sepsis_onset_dttm(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """Run clifpy's CDC Adult Sepsis Event (ASE) calculator and return the
    earliest ASE onset per hospitalization as t_zero.

    compute_ase emits one row per (hospitalization, blood culture) episode
    with a 0/1 ``sepsis`` flag and an onset timestamp (the LEAST over all
    contributing ASE criteria). We keep septic episodes and take the minimum
    onset per hospitalization. Hospitalizations that never meet ASE are absent
    here and become all-negative downstream (t_zero = null).

    Requires the CLIF microbiology_culture table (blood cultures); without it
    compute_ase cannot establish Component A and raises on load.
    """
    micro = Path(cfg["data_directory"]) / f"clif_microbiology_culture.{cfg['filetype']}"
    if not micro.exists():
        raise FileNotFoundError(
            "task4_sepsis: CDC ASE labeling needs the CLIF microbiology_culture "
            f"table (blood cultures), not found at {micro}. Add "
            "clif_microbiology_culture to the export or revert to the "
            "antibiotic-anchored label."
        )

    ase = compute_ase(
        hospitalization_ids=hosp_ids,
        data_directory=cfg["data_directory"],
        filetype=cfg["filetype"],
        timezone=cfg["timezone"],
        apply_rit=True,
        include_lactate=ASE_INCLUDE_LACTATE,
        verbose=False,
    )

    ase = pd_to_pl(ase[["hospitalization_id", "sepsis", ASE_ONSET_COL]])
    return (
        ase.filter((pl.col("sepsis") == 1) & pl.col(ASE_ONSET_COL).is_not_null())
        .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .group_by("hospitalization_id")
        .agg(pl.col(ASE_ONSET_COL).min().alias("t_zero"))
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

    t_zero = _sepsis_onset_dttm(cfg, hosp_ids)
    cohort = cohort.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).join(
        t_zero, on="hospitalization_id", how="left"
    )

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

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return finalize_cohort(df, OUTPUT_COLS, META)
