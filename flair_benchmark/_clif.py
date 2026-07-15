"""Shared CLIF loaders. Replaces FLAIRCohortBuilder.

Each task script imports the small subset it needs:

    from flair_benchmark._clif import (
        read_clif_config, base_icu_cohort,
        load_hospitalization, load_adt, load_patient,
        load_respiratory_support, load_vitals, load_labs,
        load_medication_admin_intermittent,
        load_medication_admin_continuous,
        pd_to_pl,
    )
"""

from __future__ import annotations

import json
from typing import Optional

import polars as pl


def read_clif_config(path: str) -> dict:
    """Parse a clif_config.json file.

    Expected keys: site, data_directory, filetype, timezone.
    Optional keys:
      stitch_time_interval_hours  gap (h) for encounter stitching (default 6)
      cache_directory             where the encounter index parquet is cached
                                  (default <data_directory>/_flair_cache)
    """
    with open(path) as f:
        cfg = json.load(f)
    if not cfg.get("site"):
        # Fail loudly: two behaviors key on the exact site string — the
        # DATA_CUTOFF year filter is SKIPPED only for site == "mimic"
        # (de-identified/shifted dates), and the no-dates split regime is
        # MIMIC-keyed. A defaulted "unknown" site would silently apply the real
        # date cutoff to MIMIC's shifted dates and empty the whole cohort.
        raise ValueError(
            f"clif config {path!r} is missing required key 'site'. Set it to "
            f"your site name (use 'mimic' for MIMIC-IV — it controls the "
            f"date-cutoff exemption and the random-split regime).")
    out = {
        "site": cfg["site"],
        "data_directory": cfg["data_directory"],
        "filetype": cfg["filetype"],
        "timezone": cfg.get("timezone", "US/Central"),
        "stitch_time_interval_hours": cfg.get("stitch_time_interval_hours", 6),
    }
    if cfg.get("cache_directory"):
        out["cache_directory"] = cfg["cache_directory"]
    return out


def pd_to_pl(pdf) -> pl.DataFrame:
    """Convert clifpy pandas df to polars, stripping any timezone info.

    clifpy attaches site tz to datetime columns; polars-native
    `dt.replace_time_zone(None)` makes all datetimes tz-naive so
    downstream comparisons cannot mix tz-aware and tz-naive values.
    """
    df = pl.from_pandas(pdf)
    tz_cols = [
        name for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
    ]
    if tz_cols:
        df = df.with_columns([pl.col(c).dt.replace_time_zone(None) for c in tz_cols])
    return df


def _load(table_cls, cfg: dict, filters: Optional[dict] = None,
          columns: Optional[list[str]] = None) -> pl.DataFrame:
    """Internal: instantiate a clifpy table class and return a polars DataFrame."""
    table = table_cls.from_file(
        data_directory=cfg["data_directory"],
        filetype=cfg["filetype"],
        timezone=cfg["timezone"],
        filters=filters,
    )
    df = pd_to_pl(table.df)
    if columns:
        df = df.select([c for c in columns if c in df.columns])
    return df


def load_hospitalization(cfg: dict, hospitalization_ids: Optional[list[str]] = None,
                         columns: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy Hospitalization → polars."""
    from clifpy.tables import Hospitalization
    filters = {"hospitalization_id": hospitalization_ids} if hospitalization_ids else None
    return _load(Hospitalization, cfg, filters=filters, columns=columns)


def load_patient(cfg: dict, patient_ids: Optional[list[str]] = None,
                 columns: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy Patient → polars."""
    from clifpy.tables import Patient
    filters = {"patient_id": patient_ids} if patient_ids else None
    return _load(Patient, cfg, filters=filters, columns=columns)


def load_adt(cfg: dict, hospitalization_ids: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy Adt → polars."""
    from clifpy.tables import Adt
    filters = {"hospitalization_id": hospitalization_ids} if hospitalization_ids else None
    return _load(Adt, cfg, filters=filters)


def load_respiratory_support(cfg: dict, hospitalization_ids: list[str],
                             device_categories: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy RespiratorySupport → polars. Long table — always filter by hospitalization_id."""
    from clifpy.tables import RespiratorySupport
    filters: dict = {"hospitalization_id": hospitalization_ids}
    if device_categories:
        filters["device_category"] = device_categories
    return _load(RespiratorySupport, cfg, filters=filters)


def load_vitals(cfg: dict, hospitalization_ids: list[str],
                vital_categories: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy Vitals → polars. Long table — always filter by hospitalization_id.
    Pass vital_categories=None to load all vital categories (used when the
    first-recorded-vital timestamp matters and the category is irrelevant)."""
    from clifpy.tables import Vitals
    filters: dict = {"hospitalization_id": hospitalization_ids}
    if vital_categories is not None:
        filters["vital_category"] = vital_categories
    return _load(Vitals, cfg, filters=filters)


def load_medication_admin_intermittent(
    cfg: dict,
    hospitalization_ids: list[str],
    med_categories: Optional[list[str]] = None,
    med_groups: Optional[list[str]] = None,
) -> pl.DataFrame:
    """clifpy MedicationAdminIntermittent → polars. Long table — always filter
    by hospitalization_id. Optional med_categories / med_groups are matched
    case-insensitively on the polars side (column + input both lowercased) to
    be robust against site-level casing drift in CLIF mCIDE enum values."""
    from clifpy.tables import MedicationAdminIntermittent
    df = _load(MedicationAdminIntermittent, cfg,
               filters={"hospitalization_id": hospitalization_ids})
    if med_categories is not None:
        wanted = [s.lower() for s in med_categories]
        df = df.filter(pl.col("med_category").str.to_lowercase().is_in(wanted))
    if med_groups is not None:
        wanted = [s.lower() for s in med_groups]
        df = df.filter(pl.col("med_group").str.to_lowercase().is_in(wanted))
    return df


def load_medication_admin_continuous(
    cfg: dict,
    hospitalization_ids: list[str],
    med_categories: Optional[list[str]] = None,
    med_groups: Optional[list[str]] = None,
) -> pl.DataFrame:
    """clifpy MedicationAdminContinuous → polars. Long table — always filter
    by hospitalization_id. Optional med_categories / med_groups are matched
    case-insensitively (see load_medication_admin_intermittent)."""
    from clifpy.tables import MedicationAdminContinuous
    df = _load(MedicationAdminContinuous, cfg,
               filters={"hospitalization_id": hospitalization_ids})
    if med_categories is not None:
        wanted = [s.lower() for s in med_categories]
        df = df.filter(pl.col("med_category").str.to_lowercase().is_in(wanted))
    if med_groups is not None:
        wanted = [s.lower() for s in med_groups]
        df = df.filter(pl.col("med_group").str.to_lowercase().is_in(wanted))
    return df


def load_labs(cfg: dict, hospitalization_ids: list[str],
              lab_categories: list[str]) -> pl.DataFrame:
    """clifpy Labs → polars. Long table — always filter by hospitalization_id AND lab_category."""
    from clifpy.tables import Labs
    filters = {"hospitalization_id": hospitalization_ids, "lab_category": lab_categories}
    return _load(Labs, cfg, filters=filters)


def load_hospital_diagnosis(cfg: dict,
                            hospitalization_ids: Optional[list[str]] = None) -> pl.DataFrame:
    """clifpy HospitalDiagnosis → polars. Filter by hospitalization_id when provided."""
    from clifpy.tables import HospitalDiagnosis
    filters = {"hospitalization_id": hospitalization_ids} if hospitalization_ids else None
    return _load(HospitalDiagnosis, cfg, filters=filters)


def base_icu_cohort(cfg: dict, min_age: int = 18,
                    min_los_days: float = 0.0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the standard adult ICU base cohort.

    ADT-first approach:
      1. Load ADT, filter to location_category='icu', collect hospitalization_ids.
      2. Load hospitalization filtered to those ids.
      3. Load patient filtered to those patient_ids.
      4. Apply null-date / age / LOS filters.

    Both returned frames carry hospitalization_join_id (the stitched-encounter
    key); the row grain stays one-per-hospitalization_id.

    Returns (cohort, adt) where:
      cohort columns:
        hospitalization_id, hospitalization_join_id, patient_id,
        admission_dttm, discharge_dttm, age_at_admission, sex_category,
        race_category, ethnicity_category, discharge_category, death_dttm
      adt columns: as loaded by clifpy (filtered to final cohort
        hospitalization_ids) plus hospitalization_join_id.
    """
    adt = load_adt(cfg)
    icu_hosp_ids = (
        adt.filter(pl.col("location_category").str.to_lowercase() == "icu")
        .select("hospitalization_id")
        .unique()["hospitalization_id"]
        .to_list()
    )

    hosp = load_hospitalization(
        cfg,
        hospitalization_ids=icu_hosp_ids,
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
        .filter(
            ((pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 86400.0)
            > min_los_days
        )
        .select([
            "hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm",
            "age_at_admission", "sex_category", "race_category", "ethnicity_category",
            "discharge_category", "death_dttm",
        ])
    )

    final_ids = cohort["hospitalization_id"].to_list()
    adt = adt.filter(pl.col("hospitalization_id").is_in(final_ids))

    # Stitch encounters once (cached) and tag both frames with the block key.
    from flair_benchmark._stitch import attach_join_id, load_or_build_encounter_index
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    cohort = attach_join_id(cohort, idx)
    adt = attach_join_id(adt, idx)
    return cohort, adt
