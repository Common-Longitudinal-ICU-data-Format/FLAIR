"""Shared CLIF loaders. Replaces FLAIRCohortBuilder.

Each task script imports the small subset it needs:

    from flair_benchmark._clif import (
        read_clif_config, base_icu_cohort,
        load_hospitalization, load_adt, load_patient,
        load_respiratory_support, load_vitals, load_labs,
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
    """
    with open(path) as f:
        cfg = json.load(f)
    return {
        "site": cfg.get("site", "unknown"),
        "data_directory": cfg["data_directory"],
        "filetype": cfg["filetype"],
        "timezone": cfg.get("timezone", "US/Central"),
    }


def pd_to_pl(pdf) -> pl.DataFrame:
    """Strip timezone from every datetime64 column, then convert to polars.

    clifpy returns tz-aware pandas; polars conversion requires tz stripped.
    """
    import pandas as pd

    for col in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[col]):
            if hasattr(pdf[col].dtype, "tz") and pdf[col].dtype.tz is not None:
                pdf[col] = pdf[col].dt.tz_localize(None)
    return pl.from_pandas(pdf)


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
                vital_categories: list[str]) -> pl.DataFrame:
    """clifpy Vitals → polars. Long table — always filter by hospitalization_id AND vital_category."""
    from clifpy.tables import Vitals
    filters = {"hospitalization_id": hospitalization_ids, "vital_category": vital_categories}
    return _load(Vitals, cfg, filters=filters)


def load_labs(cfg: dict, hospitalization_ids: list[str],
              lab_categories: list[str]) -> pl.DataFrame:
    """clifpy Labs → polars. Long table — always filter by hospitalization_id AND lab_category."""
    from clifpy.tables import Labs
    filters = {"hospitalization_id": hospitalization_ids, "lab_category": lab_categories}
    return _load(Labs, cfg, filters=filters)


def base_icu_cohort(cfg: dict, min_age: int = 18,
                    min_los_days: float = 0.0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the standard adult ICU base cohort.

    ADT-first approach:
      1. Load ADT, filter to location_category='icu', collect hospitalization_ids.
      2. Load hospitalization filtered to those ids.
      3. Load patient filtered to those patient_ids.
      4. Apply null-date / age / LOS filters.

    Returns (cohort, adt) where:
      cohort columns:
        hospitalization_id, patient_id, admission_dttm, discharge_dttm,
        age_at_admission, sex_category, race_category, ethnicity_category,
        discharge_category, death_dttm
      adt columns: as loaded by clifpy (filtered to final cohort hospitalization_ids).
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
    return cohort, adt
