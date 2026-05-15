"""First-24h SOFA per hospitalization (worst-in-window).

Computed once per site against a fixed reference window — the first 24h of
each hospitalization's first ICU stay — independent of which task is
being summarized. Decision-point and timestep tasks therefore inherit a
stay-level severity measure that's consistent with the one-per-stay tasks.

Follows the clifpy `sofa_score_calculation.py` skill pattern: load minimal
tables, convert vasopressor doses, build wide dataset bounded by cohort,
backfill missing medication columns, then call `compute_sofa`.
"""

from __future__ import annotations

import polars as pl


def compute_first_24h_sofa_max(orchestrator,
                               first_icu_per_hosp: pl.DataFrame) -> pl.DataFrame:
    """Return [hospitalization_id, sofa_max] for the first 24h of first ICU stay.

    first_icu_per_hosp: pl.DataFrame with columns
      hospitalization_id, first_icu_start_time
    """
    if first_icu_per_hosp.height == 0:
        return _empty_frame()

    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE, compute_sofa

    tz = getattr(orchestrator, "timezone", None)
    cohort_pd = first_icu_per_hosp.select([
        "hospitalization_id",
        pl.col("first_icu_start_time").alias("start_time"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("end_time"),
    ]).to_pandas()
    if tz:
        if cohort_pd["start_time"].dt.tz is None:
            cohort_pd["start_time"] = cohort_pd["start_time"].dt.tz_localize(tz)
        if cohort_pd["end_time"].dt.tz is None:
            cohort_pd["end_time"] = cohort_pd["end_time"].dt.tz_localize(tz)

    hosp_ids = (first_icu_per_hosp["hospitalization_id"]
                .cast(pl.Utf8).to_list())

    _load_sofa_tables(orchestrator, hosp_ids)
    _convert_vasopressor_units(orchestrator)

    orchestrator.create_wide_dataset(
        tables_to_load=list(REQUIRED_SOFA_CATEGORIES_BY_TABLE.keys()),
        category_filters=REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df=cohort_pd,
        return_dataframe=True,
    )
    wide_df = orchestrator.wide_df
    if wide_df is None or len(wide_df) == 0:
        return _empty_frame()

    for col in ("norepinephrine_mcg_kg_min", "epinephrine_mcg_kg_min",
                "dopamine_mcg_kg_min", "dobutamine_mcg_kg_min"):
        if col not in wide_df.columns:
            wide_df[col] = None

    sofa_pdf = compute_sofa(
        wide_df=wide_df,
        id_name="hospitalization_id",
        extremal_type="worst",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
    )
    if sofa_pdf is None or len(sofa_pdf) == 0:
        return _empty_frame()

    sofa_pl = pl.from_pandas(sofa_pdf[["hospitalization_id", "sofa_total"]])
    return sofa_pl.rename({"sofa_total": "sofa_max"}).filter(
        pl.col("hospitalization_id").cast(pl.Utf8).is_in(hosp_ids)
    )


def _load_sofa_tables(orchestrator, hosp_ids: list[str]) -> None:
    """Load only the tables and categories SOFA needs, scoped to cohort."""
    orchestrator.load_table(
        "labs",
        filters={
            "hospitalization_id": hosp_ids,
            "lab_category": ["creatinine", "platelet_count",
                             "po2_arterial", "bilirubin_total"],
        },
        columns=["hospitalization_id", "lab_result_dttm",
                 "lab_category", "lab_value_numeric"],
    )
    orchestrator.load_table(
        "vitals",
        filters={
            "hospitalization_id": hosp_ids,
            "vital_category": ["map", "spo2", "weight_kg", "height_cm"],
        },
        columns=["hospitalization_id", "recorded_dttm",
                 "vital_category", "vital_value"],
    )
    orchestrator.load_table(
        "patient_assessments",
        filters={
            "hospitalization_id": hosp_ids,
            "assessment_category": ["gcs_total"],
        },
        columns=["hospitalization_id", "recorded_dttm",
                 "assessment_category", "numerical_value"],
    )
    orchestrator.load_table(
        "medication_admin_continuous",
        filters={
            "hospitalization_id": hosp_ids,
            "med_category": ["norepinephrine", "epinephrine",
                             "dopamine", "dobutamine"],
        },
    )
    orchestrator.load_table(
        "respiratory_support",
        filters={"hospitalization_id": hosp_ids},
        columns=["hospitalization_id", "recorded_dttm",
                 "device_category", "fio2_set"],
    )


def _convert_vasopressor_units(orchestrator) -> None:
    """SOFA cardiovascular scoring requires vasopressor doses in mcg/kg/min."""
    med_df = orchestrator.medication_admin_continuous.df
    med_df = med_df[med_df["med_dose"].notna()]
    med_df = med_df[med_df["med_dose_unit"].notna()]
    med_df = med_df[~med_df["med_dose_unit"].astype(str)
                    .str.lower().isin(["nan", "none", ""])]
    orchestrator.medication_admin_continuous.df = med_df

    orchestrator.convert_dose_units_for_continuous_meds(
        preferred_units={
            "norepinephrine": "mcg/kg/min",
            "epinephrine": "mcg/kg/min",
            "dopamine": "mcg/kg/min",
            "dobutamine": "mcg/kg/min",
        },
        override=True,
    )
    converted = orchestrator.medication_admin_continuous.df_converted
    orchestrator.medication_admin_continuous.df_converted = (
        converted[converted["_convert_status"] == "success"].copy()
    )


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"hospitalization_id": pl.Utf8, "sofa_max": pl.Float64}
    )
