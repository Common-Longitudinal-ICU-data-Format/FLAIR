"""get_meds_stream — lightweight med ELF pull, NO conversion.

Pulls the configured med categories straight from the CLIF med tables and emits
ELF events with the RAW dose + RAW unit. Deliberately skips the heavy CLIF_MEDS
ETL (unit conversion, vitals/weight join, dose normalization) — the raw unit is
charted on every dose so a downstream step can convert later.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (
    load_medication_admin_continuous, load_medication_admin_intermittent,
)
from flair_benchmark.features._elf import (
    attach_subject_id, empty_elf, finalize,
)

_TABLES = {
    "medication_admin_continuous": ("MED_CON", load_medication_admin_continuous),
    "medication_admin_intermittent": ("MED_INT", load_medication_admin_intermittent),
}


def _horizon(task_df: pl.DataFrame) -> pl.DataFrame:
    """Per-stay latest prediction time (the no-future-data bound)."""
    return (
        task_df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .group_by("hospitalization_id")
        .agg(pl.col("prediction_dttm").max().alias("_horizon"))
    )


def _one_table(domain: str, df: pl.DataFrame, horizon: pl.DataFrame) -> pl.DataFrame:
    """CLIF med table (polars) -> ELF events, bounded at the horizon."""
    if df.height == 0:
        return empty_elf()

    unit_expr = (
        pl.col("med_dose_unit") if "med_dose_unit" in df.columns
        else pl.lit(None, dtype=pl.Utf8)
    )
    dose_expr = (
        pl.col("med_dose").cast(pl.Float64, strict=False) if "med_dose" in df.columns
        else pl.lit(None, dtype=pl.Float64)
    )

    df = (
        df.with_columns([
            pl.col("hospitalization_id").cast(pl.Utf8),
            dose_expr.alias("numeric_value"),
            unit_expr.cast(pl.Utf8).alias("unit"),
            pl.col("admin_dttm").alias("time"),
        ])
        # Chart 'UNK' only when a dose exists but no unit (CLIF_MEDS convention),
        # so a numeric value never lacks a unit to convert from.
        .with_columns(
            pl.when(pl.col("numeric_value").is_not_null() & pl.col("unit").is_null())
            .then(pl.lit("UNK"))
            .otherwise(pl.col("unit"))
            .alias("unit")
        )
        .with_columns(
            pl.concat_str(
                [pl.lit(domain), pl.col("med_category"), pl.col("unit")],
                separator="//", ignore_nulls=True,
            ).alias("code")
        )
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("text_value"))
    )

    df = (
        df.join(horizon, on="hospitalization_id", how="inner")
        .filter(pl.col("time") <= pl.col("_horizon"))
        .select(["hospitalization_id", "time", "code",
                 "numeric_value", "unit", "text_value"])
    )
    return df


def get_meds_stream(task_df: pl.DataFrame, feature_cfg: dict,
                    clif_cfg: dict) -> pl.DataFrame:
    """Build the ELF-long lightweight med stream for a task cohort."""
    meds_cfg = feature_cfg.get("meds_stream") or {}
    if not meds_cfg:
        return empty_elf()

    hosp_ids = (
        task_df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .select("hospitalization_id").unique()["hospitalization_id"].to_list()
    )
    horizon = _horizon(task_df)

    parts: list[pl.DataFrame] = []
    for table, tspec in meds_cfg.items():
        if table not in _TABLES:
            continue
        domain, loader = _TABLES[table]
        cats = (tspec or {}).get("categories")
        df = loader(clif_cfg, hospitalization_ids=hosp_ids, med_categories=cats)
        events = _one_table(domain, df, horizon)
        if events.height:
            parts.append(events)

    if not parts:
        return empty_elf()

    elf = pl.concat(parts, how="vertical")
    elf = attach_subject_id(elf, clif_cfg)
    return finalize(elf)
