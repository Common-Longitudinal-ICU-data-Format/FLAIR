"""get_meds_stream — lightweight med ELF pull, NO conversion.

Pulls the configured med categories straight from the CLIF med tables and emits
ELF events with the RAW dose + RAW unit. Deliberately skips the heavy CLIF_MEDS
ETL (unit conversion, vitals/weight join, dose normalization) — the raw unit is
charted on every dose so a downstream step can convert later.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (
    load_medication_admin_continuous,
    load_medication_admin_intermittent,
)
from flair_benchmark._stitch import JOIN_ID, attach_join_id as attach_stitched_join
from flair_benchmark.features._elf import (
    attach_subject_id,
    empty_elf,
    finalize,
)
from flair_benchmark.features._scope import HORIZON_COL, feature_scope

_TABLES = {
    "medication_admin_continuous": ("MED_CON", load_medication_admin_continuous),
    "medication_admin_intermittent": ("MED_INT", load_medication_admin_intermittent),
}


def _one_table(
    domain: str,
    df: pl.DataFrame,
    horizon: pl.DataFrame,
    idx: pl.DataFrame,
) -> pl.DataFrame:
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
        attach_stitched_join(df, idx)
        .join(horizon, on=JOIN_ID, how="inner")
        .filter(pl.col("time") <= pl.col(HORIZON_COL))
        .select(["hospitalization_id", "time", "code",
                 "numeric_value", "unit", "text_value", JOIN_ID])
    )
    return df


def get_meds_stream(task_df: pl.DataFrame, feature_cfg: dict,
                    clif_cfg: dict) -> pl.DataFrame:
    """Build the ELF-long lightweight med stream for a task cohort."""
    meds_cfg = feature_cfg.get("meds_stream") or {}
    if not meds_cfg:
        return empty_elf()

    scope = feature_scope(task_df, clif_cfg)
    if not scope.hospitalization_ids:
        return empty_elf()

    parts: list[pl.DataFrame] = []
    for table, tspec in meds_cfg.items():
        if table not in _TABLES:
            continue
        domain, loader = _TABLES[table]
        cats = (tspec or {}).get("categories")
        df = loader(
            clif_cfg,
            hospitalization_ids=scope.hospitalization_ids,
            med_categories=cats,
        )
        events = _one_table(domain, df, scope.horizons, scope.index)
        if events.height:
            parts.append(events)

    if not parts:
        return empty_elf()

    elf = pl.concat(parts, how="vertical")
    elf = attach_subject_id(elf, clif_cfg)
    return finalize(elf)
