"""get_wide_feature_dataset — clifpy wide dataset → ELF long.

Pulls every configured time-series feature for the task cohort via clifpy
``create_wide_dataset``, bounding events per stay to ``time <=
max(prediction_dttm)`` (the no-future-data guard — all history up to the
prediction time is allowed), strips tz, then melts the pivoted wide frame
back to ELF long events.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import pd_to_pl
from flair_benchmark.features._config import V1_TABLES
from flair_benchmark.features._elf import (
    attach_join_id,
    attach_subject_id,
    build_code,
    empty_elf,
    finalize,
)
from flair_benchmark.features._units import RESP_CATEGORICAL, unit_for

# CLIF table -> ELF code domain prefix.
DOMAIN = {
    "vitals": "VITAL",
    "labs": "LAB",
    "medication_admin_continuous": "MED_CON",
    "medication_admin_intermittent": "MED_INT",
    "patient_assessments": "PA",
    "respiratory_support": "RESP",
}

_IDX = ["hospitalization_id", "subject_id", "time"]


def _cohort_windows(task_df: pl.DataFrame):
    """Per-stay no-future cutoff + the hospitalization_id list (as strings)."""
    bounds = (
        task_df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .group_by("hospitalization_id")
        .agg(pl.col("prediction_dttm").max().alias("end_time"))
    )
    return bounds, bounds["hospitalization_id"].to_list()


def _feature_maps(wide_cfg: dict, units_override: dict | None):
    """Return (numeric: feature->{code,unit}, text: feature->code)."""
    numeric: dict[str, dict] = {}
    text: dict[str, str] = {}
    for table, tspec in (wide_cfg or {}).items():
        if table not in V1_TABLES or table not in DOMAIN:
            continue
        dom = DOMAIN[table]
        if table == "respiratory_support":
            for col in (tspec or {}).get("columns", []):
                if col in RESP_CATEGORICAL:
                    text[col] = build_code(dom, col)
                else:
                    u = unit_for(table, col, units_override)
                    numeric[col] = {"code": build_code(dom, col, u), "unit": u}
        else:
            for cat in (tspec or {}).get("categories", []):
                u = unit_for(table, cat, units_override)
                numeric[cat] = {"code": build_code(dom, cat, u), "unit": u}
    return numeric, text


def get_wide_feature_dataset(task_df: pl.DataFrame, feature_cfg: dict,
                             clif_cfg: dict) -> pl.DataFrame:
    """Build the ELF-long wide feature dataset for a task cohort."""
    wide_cfg = feature_cfg.get("wide") or {}
    if not wide_cfg:
        return empty_elf()

    category_filters: dict[str, list[str]] = {}
    for table, tspec in wide_cfg.items():
        if table not in V1_TABLES or table not in DOMAIN:
            continue
        vals = (tspec or {}).get("categories") or (tspec or {}).get("columns")
        if vals:
            category_filters[table] = list(vals)
    if not category_filters:
        return empty_elf()

    bounds, hosp_ids = _cohort_windows(task_df)

    # clifpy create_wide_dataset loads its own base tables (patient/hospitalization/
    # adt) — only pass the feature tables + hospitalization_ids. The per-stay time
    # truncation (the no-future-data guard) is enforced below in polars, avoiding
    # the tz-aware cohort_df column-name ambiguity across clifpy layers.
    from clifpy import ClifOrchestrator

    co = ClifOrchestrator(
        data_directory=clif_cfg["data_directory"],
        filetype=clif_cfg["filetype"],
        timezone=clif_cfg["timezone"],
    )
    co.create_wide_dataset(
        tables_to_load=list(category_filters),
        category_filters=category_filters,
        hospitalization_ids=hosp_ids,
        output_format="dataframe",
        show_progress=False,
    )

    wide = pd_to_pl(co.wide_df)
    if wide.height == 0:
        return empty_elf()

    wide = wide.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    if "patient_id" in wide.columns:
        wide = wide.with_columns(pl.col("patient_id").cast(pl.Utf8).alias("subject_id"))
    else:
        wide = attach_subject_id(wide, clif_cfg)
    wide = wide.rename({"event_time": "time"})

    # No-future-data guard: keep only events at or before each stay's
    # max(prediction_dttm). No lower bound — all history up to the
    # prediction time may be used.
    wide = (
        wide.join(bounds, on="hospitalization_id", how="inner")
        .filter(pl.col("time") <= pl.col("end_time"))
        .drop("end_time")
    )
    if wide.height == 0:
        return empty_elf()

    numeric, text = _feature_maps(wide_cfg, feature_cfg.get("units"))
    parts: list[pl.DataFrame] = []

    num_cols = [c for c in numeric if c in wide.columns]
    if num_cols:
        nmap = pl.DataFrame({
            "feature": num_cols,
            "code": [numeric[c]["code"] for c in num_cols],
            "unit": [numeric[c]["unit"] for c in num_cols],
        })
        long_n = (
            wide.select(_IDX + num_cols)
            .unpivot(index=_IDX, on=num_cols,
                     variable_name="feature", value_name="_val")
            .with_columns(pl.col("_val").cast(pl.Float64, strict=False))
            .filter(pl.col("_val").is_not_null())
            .join(nmap, on="feature", how="left")
            .with_columns([
                pl.col("_val").alias("numeric_value"),
                pl.lit(None, dtype=pl.Utf8).alias("text_value"),
            ])
            .select(_IDX + ["code", "numeric_value", "unit", "text_value"])
        )
        parts.append(long_n)

    txt_cols = [c for c in text if c in wide.columns]
    if txt_cols:
        tmap = pl.DataFrame({"feature": txt_cols,
                             "code": [text[c] for c in txt_cols]})
        long_t = (
            wide.select(_IDX + txt_cols)
            .with_columns([pl.col(c).cast(pl.Utf8) for c in txt_cols])
            .unpivot(index=_IDX, on=txt_cols,
                     variable_name="feature", value_name="_val")
            .filter(pl.col("_val").is_not_null())
            .join(tmap, on="feature", how="left")
            .with_columns([
                pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
                pl.lit(None, dtype=pl.Utf8).alias("unit"),
                pl.col("_val").alias("text_value"),
            ])
            .select(_IDX + ["code", "numeric_value", "unit", "text_value"])
        )
        parts.append(long_t)

    if not parts:
        return empty_elf()

    elf = pl.concat(parts, how="vertical")
    elf = attach_join_id(elf, clif_cfg)
    return finalize(elf)
