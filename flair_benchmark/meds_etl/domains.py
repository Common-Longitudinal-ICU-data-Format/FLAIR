"""Per-domain CLIF → MEDS runners. Each returns a polars MEDS events frame.

Two paths:
  - run_generic: VITAL, LAB, PA, RESP, CRRT, HOSP_DX, … (load → optional clamp/drop → emit)
  - run_med:     MED_CON, MED_INT (clifpy dose-unit conversion before emit)
"""
from __future__ import annotations

import threading

import polars as pl

from flair_benchmark.meds_etl._emit import (
    TableCache,
    build_clamp,
    concept_section,
    emit_events,
    hosp_filters,
    join_hospitalization,
    load_clif_pandas,
    load_clif_table,
    restrict_to_hosp,
    subject_col,
)
from flair_benchmark.meds_etl.transforms import normalize_categories, strip_tz

# Domains whose events are dropped when both numeric_value and text_value are null.
_SKIP_NULL = {"RESP", "CRRT", "ECMO_MCS"}
# Columns dropped per domain before emission (parity with CLIF_MEDS processors).
_DROP_COLS = {
    "RESP": ("vent_brand_name",),
    "CRRT": ("device_id", "dialysis_machine_name"),
}


def _apply_category_filter(df: pl.DataFrame, opts: dict) -> pl.DataFrame:
    """Filter rows to opts['categories'] on opts['category_col'] (values lowercased)."""
    cats = opts.get("categories")
    col = opts.get("category_col")
    if not cats or not col or col not in df.columns:
        return df
    wanted = [str(c).lower() for c in cats]
    return df.filter(pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(wanted))


def run_generic(domain: str, clif_cfg: dict, domain_cfg: dict,
                hosp_ids: list[str] | None, opts: dict,
                cache: TableCache | None = None) -> pl.DataFrame:
    table_name, concepts = concept_section(domain_cfg)
    subj = subject_col(domain_cfg, clif_cfg)

    df = restrict_to_hosp(load_clif_table(clif_cfg, table_name, hosp_ids=hosp_ids), hosp_ids)
    for col in _DROP_COLS.get(domain, ()):
        if col in df.columns:
            df = df.drop(col)
    df = _apply_category_filter(df, opts)
    df = join_hospitalization(df, clif_cfg, concepts, subj, hosp_ids, cache=cache)

    clamp_limits, clamp_col = build_clamp(domain_cfg)
    return emit_events(df, concepts, subj,
                       skip_null=domain in _SKIP_NULL,
                       clamp_limits=clamp_limits, clamp_category_col=clamp_col)


# clifpy's converter runs SQL on duckdb's module-level default connection, which
# is not safe for concurrent queries — serialize conversions across domain threads.
_MED_CONVERT_LOCK = threading.Lock()

# The converter only reads weight_kg rows from vitals (see clifpy
# find_most_recent_weight): project + filter at load instead of full-loading.
_WEIGHT_VITALS_COLS = ["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]


def _load_weight_vitals(clif_cfg: dict, hosp_ids: list[str] | None,
                        cache: TableCache | None):
    filters = hosp_filters(hosp_ids) or {}
    filters["vital_category"] = ["weight_kg"]

    def _load():
        return load_clif_pandas(clif_cfg, "vitals",
                                filters=filters, columns=_WEIGHT_VITALS_COLS)

    return cache.get_or_load(("vitals", "weight_kg"), _load) if cache is not None else _load()


def run_med(domain: str, clif_cfg: dict, domain_cfg: dict,
            hosp_ids: list[str] | None, opts: dict,
            cache: TableCache | None = None) -> pl.DataFrame:
    """MED_CON / MED_INT — clifpy dose-unit conversion (needs vitals for weights), then emit."""
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category

    table_name, concepts = concept_section(domain_cfg)
    subj = subject_col(domain_cfg, clif_cfg)

    med_pdf = load_clif_pandas(clif_cfg, table_name, filters=hosp_filters(hosp_ids))
    if hosp_ids is not None and "hospitalization_id" in med_pdf.columns:
        med_pdf = med_pdf[med_pdf["hospitalization_id"].astype(str).isin(hosp_ids)]

    uc = next(iter(concepts.values())).get("unit_conversion", {})
    preferred = uc.get("preferred_units", {})
    if uc.get("enabled", False) and preferred:
        if not uc.get("allow_other_meds", True):
            med_pdf = med_pdf[med_pdf["med_category"].isin(set(preferred.keys()))]
        vitals_pdf = _load_weight_vitals(clif_cfg, hosp_ids, cache)
        if hosp_ids is not None and "hospitalization_id" in vitals_pdf.columns:
            vitals_pdf = vitals_pdf[vitals_pdf["hospitalization_id"].astype(str).isin(hosp_ids)]
        with _MED_CONVERT_LOCK:
            med_pdf, _ = convert_dose_units_by_med_category(
                med_df=med_pdf, vitals_df=vitals_pdf,
                preferred_units=preferred, override=uc.get("override", False),
            )
    else:
        med_pdf["med_dose_converted"] = med_pdf["med_dose"]
        med_pdf["med_dose_unit_converted"] = "UNK"

    df = normalize_categories(strip_tz(pl.from_pandas(med_pdf)))
    df = _apply_category_filter(df, opts)
    df = join_hospitalization(df, clif_cfg, concepts, subj, hosp_ids, cache=cache)
    return emit_events(df, concepts, subj, skip_null=False)


# Dispatch — domain name → runner. MED_* use the unit-conversion path; rest are generic.
_MED_DOMAINS = {"MED_CON", "MED_INT"}


def run_domain(domain: str, clif_cfg: dict, domain_cfg: dict,
               hosp_ids: list[str] | None, opts: dict,
               cache: TableCache | None = None) -> pl.DataFrame:
    if domain in _MED_DOMAINS:
        return run_med(domain, clif_cfg, domain_cfg, hosp_ids, opts, cache=cache)
    return run_generic(domain, clif_cfg, domain_cfg, hosp_ids, opts, cache=cache)
