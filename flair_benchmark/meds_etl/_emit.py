"""Core CLIF-table → MEDS-event emission, shared by every domain.

Consolidates the near-identical per-domain processors from CLIF_MEDS into one
emit loop plus small per-domain hooks (unit conversion, column drops, null-skip,
outlier clamping). Behaviour per domain is preserved exactly; only the duplication
is removed.

Every returned frame has the MEDS event schema:
    subject_id (Int64), time (Datetime), code (Utf8),
    numeric_value (Float32), text_value (Utf8), hospitalization_id (Int64)
"""
from __future__ import annotations

import polars as pl
from clifpy.utils.io import load_data

from flair_benchmark.meds_etl.transforms import normalize_categories, strip_tz

# Keys in a domain YAML that are NOT concept sections.
_CONFIG_META_KEYS = {"subject_id_col", "elf_version", "outlier_shaping"}

MEDS_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime,
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "text_value": pl.Utf8,
    "hospitalization_id": pl.Int64,
}


def concept_section(domain_cfg: dict) -> tuple[str, dict]:
    """Return (table_name, concepts_dict) — the single concept section of a domain YAML.

    The section key doubles as the clifpy table_name (e.g. 'vitals', 'labs',
    'medication_admin_continuous').
    """
    key = [k for k in domain_cfg if k not in _CONFIG_META_KEYS][0]
    return key, domain_cfg[key]


def subject_col(domain_cfg: dict, clif_cfg: dict) -> str:
    return domain_cfg.get("subject_id_col", clif_cfg.get("subject_id_col", "patient_id"))


def load_clif_pandas(clif_cfg: dict, table_name: str):
    """Load a CLIF table via clifpy as pandas (UTC → site-local conversion happens here)."""
    return load_data(
        table_name=table_name,
        table_path=str(clif_cfg["data_directory"]),
        table_format_type=clif_cfg["filetype"],
        site_tz=clif_cfg.get("timezone"),
    )


def load_clif_table(clif_cfg: dict, table_name: str) -> pl.DataFrame:
    """Load a CLIF table → polars, tz-stripped, categories normalized."""
    return normalize_categories(strip_tz(pl.from_pandas(load_clif_pandas(clif_cfg, table_name))))


def restrict_to_hosp(df: pl.DataFrame, hosp_ids: list[str] | None) -> pl.DataFrame:
    """Filter a frame to the cohort's hospitalization_ids (matched as Utf8 for type safety)."""
    if hosp_ids is None or "hospitalization_id" not in df.columns:
        return df
    return df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).filter(
        pl.col("hospitalization_id").is_in(hosp_ids)
    )


def _needed_cols(concepts: dict) -> set[str]:
    needed: set[str] = set()
    for mapping in concepts.values():
        if not isinstance(mapping, dict):
            continue
        time_spec = mapping.get("time")
        if isinstance(time_spec, str) and time_spec.startswith("col("):
            needed.add(time_spec[4:-1])
        code_spec = mapping.get("code")
        if isinstance(code_spec, list):
            for part in code_spec:
                if isinstance(part, dict):
                    expr = next(iter(part))
                    if expr.startswith("col(") and expr.endswith(")"):
                        needed.add(expr[4:-1])
                elif isinstance(part, str) and part.startswith("col(") and part.endswith(")"):
                    needed.add(part[4:-1])
    return needed


def join_hospitalization(df: pl.DataFrame, clif_cfg: dict, concepts: dict,
                         subject_id_col: str, hosp_ids: list[str] | None) -> pl.DataFrame:
    """Left-join the hospitalization table for subject_id / any missing referenced columns."""
    needed = _needed_cols(concepts)
    missing = {c for c in needed if c not in df.columns}
    if subject_id_col not in df.columns:
        missing.add(subject_id_col)
    if not missing or "hospitalization_id" not in df.columns:
        return df

    hosp = restrict_to_hosp(load_clif_table(clif_cfg, "hospitalization"), hosp_ids)
    join_cols = ["hospitalization_id"] + [c for c in missing if c in hosp.columns]
    hosp = hosp.select(join_cols).unique(subset=["hospitalization_id"])
    df = df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    hosp = hosp.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    return df.join(hosp, on="hospitalization_id", how="left")


def build_clamp(domain_cfg: dict) -> tuple[dict, str | None]:
    """Return (clamp_limits, clamp_category_col) from an outlier_shaping config block."""
    oc = domain_cfg.get("outlier_shaping", {})
    if not oc.get("enabled", False):
        return {}, None
    limits = {k: (float(b[0]), float(b[1])) for k, b in oc.get("limits", {}).items()}
    return limits, oc.get("category_col")


def _code_expr(code_spec, available: set[str]) -> pl.Expr:
    """Vectorized ELF-code builder mirroring resolve.resolve_code.

    A ``col(X)`` part becomes the column value (→ 'UNK' when null/absent); a dict
    part ``{col(X): {raw: mapped}}`` applies the value map then 'UNK'-fills.
    """
    if isinstance(code_spec, str):
        return pl.lit(code_spec)
    parts: list[pl.Expr] = []
    for part in code_spec:
        if isinstance(part, dict):
            key = next(iter(part))
            vmap = part[key]
            col = key[4:-1] if key.startswith("col(") and key.endswith(")") else key
            if col in available:
                parts.append(pl.col(col).cast(pl.Utf8).replace(vmap).fill_null("UNK"))
            else:
                parts.append(pl.lit("UNK"))
        elif isinstance(part, str) and part.startswith("col(") and part.endswith(")"):
            col = part[4:-1]
            parts.append(pl.col(col).cast(pl.Utf8).fill_null("UNK") if col in available else pl.lit("UNK"))
        else:
            parts.append(pl.lit(str(part)))
    return pl.concat_str(parts, separator="//")


def _numeric_expr(nv_col: str, clamp_limits: dict, clamp_category_col: str | None,
                  available: set[str]) -> pl.Expr:
    """Float numeric value with optional outlier clamping (by category or per-column)."""
    base = pl.col(nv_col).cast(pl.Float64, strict=False)
    if not clamp_limits:
        return base
    if clamp_category_col and clamp_category_col in available:
        acc = base
        for key, (lo, hi) in clamp_limits.items():
            acc = pl.when(pl.col(clamp_category_col) == key).then(base.clip(lo, hi)).otherwise(acc)
        return acc
    if nv_col in clamp_limits:
        lo, hi = clamp_limits[nv_col]
        return base.clip(lo, hi)
    return base


def emit_events(df: pl.DataFrame, concepts: dict, subject_id_col: str, *,
                skip_null: bool = False,
                clamp_limits: dict | None = None,
                clamp_category_col: str | None = None) -> pl.DataFrame:
    """Emit MEDS event rows (vectorized, one frame per concept). Preserves CLIF_MEDS semantics."""
    clamp_limits = clamp_limits or {}
    available = set(df.columns)
    if subject_id_col not in available:
        return pl.DataFrame(schema=MEDS_SCHEMA)

    subj = pl.col(subject_id_col).cast(pl.Int64, strict=False)
    hosp = (pl.col("hospitalization_id").cast(pl.Int64, strict=False)
            if "hospitalization_id" in available else pl.lit(None, dtype=pl.Int64))

    frames: list[pl.DataFrame] = []
    for concept_name, mapping in concepts.items():
        if concept_name == "unit_conversion":
            continue

        sel = df
        time_spec = mapping.get("time")
        if time_spec is not None:
            tcol = time_spec[4:-1]
            if tcol not in available:
                continue
            sel = sel.filter(pl.col(tcol).is_not_null())
            time_expr = pl.col(tcol).cast(pl.Datetime("us"), strict=False)
        else:
            time_expr = pl.lit(None, dtype=pl.Datetime("us"))

        nv_col = mapping.get("numeric_value")
        if nv_col and nv_col in available:
            num_expr = _numeric_expr(nv_col, clamp_limits, clamp_category_col, available)
        else:
            num_expr = pl.lit(None, dtype=pl.Float64)

        tv_col = mapping.get("text_value")
        text_expr = (pl.col(tv_col).cast(pl.Utf8) if tv_col and tv_col in available
                     else pl.lit(None, dtype=pl.Utf8))

        out = sel.select([
            subj.alias("subject_id"),
            time_expr.alias("time"),
            _code_expr(mapping["code"], available).alias("code"),
            num_expr.cast(pl.Float32).alias("numeric_value"),
            text_expr.alias("text_value"),
            hosp.alias("hospitalization_id"),
        ]).filter(pl.col("subject_id").is_not_null())

        if skip_null:
            out = out.filter(pl.col("numeric_value").is_not_null()
                             | pl.col("text_value").is_not_null())
        frames.append(out)

    if not frames:
        return pl.DataFrame(schema=MEDS_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").select(list(MEDS_SCHEMA.keys()))
