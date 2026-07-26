"""Read a CLIF table, clean it, and emit MEDS event rows. Shared by every domain.

One emit loop plus small per-domain hooks (unit conversion, column drops,
null-skip, outlier nulling), replacing the near-identical per-domain processors
CLIF_MEDS had.

Every returned frame has the MEDS event schema:
    subject_id (Utf8), time (Datetime), code (Utf8),
    numeric_value (Float32), text_value (Utf8), hospitalization_id (Utf8)

IDs ARE STRINGS. CLIF types patient_id and hospitalization_id as strings in
mCIDE, and real sites use alphanumeric ('P101', 'HOSP_0012') or zero-padded
('007') values. Anything else here loses rows or silently renumbers patients.

The load path is: clifpy DuckDB scan -> pandas -> polars -> strip timezone ->
normalize categories. Both cleaning steps matter and are easy to skip by
accident. `strip_tz` prevents a second UTC conversion downstream; the category
normalization is what makes a concept YAML's category keys match real data.

Pushdown (`filters=`) is an OPTIMIZATION ONLY. Callers keep their in-memory
filters, and any pushdown failure falls back to a full load with a warning — so a
site whose id column has a mismatched type gets slower, never wrong.

Depends on: clifpy (loader), polars.
Used by: meds_etl/pipeline.py.
"""
from __future__ import annotations

import os
import threading
import warnings
from typing import Callable, TypeVar

import polars as pl
from clifpy.utils.io import load_data

# Keys in a domain YAML that are NOT concept sections.
_CONFIG_META_KEYS = {"subject_id_col", "elf_version", "outlier_shaping"}

# Above this many cohort ids, skip DuckDB predicate pushdown (the SQL IN literal
# gets unwieldy) and fall back to full-load + in-memory filter.
_PUSHDOWN_MAX_IDS = int(os.environ.get("FLAIR_PUSHDOWN_MAX_IDS", "50000"))

_T = TypeVar("_T")

MEDS_SCHEMA = {
    "subject_id": pl.Utf8,
    "time": pl.Datetime,
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "text_value": pl.Utf8,
    "hospitalization_id": pl.Utf8,
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


def hosp_filters(hosp_ids: list[str] | None) -> dict | None:
    """clifpy ``filters=`` dict restricting a load to the cohort, or None when
    pushdown should be skipped (no cohort, or the id list is too large for a
    SQL IN literal)."""
    if hosp_ids is None or len(hosp_ids) > _PUSHDOWN_MAX_IDS:
        return None
    return {"hospitalization_id": list(hosp_ids)}


def load_clif_pandas(clif_cfg: dict, table_name: str, *,
                     filters: dict | None = None,
                     columns: list[str] | None = None):
    """Load a CLIF table via clifpy as pandas (UTC → site-local conversion happens here).

    ``filters``/``columns`` are pushed into clifpy's DuckDB scan so only the
    cohort's rows leave disk. Pushdown is an optimization only: callers keep
    their in-memory filters, and any pushdown failure falls back to the full load.
    """
    kwargs = dict(
        table_name=table_name,
        table_path=str(clif_cfg["data_directory"]),
        table_format_type=clif_cfg["filetype"],
        site_tz=clif_cfg.get("timezone"),
    )
    if filters is None and columns is None:
        return load_data(**kwargs)
    try:
        return load_data(**kwargs, filters=filters, columns=columns)
    except Exception as exc:  # e.g. DuckDB cast error on a type-mismatched id column
        warnings.warn(f"meds_etl: filter pushdown failed for '{table_name}' "
                      f"({exc}); falling back to full load")
        return load_data(**kwargs)


def load_clif_table(clif_cfg: dict, table_name: str, *,
                    hosp_ids: list[str] | None = None) -> pl.DataFrame:
    """Load a CLIF table → polars, tz-stripped, categories normalized."""
    pdf = load_clif_pandas(clif_cfg, table_name, filters=hosp_filters(hosp_ids))
    return normalize_categories(strip_tz(pl.from_pandas(pdf)))


def strip_tz(df: pl.DataFrame) -> pl.DataFrame:
    """Drop timezone info from every datetime column, keeping local wall-clock time.

    clifpy's loader has already converted UTC -> site-local once. Dropping the tz
    here (`replace_time_zone(None)`) preserves that wall-clock reading and
    guarantees no second conversion can ever happen downstream.
    """
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Datetime or (hasattr(dtype, "time_zone") and dtype.time_zone is not None):
            df = df.with_columns(
                pl.col(col_name).dt.replace_time_zone(None).alias(col_name)
            )
    return df


def normalize_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize every *_category column: trim, lowercase, spaces/hyphens -> underscores.

    Concept YAMLs key on the normalized form, so skipping this silently drops
    every row whose category differs only in case or punctuation.
    """
    cat_cols = [c for c in df.columns if c.endswith("_category") and df[c].dtype == pl.Utf8]
    if not cat_cols:
        return df
    return df.with_columns(
        pl.col(c)
          .str.strip_chars()
          .str.to_lowercase()
          .str.replace_all(" ", "_")
          .str.replace_all("-", "_")
        for c in cat_cols
    )


class TableCache:
    """Per-run memoizer for shared table loads (hospitalization, weight vitals).

    Thread-safe with per-key locks: two domains needing the same table share one
    load, while loads of different tables can proceed concurrently.

    Scope: one cache per ``build_meds_events`` call. Keys deliberately omit
    ``hosp_ids`` (fixed within a run) — never share an instance across runs
    with different cohorts.
    """

    def __init__(self) -> None:
        self._results: dict[tuple, object] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._meta = threading.Lock()

    def get_or_load(self, key: tuple, loader: Callable[[], _T]) -> _T:
        with self._meta:
            if key in self._results:
                return self._results[key]  # type: ignore[return-value]
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            with self._meta:
                if key in self._results:
                    return self._results[key]  # type: ignore[return-value]
            value = loader()
            with self._meta:
                self._results[key] = value
            return value


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
                         subject_id_col: str, hosp_ids: list[str] | None, *,
                         cache: TableCache | None = None) -> pl.DataFrame:
    """Left-join the hospitalization table for subject_id / any missing referenced columns."""
    needed = _needed_cols(concepts)
    missing = {c for c in needed if c not in df.columns}
    if subject_id_col not in df.columns:
        missing.add(subject_id_col)
    if not missing or "hospitalization_id" not in df.columns:
        return df

    def _load() -> pl.DataFrame:
        return restrict_to_hosp(
            load_clif_table(clif_cfg, "hospitalization", hosp_ids=hosp_ids), hosp_ids)

    hosp = cache.get_or_load(("hospitalization",), _load) if cache is not None else _load()
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
    """Build the ELF code string for one concept, vectorized over the frame.

    A `col(X)` part becomes that column's value ('UNK' when null or absent); a
    dict part `{col(X): {raw: mapped}}` applies the value map first, then
    'UNK'-fills. Everything else is a literal. Parts are joined with '//'.
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


def _outside(base: pl.Expr, lo: float, hi: float) -> pl.Expr:
    """`base` with out-of-range values replaced by null (never by the limit)."""
    return pl.when(base.is_between(lo, hi)).then(base).otherwise(None)


def _numeric_expr(nv_col: str, clamp_limits: dict, clamp_category_col: str | None,
                  available: set[str]) -> pl.Expr:
    """Float numeric value, with out-of-range values NULLED (by category or per-column).

    Out-of-range means "we do not believe this measurement", so it becomes
    missing — the same thing clifpy's own outlier handling does. It must not be
    winsorised to the limit: clipping turns an implausible value into a
    plausible-looking one, and a downstream model cannot tell the difference. A
    single wrong limit then rewrites a whole column (a `height_cm: [30, 96]` typo
    silently reports every adult as 96 cm) instead of just emptying it, which is
    at least visible.
    """
    base = pl.col(nv_col).cast(pl.Float64, strict=False)
    if not clamp_limits:
        return base
    if clamp_category_col and clamp_category_col in available:
        acc = base
        for key, (lo, hi) in clamp_limits.items():
            acc = (pl.when(pl.col(clamp_category_col) == key)
                   .then(_outside(base, lo, hi)).otherwise(acc))
        return acc
    if nv_col in clamp_limits:
        lo, hi = clamp_limits[nv_col]
        return _outside(base, lo, hi)
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

    # Utf8, never Int64: a numeric cast turns every alphanumeric site id into null
    # and the is_not_null filter below would then drop the entire table silently.
    subj = pl.col(subject_id_col).cast(pl.Utf8, strict=False)
    hosp = (pl.col("hospitalization_id").cast(pl.Utf8, strict=False)
            if "hospitalization_id" in available else pl.lit(None, dtype=pl.Utf8))

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
