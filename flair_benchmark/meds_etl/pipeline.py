"""Run the CLIF -> MEDS ETL: config in, events frame + codes registry out.

The whole pipeline, in the order it executes:

  Part 1  build_meds_events()   orchestration — read the concept YAMLs, run every
                                domain (concurrently), concatenate
  Part 2  run_domain()          the per-domain runners
  Part 3  build_codes_table()   the codes registry derived from the events

Public entry: `build_meds_events(clif_cfg, elf_cfg, hosp_ids)` returns
`(events_df, domain_versions)` where events_df has the MEDS_SCHEMA columns. The
bundled per-domain concept YAMLs live in `concepts/`.

CONCURRENCY. Domains are independent single-table pulls, so they are extracted in
parallel. `max_workers` comes from the argument, `elf_cfg['max_workers']`, or the
env var FLAIR_MEDS_WORKERS; `max_workers=1` gives strictly serial execution. The
concatenated output is ALWAYS in ELF-config domain order regardless of worker
count — results are written into a pre-sized list by index, never appended.

Two things are NOT thread-safe and are locked accordingly: clifpy's dose
converter (it runs SQL on duckdb's module-level default connection), and the
shared table cache.

Depends on: meds_etl/_emit.py, clifpy, polars, pyyaml.
Used by: features/extractors.py.
"""
from __future__ import annotations

import contextlib
import itertools
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import polars as pl
import yaml

from flair_benchmark.meds_etl._emit import (
    MEDS_SCHEMA,
    TableCache,
    build_clamp,
    concept_section,
    emit_events,
    hosp_filters,
    join_hospitalization,
    load_clif_pandas,
    load_clif_table,
    normalize_categories,
    restrict_to_hosp,
    strip_tz,
    subject_col,
)

_CONCEPTS_DIR = Path(__file__).parent / "concepts"


# ===========================================================================
# Part 1 — Orchestration
# ===========================================================================


def load_concept_config(domain: str, concepts_dir: Path | None = None) -> dict:
    """Load the bundled concept YAML for a domain (e.g. VITAL -> concepts/VITAL.yaml)."""
    path = (concepts_dir or _CONCEPTS_DIR) / f"{domain}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_meds_events(clif_cfg: dict, elf_cfg: dict,
                      hosp_ids: list[str] | None, *,
                      max_workers: int | None = None) -> tuple[pl.DataFrame, dict[str, str]]:
    """Run the ETL for every domain listed in elf_cfg['domains'].

    elf_cfg schema:
        domains:
          VITAL: {}
          PA:    {category_col: assessment_category, categories: [gcs_total, RASS]}
          ...
    hosp_ids restricts every table to the cohort (matched as Utf8). Pass None for all.
    """
    concepts_dir = Path(elf_cfg["concepts_dir"]) if elf_cfg.get("concepts_dir") else None
    versions: dict[str, str] = {}

    domains: list[tuple[str, dict, dict]] = []
    for domain, opts in elf_cfg["domains"].items():
        domain_cfg = load_concept_config(domain, concepts_dir)
        versions[domain] = domain_cfg.get("elf_version", "0.0.0")
        domains.append((domain, opts or {}, domain_cfg))

    n = len(domains)
    workers = _resolve_workers(max_workers, elf_cfg, max(n, 1))
    cache = TableCache()  # hospitalization / weight-vitals loaded once per run
    frames: list[pl.DataFrame | None] = [None] * n

    state_lock = threading.Lock()
    running: dict[str, None] = {}   # insertion-ordered set of in-flight domains
    done = 0

    def status() -> str:
        with state_lock:
            names = ", ".join(running) or "-"
            return f"{done}/{n} done | running: {names}"

    def run_one(i: int) -> None:
        nonlocal done
        domain, opts, domain_cfg = domains[i]
        with state_lock:
            running[domain] = None
        try:
            frame = run_domain(domain, clif_cfg, domain_cfg, hosp_ids, opts, cache=cache)
        finally:
            with state_lock:
                running.pop(domain, None)
        with state_lock:
            done += 1
            k = done
        # Write by INDEX, not append — this is what keeps the concatenated output
        # in ELF-config domain order however the threads happen to finish.
        frames[i] = frame
        print(f"    {domain} ({k}/{n}): {frame.height} events", flush=True)

    if n:
        print(f"  meds_etl: {n} domain(s), {workers} worker(s) …", flush=True)
        with _ticker(status):
            if workers == 1:
                for i in range(n):
                    run_one(i)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(run_one, i) for i in range(n)]
                    pending = set(futures)
                    while pending:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in finished:
                            f.result()  # propagate the first failure, as in the serial path

    events = (pl.concat([f for f in frames if f is not None], how="vertical")
              if n else pl.DataFrame(schema=MEDS_SCHEMA))
    return events, versions


def _resolve_workers(max_workers: int | None, elf_cfg: dict, n_domains: int) -> int:
    if max_workers is None:
        max_workers = elf_cfg.get("max_workers")
    if max_workers is None:
        env = os.environ.get("FLAIR_MEDS_WORKERS")
        max_workers = int(env) if env else None
    if max_workers is None:
        max_workers = min(4, n_domains)
    return max(1, int(max_workers))


@contextlib.contextmanager
def _ticker(status, heartbeat: float = 15.0):
    """Show liveness during long steps that have no inner loop to hook.

    `status` is a zero-arg callable returning the current label, so one ticker can
    narrate several concurrently running domains:
      - interactive terminal: a smooth spinner + elapsed seconds, redrawn in place
      - piped / log run (e.g. `| tee`): a plain heartbeat line every `heartbeat`
        seconds, so external sites tailing the log still see motion
    """
    is_tty = sys.stderr.isatty()
    stop = threading.Event()
    t0 = time.time()

    def spin():
        if is_tty:
            width = 0
            for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if stop.is_set():
                    break
                line = f"\r  meds_etl: {status()} {ch} {int(time.time() - t0)}s "
                width = max(width, len(line))
                sys.stderr.write(line.ljust(width))
                sys.stderr.flush()
                time.sleep(0.2)
            sys.stderr.write("\r" + " " * width + "\r")
            sys.stderr.flush()
        else:
            # heartbeat lines for non-tty (logs) — coarse, but visible.
            while not stop.wait(heartbeat):
                print(f"    {status()} … still running ({int(time.time() - t0)}s)",
                      flush=True)

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1)


# ===========================================================================
# Part 2 — Per-domain runners
#
# Two paths, dispatched by domain name:
#   run_generic : VITAL, LAB, PA, RESP, CRRT, HOSP_DX, ...
#                 load -> optional clamp/drop -> emit
#   run_med     : MED_CON, MED_INT
#                 same, but with clifpy dose-unit conversion before emit
# ===========================================================================

# Domains whose events are dropped when both numeric_value and text_value are null.
_SKIP_NULL = {"RESP", "CRRT", "ECMO_MCS"}

# Columns dropped per domain before emission (parity with CLIF_MEDS processors).
_DROP_COLS = {
    "RESP": ("vent_brand_name",),
    "CRRT": ("device_id", "dialysis_machine_name"),
}

# MED_* use the unit-conversion path; everything else is generic.
_MED_DOMAINS = {"MED_CON", "MED_INT"}


def run_domain(domain: str, clif_cfg: dict, domain_cfg: dict,
               hosp_ids: list[str] | None, opts: dict,
               cache: TableCache | None = None) -> pl.DataFrame:
    if domain in _MED_DOMAINS:
        return run_med(domain, clif_cfg, domain_cfg, hosp_ids, opts, cache=cache)
    return run_generic(domain, clif_cfg, domain_cfg, hosp_ids, opts, cache=cache)


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
    if not med_pdf.empty and uc.get("enabled", False) and preferred:
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


def _load_weight_vitals(clif_cfg: dict, hosp_ids: list[str] | None,
                        cache: TableCache | None):
    filters = hosp_filters(hosp_ids) or {}
    filters["vital_category"] = ["weight_kg"]

    def _load():
        return load_clif_pandas(clif_cfg, "vitals",
                                filters=filters, columns=_WEIGHT_VITALS_COLS)

    return cache.get_or_load(("vitals", "weight_kg"), _load) if cache is not None else _load()


def _apply_category_filter(df: pl.DataFrame, opts: dict) -> pl.DataFrame:
    """Filter rows to opts['categories'] on opts['category_col'] (values lowercased)."""
    cats = opts.get("categories")
    col = opts.get("category_col")
    if not cats or not col or col not in df.columns:
        return df
    wanted = [str(c).lower() for c in cats]
    return df.filter(pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(wanted))


# ===========================================================================
# Part 3 — Codes registry (metadata/codes.parquet)
#
# One row per distinct ELF code seen in the events, with a generated description
# and the concept version of the domain it came from.
# ===========================================================================

DOMAIN_LABELS = {
    "VITAL": "Vital sign", "LAB": "Laboratory", "MED_CON": "Continuous medication",
    "MED_INT": "Intermittent medication", "RESP": "Respiratory support",
    "PA": "Patient assessment", "CODE_STATUS": "Code status",
    "HOSP": "Hospitalization", "PATIENT": "Patient", "ADT": "ADT",
    "POS": "Position", "CRRT": "CRRT", "ECMO_MCS": "ECMO/MCS",
    "PROC": "Procedure", "HOSP_DX": "Hospital diagnosis",
}

CODES_SCHEMA = {
    "code": pl.Utf8,
    "description": pl.Utf8,
    "parent_codes": pl.List(pl.Utf8),
    "concept_version": pl.Utf8,
    "event_count": pl.Int64,
    "is_numeric_value": pl.Int8,
    "is_text_value": pl.Int8,
}


def build_codes_table(events: pl.DataFrame, domain_versions: dict[str, str]) -> pl.DataFrame:
    """Build the codes registry from an in-memory MEDS events frame.

    Event counts below 10 are floored to 10 for PHI protection, and the
    numeric/text presence flags are derived per code.
    """
    if events.height == 0:
        return pl.DataFrame(schema=CODES_SCHEMA)

    code_stats = (
        events.lazy()
        .select("code", "numeric_value", "text_value")
        .filter(pl.col("code").is_not_null() & (pl.col("code") != ""))
        .group_by("code")
        .agg(
            pl.len().alias("len"),
            pl.col("numeric_value").is_not_null().any().alias("has_numeric"),
            pl.col("text_value").is_not_null().any().alias("has_text"),
        )
        .collect()
        .with_columns(
            pl.when(pl.col("len") < 10).then(10).otherwise(pl.col("len")).cast(pl.Int64).alias("event_count"),
            pl.col("has_numeric").cast(pl.Int8).alias("is_numeric_value"),
            pl.col("has_text").cast(pl.Int8).alias("is_text_value"),
        )
        .drop("len", "has_numeric", "has_text")
        .sort("code")
    )

    rows = []
    for s in code_stats.iter_rows(named=True):
        domain = code_to_domain(s["code"])
        rows.append({
            "code": s["code"],
            "description": generate_description(s["code"]),
            "parent_codes": None,
            "concept_version": domain_versions.get(domain, "0.0.0") if domain else "0.0.0",
            "event_count": s["event_count"],
            "is_numeric_value": s["is_numeric_value"],
            "is_text_value": s["is_text_value"],
        })
    return pl.DataFrame(rows, schema=CODES_SCHEMA)


def generate_description(code: str) -> str:
    """Auto-generate a human-readable description from an ELF code string."""
    if code == "MEDS_BIRTH":
        return "Patient birth event"
    if code == "MEDS_DEATH":
        return "Patient death event"

    parts = code.split("//")
    domain = parts[0]
    label = DOMAIN_LABELS.get(domain, domain)
    concept = parts[1].replace("_", " ") if len(parts) > 1 else ""
    qualifiers = [p for p in parts[2:] if p not in ("NA", "UNK")]

    desc = f"{label}: {concept}"
    if qualifiers:
        desc += f" ({', '.join(qualifiers)})"
    return desc


def code_to_domain(code: str):
    """Extract the domain prefix from an ELF code, for version lookup."""
    if code.startswith("MEDS_"):
        return "PATIENT"
    parts = code.split("//")
    return parts[0] if parts[0] in DOMAIN_LABELS else None


def write_codes_parquet(output_dir: Path, domain_versions: dict[str, str]):
    """Scan output parquet files under data/ and write metadata/codes.parquet.

    NOTE: currently unused. The FE-meds path writes a single MEDS.parquet and
    calls `build_codes_table` on the in-memory frame instead. Kept for the
    per-domain-file layout the vendored CLIF_MEDS ETL also supports.
    """
    data_dir = output_dir / "data"
    frames = [pl.read_parquet(p) for p in data_dir.glob("*.parquet")]
    events = pl.concat(frames, how="diagonal") if frames else pl.DataFrame(
        schema={"code": pl.Utf8, "numeric_value": pl.Float32, "text_value": pl.Utf8})
    codes_df = build_codes_table(events, domain_versions)
    out_path = output_dir / "metadata" / "codes.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codes_df.write_parquet(out_path)
    print(f"  codes.parquet -> {out_path} ({len(codes_df)} codes)")
    return codes_df
