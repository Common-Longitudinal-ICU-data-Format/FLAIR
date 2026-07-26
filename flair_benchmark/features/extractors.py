"""Turn a task cohort into MEDS/ELF event tables. Two extractors, two weights.

  Part 1  get_meds_stream()   LIGHT — raw dose + raw unit, no conversion
  Part 2  build_meds_tables() FULL  — the vendored CLIF->MEDS ETL
          build_shared_meds()       — one shared MEDS store across tasks

Both start from a task cohort and end in ELF-shaped events, so they share this
file; the difference is how much work happens in between. `get_meds_stream`
deliberately skips the heavy ETL (unit conversion, vitals/weight join, dose
normalization) and charts the raw unit on every dose so a downstream step can
convert later. The full path runs the real ETL and writes MEDS-standard tables.

THE LEAKAGE RULE, which differs between them and is the thing to get right:

  - get_meds_stream BOUNDS events at the cohort's feature horizon
    (`time <= feature_cutoff_dttm`) inside this module.
  - build_meds_tables leaves the table UNBOUNDED on purpose. Its point-in-time
    filter runs downstream in the count featurizer, which keeps one MEDS table
    reusable across every prediction of an encounter instead of one per row.

The cutoff column is `feature_cutoff_dttm`, stamped by tasks/_finalize.py. It
defaults to `prediction_dttm` and is set EARLIER by any task whose prediction
timestamp would itself leak the outcome. Read the cutoff column, never
`prediction_dttm` directly — that is what keeps the two paths agreeing per task.

Depends on: cohort/clif.py, cohort/stitch.py, features/_config.py,
            features/_scope.py, meds_etl/.
Used by: cli.py `fe-meds`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from flair_benchmark.cohort.clif import (
    load_medication_admin_continuous,
    load_medication_admin_intermittent,
)
from flair_benchmark.cohort.stitch import JOIN_ID, attach_join_id
from flair_benchmark.features._config import attach_subject_id, empty_elf, finalize
from flair_benchmark.features._scope import HORIZON_COL, feature_scope
from flair_benchmark.meds_etl import build_codes_table, build_meds_events

# ===========================================================================
# Part 1 — Lightweight med stream (no conversion)
# ===========================================================================

_TABLES = {
    "medication_admin_continuous": ("MED_CON", load_medication_admin_continuous),
    "medication_admin_intermittent": ("MED_INT", load_medication_admin_intermittent),
}


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
        attach_join_id(df, idx)
        .join(horizon, on=JOIN_ID, how="inner")
        .filter(pl.col("time") <= pl.col(HORIZON_COL))
        .select(["hospitalization_id", "time", "code",
                 "numeric_value", "unit", "text_value", JOIN_ID])
    )
    return df


# ===========================================================================
# Part 2 — FE-meds: the cohort -> MEDS event-table junction
#
# Given a task cohort (one row per prediction_id with hospitalization_id and
# hospitalization_join_id) and an ELF concept config, run the vendored CLIF->MEDS
# ETL over the cohort's hospitalizations and emit two MEDS-standard tables:
#
#   out_dir/data/MEDS.parquet      one row per clinical event:
#       subject_id, time, code, numeric_value, text_value,
#       hospitalization_id, hospitalization_join_id   (extraction columns only)
#   out_dir/metadata/codes.parquet code registry (description, version, counts)
#
# The data table is NOT stamped with prediction_id / prediction_dttm. The
# prediction-row join (on hospitalization_join_id) and the point-in-time filter
# happen downstream, which is what keeps this table compact and reusable across
# every prediction of an encounter.
# ===========================================================================

DATA_COLS = [
    "subject_id", "time", "code", "numeric_value", "text_value",
    "hospitalization_id", JOIN_ID,
]


def load_elf_config(path: str) -> dict:
    """Load flair_elf_config.yaml (lists the event concepts to extract)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "domains" not in cfg:
        raise ValueError(f"ELF config {path} must define a 'domains:' mapping")
    return cfg


def build_meds_tables(task_df: pl.DataFrame, clif_cfg: dict, elf_config: str,
                      out_dir: str) -> pl.DataFrame:
    """Run FE-meds and write data/MEDS.parquet + metadata/codes.parquet.

    Returns the MEDS data-table frame (DATA_COLS).
    """
    elf_cfg = load_elf_config(elf_config)

    # require_horizon=False on purpose: unlike get_meds_stream, this table is left
    # UNBOUNDED — no time<=horizon cut here. The point-in-time filter runs
    # downstream in the count featurizer (see the module docstring). Do not add a
    # horizon filter here.
    scope = feature_scope(task_df, clif_cfg, require_horizon=False)

    events, versions = build_meds_events(clif_cfg, elf_cfg, scope.hospitalization_ids)

    # Attach the encounter-block key from the full stitch index so sibling
    # hospitalizations not represented by a prediction row still map correctly.
    events = attach_join_id(
        events.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
        scope.index,
    ).select(DATA_COLS)

    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)

    data_path = out / "data" / "MEDS.parquet"
    events.write_parquet(data_path)
    print(f"  FE-meds data table -> {data_path} ({events.height} events)")

    codes = build_codes_table(events, versions)
    codes_path = out / "metadata" / "codes.parquet"
    codes.write_parquet(codes_path)
    print(f"  FE-meds codes table -> {codes_path} ({codes.height} codes)")

    return events


def build_shared_meds(clif_cfg: dict, elf_cfg: dict, hosp_ids: list[str],
                      idx: pl.DataFrame, out_dir: str,
                      batch_size: int | None = None,
                      max_workers: int | None = None) -> dict[str, str]:
    """Build ONE shared MEDS store over `hosp_ids` -> out_dir/MEDS/part-*.parquet.

    This is the cross-task ETL: `hosp_ids` is the union of every task cohort's
    join-id membership (already expanded via `members_of_joins`), so the same
    CLIF->MEDS extraction runs once instead of once per task.

    The join key is attached from the STITCH INDEX (`attach_join_id`), not a
    cohort-local mapping, so a stitched sibling hospitalization's events keep the
    encounter's `hospitalization_join_id` even when that sibling is not itself a
    prediction row. Returns the per-domain ELF `versions` for the codes registry.

    `batch_size=None` runs a single extraction -> part-0000.parquet.
    `batch_size=N` (the hidden `-pmc` / poor-man's-compute mode) extracts the
    hosp_ids in sequential N-sized batches, writing one part file per batch and
    freeing each frame, so peak memory is bounded by one batch rather than the
    whole cohort — slower, but it fits a small box.
    """
    out = Path(out_dir)
    meds_dir = out / "MEDS"
    meds_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale part files from a previous (differently-batched) run.
    for old in meds_dir.glob("part-*.parquet"):
        old.unlink()

    hosp_ids = [str(h) for h in hosp_ids]
    batches = _batches(hosp_ids, batch_size) if batch_size else [hosp_ids]

    versions: dict[str, str] = {}
    total = 0
    for i, batch in enumerate(batches):
        events, vers = build_meds_events(clif_cfg, elf_cfg, batch,
                                         max_workers=max_workers)
        versions.update(vers)
        events = attach_join_id(events, idx).select(DATA_COLS)
        part_path = meds_dir / f"part-{i:04d}.parquet"
        events.write_parquet(part_path)
        total += events.height
        n_b = len(batches)
        print(f"  shared MEDS: batch {i + 1}/{n_b} "
              f"({len(batch)} hosp) -> {part_path.name} ({events.height} events)",
              flush=True)
        del events

    print(f"  shared MEDS total -> {meds_dir} "
          f"({total} events across {len(batches)} part file(s))", flush=True)
    return versions


def _batches(items: list, size: int) -> list[list]:
    """Split `items` into consecutive chunks of at most `size` (last may be smaller)."""
    return [items[i:i + size] for i in range(0, len(items), size)]
