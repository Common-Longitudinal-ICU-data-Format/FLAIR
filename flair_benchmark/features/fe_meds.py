"""FE-meds — the cohort → MEDS event-table junction.

Given a task cohort (one row per ``prediction_id`` with ``hospitalization_id`` and
``hospitalization_join_id``) and an ELF concept config, run the vendored CLIF→MEDS
ETL over the cohort's hospitalizations and emit two MEDS-standard tables:

  out_dir/data/MEDS.parquet     — one row per clinical event:
      subject_id, time, code, numeric_value, text_value,
      hospitalization_id, hospitalization_join_id        (extraction columns only)
  out_dir/metadata/codes.parquet — code registry (description, version, counts).

The data table is **not** stamped with prediction_id / prediction_dttm: the
prediction-row join (on ``hospitalization_join_id``) and the point-in-time filter
happen in the downstream count featurizer, keeping this table compact and reusable
across every prediction of an encounter.

That downstream filter is ``time <= feature_cutoff_dttm`` — the cohort-contract
column stamped by ``tasks/_finalize.py``, which defaults to ``prediction_dttm`` and
is set EARLIER by a task whose prediction timestamp would itself leak the outcome.
Reading the cutoff column rather than ``prediction_dttm`` is what keeps this path
agreeing with ``wide.py`` / ``meds_stream.py`` per task.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from flair_benchmark._stitch import attach_join_id
from flair_benchmark.features._scope import feature_scope
from flair_benchmark.meds_etl import build_codes_table, build_meds_events

JOIN_ID = "hospitalization_join_id"

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

    # require_horizon=False on purpose: unlike wide/meds_stream, this table is
    # left UNBOUNDED — no time<=horizon cut here. The point-in-time filter runs
    # downstream in the count featurizer (see module docstring). Do not add a
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


def _batches(items: list, size: int) -> list[list]:
    """Split ``items`` into consecutive chunks of at most ``size`` (last may be smaller)."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_shared_meds(clif_cfg: dict, elf_cfg: dict, hosp_ids: list[str],
                      idx: pl.DataFrame, out_dir: str,
                      batch_size: int | None = None,
                      max_workers: int | None = None) -> dict[str, str]:
    """Build ONE shared MEDS store over ``hosp_ids`` → ``out_dir/MEDS/part-*.parquet``.

    This is the cross-task ETL: ``hosp_ids`` is the union of every task cohort's
    join-id membership (already expanded via ``members_of_joins``), so the same
    CLIF→MEDS extraction runs once instead of once per task.

    The join key is attached from the **stitch index** (``attach_join_id``), not a
    cohort-local mapping, so a stitched sibling hospitalization's events keep the
    encounter's ``hospitalization_join_id`` even when that sibling is not itself a
    prediction row. Returns the per-domain ELF ``versions`` for the codes registry.

    ``batch_size=None`` runs a single extraction → ``part-0000.parquet``.
    ``batch_size=N`` (the hidden ``-pmc`` / poor-man's-compute mode) extracts the
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
