"""FE-meds — the cohort → MEDS event-table junction.

Given a task cohort (one row per ``prediction_id`` with ``hospitalization_id`` and
``hospitalization_join_id``) and an ELF concept config, run the vendored CLIF→MEDS
ETL over the cohort's hospitalizations and emit two MEDS-standard tables:

  out_dir/data/MEDS.parquet     — one row per clinical event:
      subject_id, time, code, numeric_value, text_value,
      hospitalization_id, hospitalization_join_id        (extraction columns only)
  out_dir/metadata/codes.parquet — code registry (description, version, counts).

The data table is **not** stamped with prediction_id / prediction_dttm: the
prediction-row join (on ``hospitalization_join_id``) and the ``time < prediction_dttm``
point-in-time filter happen in the downstream count featurizer, keeping this table
compact and reusable across every prediction of an encounter.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

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

    # Cohort hospitalization_id ↔ join_id mapping (canonical from the task itself).
    mapping = (
        task_df.select(
            pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col(JOIN_ID).cast(pl.Utf8),
        )
        .unique(subset=["hospitalization_id"])
    )
    hosp_ids = mapping["hospitalization_id"].to_list()

    events, versions = build_meds_events(clif_cfg, elf_cfg, hosp_ids)

    # Attach the encounter-block key; a hosp_id missing from the cohort maps to itself.
    events = (
        events.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .join(mapping, on="hospitalization_id", how="left")
        .with_columns(pl.col(JOIN_ID).fill_null(pl.col("hospitalization_id")))
        .select(DATA_COLS)
    )

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
