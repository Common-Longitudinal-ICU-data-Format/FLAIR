"""Vendored CLIF -> MEDS ETL, consolidated for in-process use.

FLAIR is all-in-one: this subpackage converts CLIF tables into ELF-coded MEDS
events without requiring the standalone CLIF_MEDS project.

    from flair_benchmark.meds_etl import build_meds_events
    events, versions = build_meds_events(clif_cfg, elf_cfg, hosp_ids)

    pipeline.py   orchestration, per-domain runners, codes registry
    _emit.py      table loading, cleaning, and MEDS-row emission
    concepts/     one YAML per clinical domain — what to extract and its units

The concept YAMLs are kept per-domain on purpose: they hold clinical ranges and
category lists that a clinician reviews one domain at a time.
"""
from flair_benchmark.meds_etl._emit import MEDS_SCHEMA
from flair_benchmark.meds_etl.pipeline import (
    build_codes_table,
    build_meds_events,
    load_concept_config,
)

__all__ = ["build_meds_events", "build_codes_table", "load_concept_config", "MEDS_SCHEMA"]
