"""Vendored CLIF → MEDS ETL (from CLIF_MEDS), consolidated for in-process use.

FLAIR is all-in-one: this subpackage converts CLIF tables to ELF-coded MEDS events
without requiring the standalone CLIF_MEDS project. Primary entry point:

    from flair_benchmark.meds_etl import build_meds_events
    events, versions = build_meds_events(clif_cfg, elf_cfg, hosp_ids)
"""
from flair_benchmark.meds_etl._emit import MEDS_SCHEMA
from flair_benchmark.meds_etl.metadata import build_codes_table
from flair_benchmark.meds_etl.pipeline import build_meds_events, load_concept_config

__all__ = ["build_meds_events", "build_codes_table", "load_concept_config", "MEDS_SCHEMA"]
