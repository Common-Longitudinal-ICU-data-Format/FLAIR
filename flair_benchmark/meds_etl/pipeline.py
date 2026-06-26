"""Build a MEDS events frame from CLIF, driven by an ELF concept config.

Public entry: ``build_meds_events(clif_cfg, elf_cfg, hosp_ids)`` returns
``(events_df, domain_versions)`` where events_df has the MEDS_SCHEMA columns.
The bundled per-domain concept YAMLs live in ``concepts/``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from flair_benchmark.meds_etl._emit import MEDS_SCHEMA
from flair_benchmark.meds_etl.domains import run_domain

_CONCEPTS_DIR = Path(__file__).parent / "concepts"


def load_concept_config(domain: str, concepts_dir: Path | None = None) -> dict:
    """Load the bundled concept YAML for a domain (e.g. VITAL → concepts/VITAL.yaml)."""
    path = (concepts_dir or _CONCEPTS_DIR) / f"{domain}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_meds_events(clif_cfg: dict, elf_cfg: dict,
                      hosp_ids: list[str] | None) -> tuple[pl.DataFrame, dict[str, str]]:
    """Run the ETL for every domain listed in elf_cfg['domains'].

    elf_cfg schema:
        domains:
          VITAL: {}
          PA:    {category_col: assessment_category, categories: [gcs_total, rass]}
          ...
    hosp_ids restricts every table to the cohort (matched as Utf8). Pass None for all.
    """
    concepts_dir = Path(elf_cfg["concepts_dir"]) if elf_cfg.get("concepts_dir") else None
    frames: list[pl.DataFrame] = []
    versions: dict[str, str] = {}

    for domain, opts in elf_cfg["domains"].items():
        opts = opts or {}
        domain_cfg = load_concept_config(domain, concepts_dir)
        versions[domain] = domain_cfg.get("elf_version", "0.0.0")
        print(f"  meds_etl: {domain} …")
        frame = run_domain(domain, clif_cfg, domain_cfg, hosp_ids, opts)
        print(f"    {domain}: {frame.height} events")
        frames.append(frame)

    events = pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=MEDS_SCHEMA)
    return events, versions
