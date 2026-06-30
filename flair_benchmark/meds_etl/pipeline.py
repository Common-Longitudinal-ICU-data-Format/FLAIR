"""Build a MEDS events frame from CLIF, driven by an ELF concept config.

Public entry: ``build_meds_events(clif_cfg, elf_cfg, hosp_ids)`` returns
``(events_df, domain_versions)`` where events_df has the MEDS_SCHEMA columns.
The bundled per-domain concept YAMLs live in ``concepts/``.
"""
from __future__ import annotations

import contextlib
import itertools
import sys
import threading
import time
from pathlib import Path

import polars as pl
import yaml

from flair_benchmark.meds_etl._emit import MEDS_SCHEMA
from flair_benchmark.meds_etl.domains import run_domain

_CONCEPTS_DIR = Path(__file__).parent / "concepts"


@contextlib.contextmanager
def _ticker(label: str, heartbeat: float = 15.0):
    """Liveness for a long step that has no inner loop to hook.

    Each domain's extraction is a single polars collect, so a background thread
    shows it isn't frozen:
      - interactive terminal: a smooth spinner + elapsed seconds, redrawn in place.
      - piped / log run (e.g. ``| tee``): a plain heartbeat line every ``heartbeat``
        seconds, so external sites tailing the log still see motion.
    """
    is_tty = sys.stderr.isatty()
    stop = threading.Event()
    t0 = time.time()

    def spin():
        if is_tty:
            for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if stop.is_set():
                    break
                sys.stderr.write(f"\r  meds_etl: {label} {ch} {int(time.time() - t0)}s ")
                sys.stderr.flush()
                time.sleep(0.2)
        else:
            # heartbeat lines for non-tty (logs) — coarse, but visible.
            while not stop.wait(heartbeat):
                print(f"    {label} … still running ({int(time.time() - t0)}s)",
                      flush=True)

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1)
        if is_tty:
            sys.stderr.write("\r" + " " * 48 + "\r")  # clear the spinner line
            sys.stderr.flush()


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

    domains = list(elf_cfg["domains"].items())
    for i, (domain, opts) in enumerate(domains, 1):
        opts = opts or {}
        domain_cfg = load_concept_config(domain, concepts_dir)
        versions[domain] = domain_cfg.get("elf_version", "0.0.0")
        print(f"  meds_etl: {domain} ({i}/{len(domains)}) …", flush=True)
        with _ticker(f"{domain} ({i}/{len(domains)})"):
            frame = run_domain(domain, clif_cfg, domain_cfg, hosp_ids, opts)
        print(f"    {domain}: {frame.height} events", flush=True)
        frames.append(frame)

    events = pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=MEDS_SCHEMA)
    return events, versions
