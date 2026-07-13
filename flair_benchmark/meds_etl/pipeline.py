"""Build a MEDS events frame from CLIF, driven by an ELF concept config.

Public entry: ``build_meds_events(clif_cfg, elf_cfg, hosp_ids)`` returns
``(events_df, domain_versions)`` where events_df has the MEDS_SCHEMA columns.
The bundled per-domain concept YAMLs live in ``concepts/``.

Domains are extracted concurrently (they are independent single-table pulls);
``max_workers`` — argument, ``elf_cfg['max_workers']``, or env
``FLAIR_MEDS_WORKERS`` — caps the thread pool, and ``max_workers=1`` gives the
strictly serial order. The concatenated output is always in ELF-config domain
order regardless of worker count.
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

from flair_benchmark.meds_etl._emit import MEDS_SCHEMA, TableCache
from flair_benchmark.meds_etl.domains import run_domain

_CONCEPTS_DIR = Path(__file__).parent / "concepts"


@contextlib.contextmanager
def _ticker(status, heartbeat: float = 15.0):
    """Liveness for long steps that have no inner loop to hook.

    ``status`` is a zero-arg callable returning the current label (so one ticker
    can narrate several concurrently running domains):
      - interactive terminal: a smooth spinner + elapsed seconds, redrawn in place.
      - piped / log run (e.g. ``| tee``): a plain heartbeat line every ``heartbeat``
        seconds, so external sites tailing the log still see motion.
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


def load_concept_config(domain: str, concepts_dir: Path | None = None) -> dict:
    """Load the bundled concept YAML for a domain (e.g. VITAL → concepts/VITAL.yaml)."""
    path = (concepts_dir or _CONCEPTS_DIR) / f"{domain}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_workers(max_workers: int | None, elf_cfg: dict, n_domains: int) -> int:
    if max_workers is None:
        max_workers = elf_cfg.get("max_workers")
    if max_workers is None:
        env = os.environ.get("FLAIR_MEDS_WORKERS")
        max_workers = int(env) if env else None
    if max_workers is None:
        max_workers = min(4, n_domains)
    return max(1, int(max_workers))


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
