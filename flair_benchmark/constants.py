"""Package version, shared constants, config template names, and the task audit hash.

Everything here is either a fixed value or a pure function over its arguments.
Nothing in this module imports from `flair_benchmark`, which is what makes it safe
to import from `__init__.py`, the task modules, and the report code without any
circular-import risk. Keep it that way.

Edit this file to bump the PyPI version (`__version__` — pyproject reads it via
[tool.hatch.version]), move the benchmark window, or change the stitch gap. The
whole file is folded into `task_script_sha256`, so ANY edit here moves every
task digest.

Per-task clinical policy (threshold, direction, report mode, ...) does NOT live
here. Each task owns it inline in its own META dict; tasks/__init__.py validates
it, and `task_script_sha256` below folds it into the audit digest.

Depends on: nothing (stdlib only).
Used by: __init__.py, cli.py, tasks/, report/, cohort/.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import ModuleType

__version__ = "0.0.1"


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Benchmark window (inclusive local calendar dates). A stitched encounter enters a
# cohort only if EVERY member hospitalization's admission_dttm and discharge_dttm
# fall inside it — no encounter with one leg outside (see
# cohort/clif.py:base_icu_cohort). The current calendar year is not fully ETL'd
# at most sites, so querying it yields partial, non-comparable cohorts.
# Bump END when a new year is fully loaded. Skipped for site == "mimic",
# whose dates are de-identified/shifted (a real-date window would empty the
# cohort).
BENCHMARK_WINDOW_START = date(2017, 12, 31)
BENCHMARK_WINDOW_END = date(2025, 12, 31)

# Gap (hours) under which back-to-back hospitalizations are stitched into one
# encounter (cohort/stitch.py). Fixed, not a site config: a different gap builds
# different encounters, so cohorts and splits would stop matching across sites.
STITCH_TIME_INTERVAL_HOURS = 6

# Scoring units a task's META may declare in `report_mode`. Each task picks one
# inline in its META; tasks/__init__.py validates the choice against this set.
REPORT_MODES = {"episodic", "continuous"}

# ---------------------------------------------------------------------------
# Config templates  (written out by `flair init-config`)
# ---------------------------------------------------------------------------
# Shipped as package data under flair_benchmark/templates/ (hatchling includes
# every non-ignored file in the package), so a fresh `pip install flair-benchmark`
# user can scaffold a working config entirely through the CLI. Those files are the
# single source of truth.

TEMPLATE_FILES = ("clif_config.json", "feature_config.yaml")


# ---------------------------------------------------------------------------
# Task audit digest
# ---------------------------------------------------------------------------

# The resolved META keys that define "the same task" for cross-site comparison.
# A key absent from a task's META serializes as null (e.g. landmark_horizon on
# an episodic task). Extending this tuple changes every task's digest, so
# cross-site comparisons must be regenerated on the same version.
_AUDIT_KEYS = (
    "name", "label_column", "task_type", "temporal_pattern",
    "prediction_unit", "clinical_threshold", "clinical_direction", "report_mode",
    "landmark_horizon",
)


def task_script_sha256(task_module: ModuleType) -> str:
    """SHA-256 of the task source, its resolved clinical policy, and this constants file.

    Stamped into every report JSON so the central aggregator can prove two sites
    ran the same task — the label logic (the task .py source), the policy it was
    evaluated under (threshold, direction, report mode, inline in the task's META),
    and the shared constants every cohort is cut with (benchmark window, stitch
    gap, ...). The whole of constants.py is hashed rather than a hand-picked list,
    so a new constant can't be forgotten; the cost is that any edit to this file,
    `__version__` bumps and comments included, changes every digest. The train/test split is fixed (always per stitched
    encounter, cohort/stitch.py:assign_split), so it carries no version here.
    """
    # Line endings normalised: a Windows checkout (git autocrlf) is the same code.
    src = Path(task_module.__file__).read_bytes().replace(b"\r\n", b"\n")
    meta = getattr(task_module, "META", {}) or {}
    policy = json.dumps(
        {k: meta.get(k) for k in _AUDIT_KEYS}, sort_keys=True
    ).encode()
    shared = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(src + b"\x00" + policy + b"\x00" + shared).hexdigest()


def dependency_versions() -> dict:
    """Installed clifpy / polars versions, stamped into every audit block.

    Both change results (clifpy: loaders, SOFA, comorbidity; polars: sort, cast,
    hash), so two sites' numbers are only comparable on the same versions — the
    `[benchmark]` extra pins them for leaderboard runs. Read from package metadata
    so clifpy is never imported here; None when a package is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("clifpy", "polars"):
        try:
            out[f"{pkg}_version"] = version(pkg)
        except PackageNotFoundError:
            out[f"{pkg}_version"] = None
    return out
