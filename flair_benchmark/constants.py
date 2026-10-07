"""Package version, shared constants, config template names, and the task audit hash.

Everything here is either a fixed value or a pure function over its arguments.
Nothing in this module imports from `flair_benchmark`, which is what makes it safe
to import from `__init__.py`, the task modules, and the report code without any
circular-import risk. Keep it that way.

Edit this file to bump the PyPI version (`__version__` — pyproject reads it via
[tool.hatch.version]) or move the global data cutoff.

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

# Rows whose split-temporal date falls after this are dropped from every cohort
# (see cohort/stitch.py:assign_split). The current calendar year is not fully
# ETL'd at most sites, so querying it yields partial, non-comparable cohorts.
# Bump this one line when a new year is fully loaded. Skipped for site == "mimic",
# whose admission dates are de-identified/shifted (a real-date cap would empty
# the cohort).
DATA_CUTOFF = date(2025, 12, 31)

# Scoring units a task's META may declare in `report_mode`. Each task picks one
# inline in its META; tasks/__init__.py validates the choice against this set.
REPORT_MODES = {"episodic", "continuous"}

# How the train/test deal is made (cohort/stitch.py:assign_split). Bump on ANY
# change to the deal — grain, ordering, ratio, regime priority.
#
#   1  encounter grain, ordered by pl.Expr.hash(seed=0)   (pre-2026-07)
#   2  patient grain, ordered by sha256                   (2026-07 .. 2026-08)
#   3  encounter grain, ordered by sha256                 (current)
#
# Folded into task_script_sha256 below because stitch.py is SHARED code and sits
# outside the task source the digest otherwise hashes: without this, changing the
# splitter silently re-deals every cohort while every report, table1 and model
# keeps its old digest and no artifact anywhere flags itself stale. A digest that
# moves is the signal to re-score — a model evaluated across a version boundary is
# being tested on stays it trained on.
SPLIT_VERSION = 3

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
    """SHA-256 of the task source, its resolved clinical policy, and SPLIT_VERSION.

    Stamped into every report JSON so the central aggregator can prove two sites
    ran the same task — the label logic (the task .py source), the policy it was
    evaluated under (threshold, direction, report mode, inline in the task's META),
    and the train/test deal the numbers were produced against. A change to any of
    the three changes the digest.

    SPLIT_VERSION is in here because it is the one input that lives in SHARED code
    (cohort/stitch.py), invisible to a hash over the task file alone — see the
    constant.
    """
    src = Path(task_module.__file__).read_bytes()
    meta = getattr(task_module, "META", {}) or {}
    policy = json.dumps(
        {k: meta.get(k) for k in _AUDIT_KEYS}, sort_keys=True
    ).encode()
    split = f"split_version={SPLIT_VERSION}".encode()
    return hashlib.sha256(src + b"\x00" + policy + b"\x00" + split).hexdigest()
