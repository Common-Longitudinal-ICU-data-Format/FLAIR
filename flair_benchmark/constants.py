"""Package version, shared constants, config templates, and the task audit hash.

Everything here is either a fixed value or a pure function over its arguments.
Nothing in this module imports from `flair_benchmark`, which is what makes it safe
to import from `__init__.py`, the task modules, and the report code without any
circular-import risk. Keep it that way.

Edit this file to bump the PyPI version (`__version__` — pyproject reads it via
[tool.hatch.version]), move the global data cutoff, or change the shared
selection defaults.

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

# Defaults for the data-driven threshold-locator menu + operating-budget view
# (report/operating_points.py, report/_locators.py — design doc §3.2, §4.3). Any
# task may override one by adding the same key to its META; the values actually
# used are recorded in operating_points.json's metadata so a reader knows what
# was asked for. cost_ratio ships at 1 (which reduces the cost-based rule toward
# the prevalence-weighted point) until clinicians state a real ratio.
SELECTION_DEFAULTS: dict = {
    "sens_target": 0.90,           # rule 6:  max specificity s.t. sensitivity >= this
    "spec_target": 0.90,           # rule 6b: max sensitivity s.t. specificity >= this
    "ppv_target": 0.30,            # rule 8:  smallest threshold clearing this PPV
    "alert_rate_target": 0.10,     # rule 9:  flag ~this fraction of the cohort
    "alert_rate_targets": (0.05, 0.10, 0.20),   # operating_budget: equal-workload grid
    "f_beta": 1,                   # rule 5:  F-beta (1 = F1)
    "cost_ratio": 1,               # rule 7:  C_FP / C_FN
}


# ---------------------------------------------------------------------------
# Config templates  (written out by `flair init-config`)
# ---------------------------------------------------------------------------
# Stored as string constants rather than data files so they are guaranteed to be
# present in the PyPI wheel — a fresh `pip install flair-benchmark` user can
# scaffold a working config entirely through the CLI. These strings are the
# single source of truth; keep FEATURE_CONFIG_TEMPLATE in sync with
# feature_config.yaml at the repo root.

CLIF_CONFIG_TEMPLATE = """\
{
    "site": "sitename",
    "data_directory": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central",
    "stitch_time_interval_hours": 6,
    "cache_directory": "./output"
}
"""

FEATURE_CONFIG_TEMPLATE = """\
# FLAIR feature-engineering config (drives `flair fe-meds`).
# Categories are CLIF 2.1 mCIDE values; unknown ones warn but don't fail.
#
# For wide time-series data (vitals, labs, assessments, respiratory support) use
# clifpy's ClifOrchestrator.create_wide_dataset directly — and filter its output
# to `time <= feature_cutoff_dttm` yourself; clifpy applies no leakage guard.
clif_config: clif_config.json              # the JSON clif config (site/data_dir/tz)
output_directory: ./output/features        # convention; --out overrides

# Lightweight med event stream: raw dose + raw unit, NO unit-conversion ETL.
meds_stream:
  medication_admin_continuous:
    categories: [norepinephrine, propofol]
  medication_admin_intermittent:
    categories: [vancomycin, cefepime, piperacillin_tazobactam]
"""

# Filename -> template, written by `flair init-config`.
TEMPLATES = {
    "clif_config.json": CLIF_CONFIG_TEMPLATE,
    "feature_config.yaml": FEATURE_CONFIG_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Task audit digest
# ---------------------------------------------------------------------------

# The resolved META keys that define "the same task" for cross-site comparison.
# A key absent from a task's META serializes as null (e.g. landmark_horizon or
# intervention on an episodic task). Extending this tuple changes every task's
# digest, so cross-site comparisons must be regenerated on the same version.
_AUDIT_KEYS = (
    "name", "label_column", "task_type", "temporal_pattern",
    "prediction_unit", "clinical_threshold", "clinical_direction", "report_mode",
    "landmark_horizon", "intervention",
)


def task_script_sha256(task_module: ModuleType) -> str:
    """SHA-256 of the task source plus its resolved clinical policy.

    Stamped into every report JSON so the central aggregator can prove two sites
    ran the same task — both the label logic (the task .py source) and the policy
    it was evaluated under (threshold, direction, report mode), which lives inline
    in the task's META. Hashing both halves means a change to either one changes
    the digest.
    """
    src = Path(task_module.__file__).read_bytes()
    meta = getattr(task_module, "META", {}) or {}
    policy = json.dumps(
        {k: meta.get(k) for k in _AUDIT_KEYS}, sort_keys=True
    ).encode()
    return hashlib.sha256(src + b"\x00" + policy).hexdigest()
