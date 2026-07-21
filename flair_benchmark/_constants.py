"""Package version + genuinely shared, task-independent constants.

Pure data — imports nothing from the package, so it is safe to import from
``__init__``, the task modules, the report code, and the baseline without any
circular-import risk.

Per-task clinical policy (threshold, action, direction, report mode, dca_axis,
…) does NOT live here — each task owns it inline in its own ``META`` (see
``tasks/taskN_*.py``), value-validated in ``tasks/__init__.py`` and folded into
the audit hash (``_hash.py``). This file holds only what several tasks (or the
whole benchmark) share.

Edit THIS file to bump the PyPI version (``__version__``; root pyproject reads it
dynamically), move the global data cutoff, or change the shared selection
defaults.
"""

from __future__ import annotations

from datetime import date

__version__ = "0.0.1"

# Global data cutoff: rows whose split-temporal date falls after this are dropped from
# every cohort (see _split.assign_split). The current calendar year is not fully
# ETL'd at most sites, so querying it yields partial, non-comparable cohorts. Bump
# this one line when a new year is fully loaded. Skipped for site == "mimic", whose
# admission dates are de-identified/shifted (a real-date cap would empty the cohort).
DATA_CUTOFF = date(2025, 12, 31)

# Scoring units a task's META may declare in ``report_mode``. The per-task policy
# that picks one of these lives inline in each task's META (tasks/taskN_*.py);
# tasks/__init__.py validates the choice against this set.
REPORT_MODES = {"episodic", "continuous"}

# Defaults for the data-driven threshold-locator menu + operating-budget view
# (report/operating_points.py, report/_locators.py — design doc §3.2, §4.3). Each
# is overridable per task by adding the same key to that task's META;
# the values actually used are recorded in operating_points.json's metadata so a
# reader knows what was asked for. cost_ratio ships at 1 (cost-based then reduces
# toward the prevalence-weighted point) until clinicians state a real ratio.
SELECTION_DEFAULTS: dict = {
    "sens_target": 0.90,           # rule 6:  max specificity s.t. sensitivity >= this
    "spec_target": 0.90,           # rule 6b: max sensitivity s.t. specificity >= this
    "ppv_target": 0.30,            # rule 8:  smallest threshold clearing this PPV
    "alert_rate_target": 0.10,     # rule 9:  flag ~this fraction of the cohort
    "alert_rate_targets": (0.05, 0.10, 0.20),   # operating_budget: equal-workload grid
    "f_beta": 1,                   # rule 5:  F-beta (1 = F1)
    "cost_ratio": 1,               # rule 7:  C_FP / C_FN
}
