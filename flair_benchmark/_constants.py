"""Single source of truth: package version + per-task clinical policy.

Pure data — imports nothing from the package, so it is safe to import from
``__init__``, the task modules, the report code, and the baseline without any
circular-import risk.

Edit THIS file to:
  - bump the PyPI version (``__version__``; root pyproject reads it dynamically), or
  - change a task's operating threshold, clinical direction, or recommended
    report mode (``TASK_POLICY``).

Everything else derives from here. The values in ``TASK_POLICY`` are spread into
each task's ``META`` at import time (see ``tasks/taskN_*.py``) and folded into the
audit hash (see ``_hash.py``), so a change here is captured by the cross-site
``task_script_sha256`` audit.
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

# Per-task clinical policy. report_mode is the task's scoring unit
# (episodic | continuous). Continuous tasks additionally declare:
#   landmark_horizon : how many landmarks (in the task's window_unit) the
#                      landmark/leadtime sections evaluate — 30 days for the
#                      daily tasks, 168 hours (7 days) for hourly sepsis.
#   intervention     : (optional) the cohort column holding the per-stay
#                      clinical-action timestamp, for the alert-vs-intervention
#                      timing analysis. When set, that column MUST be present in
#                      the cohort/preds at report time.
#
# dca_axis (optional) names the decision curve's axis in clinical language. A
# `clinical_direction: "below"` task acts on LOW risk, so its decision curve is
# built on the COMPLEMENT of the label: for task3 the model predicts extubation
# FAILURE, but the decision is whether to EXTUBATE, so the DCA's event is
# "successful extubation" and its x-axis is the probability of success a clinician
# would require before extubating — 0 (unworried about failure, worried about the
# harms of continued intubation: extubate everyone) to 1 (terrified of failure:
# extubate nobody). That complement event has no name derivable from the label, so
# it is declared here. Omit for an "above" task: the DCA event IS the label event.
TASK_POLICY: dict[str, dict] = {
    "task1_icu_daily_mortality":    {"clinical_threshold": 0.95, "clinical_action": "withdraw life support", "clinical_direction": "above", "report_mode": "continuous", "landmark_horizon": 30,
                                     "dca_axis": {"event": "in-hospital mortality", "action": "withdraw life support"}},
    "task2_icu_daily_ltach":        {"clinical_threshold": 0.5, "clinical_action": "perform early tracheostomy", "clinical_direction": "above", "report_mode": "continuous", "landmark_horizon": 30,
                                     "dca_axis": {"event": "LTACH discharge", "action": "perform early tracheostomy"}},
    "task3_extubation_failure_24h": {"clinical_threshold": 0.15, "clinical_action": "extubate", "clinical_direction": "below", "report_mode": "episodic",
                                     "dca_axis": {"event": "successful extubation", "action": "extubate"}},
    "task4_sepsis_abx_6h":          {"clinical_threshold": 0.1, "clinical_action": "treat with antibiotics", "clinical_direction": "above", "report_mode": "continuous", "landmark_horizon": 168,
                                     "intervention": {"column": "first_abx_dttm", "display": "treatment"},
                                     "dca_axis": {"event": "sepsis onset within 6h", "action": "treat with antibiotics"}},
    "task5_icu_readmission":        {"clinical_threshold": 0.2, "clinical_action": "keep in ICU", "clinical_direction": "above", "report_mode": "episodic",
                                     "dca_axis": {"event": "unplanned ICU readmission", "action": "keep in ICU"}},
}

REPORT_MODES = {"episodic", "continuous"}

# Defaults for the data-driven threshold-locator menu + operating-budget view
# (report/operating_points.py, report/_locators.py — design doc §3.2, §4.3). Each
# is overridable per task by adding the same key to that task's TASK_POLICY entry;
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
