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

__version__ = "0.0.1"

# Per-task clinical policy. report_mode is the baseline's recommended scoring
# unit (episodic | peak | landmark); callers can still override via build_report.
TASK_POLICY: dict[str, dict] = {
    "task1_icu_daily_mortality":    {"clinical_threshold": 0.95, "clinical_direction": "above", "report_mode": "landmark"},
    "task2_icu_daily_ltach":        {"clinical_threshold": 0.5,  "clinical_direction": "above", "report_mode": "landmark"},
    "task3_extubation_failure_24h": {"clinical_threshold": 0.15, "clinical_direction": "below", "report_mode": "episodic"},
    "task4_sepsis_abx_6h":          {"clinical_threshold": 0.1,  "clinical_direction": "above", "report_mode": "peak"},
    "task5_icu_readmission":        {"clinical_threshold": 0.2,  "clinical_direction": "above", "report_mode": "episodic"},
}

REPORT_MODES = {"episodic", "peak", "landmark"}
