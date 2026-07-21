"""Auto-discover task modules. Adding a task = drop a new .py file here.

Each task module must expose:
    META: dict          # see docs/tasks for required keys
    build(clif_config, train_end, test_start) -> pl.DataFrame

META taxonomy fields (required, enum-validated):
    task_class       ∈ {outcome, deterioration, intervention, operational}
    temporal_pattern ∈ {episode, fixed_horizon, continuous}
    prediction_unit  ∈ {one-per-stay, one-per-window}
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from flair_benchmark._constants import REPORT_MODES

TASK_CLASS_VALUES = {"outcome", "deterioration", "intervention", "operational"}
TEMPORAL_PATTERN_VALUES = {"episode", "fixed_horizon", "continuous"}
PREDICTION_UNIT_VALUES = {"one-per-stay", "one-per-window"}

_TAXONOMY_ENUMS = {
    "task_class": TASK_CLASS_VALUES,
    "temporal_pattern": TEMPORAL_PATTERN_VALUES,
    "prediction_unit": PREDICTION_UNIT_VALUES,
}


def list_tasks() -> list[str]:
    """Return sorted task names (filenames without .py, no underscore prefix)."""
    pkg = importlib.import_module(__name__)
    return sorted(
        name for _, name, _ in pkgutil.iter_modules(pkg.__path__)
        if not name.startswith("_")
    )


def get_task(name: str) -> ModuleType:
    """Import and return the task module. Validates META + build()."""
    if name not in list_tasks():
        raise ValueError(
            f"Unknown task: {name!r}. Available: {list_tasks()}"
        )
    module = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(module, "META") or not hasattr(module, "build"):
        raise TypeError(
            f"Task module {name!r} must define META dict and build() function."
        )
    _validate_taxonomy(name, module.META)
    return module


def _validate_taxonomy(task_name: str, meta: dict) -> None:
    # META['name'] must match the filename: the auto-discovery in list_tasks()
    # keys everything (report routing, the audit hash) off the module name, so a
    # mismatched META['name'] would mislabel the task everywhere downstream.
    if meta.get("name") != task_name:
        raise ValueError(
            f"Task {task_name!r}: META['name'] = {meta.get('name')!r} must "
            f"match the module filename."
        )
    if meta.get("task_type") == "binary":
        t = meta.get("clinical_threshold")
        if not isinstance(t, (int, float)) or not 0 < t < 1:
            raise ValueError(
                f"Task {task_name!r}: META['clinical_threshold'] must be a float "
                f"in (0, 1); got {t!r}."
            )
        # The operating-point sweep is a fixed 0.01-step grid (report/_sweep.py),
        # and the clinical policy is one flagged row inside it. A threshold off the
        # grid would have no is_clinical row — reject it here so the invariant holds
        # by construction (all shipped thresholds are multiples of 0.01).
        if round(t * 100) != round(t * 100, 6):
            raise ValueError(
                f"Task {task_name!r}: META['clinical_threshold'] = {t!r} must be a "
                f"multiple of 0.01 so it lands on the operating-point sweep grid."
            )
        if meta.get("clinical_direction") not in ("above", "below"):
            raise ValueError(
                f"Task {task_name!r}: META['clinical_direction'] must be "
                f"'above' or 'below'; got {meta.get('clinical_direction')!r}."
            )
    if "report_mode" in meta and meta["report_mode"] not in REPORT_MODES:
        raise ValueError(
            f"Task {task_name!r}: META['report_mode'] = {meta['report_mode']!r} "
            f"is not one of {sorted(REPORT_MODES)}."
        )
    if meta.get("report_mode") == "continuous":
        h = meta.get("landmark_horizon")
        if not isinstance(h, int) or isinstance(h, bool) or h < 1:
            raise ValueError(
                f"Task {task_name!r}: continuous report_mode requires "
                f"META['landmark_horizon'] to be a positive int (landmarks in "
                f"the task's window_unit); got {h!r}."
            )
    if "intervention" in meta:
        iv = meta["intervention"]
        if (not isinstance(iv, dict) or not isinstance(iv.get("column"), str)
                or not iv["column"]):
            raise ValueError(
                f"Task {task_name!r}: META['intervention'] must be a dict with "
                f"a non-empty str 'column' (optional 'display'); got {iv!r}."
            )
    for field, allowed in _TAXONOMY_ENUMS.items():
        if field not in meta:
            raise TypeError(
                f"Task {task_name!r}: META is missing required key {field!r}. "
                f"Allowed values: {sorted(allowed)}."
            )
        if meta[field] not in allowed:
            raise ValueError(
                f"Task {task_name!r}: META[{field!r}] = {meta[field]!r} "
                f"is not one of {sorted(allowed)}."
            )
