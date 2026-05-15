"""Auto-discover task modules. Adding a task = drop a new .py file here.

Each task module must expose:
    META: dict          # see docs/tasks for required keys
    build(clif_config, train_end, test_start) -> pl.DataFrame

META taxonomy fields (required, enum-validated):
    task_class       ∈ {outcome, deterioration, intervention, operational}
    temporal_pattern ∈ {episode, fixed_horizon, continuous}
    prediction_unit  ∈ {one-per-stay, one-per-decision-point, one-per-timestep}
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

TASK_CLASS_VALUES = {"outcome", "deterioration", "intervention", "operational"}
TEMPORAL_PATTERN_VALUES = {"episode", "fixed_horizon", "continuous"}
PREDICTION_UNIT_VALUES = {"one-per-stay", "one-per-decision-point", "one-per-timestep"}

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
