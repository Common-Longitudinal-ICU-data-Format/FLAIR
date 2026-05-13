"""Auto-discover task modules. Adding a task = drop a new .py file here.

Each task module must expose:
    META: dict          # see docs/tasks for required keys
    build(clif_config, train_end, test_start) -> pl.DataFrame
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


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
    return module
