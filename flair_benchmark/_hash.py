"""SHA-256 of a task .py file, stamped into every report JSON for audit.

Lets the central aggregator prove two sites ran byte-identical task code
regardless of the installed package version.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType


def task_script_sha256(task_module: ModuleType) -> str:
    """Return the SHA-256 hex digest of the task module's source file."""
    path = Path(task_module.__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()
