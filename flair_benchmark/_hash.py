"""Audit digest of a task, stamped into every report JSON.

Lets the central aggregator prove two sites ran the same task — both the label
logic (the task .py source) AND the clinical policy it was evaluated under
(threshold, direction, report mode), which lives inline in each task's ``META``.
The digest is the SHA-256 of the task source bytes concatenated with a canonical
JSON of the audit-relevant resolved META, so a change to either side changes the
hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

# Resolved META keys that define "the same task" for cross-site comparison.
# (Keys absent from a task's META serialize as null — e.g. landmark_horizon /
# intervention on episodic tasks. Extending this tuple changes every task's
# digest, so cross-site comparisons must regenerate on the same version.)
_AUDIT_KEYS = (
    "name", "label_column", "task_type", "temporal_pattern",
    "prediction_unit", "clinical_threshold", "clinical_direction", "report_mode",
    "landmark_horizon", "intervention",
)


def task_script_sha256(task_module: ModuleType) -> str:
    """SHA-256 of the task source + its resolved clinical policy (audit digest)."""
    src = Path(task_module.__file__).read_bytes()
    meta = getattr(task_module, "META", {}) or {}
    policy = json.dumps(
        {k: meta.get(k) for k in _AUDIT_KEYS}, sort_keys=True
    ).encode()
    return hashlib.sha256(src + b"\x00" + policy).hexdigest()
