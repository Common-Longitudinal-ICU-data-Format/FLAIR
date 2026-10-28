"""Snapshot reference hashes — record them once, audit every later run against them.

A snapshot (`<site>_flair_v<version>/`, see cohort/stitch.py:snapshot_dir and
docs/101/snapshot.md) freezes a site's encounter IDs. Its `hashes.json` holds,
per task, the four hashes of the first cohort built on those IDs:

    {"flair_version": "0.0.1", "site": "mimic",
     "encounter_index_sha256": "...",
     "tasks": {"icu_readmission": {"task_script_sha256": "...",
                                   "cohort_sha256": "...",
                                   "train_cohort_sha256": "...",
                                   "test_cohort_sha256": "...",
                                   "created_at": "..."}}}

Every hash is over content (code bytes, sorted string-cast rows), never file
times or paths, so a snapshot copied to another machine audits the same. The
reference is written once per task and never overwritten; a later run only
compares. Nothing here blocks a run — a mismatch is printed and stamped into
Table 1, and the leaderboard flags it.

Depends on: nothing beyond stdlib.
Used by: cli.py `load-task`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HASH_KEYS = ("task_script_sha256", "cohort_sha256",
             "train_cohort_sha256", "test_cohort_sha256")


def _path(snap: Path) -> Path:
    return Path(snap) / "hashes.json"


def read_hashes(snap: Path) -> dict:
    """The snapshot's hashes.json, or {} if it has none yet."""
    p = _path(snap)
    return json.loads(p.read_text()) if p.exists() else {}


def write_task_hashes(snap: Path, *, site: str, flair_version: str,
                      index_sha256: str | None, task: str, metadata: dict) -> bool:
    """Record `task`'s four hashes as the reference, unless it already has one.

    Returns True if written. The encounter_index_sha256 is set only the first time,
    for the same reason: the first build is the reference.
    """
    doc = read_hashes(snap)
    doc.setdefault("flair_version", flair_version)
    doc.setdefault("site", site)
    if doc.get("encounter_index_sha256") is None:
        doc["encounter_index_sha256"] = index_sha256
    tasks = doc.setdefault("tasks", {})
    if task in tasks:
        return False
    tasks[task] = {**{k: metadata.get(k) for k in HASH_KEYS},
                   "created_at": datetime.now(timezone.utc).isoformat()}
    _path(snap).write_text(json.dumps(doc, indent=2))
    return True


def audit(reference: dict | None, metadata: dict) -> dict | None:
    """{hash: bool} — does this run's hash equal the snapshot's reference? None if no reference."""
    if not reference:
        return None
    return {k: reference.get(k) == metadata.get(k) for k in HASH_KEYS}
