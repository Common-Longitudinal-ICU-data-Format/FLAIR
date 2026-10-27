"""Snapshot: frozen IDs + reference hashes, audited by content (cohort/snapshot.py, stitch.py)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest

from flair_benchmark.cohort import snapshot as snap
from flair_benchmark.cohort.stitch import (
    encounter_index_sha256,
    load_or_build_encounter_index,
    use_snapshot,
)
from flair_benchmark.cohort.table1 import cohort_sha256
from flair_benchmark.constants import task_script_sha256

REF = {"task_script_sha256": "t", "cohort_sha256": "c",
       "train_cohort_sha256": "tr", "test_cohort_sha256": "te"}


def test_audit_flags_only_the_differing_hash():
    assert snap.audit(None, REF) is None                       # first build: nothing to compare
    assert all(snap.audit(REF, REF).values())
    got = snap.audit(REF, {**REF, "test_cohort_sha256": "other"})
    assert got == {"task_script_sha256": True, "cohort_sha256": True,
                   "train_cohort_sha256": True, "test_cohort_sha256": False}


def test_reference_hashes_are_never_overwritten(tmp_path):
    kw = dict(site="s", flair_version="0.0.1", task="icu_readmission")
    assert snap.write_task_hashes(tmp_path, index_sha256="i1", metadata=REF, **kw)
    assert not snap.write_task_hashes(tmp_path, index_sha256="i2",
                                      metadata={**REF, "cohort_sha256": "new"}, **kw)
    doc = snap.read_hashes(tmp_path)
    assert doc["encounter_index_sha256"] == "i1"
    assert doc["tasks"]["icu_readmission"]["cohort_sha256"] == "c"


def test_content_hashes_ignore_row_order_and_file_times(tmp_path):
    idx = pl.DataFrame({"hospitalization_id": ["h1", "h2", "h3"],
                        "hospitalization_join_id": ["e1", "e1", "e3"],
                        "patient_id": ["p1", "p1", "p2"]})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    idx.write_parquet(a)
    idx.reverse().write_parquet(b)                               # other order, later mtime
    assert encounter_index_sha256(pl.read_parquet(a)) == encounter_index_sha256(pl.read_parquet(b))

    t = datetime(2024, 1, 1)
    cohort = pl.DataFrame({"hospitalization_join_id": ["e1", "e3"], "prediction_dttm": [t, t],
                           "feature_cutoff_dttm": [t, t], "y": [0, 1], "split": ["train", "test"]})
    assert cohort_sha256(cohort, "y") == cohort_sha256(cohort.reverse(), "y")


def test_task_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    meta = {"name": "t"}
    assert (task_script_sha256(SimpleNamespace(__file__=str(lf), META=meta))
            == task_script_sha256(SimpleNamespace(__file__=str(crlf), META=meta)))


def test_frozen_snapshot_never_restitches(tmp_path):
    try:
        use_snapshot(tmp_path, frozen=True)
        with pytest.raises(FileNotFoundError, match="encounter_index"):
            load_or_build_encounter_index({"site": "s", "data_directory": "/nowhere"})
    finally:
        use_snapshot(None)
