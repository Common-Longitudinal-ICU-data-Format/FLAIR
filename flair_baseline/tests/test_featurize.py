"""count_features null/zero invariants (synthetic data — no real/MIMIC access).

Locks the "null = no data, count only where data" contract:
  * the CSR never materializes a 0 — a cell exists only where events exist;
  * a vocab code the site lacks produces an all-missing (empty) column, not zeros;
  * an encounter with no events for an existing code leaves that cell unstored (null).
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from flair_baseline.featurize import count_features

# Two encounters, prediction at 2020-01-02. Events strictly before are counted.
# A: 2 lactates + 1 imv before, 1 lactate after (must be dropped).
# B: 1 lactate before, NO imv at all (its imv cell must stay null/unstored).
_EVENTS = pl.DataFrame(
    {
        "hospitalization_join_id": ["A", "A", "A", "A", "B"],
        "time": [
            datetime(2020, 1, 1, 6),   # A lactate (before)
            datetime(2020, 1, 1, 8),   # A lactate (before)
            datetime(2020, 1, 1, 9),   # A imv     (before)
            datetime(2020, 1, 3, 0),   # A lactate (AFTER prediction — excluded)
            datetime(2020, 1, 1, 7),   # B lactate (before)
        ],
        "code": [
            "LAB//lactate//mmol/l",
            "LAB//lactate//mmol/l",
            "RESP//device_category//imv",
            "LAB//lactate//mmol/l",
            "LAB//lactate//mmol/l",
        ],
        "numeric_value": [1.0, 1.2, None, 0.9, 1.1],
    }
)

_COHORT = pl.DataFrame(
    {
        "prediction_id": ["pA", "pB"],
        "hospitalization_join_id": ["A", "B"],
        "prediction_dttm": [datetime(2020, 1, 2), datetime(2020, 1, 2)],
        "split": ["train", "test"],
        "label": [1, 0],
    }
)

# truncated codes: numeric→2 levels, categorical→3 levels
_LACTATE = "LAB//lactate"
_IMV = "RESP//device_category//imv"


def test_no_stored_zeros():
    """A count cell exists only where there is data — the CSR stores no 0."""
    X, _ids, vocab = count_features(_EVENTS, _COHORT, "label")
    assert (X.data == 0).sum() == 0
    assert X.nnz == 0 or X.data.min() > 0
    # vocab is fit on the train split (encounter A) only.
    assert set(vocab) == {_LACTATE, _IMV}


def test_missing_event_cell_is_null_not_zero():
    """Encounter B has no imv → its imv cell is unstored (null), not a 0."""
    X, ids, vocab = count_features(_EVENTS, _COHORT, "label")
    row = {pid: i for i, pid in enumerate(ids)}
    imv = vocab.index(_IMV)
    lac = vocab.index(_LACTATE)
    # A: 2 lactates (the post-prediction one is excluded) + 1 imv.
    assert X[row["pA"], lac] == 2.0
    assert X[row["pA"], imv] == 1.0
    # B: 1 lactate, and NO imv entry at all (null = no data).
    assert X[row["pB"], lac] == 1.0
    assert X[:, imv].nnz == 1  # only encounter A, B's cell is unstored


def test_missing_vocab_column_is_empty():
    """A supplied vocab code absent from the site → all-missing column, never 0."""
    vocab = [_LACTATE, _IMV, "MED_INT//ghostdrug"]  # ghostdrug never extracted here
    X, _ids, out_vocab = count_features(_EVENTS, _COHORT, "label", vocab=vocab)
    assert out_vocab == vocab
    ghost = vocab.index("MED_INT//ghostdrug")
    assert X[:, ghost].nnz == 0
    assert (X.data == 0).sum() == 0
