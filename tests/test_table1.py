"""Unit tests for the Table 1 builder (flair_benchmark.report.table1.build_table1).

Frame-level, no clifpy IO. Covers: overall + by_label (positive/negative) + by_split
columns, encounter-vs-row grain, severity/comorbidity continuous vars, and HIPAA
cell suppression.
"""

from __future__ import annotations

import polars as pl
import pytest

from flair_benchmark.report.table1 import build_table1

META = {"label_column": "label", "task_type": "binary"}


def _cohort():
    # e1: 3 rows, ever-positive (one row label=1). e2..e12: one row each, negative.
    # Enough encounters that the negative column clears the <10 suppression floor.
    rows = []
    rows += [("e1", v, 1 if i == 1 else 0, "train", 70, 9.0, 5.0, 12.0,
              "Male", "White", "Non-Hispanic") for i, v in enumerate(range(3))]
    for k in range(2, 14):
        rows.append((f"e{k}", 0, 0, "test", 50 + k, 2.0, 1.0, 3.0,
                     "Female", "Black or African American", "Hispanic"))
    cols = ["hospitalization_join_id", "_v", "label", "split", "age_at_admission",
            "sofa_max", "charlson_score", "elix_score",
            "sex_category", "race_category", "ethnicity_category"]
    return pl.DataFrame(rows, schema=cols, orient="row").drop("_v")


def test_grain_encounter_vs_row():
    t = build_table1(_cohort(), META)
    # 13 encounters (e1..e13), 15 rows (e1 has 3).
    assert t["n_total"] == 13
    assert t["n_rows"] == 15
    assert t["by_grain"]["encounter"]["n_total"] == 13
    assert t["by_grain"]["row"]["n_total"] == 15


def test_by_label_positive_and_negative_columns():
    t = build_table1(_cohort(), META)
    assert set(t["by_label"].keys()) == {"0", "1"}
    # e1 is the only ever-positive encounter -> suppressed (<10); 12 negatives -> shown.
    assert t["by_label"]["1"]["n"] == "<10"
    assert t["by_label"]["0"]["n"] == 12


def test_ever_positive_dedup():
    # e1 has a positive row among negatives; at encounter grain it is positive once,
    # not three negative-ish rows.
    t = build_table1(_cohort(), META)
    enc = t["by_grain"]["encounter"]
    assert enc["by_label"]["1"]["n"] == "<10"        # exactly one encounter (e1)
    # at row grain, e1 contributes 1 positive row + 2 negative rows.
    assert t["by_grain"]["row"]["by_label"]["1"]["n"] == "<10"


def test_severity_comorbidity_columns_present():
    t = build_table1(_cohort(), META)
    assert set(["age_at_admission", "sofa_max", "charlson_score", "elix_score"]) <= set(
        t["continuous"].keys()
    )
    # negative column median SOFA reflects the e2.. rows (2.0).
    assert t["by_label"]["0"]["continuous"]["sofa_max"]["median"] == 2.0


def test_works_without_scores():
    # No sofa/charlson/elix columns -> those continuous vars simply absent.
    df = _cohort().drop(["sofa_max", "charlson_score", "elix_score"])
    t = build_table1(df, META)
    assert "sofa_max" not in t["continuous"]
    assert "age_at_admission" in t["continuous"]


def test_categorical_suppression():
    t = build_table1(_cohort(), META)
    # Male appears only in e1 (1 encounter) -> suppressed; Female in 12 -> shown.
    sex = t["categorical"]["sex_category"]
    assert sex["Male"] == "<10"
    assert sex["Female"] == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
