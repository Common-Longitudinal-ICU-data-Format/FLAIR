"""STROBE flow recorder (flair_benchmark.cohort.strobe): counts, exclusions, suppression."""

from __future__ import annotations

import polars as pl

from flair_benchmark.cohort import strobe


def test_record_steps_and_suppression():
    strobe.reset()
    df = pl.DataFrame({"hospitalization_id": [f"h{i}" for i in range(30)],
                       "patient_id": [f"p{i // 2}" for i in range(30)]})
    out = df.pipe(strobe.record, "all")
    assert out is df                                   # pass-through for .pipe chains
    df.head(25).pipe(strobe.record, "kept 25")
    df.head(25).pipe(strobe.record, "nothing dropped")
    pl.DataFrame({"hospitalization_id": ["e1", "e2"]}).pipe(
        strobe.record, "encounter keyed", encounter_keyed=True)

    pl.DataFrame({"hospitalization_id": ["h1", "h2"], "hospitalization_join_id": ["e1", "e1"]}).pipe(
        strobe.record, "stitched")

    s = strobe.steps()
    assert s[4] == {"step": "stitched", "n_rows": "<10", "n_encounters": "<10",
                    "n_excluded_encounters": "<10"}   # representative hosp ids not counted
    assert s[0] == {"step": "all", "n_rows": 30, "n_hospitalizations": 30, "n_patients": 15}
    assert s[1]["n_hospitalizations"] == 25
    assert s[1]["n_excluded_hospitalizations"] == "<10"   # 5 dropped -> suppressed
    assert s[2]["n_excluded_hospitalizations"] == 0       # a true zero is published
    assert s[3] == {"step": "encounter keyed", "n_rows": "<10", "n_encounters": "<10"}

    strobe.reset()
    assert strobe.steps() == []


def test_render_strobe_pngs(tmp_path):
    from flair_benchmark.report._viz import render_strobe_pngs

    table1 = {
        "attrition": [
            {"step": "all", "n_rows": 100, "n_hospitalizations": 100, "n_patients": 80},
            {"step": "stitched", "n_rows": 100, "n_encounters": 90, "n_patients": 80},
            {"step": "kept", "n_rows": 85, "n_encounters": 85, "n_patients": 76,
             "n_excluded_encounters": "<10", "n_excluded_patients": "<10"},
        ],
        "by_split": {"train": {"n": 60}, "test": {"n": 25}},
        "by_label": {"1": {"n": "<10"}, "0": {"n": 80}},
        "metadata": {"task": "toy"},
    }
    paths = render_strobe_pngs(table1, tmp_path)
    assert [p.name for p in paths] == ["strobe.png", "strobe_attrition.png"]
    assert all(p.stat().st_size > 0 for p in paths)
