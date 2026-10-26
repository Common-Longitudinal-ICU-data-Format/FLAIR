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
