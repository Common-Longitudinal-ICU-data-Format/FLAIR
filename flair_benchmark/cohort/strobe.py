"""STROBE cohort flow — the count left after every inclusion/exclusion step.

Published as `attrition` in Table 1 (`table1.json`, non-PHI folder) at `flair load-task` time, so
each Table 1 carries its own flow record. Like the rest of Table 1 it is
publishable: every count passes through `suppress()`.

The filters themselves stay where they are (cohort/clif.py, cohort/encounter.py,
each task's build()); after each one the frame is piped through `record`, which
appends a step and hands the frame back unchanged:

    cohort = cohort.filter(...).pipe(record, "age >= 18")

Each step carries n_rows plus whichever of n_hospitalizations / n_encounters /
n_patients the frame has a key for, and n_excluded_* versus the previous step on
the same key.

Output (`steps()`, stored as table1["attrition"]):
  [ {"step": str, "n_rows": int, "n_encounters": int, ...,
     "n_excluded_encounters": int, ...}, ... ]
The final split / label counts are Table 1's own by_split / by_label.

Depends on: report/_stats.py (suppression).
Used by: cohort/clif.py, cohort/encounter.py, tasks/*, cli.py `load-task`.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report._stats import suppress

# ponytail: module-global, one build() per process (the CLI); pass a list through
# build() if cohorts are ever built concurrently.
_steps: list[dict] = []

_KEYS = {"hospitalization_id": "hospitalizations",
         "hospitalization_join_id": "encounters",
         "patient_id": "patients"}


def reset() -> None:
    _steps.clear()


def record(df: pl.DataFrame, step: str, *,
           encounter_keyed: bool = False) -> pl.DataFrame:
    """Append the counts of `df` as STROBE step `step`; return `df` unchanged.

    encounter_keyed=True for frames from `as_encounter_keyed`, whose
    'hospitalization_id' column actually holds the stitched-encounter id.
    """
    counts = {"n_rows": df.height}
    for col, name in _KEYS.items():
        # Once a frame carries the encounter key, its hospitalization_id is a
        # representative member, not a population: count encounters from then on.
        if col == "hospitalization_id" and "hospitalization_join_id" in df.columns:
            continue
        if col in df.columns:
            if encounter_keyed and col == "hospitalization_id":
                name = "encounters"
            counts[f"n_{name}"] = df[col].n_unique()
    _steps.append({"step": step, **counts})
    return df


def _pub(n: int) -> int | str:
    return 0 if n == 0 else suppress(n)   # a true zero is not PHI


def steps() -> list[dict]:
    """Recorded steps with n_excluded_* added and every count suppressed."""
    out, prev = [], {}
    for s in _steps:
        row = {"step": s["step"]}
        for k, v in s.items():
            if k.startswith("n_"):
                row[k] = _pub(v)
        for k, v in s.items():
            if k.startswith("n_") and k != "n_rows" and k in prev:
                row[f"n_excluded_{k[2:]}"] = _pub(prev[k] - v)
        out.append(row)
        prev = s
    return out

