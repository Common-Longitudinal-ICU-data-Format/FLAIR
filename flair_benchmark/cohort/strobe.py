"""STROBE cohort flow — the count left after every inclusion/exclusion step.

Written next to the cohort parquet at `flair load-task` time as
`<cohort>.strobe.json`, alongside Table 1. Like Table 1 it is publishable: every
count passes through `suppress()`.

The filters themselves stay where they are (cohort/clif.py, cohort/encounter.py,
each task's build()); after each one the frame is piped through `record`, which
appends a step and hands the frame back unchanged:

    cohort = cohort.filter(...).pipe(record, "age >= 18")

Each step carries n_rows plus whichever of n_hospitalizations / n_encounters /
n_patients the frame has a key for, and n_excluded_* versus the previous step on
the same key.

Output JSON shape:
  {
    "steps": [ {"step": str, "n_rows": int, "n_encounters": int, ...,
                "n_excluded_encounters": int, ...}, ... ],
    "final": { "n_encounters", "n_rows",
               "by_split": {"train": n, "test": n},          # encounter grain
               "by_label": {"1": n, "0": n} },               # binary, ever-positive
    "metadata": { task, task_script_sha256, flair_version, generated_at, min_cell_count }
  }

Depends on: report/_stats.py (suppression), constants.py.
Used by: cohort/clif.py, cohort/encounter.py, tasks/*, cli.py `load-task`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from flair_benchmark.constants import task_script_sha256
from flair_benchmark.report._stats import MIN_CELL_COUNT, suppress

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


def build_strobe(cohort: pl.DataFrame, task_module) -> dict:
    """The full STROBE dict: recorded steps, final split/label counts, metadata."""
    from flair_benchmark import __version__

    meta = task_module.META
    label = meta["label_column"]
    enc = cohort.group_by("hospitalization_join_id").agg(
        pl.col("split").first(),
        *([pl.col(label).max()] if meta["task_type"] == "binary" else []),
    )
    final = {
        "n_encounters": _pub(enc.height),
        "n_rows": _pub(cohort.height),
        "by_split": {s: _pub(enc.filter(pl.col("split") == s).height)
                     for s in ("train", "test")},
    }
    if meta["task_type"] == "binary":
        final["by_label"] = {str(v): _pub(enc.filter(pl.col(label) == v).height)
                             for v in (1, 0)}
    return {
        "steps": steps(),
        "final": final,
        "metadata": {
            "task": meta["name"],
            "task_script_sha256": task_script_sha256(task_module),
            "flair_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_cell_count": MIN_CELL_COUNT,
        },
    }
