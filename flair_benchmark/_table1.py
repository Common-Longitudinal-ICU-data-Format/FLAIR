"""Generate the per-task Table 1 JSON at task-generation time.

A task's cohort carries demographics (age/sex/race/ethnicity) but not severity or
comorbidity, so this module computes SOFA max + Charlson/Elixhauser for the
cohort's representative hospitalizations, joins them on, and calls
``report.table1.build_table1``. The scores are the heavy part (clifpy SOFA +
HospitalDiagnosis), so they are cached per site to ``<cache_dir>/cohort_scores.parquet``
— the same build-once-reuse pattern as the encounter index — and reused across all
five tasks.

Score failures (a site missing the tables clifpy needs) are non-fatal: the Table 1
is still written from the demographics it has, with the missing severity/comorbidity
columns simply absent.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark._clif import load_hospitalization
from flair_benchmark._hash import task_script_sha256
from flair_benchmark._stitch import _cache_dir
from flair_benchmark.report._suppress import MIN_CELL_COUNT
from flair_benchmark.report.table1 import build_table1

_SCORES_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "sofa_max": pl.Float64,
    "charlson_score": pl.Float64,
    "elix_score": pl.Float64,
}


def _warn(msg: str) -> None:
    print(f"[flair table1] WARN: {msg}", file=sys.stderr)


def _scores_cache_path(cfg: dict) -> Path:
    return _cache_dir(cfg) / "cohort_scores.parquet"


def _compute_scores(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """[hospitalization_id, sofa_max, charlson_score, elix_score] for hosp_ids.
    Each score is best-effort: a failure leaves that column null."""
    from flair_benchmark.report._comorbidity import compute_cci_elix
    from flair_benchmark.report._severity import compute_hospitalization_sofa_max

    base = (
        load_hospitalization(
            cfg, hospitalization_ids=hosp_ids,
            columns=["hospitalization_id", "admission_dttm", "discharge_dttm"],
        )
        .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    )

    try:
        sofa = compute_hospitalization_sofa_max(cfg, base)
    except Exception as exc:  # noqa: BLE001
        _warn(f"SOFA unavailable: {exc}")
        sofa = pl.DataFrame(schema={"hospitalization_id": pl.Utf8, "sofa_max": pl.Float64})

    try:
        comorb = compute_cci_elix(cfg, hosp_ids)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Charlson/Elixhauser unavailable: {exc}")
        comorb = pl.DataFrame(schema={
            "hospitalization_id": pl.Utf8,
            "charlson_score": pl.Float64, "elix_score": pl.Float64,
        })

    return (
        base.select("hospitalization_id")
        .join(sofa.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
              on="hospitalization_id", how="left")
        .join(comorb.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
              on="hospitalization_id", how="left")
    )


def load_or_build_cohort_scores(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """Per-site cached severity/comorbidity scores. Computes only the
    hospitalization_ids not already cached, appends, and returns the requested
    subset."""
    hosp_ids = [str(h) for h in hosp_ids]
    path = _scores_cache_path(cfg)
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=_SCORES_SCHEMA)

    cached = set(have["hospitalization_id"].to_list()) if have.height else set()
    missing = [h for h in hosp_ids if h not in cached]
    if missing:
        new = _compute_scores(cfg, missing)
        have = pl.concat([have, new], how="diagonal_relaxed").unique(
            subset=["hospitalization_id"], keep="first"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        have.write_parquet(path)

    return have.filter(pl.col("hospitalization_id").is_in(hosp_ids))


def generate_table1(cfg: dict, cohort: pl.DataFrame, task_module) -> dict:
    """Build the Table 1 dict for a generated task cohort (overall + by_label +
    by_split, at both encounter and row grain), with severity/comorbidity joined
    and an audit ``metadata`` stamp."""
    from flair_benchmark import __version__

    meta = task_module.META
    cohort_x = cohort
    if "hospitalization_id" in cohort.columns and cohort.height:
        hosp_ids = cohort["hospitalization_id"].cast(pl.Utf8).unique().to_list()
        try:
            scores = load_or_build_cohort_scores(cfg, hosp_ids)
            cohort_x = cohort.with_columns(
                pl.col("hospitalization_id").cast(pl.Utf8)
            ).join(scores, on="hospitalization_id", how="left")
        except Exception as exc:  # noqa: BLE001 — table1 must not break generation
            _warn(f"cohort scores skipped: {exc}")

    t1 = build_table1(cohort_x, meta)
    t1["metadata"] = {
        "task": meta["name"],
        "task_script_sha256": task_script_sha256(task_module),
        "flair_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_cell_count": MIN_CELL_COUNT,
    }
    return t1
