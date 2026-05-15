"""Per-site cohort metadata extractor for the FLAIR leaderboard.

Usage (CLI):
  flair extract-metadata --clif-config <path> --out <dir> [--task task6_...]
                         [--skip-sofa] [--skip-comorb]

Output: one `<task_name>.json` per task in `<dir>`. Each file is a
self-contained summary of the site's cohort for that task — label
distribution, demographics, severity, comorbidity, and yearly counts —
HIPAA-suppressed and audit-stamped (task SHA-256, flair_version,
generated_at).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_hospitalization,
                                    read_clif_config)
from flair_benchmark._icu_timing import compute_icu_timing
from flair_benchmark.metadata._comorbidity import compute_cci_elix
from flair_benchmark.metadata._severity import compute_first_24h_sofa_max
from flair_benchmark.metadata._summarize import summarize_task
from flair_benchmark.report._suppress import MIN_CELL_COUNT

__all__ = ["extract_site_metadata"]

REFERENCE_WINDOW_DESC = "first 24h of first ICU stay (SOFA max-in-window)"


def extract_site_metadata(clif_config_path: str,
                          tasks: list[str] | None = None,
                          skip_sofa: bool = False,
                          skip_comorb: bool = False) -> dict[str, dict]:
    """Return {task_name: per_task_json_dict} for all (or selected) tasks.

    Heavy steps (CLIF cohort load, SOFA, comorbidity) run once and are
    reused across all tasks.
    """
    from flair_benchmark import __version__
    from flair_benchmark._hash import task_script_sha256
    from flair_benchmark.tasks import get_task, list_tasks

    cfg = read_clif_config(clif_config_path)
    task_names = tasks or list_tasks()

    cohort, adt = base_icu_cohort(cfg)
    timing = compute_icu_timing(adt)
    cohort = cohort.join(timing, on="hospitalization_id", how="inner")
    hosp_ids = cohort["hospitalization_id"].to_list()

    admission_dttm = cohort.select(["hospitalization_id", "admission_dttm"])

    sofa_per_stay = None
    if not skip_sofa:
        orchestrator = _build_orchestrator(cfg)
        first_icu = cohort.select(["hospitalization_id", "first_icu_start_time"])
        sofa_per_stay = compute_first_24h_sofa_max(orchestrator, first_icu)

    comorb_per_stay = None
    if not skip_comorb:
        try:
            comorb_per_stay = compute_cci_elix(cfg, hosp_ids)
        except Exception as exc:  # noqa: BLE001
            comorb_per_stay = None
            _print_warn(f"Charlson/Elixhauser skipped (HospitalDiagnosis "
                        f"unavailable at site {cfg['site']!r}): {exc}")

    audit = {
        "site": cfg["site"],
        "flair_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_cell_n": MIN_CELL_COUNT,
        "reference_window": REFERENCE_WINDOW_DESC,
    }

    out: dict[str, dict] = {}
    for task_name in task_names:
        task_module = get_task(task_name)
        task_cohort = task_module.build(
            clif_config=clif_config_path, train_end=None, test_start=None,
        )
        block = summarize_task(
            task_module, task_cohort, admission_dttm,
            sofa_per_stay, comorb_per_stay,
        )
        out[task_name] = {
            **audit,
            "task": {
                "name": task_module.META["name"],
                "task_class": task_module.META["task_class"],
                "temporal_pattern": task_module.META["temporal_pattern"],
                "prediction_unit": task_module.META["prediction_unit"],
                "task_type": task_module.META["task_type"],
                "label_column": task_module.META["label_column"],
                "task_script_sha256": task_script_sha256(task_module),
            },
            **block,
        }
    return out


def _build_orchestrator(cfg: dict):
    """Instantiate a clifpy.ClifOrchestrator from a FLAIR clif_config dict."""
    from clifpy import ClifOrchestrator
    return ClifOrchestrator(
        data_directory=cfg["data_directory"],
        filetype=cfg["filetype"],
        timezone=cfg["timezone"],
    )


def _print_warn(msg: str) -> None:
    import sys
    print(f"[flair extract-metadata] WARN: {msg}", file=sys.stderr)
