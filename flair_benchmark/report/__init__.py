"""Report bundle builder. Emits 9 JSON files; never images.

Optional --viz renders local sanity-check PNGs from the same JSON dicts
(see report/_viz.py); the FLAIR website renders the JSONs interactively.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark.report.curves import build_curves
from flair_benchmark.report.dca import build_dca
from flair_benchmark.report.encounter import build_encounter
from flair_benchmark.report.fairness import build_fairness
from flair_benchmark.report.scorecard import build_scorecard
from flair_benchmark.report.table1 import build_table1
from flair_benchmark.report.temporal import build_temporal
from flair_benchmark.report.threshold_analysis import build_threshold_analysis
from flair_benchmark.report.tripod_ai import build_tripod_ai

__all__ = ["build_report"]

# cohort columns the report can use when they flow through (or are joined in).
# Demographics feed fairness.py / table1.py; without them fairness emits no subgroups.
_CONTEXT_COLS = ["window_number", "prediction_dttm", "admission_dttm",
                 "hospitalization_join_id",
                 "age_at_admission", "sex_category", "race_category",
                 "ethnicity_category"]


def build_report(preds_path: str, task_module, out_dir: str,
                 cohort_path: str | None = None,
                 viz: bool = False, site: str | None = None) -> dict[str, Path]:
    """Read preds parquet, look up META on task_module, write 9 JSONs."""
    from flair_benchmark import __version__
    from flair_benchmark._hash import task_script_sha256

    preds = pl.read_parquet(preds_path)
    meta = task_module.META
    _validate_preds_schema(preds, meta)
    preds = _attach_cohort_context(preds, cohort_path, meta)
    preds = _ensure_y_pred(preds, meta)

    audit = {
        "task": meta["name"],
        "task_script_sha256": task_script_sha256(task_module),
        "flair_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    table1 = build_table1(preds, meta)
    table1["metadata"] = {**audit, "min_cell_count": 10}

    fairness = build_fairness(preds, meta)
    fairness["metadata"] = {**audit, **fairness["metadata"]}

    tripod = build_tripod_ai(preds, meta, table1)
    tripod["metadata"] = audit

    curves = build_curves(preds, meta)
    curves["metadata"] = {**audit, **curves["metadata"]}

    temporal = build_temporal(preds, meta)
    temporal["metadata"] = {**audit, **temporal["metadata"]}

    dca = build_dca(preds, meta)
    dca["metadata"] = {**audit, **dca["metadata"]}

    threshold = build_threshold_analysis(preds, meta)
    threshold["metadata"] = {**audit, **threshold["metadata"]}

    encounter = build_encounter(preds, meta)
    encounter["metadata"] = {**audit, **encounter["metadata"]}

    # scorecard reads the already-computed dicts (no recompute) → the flat,
    # cross-site-poolable headline row.
    scorecard = build_scorecard(meta, curves, threshold, encounter, site=site)
    scorecard["metadata"] = {**audit, **scorecard.get("metadata", {})}

    reports = {"table1": table1, "tripod_ai": tripod, "fairness": fairness,
               "curves": curves, "temporal": temporal, "dca": dca,
               "threshold_analysis": threshold, "encounter": encounter,
               "scorecard": scorecard}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for kind, payload in reports.items():
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(payload, indent=2, default=str))

    if viz:
        from flair_benchmark.report._viz import render_sanity_pngs
        for png in render_sanity_pngs(reports, out / "viz"):
            paths[png.stem + "_png"] = png
    return paths


def _ensure_y_pred(preds: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """Derive y_pred from y_prob (>= 0.5) for binary tasks when it is absent.

    Lets a site ship a single file — the cohort parquet plus a y_prob column —
    instead of authoring a separate predictions parquet (which risks row drift
    against window_number / prediction_id). y_pred is honored if provided.
    """
    if (meta["task_type"] == "binary"
            and "y_pred" not in preds.columns
            and "y_prob" in preds.columns):
        return preds.with_columns(
            (pl.col("y_prob") >= 0.5).cast(pl.Int8).alias("y_pred")
        )
    return preds


def _attach_cohort_context(preds: pl.DataFrame, cohort_path: str | None,
                           meta: dict) -> pl.DataFrame:
    """Bring window_number / prediction_dttm / admission_dttm onto preds.

    Preference order:
      1. columns already present in preds (sites copy cohort columns through)
      2. join the cohort parquet on prediction_id, else
         (hospitalization_id, prediction_dttm), else hospitalization_id
         (safe for one-per-stay tasks only)
    Missing context never fails — temporal sections degrade gracefully.
    """
    needed = [c for c in _CONTEXT_COLS if c not in preds.columns]
    if not needed or cohort_path is None:
        return preds

    cohort = pl.read_parquet(cohort_path)
    available = [c for c in needed if c in cohort.columns]
    if not available:
        return preds

    if "prediction_id" in preds.columns and "prediction_id" in cohort.columns:
        keys = ["prediction_id"]
    elif "prediction_dttm" in preds.columns and "prediction_dttm" in cohort.columns:
        keys = ["hospitalization_id", "prediction_dttm"]
    elif meta["prediction_unit"] == "one-per-stay":
        keys = ["hospitalization_id"]
    else:
        warnings.warn(
            "Cohort provided but predictions lack prediction_id/prediction_dttm "
            "for a multi-row-per-stay task; skipping cohort join. "
            "Carry prediction_id through your predictions parquet.",
            stacklevel=2,
        )
        return preds
    return preds.join(cohort.select([*keys, *available]).unique(subset=keys),
                      on=keys, how="left")


def _validate_preds_schema(preds: pl.DataFrame, meta: dict) -> None:
    """Enforce the strict preds parquet contract."""
    required = {"hospitalization_id", "split", meta["label_column"]}
    task_type = meta["task_type"]
    if task_type == "multiclass":
        required.add("y_pred")
    elif task_type == "regression":
        required.add("y_pred_value")
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"Predictions parquet missing required columns for task "
            f"'{meta['name']}' ({task_type}): {sorted(missing)}. "
            f"Got columns: {sorted(preds.columns)}"
        )

    # Binary: the site only has to add y_prob to the cohort file — y_pred is
    # derived at 0.5 when absent (see _ensure_y_pred). Require at least one.
    if task_type == "binary" and not ({"y_pred", "y_prob"} & set(preds.columns)):
        raise ValueError(
            f"Predictions parquet for binary task '{meta['name']}' must contain "
            f"'y_prob' (recommended) or 'y_pred'. Add y_prob to the cohort file: "
            f"cohort.with_columns(pl.Series('y_prob', probs))."
        )

    # y_prob must be a genuine probability: calibration bins, DCA net-benefit,
    # and Brier score all silently corrupt on out-of-range / null values.
    if "y_prob" in preds.columns:
        n_null = preds["y_prob"].null_count()
        if n_null > 0:
            raise ValueError(
                f"Predictions parquet for task '{meta['name']}': y_prob must be a "
                f"probability in [0, 1], but {n_null} of {preds.height} values are "
                f"null. Drop those rows or fill in a probability."
            )
        lo, hi = preds["y_prob"].min(), preds["y_prob"].max()
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(
                f"Predictions parquet for task '{meta['name']}': y_prob must be a "
                f"probability in [0, 1], but the observed range is [{lo}, {hi}]. "
                f"Pass calibrated probabilities (e.g. model.predict_proba(...)[:, 1]), "
                f"not logits or scores."
            )
