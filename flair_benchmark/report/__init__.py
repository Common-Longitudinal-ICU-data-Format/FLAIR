"""Report bundle builder, keyed on the unit of prediction (`--mode`).

TWO JSON FILES PER MODE. Always.

  continuous : landmark.json + hospitalization_level.json
  episodic   : overall.json  + hospitalization_level.json

  landmark.json              four pooled numbers (AUROC, AUPRC, Brier,
                             calibration slope) with CIs, plus the per-landmark
                             curve behind each. No thresholds, no fairness, no
                             curves. See report/landmark.py.
  overall.json               the episodic equivalent: ranking + calibration,
                             threshold-free, curves stored on a fixed grid.
                             See report/overall.py.
  hospitalization_level.json THE decision file, one schema in BOTH modes: the
                             fixed 99-point threshold sweep with every metric
                             and its CI band, Youden's J beside the shipped
                             policy, and fairness at the clinical threshold.
                             See report/hospitalization_level.py.

Only JSON is exported to the FLAIR website (cross-site aggregation). Optional
--viz renders local site-validation PNGs from the same dicts (report/_viz.py) —
never exported.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark.constants import REPORT_MODES

__all__ = ["build_report"]

# 4: two files per mode — landmark/overall + hospitalization_level. Net benefit
#    folded into the sweep (dca.json gone), kpi/leadtime gone, threshold rules
#    cut to Youden alone, fairness computed at the clinical threshold only,
#    curves stored on fixed grids, calibration reduced to quantile bins.
# 3: operating_points.json as a fixed 99-point sweep with 11 threshold locators,
#    alert burden, operating budget and treatment sub-analyses.
# 2: episodic hoisted every threshold-dependent metric into operating_points.json.
# 1: the original four-pillar shape.
REPORT_SCHEMA = 4


def _blas_single_thread():
    """Context manager pinning ALL native thread pools (BLAS + OpenMP) to one
    thread, so report numbers are reproducible regardless of a machine's core
    count / thread-pool state."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception:                            # pragma: no cover
        from contextlib import nullcontext
        return nullcontext()

# Cohort columns the report can use when they flow through (or are joined in).
# window_number places a row on the landmark axis; the demographics feed the
# fairness block of hospitalization_level.json (without them it emits no
# subgroups). prediction_dttm doubles as a join key for multi-row-per-stay tasks.
# (The Table 1 is generated at task-generation time — see cohort/table1.py.)
_CONTEXT_COLS = ["window_number", "prediction_dttm", "admission_dttm",
                 "hospitalization_join_id",
                 "age_at_admission", "sex_category", "race_category",
                 "ethnicity_category"]


def build_report(preds_path: str, task_module, out_dir: str,
                 cohort_path: str | None = None,
                 viz: bool = True, site: str | None = None,
                 mode: str | None = None) -> dict[str, Path]:
    """Read preds parquet, look up META, write the report JSON bundle.

    Reports are ALWAYS computed on the held-out `test` split — the home site trains on
    `train`, every site (incl. inference-only) is scored on `test`, so cross-site
    numbers are comparable. `mode` is the unit of prediction:

      episodic   : one prediction per stay  -> overall.json  + hospitalization_level.json
      continuous : multi-row-per-stay       -> landmark.json + hospitalization_level.json

    When `mode` is omitted it is inferred from META.temporal_pattern
    (episode -> episodic; continuous -> continuous).
    """
    from flair_benchmark import __version__
    from flair_benchmark.constants import task_script_sha256

    preds = pl.read_parquet(preds_path)
    meta = task_module.META
    _validate_preds_schema(preds, meta)
    _validate_split_integrity(preds, meta)
    preds = _attach_cohort_context(preds, cohort_path, meta)
    preds = _ensure_y_pred(preds, meta)

    # Reports are always scored on the held-out 'test' split only.
    split = "test"
    available = sorted(preds["split"].unique().drop_nulls().to_list())
    if split not in available:
        raise ValueError(
            f"no 'test' rows in predictions; available splits: {available}. "
            f"Cohorts from `flair load-task` always carry train + test.")
    preds = preds.filter(pl.col("split") == split)
    if preds.height == 0:
        raise ValueError("'test' split is empty after filtering")

    block = "hospitalization_join_id"
    mode = _resolve_mode(mode, meta.get("temporal_pattern"))

    if mode == "episodic":
        n_stays = preds[block].n_unique()
        if n_stays != preds.height:
            raise ValueError(
                f"Episodic report expects one row per stay, but split '{split}' has "
                f"{preds.height} rows for {n_stays} stays. Deduplicate to one "
                f"prediction per stay, or this is a continuous task "
                f"(use --mode continuous).")

    prevalence = (round(float((preds[meta["label_column"]] > 0).mean()), 6)
                  if meta["task_type"] == "binary" else None)
    audit = {
        "task": meta["name"],
        "task_script_sha256": task_script_sha256(task_module),
        "flair_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_mode": mode,
        "split": split,
        "n_stays": preds[block].n_unique(),
        "prevalence": prevalence,
        # Shape of this bundle. Bump whenever a key moves between files: sites
        # have already uploaded bundles in every earlier schema, and a reader
        # needs to be able to tell them apart. See REPORT_SCHEMA above.
        "report_schema": REPORT_SCHEMA,
        # The task's shipped policy; null for tasks without one. It is ONE
        # flagged row of the 99-point sweep, not the only point scored.
        "evaled_at": meta.get("clinical_threshold"),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pin linear algebra to ONE thread for the whole build so the report numbers
    # are reproducible across machines: a degenerate logistic fit (e.g. a tiny
    # fairness subgroup) is sensitive to the BLAS thread count, which otherwise
    # varies with each site's core count.
    with _blas_single_thread():
        from flair_benchmark.report.hospitalization_level import (
            build_hospitalization_level,
        )
        collapse = mode == "continuous"
        if collapse:
            from flair_benchmark.report.landmark import build_landmark
            payloads = {"landmark": build_landmark(preds, meta)}
        else:
            from flair_benchmark.report.overall import build_overall
            payloads = {"overall": build_overall(preds, meta)}
        payloads["hospitalization_level"] = build_hospitalization_level(
            preds, meta, collapse=collapse)
    for payload in payloads.values():
        payload["metadata"] = {**audit, **payload.get("metadata", {})}

    paths: dict[str, Path] = {}
    for kind, payload in payloads.items():
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(payload, indent=2, default=str))

    if viz:
        try:
            from flair_benchmark.report._viz import render_pngs
            for png in render_pngs(payloads, out / "viz", mode):
                paths[png.stem + "_png"] = png
        except ImportError:
            # viz is default-on but matplotlib is an optional extra; the JSON
            # bundle (the exported artifact) is already written, so skip PNGs.
            warnings.warn(
                "matplotlib not installed; skipping --viz PNGs "
                "(pip install 'flair-benchmark[viz]', or pass --no-viz).",
                stacklevel=2)
    return paths


def _resolve_mode(mode: str | None, pattern: str | None) -> str:
    """Pick episodic / continuous from the explicit flag or META."""
    if mode is None:
        if pattern == "continuous":
            return "continuous"
        if pattern == "episode":
            warnings.warn(
                "Task META temporal_pattern=='episode'; using --mode episodic "
                "(pass --mode to silence this).", stacklevel=3)
        return "episodic"
    if mode in {"peak", "landmark"}:
        raise ValueError(
            f"--mode {mode} was merged into --mode continuous, which emits "
            f"landmark.json + hospitalization_level.json.")
    if mode not in REPORT_MODES:
        raise ValueError(
            f"Unknown --mode '{mode}'; expected one of {sorted(REPORT_MODES)}.")
    if mode == "episodic" and pattern == "continuous":
        warnings.warn(
            "--mode episodic on a continuous task collapses many rows per stay; "
            "this errors unless predictions are already one-per-stay.", stacklevel=3)
    if mode == "continuous" and pattern == "episode":
        warnings.warn(
            "--mode continuous on a one-per-stay (episode) task; --mode episodic "
            "is the usual choice.", stacklevel=3)
    return mode


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
         (hospitalization_join_id, prediction_dttm), else hospitalization_join_id
          (safe for one-per-stay tasks only)
    Missing context never fails — sections that need it degrade gracefully. The
    report only ever consumes cohort columns, never raw CLIF data.
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
    elif ("prediction_dttm" in preds.columns and "prediction_dttm" in cohort.columns
          and "hospitalization_join_id" in cohort.columns):
        keys = ["hospitalization_join_id", "prediction_dttm"]
    elif meta["prediction_unit"] == "one-per-stay":
        keys = ["hospitalization_join_id"]
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
    """Enforce the strict preds parquet contract.

    Every predictions parquet must always carry: prediction_id, split, the task label,
    hospitalization_join_id, and y_prob (the probability). hospitalization_id may
    be present for audit traceability, but it is not the report/evaluation unit.
    y_pred is derived from y_prob at 0.5 when absent (see _ensure_y_pred).
    """
    required = {"hospitalization_join_id", "prediction_id", "split", meta["label_column"]}
    task_type = meta["task_type"]
    if task_type == "binary":
        required.add("y_prob")
    elif task_type == "multiclass":
        required.add("y_pred")
    elif task_type == "regression":
        required.add("y_pred_value")
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"Predictions parquet missing required columns for task "
            f"'{meta['name']}' ({task_type}): {sorted(missing)}. "
            f"Got columns: {sorted(preds.columns)}. "
            f"Every preds parquet must carry hospitalization_join_id, prediction_id, split, "
            f"{meta['label_column']}, and y_prob — add y_prob to the cohort file: "
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
        # NaN is neither null nor range-detectable: null_count() ignores it and
        # min()/max() skip it, so a NaN score clears both other guards here and
        # then dies deep inside sklearn with an unrelated-looking message.
        n_nan = int(preds["y_prob"].cast(pl.Float64, strict=False).is_nan().sum())
        if n_nan > 0:
            raise ValueError(
                f"Predictions parquet for task '{meta['name']}': y_prob must be a "
                f"probability in [0, 1], but {n_nan} of {preds.height} values are "
                f"NaN. Drop those rows or fill in a probability."
            )
        lo, hi = preds["y_prob"].min(), preds["y_prob"].max()
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(
                f"Predictions parquet for task '{meta['name']}': y_prob must be a "
                f"probability in [0, 1], but the observed range is [{lo}, {hi}]. "
                f"Pass calibrated probabilities (e.g. model.predict_proba(...)[:, 1]), "
                f"not logits or scores."
            )


def _validate_split_integrity(preds: pl.DataFrame, meta: dict) -> None:
    """A stitched encounter must belong to exactly one split.

    Continuous tasks copy or derive labels across multiple prediction windows, so
    putting rows from one hospitalization_join_id in both train and test leaks
    patient-level signal into the test set. Cohort generation prevents this; this
    guard catches hand-authored prediction files that bypass the cohort splitter.
    """
    bad = (
        preds.group_by("hospitalization_join_id")
        .agg(pl.col("split").drop_nulls().n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    if bad.height:
        examples = bad["hospitalization_join_id"].head(5).to_list()
        raise ValueError(
            f"Predictions parquet for task '{meta['name']}' has "
            f"{bad.height} hospitalization_join_id values appearing in multiple "
            f"splits (examples: {examples}). Split once per hospitalization_join_id; "
            f"do not split prediction rows from the same stay across train/test."
        )
