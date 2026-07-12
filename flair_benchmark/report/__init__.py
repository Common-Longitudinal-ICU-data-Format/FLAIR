"""Report bundle builder, keyed on the unit of prediction (`--mode`).

  episodic   : one prediction per stay -> discrimination, calibration, dca, fairness.
  continuous : multi-row-per-stay task -> ONE bundle holding the stay-peak view
               (discrimination + fairness + NNE, as a "peak" section) AND the
               per-landmark four pillars (META['landmark_horizon'] landmarks in
               the task's window_unit) AND leadtime + alert-vs-intervention
               timing. See report/continuous.py.

All output is JSON with raw curve arrays + 95% CIs, at the task's fixed clinical
threshold. Only JSON is exported to the FLAIR website (cross-site aggregation).

Optional --viz renders local site-validation PNGs from the same JSON dicts
(see report/_viz.py) — never exported.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark.report.episodic import build_episodic_report

__all__ = ["build_report"]

# cohort columns the report can use when they flow through (or are joined in).
# Demographics feed fairness.py; without them fairness emits no subgroups.
# (The Table 1 is generated at task-generation time — see flair_benchmark/_table1.py.)
_CONTEXT_COLS = ["window_number", "prediction_dttm", "admission_dttm",
                 "hospitalization_join_id",
                 "age_at_admission", "sex_category", "race_category",
                 "ethnicity_category"]


def build_report(preds_path: str, task_module, out_dir: str,
                 episodic: bool = False,
                 cohort_path: str | None = None,
                 viz: bool = True, site: str | None = None,
                 mode: str | None = None) -> dict[str, Path]:
    """Read preds parquet, look up META, write the report JSON bundle.

    Reports are ALWAYS computed on the held-out `test` split — the home site trains on
    `train`, every site (incl. inference-only) is scored on `test`, so cross-site
    numbers are comparable. `mode` is the unit of prediction:

      episodic   : one prediction per stay -> discrimination/calibration/dca/fairness.
      continuous : multi-row-per-stay -> one bundle: peak section + per-landmark
                   pillars (META['landmark_horizon'] landmarks) + leadtime with
                   the alert-vs-intervention timing arms.

    `episodic=True` is a deprecated alias for `mode="episodic"`. When neither is
    given, the mode is inferred from META.temporal_pattern (episode -> episodic;
    continuous -> continuous).
    """
    from flair_benchmark import __version__
    from flair_benchmark._hash import task_script_sha256

    preds = pl.read_parquet(preds_path)
    meta = task_module.META
    _validate_preds_schema(preds, meta)
    _validate_split_integrity(preds, meta)
    preds = _attach_cohort_context(preds, cohort_path, meta)
    _validate_intervention_column(preds, meta)
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
    mode = _resolve_mode(mode, episodic, meta.get("temporal_pattern"))

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
        # operating threshold the threshold-dependent sections (discrimination
        # confusion point, dca clinical point, fairness) were evaluated at;
        # null for tasks with no clinical_threshold in META.
        "evaled_at": meta.get("clinical_threshold"),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if mode == "continuous":
        from flair_benchmark.report.continuous import build_continuous_report
        payloads = build_continuous_report(preds, meta)
    else:
        payloads = build_episodic_report(preds, meta)
    for payload in payloads.values():
        payload["metadata"] = {**audit, **payload.get("metadata", {})}

    paths: dict[str, Path] = {}
    for kind, payload in payloads.items():
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(payload, indent=2, default=str))

    if viz:
        try:
            if mode == "continuous":
                from flair_benchmark.report._viz import render_continuous_pngs
                pngs = render_continuous_pngs(payloads, out / "viz")
            else:
                from flair_benchmark.report._viz import render_sanity_pngs
                pngs = render_sanity_pngs(payloads, out / "viz")
            for png in pngs:
                paths[png.stem + "_png"] = png
        except ImportError:
            # viz is default-on but matplotlib is an optional extra; the JSON
            # bundle (the exported artifact) is already written, so skip PNGs.
            warnings.warn(
                "matplotlib not installed; skipping --viz PNGs "
                "(pip install 'flair-benchmark[viz]', or pass --no-viz).",
                stacklevel=2)
    return paths


def _resolve_mode(mode: str | None, episodic: bool,
                  pattern: str | None) -> str:
    """Pick episodic / continuous from the explicit flag, alias, or META."""
    if mode is None and episodic:
        mode = "episodic"            # deprecated --episodic alias
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
            f"--mode {mode} was merged into --mode continuous: one bundle with "
            f"the peak section + per-landmark pillars + leadtime + "
            f"alert-vs-intervention timing.")
    if mode not in {"episodic", "continuous"}:
        raise ValueError(
            f"Unknown --mode '{mode}'; expected episodic or continuous.")
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
    Missing context never fails — temporal sections degrade gracefully — EXCEPT
    the META-declared intervention column, which is enforced afterwards by
    `_validate_intervention_column` (the report only ever consumes cohort
    columns, never raw CLIF data, so it must arrive here or already be on preds).
    """
    context_cols = list(_CONTEXT_COLS)
    if meta.get("intervention"):
        context_cols.append(meta["intervention"]["column"])
    needed = [c for c in context_cols if c not in preds.columns]
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
        lo, hi = preds["y_prob"].min(), preds["y_prob"].max()
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(
                f"Predictions parquet for task '{meta['name']}': y_prob must be a "
                f"probability in [0, 1], but the observed range is [{lo}, {hi}]. "
                f"Pass calibrated probabilities (e.g. model.predict_proba(...)[:, 1]), "
                f"not logits or scores."
            )


def _validate_intervention_column(preds: pl.DataFrame, meta: dict) -> None:
    """META-declared intervention column must be present after the cohort join.

    The column (e.g. task4's `first_abx_dttm`) is generated at task-BUILD time
    and stored in the cohort parquet; the report never touches raw CLIF data.
    Null values are fine (never-treated stays are the no-intervention arm), but
    a missing column means the eval cannot answer "did the model fire before
    the intervention" — fail loudly rather than silently degrade.
    """
    iv = meta.get("intervention")
    if iv and iv["column"] not in preds.columns:
        raise ValueError(
            f"Task '{meta['name']}' declares META['intervention'] on column "
            f"'{iv['column']}', but it is missing from the predictions parquet. "
            f"Regenerate the cohort with this flair version (the task build "
            f"emits it) and either carry the column through your predictions "
            f"or pass the cohort parquet via --cohort so it can be joined in."
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
