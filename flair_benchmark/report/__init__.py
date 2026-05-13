"""Report bundle builder. Emits 4 JSON files; never images."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark.report.curves import build_curves
from flair_benchmark.report.fairness import build_fairness
from flair_benchmark.report.table1 import build_table1
from flair_benchmark.report.tripod_ai import build_tripod_ai

__all__ = ["build_report"]


def build_report(preds_path: str, task_module, out_dir: str) -> dict[str, Path]:
    """Read preds parquet, look up META on task_module, write 4 JSONs."""
    from flair_benchmark import __version__
    from flair_benchmark._hash import task_script_sha256

    preds = pl.read_parquet(preds_path)
    meta = task_module.META
    _validate_preds_schema(preds, meta)

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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "table1": out / "table1.json",
        "tripod_ai": out / "tripod_ai.json",
        "fairness": out / "fairness.json",
        "curves": out / "curves.json",
    }
    paths["table1"].write_text(json.dumps(table1, indent=2, default=str))
    paths["tripod_ai"].write_text(json.dumps(tripod, indent=2, default=str))
    paths["fairness"].write_text(json.dumps(fairness, indent=2, default=str))
    paths["curves"].write_text(json.dumps(curves, indent=2, default=str))
    return paths


def _validate_preds_schema(preds: pl.DataFrame, meta: dict) -> None:
    """Enforce the strict preds parquet contract."""
    required = {"hospitalization_id", "split", meta["label_column"]}
    task_type = meta["task_type"]
    if task_type in ("binary", "multiclass"):
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
