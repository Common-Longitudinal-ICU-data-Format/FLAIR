"""TRIPOD-AI report skeleton, auto-populated from META + computed metrics.

Output sections mirror the TRIPOD-AI checklist:
  title, abstract, clinical_setting, prediction_task, intended_use,
  methods: { data, model, evaluation },
  results: { performance, calibration (binary only) },
  discussion: { limitations, generalizability },
  metadata: { ... }
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def build_tripod_ai(preds: pl.DataFrame, meta: dict, table1_json: dict) -> dict:
    """Return the populated TRIPOD-AI dict."""
    task_type = meta["task_type"]
    label_col = meta["label_column"]
    test = preds.filter(pl.col("split") == "test")

    if task_type == "binary":
        performance = _binary_metrics(test, label_col)
        calibration = _calibration_summary(test, label_col)
    elif task_type == "multiclass":
        performance = _multiclass_metrics(test, label_col, meta.get("class_names"))
        calibration = None
    elif task_type == "regression":
        performance = _regression_metrics(test, label_col)
        calibration = None
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return {
        "title": f"FLAIR Benchmark Report: {meta.get('display_name', meta['name'])}",
        "abstract": (
            f"Prediction model report for the FLAIR benchmark task "
            f"'{meta['name']}' ({task_type}). "
            f"N_total={preds.height}, N_test={test.height}."
        ),
        "clinical_setting": "Intensive Care Unit (ICU)",
        "prediction_task": meta.get("display_name", meta["name"]),
        "intended_use": meta.get("intended_use",
                                 "Research benchmark; not for clinical decision-making."),
        "methods": {
            "data": {
                "source": "CLIF-formatted ICU data",
                "n_total": preds.height,
                "n_train": preds.filter(pl.col("split") == "train").height,
                "n_test": test.height,
                "inclusion_criteria": meta.get("inclusion_criteria", []),
                "exclusion_criteria": meta.get("exclusion_criteria", []),
                "input_window_hours": meta.get("input_window_hours"),
                "prediction_window": meta.get("prediction_window"),
            },
            "model": {"type": "unknown", "features": "unknown",
                      "preprocessing": "unknown", "missing_data_handling": "unknown"},
            "evaluation": {
                "task_type": task_type,
                "validation_strategy": "temporal split",
                "metrics_computed": meta.get("evaluation_metrics", []),
            },
        },
        "results": {
            "performance": performance,
            "calibration": calibration,
        },
        "discussion": {
            "limitations": [
                "Single-site model performance may not generalize.",
                "Temporal split does not capture future distribution shift.",
                "CLIF variables limited to harmonized subset.",
            ],
            "generalizability":
                "Performance at other CLIF sites may differ; federated evaluation recommended.",
        },
    }


def _binary_metrics(test: pl.DataFrame, label_col: str) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    y_true = test[label_col].to_numpy()
    y_pred = test["y_pred"].to_numpy()
    y_prob = test["y_prob"].to_numpy() if "y_prob" in test.columns else None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    out: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
        "npv": round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if y_prob is not None:
        out["auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        out["auprc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    return out


def _multiclass_metrics(test: pl.DataFrame, label_col: str,
                        class_names: list | None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    y_true = test[label_col].to_numpy()
    y_pred = test["y_pred"].to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    display = ([class_names[int(i)] for i in labels]
               if class_names and all(isinstance(i, (int, np.integer)) for i in labels)
                  and max(labels) < len(class_names)
               else [str(c) for c in labels])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {
            display[i]: {"precision": round(float(p[i]), 4),
                         "recall": round(float(r[i]), 4),
                         "f1": round(float(f[i]), 4)}
            for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "class_order": display,
    }


def _regression_metrics(test: pl.DataFrame, label_col: str) -> dict:
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    y_true = test[label_col].to_numpy()
    y_pred = test["y_pred_value"].to_numpy()
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": round(mse, 4),
        "rmse": round(mse ** 0.5, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "explained_variance": round(float(explained_variance_score(y_true, y_pred)), 4),
    }


def _calibration_summary(test: pl.DataFrame, label_col: str,
                         n_bins: int = 10) -> dict | None:
    """Brief reliability summary; full bins live in curves.json."""
    if "y_prob" not in test.columns:
        return None
    y_true = test[label_col].to_numpy().astype(float)
    y_prob = test["y_prob"].to_numpy().astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())

    from flair_benchmark.report._calib import calibration_extended
    ext = calibration_extended(y_true, y_prob)
    return {"n_bins": n_bins, "ece": round(float(ece), 4),
            "slope": ext["slope"], "intercept": ext["intercept"],
            "o_e_ratio": ext["o_e_ratio"], "brier": ext["brier"]}
