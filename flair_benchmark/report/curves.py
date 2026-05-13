"""ROC / PR / calibration as downsampled raw points.

Sites never emit images — the central dashboard renders. JSON is kept
under ~200 kB by downsampling roc/pr to ≤500 points.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MAX_POINTS = 500


def build_curves(preds: pl.DataFrame, meta: dict) -> dict:
    """Switch on task_type and emit the relevant curve sections."""
    task_type = meta["task_type"]
    label_col = meta["label_column"]
    test = preds.filter(pl.col("split") == "test")

    out: dict = {"task_type": task_type, "metadata": {"n_test": test.height,
                                                       "downsample_max": MAX_POINTS}}

    if task_type == "binary" and "y_prob" in test.columns:
        y_true = test[label_col].to_numpy().astype(int)
        y_prob = test["y_prob"].to_numpy().astype(float)
        out["roc"] = _roc_points(y_true, y_prob)
        out["pr"] = _pr_points(y_true, y_prob)
        out["calibration"] = _calibration_bins(y_true, y_prob, n_bins=10)
    elif task_type == "multiclass":
        out["multiclass"] = _multiclass_curves(test, label_col, meta.get("class_names"))
    elif task_type == "regression":
        out["regression"] = _regression_points(test, label_col)
    return out


def _roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    fpr, tpr, thr = _downsample3(fpr, tpr, thr, MAX_POINTS)
    return {"fpr": _r(fpr), "tpr": _r(tpr), "thresholds": _r(thr)}


def _pr_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thr with one fewer element than prec/rec; pad
    thr = np.concatenate([thr, [1.0]])
    rec, prec, thr = _downsample3(rec, prec, thr, MAX_POINTS)
    return {"recall": _r(rec), "precision": _r(prec), "thresholds": _r(thr)}


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> dict:
    """Equal-width bins. Any bin with count<10 has bin_true_prob suppressed to null."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    bin_pred, bin_true, bin_count = [], [], []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        bin_count.append(c)
        if c == 0:
            bin_pred.append(None)
            bin_true.append(None)
            continue
        mean_pred = float(y_prob[mask].mean())
        mean_true = float(y_true[mask].mean())
        bin_pred.append(round(mean_pred, 4))
        bin_true.append(round(mean_true, 4) if c >= 10 else None)
        ece += (c / n) * abs(mean_true - mean_pred)
    return {"n_bins": n_bins, "bin_edges": _r(edges),
            "bin_pred_prob": bin_pred, "bin_true_prob": bin_true,
            "bin_counts": bin_count, "ece": round(ece, 4)}


def _multiclass_curves(test: pl.DataFrame, label_col: str,
                       class_names: list | None) -> dict:
    from sklearn.metrics import confusion_matrix, roc_curve
    y_true = test[label_col].to_numpy()
    y_pred = test["y_pred"].to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    display = ([class_names[int(i)] for i in labels]
               if class_names and all(isinstance(i, (int, np.integer)) for i in labels)
                  and max(labels) < len(class_names)
               else [str(c) for c in labels])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class: dict = {}
    for lbl, name in zip(labels, display):
        for cand in (f"class_probs_{lbl}", f"class_probs_{name}"):
            if cand in test.columns:
                prob_col = cand
                break
        else:
            continue
        y_bin = (y_true == lbl).astype(int)
        y_prob_c = test[prob_col].to_numpy().astype(float)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, thr = roc_curve(y_bin, y_prob_c)
        fpr, tpr, thr = _downsample3(fpr, tpr, thr, MAX_POINTS)
        per_class[name] = {"fpr": _r(fpr), "tpr": _r(tpr), "thresholds": _r(thr)}
    return {"class_order": display,
            "confusion_matrix": cm.tolist(),
            "per_class_roc": per_class}


def _regression_points(test: pl.DataFrame, label_col: str) -> dict:
    y_true = test[label_col].to_numpy().astype(float)
    y_pred = test["y_pred_value"].to_numpy().astype(float)
    residual = y_pred - y_true
    qs = np.linspace(0.0, 1.0, 21)
    return {
        "quantiles": _r(qs),
        "y_true_quantiles": _r(np.quantile(y_true, qs)),
        "y_pred_quantiles": _r(np.quantile(y_pred, qs)),
        "residual_quantiles": _r(np.quantile(residual, qs)),
    }


def _downsample3(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(a)
    if n <= k:
        return a, b, c
    idx = np.linspace(0, n - 1, k).astype(int)
    return a[idx], b[idx], c[idx]


def _r(xs) -> list:
    return [round(float(x), 6) for x in xs]
