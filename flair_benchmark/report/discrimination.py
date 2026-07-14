"""Discrimination → discrimination.json.

How well the model RANKS stays, independent of calibration:
  - AUROC, AUPRC (+ 95% bootstrap CI), AUPRC lift over prevalence.
  - ROC / PR curves as downsampled raw points for downstream plotting, plus a
    fixed-FPR-grid ROC form (`grid_fpr` / `tpr_at_grid`) for cross-site pooling.
  - operating_point: sensitivity / specificity / PPV / NPV / F1 + confusion
    counts at a threshold, each with a 95% CI. Emitted only when
    ``operating_point=True`` (the default, used by the continuous report's peak
    and landmark sections). The EPISODIC report passes False and hoists every
    threshold-dependent number into operating_points.json instead, which sweeps
    the clinical threshold ± 0.05 / ± 0.10 — so discrimination.json is pure,
    threshold-free ranking.

The frame is one row per stay (split-filtered upstream), so the row bootstrap in
report/_ci.py is a stay-level bootstrap.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report._suppress import suppress

MAX_POINTS = 500
GRID_FPR = np.round(np.arange(0.0, 1.0001, 0.01), 2)   # fixed, site-invariant


def build_discrimination(preds: pl.DataFrame, meta: dict,
                         operating_point: bool = True) -> dict:
    """Switch on task_type; emit ranking metrics + curves (+ CI for binary).

    ``operating_point=False`` drops the threshold-dependent block (the episodic
    report reports it in operating_points.json, across five thresholds).
    """
    task_type = meta["task_type"]
    label_col = meta["label_column"]
    out: dict = {"task_type": task_type, "metadata": {"n": preds.height,
                                                       "downsample_max": MAX_POINTS}}

    if task_type == "binary" and "y_prob" in preds.columns:
        y_true = preds[label_col].to_numpy().astype(int)
        y_prob = preds["y_prob"].to_numpy().astype(float)
        prevalence = round(float(y_true.mean()), 6) if len(y_true) else None
        # A single-class cohort (all events or all non-events) has no contrast:
        # AUROC/AUPRC and the ROC/PR curves are undefined. Mirror landmark.py's
        # guard so episodic/peak degrade to a marker instead of emitting nan
        # arrays + sklearn UndefinedMetricWarning.
        if len(np.unique(y_true)) < 2:
            out["prevalence"] = prevalence
            out["metadata"]["reason"] = "single label class (no positive/negative contrast)"
            return out
        roc_curve, roc_metrics = _roc_points(y_true, y_prob)
        pr_curve, pr_metrics = _pr_points(y_true, y_prob, prevalence)
        out["prevalence"] = prevalence
        # scalars (each with its 95% CI) first, raw curve arrays below.
        out["metrics"] = {**roc_metrics, **pr_metrics}
        if operating_point:
            out["metrics"]["operating_point"] = _operating_point(
                y_true, y_prob, meta)
        out["curves"] = {"roc": roc_curve, "pr": pr_curve}
    elif task_type == "multiclass":
        out["curves"] = {
            "multiclass": _multiclass_curves(preds, label_col, meta.get("class_names"))}
    elif task_type == "regression":
        out["curves"] = {"regression": _regression_points(preds, label_col)}
    return out


def _auroc(yt: np.ndarray, yp: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else None


def _auprc(yt: np.ndarray, yp: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score
    return average_precision_score(yt, yp) if len(np.unique(yt)) > 1 else None


def _roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[dict, dict]:
    """Return (curve, metrics): raw ROC arrays, and AUROC + 95% CI."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    auroc = _auroc(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob, {"auroc": _auroc})["auroc"]
    # fixed-grid TPR for cross-site vertical averaging (fpr is increasing).
    tpr_at_grid = np.interp(GRID_FPR, fpr, tpr)
    dfpr, dtpr, dthr = _downsample3(fpr, tpr, thr, MAX_POINTS)
    curve = {
        "fpr": _r(dfpr), "tpr": _r(dtpr), "thresholds": _r(dthr),
        "grid_fpr": _r(GRID_FPR), "tpr_at_grid": _r(tpr_at_grid),
    }
    metrics = {"auroc": None if auroc is None else round(float(auroc), 4),
               "auroc_ci": ci}
    return curve, metrics


def _pr_points(y_true: np.ndarray, y_prob: np.ndarray,
               prevalence: float | None) -> tuple[dict, dict]:
    """Return (curve, metrics): raw PR arrays, and AUPRC + 95% CI + lift."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thr with one fewer element than prec/rec; pad
    thr = np.concatenate([thr, [1.0]])
    rec, prec, thr = _downsample3(rec, prec, thr, MAX_POINTS)
    auprc = _auprc(y_true, y_prob)
    ci = bootstrap_ci(y_true, y_prob, {"auprc": _auprc})["auprc"]
    # baseline = prevalence (a no-skill classifier's precision); lift decouples
    # discrimination from how rare the event is — the cross-site comparable number.
    lift = (round(float(auprc) / prevalence, 2)
            if auprc is not None and prevalence else None)
    curve = {"recall": _r(rec), "precision": _r(prec), "thresholds": _r(thr)}
    metrics = {"auprc": None if auprc is None else round(float(auprc), 4),
               "auprc_ci": ci, "auprc_baseline": prevalence, "auprc_lift": lift}
    return curve, metrics


def _operating_point(y_true: np.ndarray, y_prob: np.ndarray, meta: dict,
                     t: float | None = None) -> dict:
    """Confusion-derived metrics at a threshold (label space).

    ``t`` defaults to the task's clinical threshold (META `clinical_threshold`).
    Pass it explicitly to evaluate the same block at another point on the grid —
    that is how operating_points.py sweeps clinical ± 0.05 / ± 0.10.

    ALARM CONVENTION: classifies y_prob >= t in BOTH directions, so sensitivity /
    specificity / PPV always mean "how well do we detect the event". The
    direction-aware clinical decision (which flips for a `below` task like task3,
    where the action fires on LOW risk) lives in the separate `action` block.

    Each rate carries a 95% bootstrap CI; confusion counts pass through
    suppress() for PHI safety.
    """
    if t is None:
        t = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    if t is None:
        return {"reason": "no clinical_threshold in META"}

    pred = y_prob >= t
    n = len(y_true)
    n_flag = int(pred.sum())
    tp = int((pred & (y_true == 1)).sum())
    fp = n_flag - tp
    fn = int((~pred & (y_true == 1)).sum())
    tn = n - n_flag - fn

    fns = {"sensitivity": _mk_stat(t, "sensitivity"),
           "specificity": _mk_stat(t, "specificity"),
           "ppv": _mk_stat(t, "ppv"), "npv": _mk_stat(t, "npv"),
           "f1": _mk_stat(t, "f1")}
    ci = bootstrap_ci(y_true, y_prob, fns)
    point = {k: _confusion_metric(tp, fp, fn, tn, k)
             for k in ("sensitivity", "specificity", "ppv", "npv", "f1")}
    return {
        "threshold": t,
        "direction": direction,
        # sens/spec/PPV/NPV/F1 are reported in risk-classifier convention
        # (predicted positive = y_prob >= t, label = event), independent of
        # clinical_direction — these are the standard model-performance numbers.
        **{k: _r1(v) for k, v in point.items()},
        **{f"{k}_ci": ci[k] for k in point},
        "flag_rate": round(n_flag / n, 4) if n else None,
        "tp": suppress(tp), "fp": suppress(fp),
        "fn": suppress(fn), "tn": suppress(tn),
        "n_flagged": n_flag,
        # the actual clinical decision (direction-aware): act on p<t when the
        # action fires for LOW risk (e.g. extubate when failure risk < t).
        "action": _action_block(y_true, y_prob, t, direction),
    }


def _action_block(y_true: np.ndarray, y_prob: np.ndarray, t: float,
                  direction: str) -> dict:
    """Direction-aware view of the decision actually taken at threshold t.

    `acted` = p < t when direction == "below" (act on low risk, e.g. extubate),
    else p >= t. Reports how many are acted on and the observed event rate in
    each arm — the numbers that flip with direction (unlike sens/spec above).
    """
    acted = (y_prob < t) if direction == "below" else (y_prob >= t)
    n = len(y_true)
    n_act = int(acted.sum())
    ev_act = float(y_true[acted].mean()) if n_act else None
    ev_not = float(y_true[~acted].mean()) if (n - n_act) else None
    return {
        "acts_on": "y_prob < threshold" if direction == "below"
                   else "y_prob >= threshold",
        "n_acted": suppress(n_act),
        "act_rate": round(n_act / n, 4) if n else None,
        "event_rate_in_acted": None if ev_act is None else round(ev_act, 4),
        "event_rate_in_not_acted": None if ev_not is None else round(ev_not, 4),
    }


def _confusion_metric(tp: int, fp: int, fn: int, tn: int, name: str) -> float | None:
    if name == "sensitivity":
        return tp / (tp + fn) if (tp + fn) else None
    if name == "specificity":
        return tn / (tn + fp) if (tn + fp) else None
    if name == "ppv":
        return tp / (tp + fp) if (tp + fp) else None
    if name == "npv":
        return tn / (tn + fn) if (tn + fn) else None
    if name == "f1":
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return None


def _mk_stat(t: float, name: str):
    """Bootstrap stat_fn: recompute one confusion metric at threshold t."""
    def _fn(yt: np.ndarray, yp: np.ndarray) -> float | None:
        pred = yp >= t
        n_flag = int(pred.sum())
        tp = int((pred & (yt == 1)).sum())
        fp = n_flag - tp
        fn = int((~pred & (yt == 1)).sum())
        tn = len(yt) - n_flag - fn
        return _confusion_metric(tp, fp, fn, tn, name)
    return _fn


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


def _r1(x) -> float | None:
    return None if x is None else round(float(x), 4)
