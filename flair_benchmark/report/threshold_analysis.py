"""Fixed clinical-threshold analysis → threshold_analysis.json.

Each binary task carries a clinically consulted decision threshold in META
(e.g. sepsis: treat with antibiotics when predicted risk > 10%; extubation:
extubate when failure risk < 15%). DCA sweeps every threshold; clinicians
act at ONE — this report evaluates the model exactly there:

  label_metrics    sensitivity/specificity/PPV/NPV/F1 + confusion matrix for
                   the positive label, classifying y_prob >= t. Always in
                   label space, regardless of action direction.
  action_metrics   the decision actually taken: act when y_prob > t
                   (direction "above") or y_prob < t (direction "below",
                   e.g. extubate when failure risk is LOW). Reports how many
                   patients are acted on and the observed event rate among
                   them.
  partial_auroc    AUROC restricted to the ROC segment spanned by score
                   thresholds within ±0.05 of t — discrimination *near the
                   operating point*, where it clinically matters. Both raw
                   (area / band width) and McClish-standardized (0.5 =
                   chance, 1.0 = perfect) values.
  sweep            label metrics at t ± 0.02 and ± 0.05, showing how fragile
                   the operating point is.

Suppression: confusion cells pass through suppress(); the block is nulled
entirely when fewer than 10 patients fall on either side of the threshold.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress

PAUC_BAND = 0.05
SWEEP_DELTAS = (-0.05, -0.02, 0.02, 0.05)


def build_threshold_analysis(preds: pl.DataFrame, meta: dict) -> dict:
    out: dict = {"task_type": meta["task_type"], "metadata": {"pauc_band": PAUC_BAND}}
    threshold = meta.get("clinical_threshold")
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or threshold is None:
        out["metadata"]["reason"] = "requires binary task with y_prob and clinical_threshold"
        return out

    test = preds.filter(pl.col("split") == "test")
    y_true = test[meta["label_column"]].to_numpy().astype(int)
    y_prob = test["y_prob"].to_numpy().astype(float)
    direction = meta.get("clinical_direction", "above")

    out.update({
        "clinical_threshold": threshold,
        "clinical_action": meta.get("clinical_action"),
        "clinical_direction": direction,
        "label_metrics": _label_metrics(y_true, y_prob, threshold),
        "action_metrics": _action_metrics(y_true, y_prob, threshold, direction),
        "partial_auroc": _partial_auroc(y_true, y_prob, threshold),
        "sweep": [
            {"threshold": round(threshold + d, 4),
             **(_label_metrics(y_true, y_prob, threshold + d) or {"suppressed": True})}
            for d in SWEEP_DELTAS if 0 < threshold + d < 1
        ],
    })
    return out


def _label_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float) -> dict | None:
    """Confusion metrics for label=1 classifying at y_prob >= threshold."""
    pred = y_prob >= threshold
    n = len(y_true)
    n_flagged = int(pred.sum())
    if n_flagged < MIN_CELL_COUNT or (n - n_flagged) < MIN_CELL_COUNT:
        return None
    tp = int((pred & (y_true == 1)).sum())
    fp = n_flagged - tp
    fn = int((~pred & (y_true == 1)).sum())
    tn = n - n_flagged - fn
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {
        "sensitivity": _r4(sens), "specificity": _r4(spec),
        "ppv": _r4(ppv), "npv": _r4(npv),
        "accuracy": _r4((tp + tn) / n), "f1": _r4(f1),
        "tp": suppress(tp), "fp": suppress(fp),
        "fn": suppress(fn), "tn": suppress(tn),
        "n_flagged": n_flagged, "flag_rate": _r4(n_flagged / n),
    }


def _action_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float, direction: str) -> dict | None:
    """The clinical decision itself: act on patients past the threshold."""
    act = y_prob > threshold if direction == "above" else y_prob < threshold
    n = len(y_true)
    n_act = int(act.sum())
    if n_act < MIN_CELL_COUNT or (n - n_act) < MIN_CELL_COUNT:
        return None
    return {
        "n_act": n_act,
        "act_rate": _r4(n_act / n),
        "event_rate_in_acted": _r4(float(y_true[act].mean())),
        "event_rate_in_not_acted": _r4(float(y_true[~act].mean())),
    }


def _partial_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float) -> dict | None:
    """AUROC over the ROC segment whose score thresholds lie in t ± PAUC_BAND.

    Trapezoidal area between the FPR bounds reached at (t+band) and (t-band),
    normalized by the FPR span, plus the McClish (1989) standardization so
    0.5 = chance and 1.0 = perfect within the band.
    """
    from sklearn.metrics import roc_curve

    if y_true.min() == y_true.max():
        return None
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # roc_curve thresholds are decreasing; FPR is increasing
    lo_fpr = float(np.interp(-(threshold + PAUC_BAND), -thr, fpr))
    hi_fpr = float(np.interp(-(threshold - PAUC_BAND), -thr, fpr))
    if hi_fpr - lo_fpr < 1e-6:
        return {"value": None, "reason": "no ROC support within band",
                "band": [round(threshold - PAUC_BAND, 4),
                         round(threshold + PAUC_BAND, 4)]}
    grid = np.linspace(lo_fpr, hi_fpr, 100)
    tpr_grid = np.interp(grid, fpr, tpr)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 compat
    area = float(trapezoid(tpr_grid, grid))
    span = hi_fpr - lo_fpr
    mean_tpr = area / span
    # McClish: rescale against the chance diagonal within the same FPR window
    chance_area = (lo_fpr + hi_fpr) / 2 * span
    max_area = span
    standardized = 0.5 * (1 + (area - chance_area) / (max_area - chance_area)) \
        if max_area > chance_area else None
    return {
        "band": [round(threshold - PAUC_BAND, 4), round(threshold + PAUC_BAND, 4)],
        "fpr_range": [round(lo_fpr, 4), round(hi_fpr, 4)],
        "mean_tpr_in_band": _r4(mean_tpr),
        "pauc_standardized": _r4(standardized),
    }


def _r4(x) -> float | None:
    return None if x is None else round(float(x), 4)
