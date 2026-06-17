"""Decision Curve Analysis → dca.json (binary tasks).

The frame is split-filtered upstream (episodic: one row per stay).

Brute-force sweep over threshold probabilities t = 0.01..0.99 (step 0.01):

  nb_model = TP/n - FP/n * t/(1-t)
  nb_all   = prevalence - (1-prevalence) * t/(1-t)   (treat everyone)
  nb_none  = 0                                        (treat no one)

`useful_range` is the widest contiguous threshold interval where the model
has strictly higher net benefit than both default strategies;
`all_useful_intervals` lists every such interval.

The net benefit at the task's clinical threshold carries a 95% bootstrap CI
(report/_ci.py). The full-curve CI band is left out by cost (one resample per
threshold × grid); add later if a band is needed.

Suppression: net benefit at a threshold flagging < 10 patients positive is
derived from too few patients — nulled and listed in suppressed_thresholds.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report._suppress import MIN_CELL_COUNT

# Uniform 0.01..0.99 grid, plus a denser low-threshold tail (0.001..0.05) so the
# rare-event decision region — where a 0.15%-prevalence model actually operates —
# is resolved instead of collapsing into one suppressed point near 0.
_LOW = np.round(np.arange(0.001, 0.05, 0.002), 3)
THRESHOLDS = np.unique(np.concatenate([_LOW, np.round(np.arange(0.01, 1.00, 0.01), 2)]))
_TOL = 1e-9


def build_dca(preds: pl.DataFrame, meta: dict) -> dict:
    out: dict = {"task_type": meta["task_type"], "metadata": {}}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "requires binary task with y_prob"
        return out

    y_true = preds[meta["label_column"]].to_numpy().astype(int)
    y_prob = preds["y_prob"].to_numpy().astype(float)
    n = len(y_true)
    if n == 0:
        out["metadata"]["reason"] = "empty split"
        return out
    prev = float(y_true.mean())

    nb_model: list[float | None] = []
    nb_all: list[float] = []
    suppressed: list[float] = []
    useful_mask: list[bool] = []
    for t in THRESHOLDS:
        odds = t / (1 - t)
        all_nb = prev - (1 - prev) * odds
        nb_all.append(round(all_nb, 6))
        pred_pos = y_prob >= t
        n_flagged = int(pred_pos.sum())
        if n_flagged < MIN_CELL_COUNT:
            nb_model.append(None)
            suppressed.append(float(t))
            useful_mask.append(False)
            continue
        tp = int((pred_pos & (y_true == 1)).sum())
        fp = n_flagged - tp
        model_nb = tp / n - (fp / n) * odds
        nb_model.append(round(model_nb, 6))
        useful_mask.append(model_nb > max(all_nb, 0.0) + _TOL)

    intervals = _contiguous_intervals(THRESHOLDS, useful_mask)
    clinical = _clinical_marker(meta, nb_model, nb_all, intervals, y_true, y_prob)
    # zoom the plot to where rare-event decisions live (treating at ~10x prevalence
    # already implies a high alarm rate), but always include the clinical threshold;
    # capped at 1.0. Keeps the full grid in JSON.
    clin = meta.get("clinical_threshold") or 0.0
    focus_hi = min(1.0, max(0.05, round(10 * prev, 3), round(clin + 0.05, 3)))
    useful_range = (max(intervals, key=lambda iv: iv["high"] - iv["low"])
                    if intervals else None)
    return {
        "task_type": "binary",
        "prevalence": round(prev, 4),
        # scalars (clinical operating point + CI, useful range) first, curves below.
        "metrics": {
            **clinical,
            "useful_range": useful_range,
            "all_useful_intervals": [[iv["low"], iv["high"]] for iv in intervals],
        },
        "curves": {
            "thresholds": [float(t) for t in THRESHOLDS],
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": 0.0,
            "focus_range": [0.0, focus_hi],
            "suppressed_thresholds": suppressed,
        },
        "metadata": {"grid": "0.001-0.05 step .002 + 0.01-0.99 step .01",
                     "min_flagged_n": MIN_CELL_COUNT, "n_test": n},
    }


def _clinical_marker(meta: dict, nb_model: list, nb_all: list,
                     intervals: list[dict], y_true: np.ndarray,
                     y_prob: np.ndarray) -> dict:
    """Net benefit at the task's clinical decision threshold (META) + 95% CI."""
    t = meta.get("clinical_threshold")
    if t is None:
        return {}
    idx = int(np.argmin(np.abs(THRESHOLDS - t)))
    ci = bootstrap_ci(y_true, y_prob,
                      {"model": lambda yt, yp: _net_benefit(yt, yp, t)})["model"]
    return {
        "clinical_threshold": t,
        "net_benefit_at_clinical_threshold": {
            "model": nb_model[idx],
            "model_ci": ci,
            "treat_all": nb_all[idx],
            "treat_none": 0.0,
        },
        "clinically_useful": any(iv["low"] <= t <= iv["high"] for iv in intervals),
    }


def _net_benefit(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> float | None:
    """Model net benefit at threshold t: TP/n - FP/n * t/(1-t)."""
    n = len(y_true)
    if n == 0 or t >= 1:
        return None
    pred_pos = y_prob >= t
    tp = int((pred_pos & (y_true == 1)).sum())
    fp = int(pred_pos.sum()) - tp
    return tp / n - (fp / n) * (t / (1 - t))


def _contiguous_intervals(thresholds: np.ndarray, mask: list[bool]) -> list[dict]:
    intervals: list[dict] = []
    start_idx = None
    for i, ok in enumerate(mask):
        if ok and start_idx is None:
            start_idx = i
        elif not ok and start_idx is not None:
            intervals.append({"low": float(thresholds[start_idx]),
                              "high": float(thresholds[i - 1])})
            start_idx = None
    if start_idx is not None:
        intervals.append({"low": float(thresholds[start_idx]),
                          "high": float(thresholds[-1])})
    return intervals
