"""Calibration → calibration.json.

Whether predicted probabilities match observed event rates:
  - reliability bins (10 equal-width bins; bins with <10 stays have their
    observed rate suppressed to null for PHI safety) + ECE.
  - weak calibration: logistic slope / intercept and observed/expected ratio.
  - Brier score.
  - a simple straight calibration line (least-squares fit of observed on
    predicted; 2 fitted params, PHI-safe).
ECE, Brier, slope and intercept each carry a 95% bootstrap CI.

Episodic only: one row per stay (split-filtered upstream).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._calib import (
    calibration_extended,
    calibration_slope_intercept,
)
from flair_benchmark.report._ci import bootstrap_ci

N_BINS = 10


def build_calibration(preds: pl.DataFrame, meta: dict) -> dict:
    out: dict = {"task_type": meta["task_type"],
                 "metadata": {"n": preds.height, "n_bins": N_BINS,
                              # calibration is assessed across ALL predicted
                              # probabilities, not at the clinical operating point.
                              "threshold_used": None,
                              "note": "threshold-free (reliability across all "
                                      "predicted probabilities)"}}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "calibration requires binary task with y_prob"
        return out

    y_true = preds[meta["label_column"]].to_numpy().astype(int)
    y_prob = preds["y_prob"].to_numpy().astype(float)
    if len(y_true) == 0:
        out["metadata"]["reason"] = "empty split"
        return out

    bins = _calibration_bins(y_true, y_prob, n_bins=N_BINS)
    ext = calibration_extended(y_true.astype(float), y_prob)
    slope_stat, intercept_stat = _paired_slope_intercept_stats()
    ci = bootstrap_ci(y_true, y_prob, {
        "ece": lambda yt, yp: _ece(yt, yp, N_BINS),
        "brier": _brier,
        "slope": slope_stat,
        "intercept": intercept_stat,
    })

    out["prevalence"] = round(float(y_true.mean()), 6)
    # scalars (each with its 95% CI) first, raw curve arrays below.
    out["metrics"] = {
        "ece": bins["ece"],
        "ece_ci": ci["ece"],
        "brier": ext["brier"],
        "brier_ci": ci["brier"],
        "slope": ext["slope"],
        "slope_ci": ci["slope"],
        "intercept": ext["intercept"],
        "intercept_ci": ci["intercept"],
        "o_e_ratio": ext["o_e_ratio"],
    }
    out["curves"] = {"reliability": bins, "smoothed": ext["smoothed"]}
    return out


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
    return {"n_bins": n_bins, "bin_edges": [round(float(x), 6) for x in edges],
            "bin_pred_prob": bin_pred, "bin_true_prob": bin_true,
            "bin_counts": bin_count, "ece": round(ece, 4)}


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true.astype(float), y_prob))


def _paired_slope_intercept_stats():
    """(slope_stat, intercept_stat) sharing ONE logistic fit per resample.

    bootstrap_ci evaluates every stat on the SAME resampled array objects
    within a replicate, so a one-slot cache keyed on array identity lets the
    intercept stat reuse the slope stat's fit — halving the LogisticRegression
    fits (the dominant replicate cost; a landmark report runs this per slice).
    Holding a reference to the cached arrays guarantees their ids are not
    recycled across replicates.
    """
    cache: dict = {"arrays": None, "fit": (None, None)}

    def _fit(yt: np.ndarray, yp: np.ndarray) -> tuple[float | None, float | None]:
        arrays = cache["arrays"]
        if arrays is None or arrays[0] is not yt or arrays[1] is not yp:
            cache["arrays"] = (yt, yp)
            cache["fit"] = calibration_slope_intercept(yt, yp)
        return cache["fit"]

    return (lambda yt, yp: _fit(yt, yp)[0],
            lambda yt, yp: _fit(yt, yp)[1])
