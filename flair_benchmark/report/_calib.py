"""Shared calibration helpers (used by curves, fairness, tripod_ai).

All summaries here are model-derived (a 2-parameter logistic recalibration
or scalar means), so they carry no patient-level information and need no
cell suppression.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float | None, float | None]:
    """Logistic regression of y_true on logit(y_prob). Returns (slope, intercept).

    A perfectly calibrated model yields slope=1, intercept=0.
    """
    logit = _logit(y_prob)
    if len(np.unique(y_true)) < 2 or np.std(logit) == 0:
        return None, None
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y_true.astype(int))
    return round(float(lr.coef_[0, 0]), 4), round(float(lr.intercept_[0]), 4)


def calibration_extended(y_true: np.ndarray, y_prob: np.ndarray,
                         grid_points: int = 50) -> dict:
    """TRIPOD-style calibration summary + smoothed curve.

    slope / intercept : logistic recalibration (weak calibration)
    o_e_ratio         : observed / expected event ratio (calibration-in-the-large)
    brier             : Brier score
    smoothed          : a simple straight calibration line — least-squares fit of
                        observed event on predicted probability (no logistic
                        recalibration), evaluated on a fixed probability grid.
                        Model-derived (2 fitted params), PHI-safe.
    """
    from sklearn.metrics import brier_score_loss

    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)

    expected = float(y_prob.mean())
    observed = float(y_true.mean())
    o_e = round(observed / expected, 4) if expected > 0 else None

    out: dict = {
        "slope": slope,
        "intercept": intercept,
        "o_e_ratio": o_e,
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "smoothed": None,
    }
    if len(np.unique(y_true)) >= 2 and np.std(y_prob) > 0:
        # plain straight line: observed ~ a + b * predicted, clipped to [0, 1].
        b, a = np.polyfit(y_prob, y_true, 1)
        grid = np.linspace(0.01, 0.99, grid_points)
        line = np.clip(a + b * grid, 0.0, 1.0)
        out["smoothed"] = {
            "method": "linear_fit",
            "fit_slope": round(float(b), 4),
            "fit_intercept": round(float(a), 4),
            "pred_grid": [round(float(x), 4) for x in grid],
            "obs_smoothed": [round(float(x), 4) for x in line],
        }
    return out
