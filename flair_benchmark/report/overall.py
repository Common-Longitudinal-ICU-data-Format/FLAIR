"""Overall model quality -> overall.json (episodic mode).

Threshold-FREE. Two questions, and nothing else:

  discrimination  does the model RANK stays correctly — AUROC, AUPRC, and the
                  two curves behind them.
  calibration     do the probabilities MEAN what they say — Brier, the logistic
                  slope/intercept, and a reliability curve.

Every number that moves when the decision threshold moves lives in
hospitalization_level.json instead, including net benefit: the decision curve IS
``net_benefit`` + ``net_benefit_treat_all`` across that file's 99 rows, so
publishing a separate DCA block would be storing the same array twice.

CURVES ARE ONE ARRAY EACH. ROC is stored as TPR at a fixed FPR grid, PR as
precision at a fixed recall grid — both on ``GRID`` (0.00 .. 1.00 step 0.01).
The x-axes are known constants, so they are never written down. Two consequences:
the payload drops from thousands of points to 202 numbers, and two sites' curves
can be averaged elementwise without interpolating one onto the other's
thresholds, which variable-length curves cannot.

CALIBRATION USES EQUAL-COUNT (QUANTILE) BINS, not equal-width. For a rare event
almost every prediction falls in the first equal-width bin, so the curve
collapses to one point; equal-count binning puts ~n/10 stays in each bin and
spreads ten readable points across the range the model actually uses. Only the
two probability arrays are published — no counts, no edges — and a bin holding
fewer than MIN_CELL_COUNT stays has its OBSERVED rate nulled (the predicted
mean is model-derived and carries no patient information).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._stats import (
    MIN_CELL_COUNT,
    bootstrap_ci,
)
from flair_benchmark.report.landmark import logistic_slope_intercept

# The one fixed axis both curves are sampled on: 0.00 .. 1.00, step 0.01.
GRID: list[float] = [round(x, 2) for x in np.arange(0, 101) / 100.0]
_GRID = np.array(GRID)

N_BINS = 10


def build_overall(preds: pl.DataFrame, meta: dict) -> dict:
    """Ranking + calibration, with no threshold anywhere in sight."""
    out: dict = {
        "task_type": meta["task_type"],
        "metadata": {
            "roc_grid": "tpr sampled at fpr 0.00..1.00 step 0.01",
            "pr_grid": "precision sampled at recall 0.00..1.00 step 0.01",
            "n_bins": N_BINS,
            "binning": "quantile (equal-count)",
            "min_cell_count": MIN_CELL_COUNT,
        },
    }
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "overall requires a binary task with y_prob"
        return _degenerate(out, preds)

    y = (preds[meta["label_column"]].to_numpy() > 0).astype(int)
    p = preds["y_prob"].to_numpy().astype(float)
    out["n"] = len(y)
    out["prevalence"] = round(float(y.mean()), 6) if len(y) else None

    if len(y) == 0 or len(np.unique(y)) < 2:
        out["metadata"]["reason"] = (
            "single label class (no positive/negative contrast)")
        return _degenerate(out, preds)

    out["discrimination"] = _discrimination(y, p, out["prevalence"])
    out["calibration"] = _calibration(y, p)
    return out


def _degenerate(out: dict, preds: pl.DataFrame) -> dict:
    """Emit the full shape with nulls, so a reader never special-cases keys."""
    out.setdefault("n", preds.height)
    out.setdefault("prevalence", None)
    out["discrimination"] = {
        "auroc": None, "auroc_ci": None, "auprc": None, "auprc_ci": None,
        "auprc_baseline": out.get("prevalence"),
        "tpr_at_grid": None, "precision_at_grid": None,
    }
    out["calibration"] = {
        "brier": None, "brier_ci": None, "slope": None, "slope_ci": None,
        "intercept": None, "intercept_ci": None,
        "quantile_bins": {"bin_pred_prob": None, "bin_true_prob": None},
    }
    return out


# ------------------------------------------------------------- discrimination


def _discrimination(y: np.ndarray, p: np.ndarray,
                    prevalence: float | None) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    ci = bootstrap_ci(y, p, {
        "auroc": lambda yt, yp: (roc_auc_score(yt, yp)
                                 if len(np.unique(yt)) > 1 else None),
        "auprc": lambda yt, yp: (average_precision_score(yt, yp)
                                 if len(np.unique(yt)) > 1 else None),
    })

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    return {
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "auroc_ci": ci["auroc"],
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auprc_ci": ci["auprc"],
        "auprc_baseline": prevalence,
        "tpr_at_grid": _r(sample_curve(fpr, tpr)),
        "precision_at_grid": _r(sample_curve(rec, prec)),
    }


def sample_curve(x: np.ndarray, y: np.ndarray,
                 grid: np.ndarray = _GRID) -> np.ndarray:
    """Sample a (possibly multivalued) curve onto the fixed grid.

    BOTH curves are multivalued at their endpoints, and it is not an edge case:
    at FPR 0 every top-ranked-positive gives another valid TPR (the ROC's
    vertical rise), and at recall 1.0 every threshold that still catches all
    events gives another valid precision. Feeding duplicate x values straight to
    ``np.interp`` makes the answer depend on which duplicate it happens to
    land on — undocumented, and it silently differs between the two curves.

    So collapse to ONE y per x first, taking the MAXIMUM: the upper envelope is
    the conventional reading of both curves (best TPR achievable at that false
    alarm rate; best precision achievable at that recall), and it makes the
    output a deterministic function of the data rather than of tie handling.
    """
    order = np.lexsort((y, x))              # x ascending, y ascending within x
    xs, ys = x[order], y[order]
    keep = np.append(np.diff(xs) > 0, True)  # last (== max y) of each x run
    return np.interp(grid, xs[keep], ys[keep])


# --------------------------------------------------------------- calibration


def _calibration(y: np.ndarray, p: np.ndarray) -> dict:
    slope, intercept = logistic_slope_intercept(y, p)
    ci = bootstrap_ci(y, p, {
        "brier": lambda yt, yp: float(((yp - yt) ** 2).mean()),
        "slope": lambda yt, yp: logistic_slope_intercept(yt, yp)[0],
        "intercept": lambda yt, yp: logistic_slope_intercept(yt, yp)[1],
    })
    return {
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "brier_ci": ci["brier"],
        "slope": None if slope is None else round(slope, 4),
        "slope_ci": ci["slope"],
        "intercept": None if intercept is None else round(intercept, 4),
        "intercept_ci": ci["intercept"],
        "quantile_bins": _quantile_bins(y, p),
    }


def _quantile_bins(y: np.ndarray, p: np.ndarray,
                   n_bins: int = N_BINS) -> dict:
    """Equal-COUNT reliability points: mean predicted vs mean observed per bin.

    Counts and edges are deliberately not published — the two probability arrays
    are all a reliability plot needs, and a per-bin count is one more small
    number about real patients for no analytic gain. A bin below MIN_CELL_COUNT
    still nulls its OBSERVED rate.
    """
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    # assign by the interior edges; clip guards the closed upper endpoint.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    pred: list[float | None] = []
    true: list[float | None] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            pred.append(None)
            true.append(None)
            continue
        pred.append(round(float(p[mask].mean()), 4))
        true.append(round(float(y[mask].mean()), 4)
                    if count >= MIN_CELL_COUNT else None)
    return {"bin_pred_prob": pred, "bin_true_prob": true}


def _r(xs) -> list[float]:
    return [round(float(x), 6) for x in xs]
