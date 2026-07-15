"""Shared confusion-matrix arithmetic + small numeric helpers.

Single owner for the 2x2-derived rates the report publishes, the bootstrap
stat factory that recomputes them per resample, and the tiny rounding /
percentile / NNE helpers that were previously copy-pasted per module. All
flagging here is the ALARM convention (``y_prob >= t``; see report/_basis.py) —
that is what keeps "sensitivity" meaning "how well do we detect the event"
across tasks and report modes.
"""

from __future__ import annotations

import numpy as np

from flair_benchmark.report._basis import alarm_mask


def confusion_counts(y_true: np.ndarray,
                     flagged: np.ndarray) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) of a boolean flagged mask against a 0/1 label."""
    n_flag = int(flagged.sum())
    tp = int((flagged & (y_true == 1)).sum())
    fp = n_flag - tp
    fn = int((~flagged & (y_true == 1)).sum())
    tn = len(y_true) - n_flag - fn
    return tp, fp, fn, tn


def confusion_metric(tp: int, fp: int, fn: int, tn: int,
                     name: str) -> float | None:
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
    if name == "fpr":
        return fp / (fp + tn) if (fp + tn) else None
    return None


def mk_stat(t: float, name: str):
    """Bootstrap stat_fn: recompute one confusion metric at threshold t.

    ALARM convention (``yp >= t``). Also serves collapsed fired-as-0/1 arrays:
    ``mk_stat(0.5, name)`` on a {0.0, 1.0} column recovers the fired mask.
    """
    def _fn(yt: np.ndarray, yp: np.ndarray) -> float | None:
        tp, fp, fn, tn = confusion_counts(yt, alarm_mask(yp, t))
        return confusion_metric(tp, fp, fn, tn, name)
    return _fn


def nne_from_ppv(ppv: float | None,
                 ci: list[float] | None) -> tuple[float | None, list | None]:
    """NNE = 1/PPV (alarms per true positive); CI is the inverted PPV CI.

    1/x is decreasing, so [ppv_lo, ppv_hi] -> [1/ppv_hi, 1/ppv_lo]; a zero
    lower PPV bound means an unbounded NNE upper bound (null).
    """
    if not ppv or ppv <= 0:
        return None, None
    nne = round(1.0 / ppv, 1)
    if ci and ci[0] is not None and ci[1] is not None:
        hi = round(1.0 / ci[0], 1) if ci[0] > 0 else None
        lo = round(1.0 / ci[1], 1) if ci[1] > 0 else None
        return nne, [lo, hi]
    return nne, None


def r1(x) -> float | None:
    """Round to 4 dp, passing None through — THE report scalar precision."""
    return None if x is None else round(float(x), 4)


def pct(vals: list[float], reps: int) -> list[float] | None:
    """Percentile (2.5, 97.5) over bootstrap samples; None below the reps//2
    valid-replicate floor (same rule as _ci.bootstrap_ci)."""
    if len(vals) < reps // 2:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]
