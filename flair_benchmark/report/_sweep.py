"""Vectorized threshold-sweep engine for the operating-points section.

ONE fixed, task- and site-invariant grid of 99 thresholds (0.01 … 0.99), swept
in a single sorted pass instead of a Python loop over thresholds. This is the
primitive behind ``operating_points.py``'s universal sweep (design doc §2):

  sweep_counts : sort the scores ONCE, then ``np.searchsorted`` reads the
                 cumulative TP/FP/FN/TN at all 99 thresholds at array speed —
                 the same trick ``sklearn.roc_curve`` uses, hand-rolled onto a
                 fixed threshold grid (not an FPR grid) so two sites' sweeps line
                 up row-for-row and pool without interpolation.
  sweep_metrics: the 2x2-derived rates (sensitivity / specificity / PPV / NPV /
                 F1 / FPR / flag-rate) as vectors, ``nan`` where a denominator is
                 zero. Formulas match ``_confusion.confusion_metric`` exactly.
  nb_curve     : net benefit across a decision-axis grid, vectorized — the same
                 ``TP/n - FP/n * t/(1-t)`` as ``_basis.net_benefit``, cutting the
                 acted arm on the same LABEL-scale grid, so the sweep and dca.json
                 agree by construction including at a tie.
  band_ci      : ONE shared bootstrap resample stream produces a 95% CI *band*
                 across the whole grid — a confidence band, not 99 independent
                 CIs. Mirrors ``_ci.bootstrap_ci`` exactly (same seed, one
                 ``rng.integers(0, n, n)`` index draw per replicate, same
                 without-replacement subsample above ``max_n``, same reps//2
                 valid-replicate floor), so the CI at any single threshold is
                 bit-identical to the old single-point bootstrap at that
                 threshold.

ALARM convention throughout the counts (``p >= t`` in both directions).
``nb_curve`` is the one direction-aware primitive: it cuts its acted arm with the
task's own rule and weighs it on the decision axis.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from flair_benchmark.report._ci import BOOT_MAX_N, BOOT_REPS, BOOT_SEED

# 0.01 … 0.99 inclusive, 99 points. Rounded so the array holds exact 2-dp values
# (float arange drifts) — the clinical threshold, always a multiple of 0.01,
# therefore compares equal to exactly one grid entry.
SWEEP_GRID: np.ndarray = np.round(np.arange(1, 100) / 100.0, 2)
GRID_N = len(SWEEP_GRID)

# the rate keys a bootstrap band is drawn for (nne is derived from the ppv band
# by inversion, not resampled; flag_rate is a workload count with no published CI)
RATE_KEYS = ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr")


def sweep_counts(label: np.ndarray, prob: np.ndarray,
                 grid: np.ndarray = SWEEP_GRID
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(tp, fp, fn, tn) int vectors over ``grid``, ALARM basis (``prob >= t``).

    Sort once; ``searchsorted`` gives, for every threshold at once, the split
    point below which scores fall under ``t``. ``cum_pos`` carries the running
    count of positives up that sorted order, so the positives at or above each
    threshold is a single fancy-index — no per-threshold data pass.
    """
    n = len(label)
    p_total = int(label.sum())
    order = np.argsort(prob, kind="stable")
    p_sorted = prob[order]
    y_sorted = label[order].astype(np.int64)
    cum_pos = np.concatenate(([0], np.cumsum(y_sorted)))   # positives with prob < grid[i]
    idx = np.searchsorted(p_sorted, grid, side="left")     # first index with prob >= t
    n_ge = n - idx                                         # stays with prob >= t (fired)
    tp = p_total - cum_pos[idx]
    fp = n_ge - tp
    fn = p_total - tp
    tn = n - n_ge - fn
    return tp, fp, fn, tn


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise num/den, ``nan`` where den == 0 (matches confusion_metric)."""
    out = np.full(len(num), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def sweep_metrics(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                  tn: np.ndarray) -> dict[str, np.ndarray]:
    """The published 2x2 rates as vectors (``nan`` on a zero denominator)."""
    tp = tp.astype(float); fp = fp.astype(float)
    fn = fn.astype(float); tn = tn.astype(float)
    n = tp + fp + fn + tn
    return {
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "ppv": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn),
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": _safe_ratio(fp, fp + tn),
        "flag_rate": _safe_ratio(tp + fp, n),
    }


def nb_curve(y_true_dec: np.ndarray, y_prob_label: np.ndarray,
             label_grid: np.ndarray, axis_grid: np.ndarray,
             direction: str) -> np.ndarray:
    """Net benefit ``TP/n - FP/n * t/(1-t)`` over a DECISION-axis grid, vectorized.

    The vectorized twin of ``_basis.net_benefit``, and it splits the two
    thresholds the same way: the acted arm is cut on ``label_grid`` (the LABEL
    scale, one searchsorted), and ``axis_grid`` — ``axis_threshold(t, direction)``,
    so ``t`` for "above" and ``1-t`` for "below" — supplies only the odds weight.
    ``y_true_dec`` is on the decision basis (``_basis.to_decision_basis``).

    Cutting on the label scale is what keeps this identical to ``net_benefit`` at
    a tie: the complemented probabilities are never compared (see the boundary
    convention in ``_basis``). For "above" the two grids are the same array and
    this reduces to the plain ``p >= t`` sweep.
    """
    n = len(y_true_dec)
    if n == 0:
        return np.full(len(axis_grid), np.nan)
    p_total = int(y_true_dec.sum())
    order = np.argsort(y_prob_label, kind="stable")
    p_sorted = y_prob_label[order]
    y_sorted = y_true_dec[order].astype(np.int64)
    cum_pos = np.concatenate(([0], np.cumsum(y_sorted)))
    idx = np.searchsorted(p_sorted, label_grid, side="left")   # count of p < t
    if direction == "below":                      # acted = p < t
        n_acted, tp = idx, cum_pos[idx]
    else:                                         # acted = p >= t
        n_acted, tp = n - idx, p_total - cum_pos[idx]
    fp = n_acted - tp
    ok = axis_grid < 1.0                          # t/(1-t) diverges at t == 1
    odds = np.where(ok, axis_grid / np.where(ok, 1.0 - axis_grid, 1.0), np.nan)
    nb = tp / n - (fp / n) * odds
    nb[~ok] = np.nan
    return nb


def arm_fire_rate(values: np.ndarray, grid: np.ndarray,
                  direction: str) -> np.ndarray:
    """Per-threshold fraction of an arm that ACTS (direction-aware), vectorized.

    ``values`` is one action-scalar per stay in the arm (max risk for an "above"
    task, min risk for "below" — the collapse that makes "ever acted at t" a
    single searchsorted). Empty arm → all ``nan``.
    """
    m = len(values)
    if m == 0:
        return np.full(len(grid), np.nan)
    s = np.sort(values)
    below = np.searchsorted(s, grid, side="left")    # values < t
    count = below if direction == "below" else (m - below)   # else values >= t
    return count / m


def band_ci(label: np.ndarray, prob_arrays: dict[str, np.ndarray],
            compute: Callable[[np.ndarray, dict[str, np.ndarray]], dict[str, np.ndarray]],
            keys: tuple[str, ...], grid_len: int = GRID_N,
            reps: int = BOOT_REPS, seed: int = BOOT_SEED,
            max_n: int = BOOT_MAX_N) -> dict[str, list]:
    """95% CI band across the grid for each key, from ONE resample stream.

    ``compute(label_resampled, {name: array_resampled})`` returns a dict of
    length-``grid_len`` vectors (``nan`` where a metric is undefined on that
    resample). Every array in ``prob_arrays`` is resampled by the SAME per-rep
    index as ``label`` (and each other), so metrics that share rows stay
    coherent. Reduction is the (2.5, 97.5) percentile with the ``reps//2``
    valid-replicate floor — a column that was undefined on too many resamples
    returns ``None``, exactly like ``_ci.bootstrap_ci`` / ``_confusion.pct``.
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(label)
    arrs = {k: np.asarray(v) for k, v in prob_arrays.items()}
    n = len(lab)
    if n == 0:
        return {k: [None] * grid_len for k in keys}
    if n > max_n:
        base = rng.choice(n, max_n, replace=False)
        lab = lab[base]
        arrs = {k: v[base] for k, v in arrs.items()}
        n = max_n

    mats = {k: np.full((reps, grid_len), np.nan) for k in keys}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        res = compute(lab[idx], {k: v[idx] for k, v in arrs.items()})
        for k in keys:
            mats[k][r] = res[k]

    out: dict[str, list] = {}
    floor = reps // 2
    for k in keys:
        m = mats[k]
        valid = (~np.isnan(m)).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")       # all-nan column -> nan (guarded below)
            lo = np.nanpercentile(m, 2.5, axis=0)
            hi = np.nanpercentile(m, 97.5, axis=0)
        out[k] = [None if valid[i] < floor
                  else [round(float(lo[i]), 4), round(float(hi[i]), 4)]
                  for i in range(grid_len)]
    return out
