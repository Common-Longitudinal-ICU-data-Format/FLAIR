"""The statistical primitives every report section shares.

Four groups of small, widely-used helpers. They live in one file because they are
all "the arithmetic the report agrees on" — the point is that there is exactly ONE
definition of each, so two sections can never disagree about what sensitivity
means or how wide a confidence interval is.

    Part 1  suppression   HIPAA cell-count masking            (used by ~10 modules)
    Part 2  bootstrap     95% confidence intervals            (used by ~10 modules)
    Part 3  confusion     2x2 arithmetic and rounding         (used by ~5 modules)
    Part 4  calibration   logistic recalibration summaries    (used by ~3 modules)

Everything here is either a pure function of its arguments or a fixed constant.

Depends on: report/_basis.py (the alarm convention), numpy, sklearn.
Used by: every report section, plus cohort/table1.py for suppression only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from flair_benchmark.report._basis import alarm_mask

# ===========================================================================
# Part 1 — HIPAA cell-count suppression
#
# Any count below 10 is published as the string '<10' rather than the number.
# ===========================================================================

MIN_CELL_COUNT = 10


def suppress(count: int, threshold: int = MIN_CELL_COUNT) -> int | str:
    """Return the count, or the string '<{threshold}' if below threshold."""
    return count if count >= threshold else f"<{threshold}"


def suppress_dict(counts: dict, threshold: int = MIN_CELL_COUNT) -> dict:
    """Vectorized version for {category: count} dicts."""
    return {k: suppress(int(v), threshold) for k, v in counts.items()}


def suppress_confusion(tp: int, fp: int, fn: int, tn: int,
                       threshold: int = MIN_CELL_COUNT) -> dict:
    """The 2x2 as a publishable dict, with SECONDARY suppression applied.

    Masking cells one at a time is not enough here, because the four cells are
    not independent of what the report publishes alongside them:

        tp + fp + fn + tn == n            (the stratum size)
        tp + fp           == n_flagged    (recoverable from flag_rate x n)

    So a single masked cell is recoverable by subtraction, and the mask
    accomplishes nothing. This applies the standard second step: whenever a cell
    is masked, mask its PARTNER in the same total as well — tp with fp, fn with
    tn — so each published total is short two unknowns and neither can be solved
    for.

    A cell of exactly 0 is published as 0 and never triggers partner masking. A
    true zero identifies nobody (the same rule `table1._prevalence` applies to a
    0.0 prevalence), and hiding it would be pointless anyway: with the totals in
    the clear, "the other three sum to n" gives it away immediately.

    Returns {"tp", "fp", "fn", "tn"}, each an int or the string '<threshold'.
    """
    cells = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    disclosive = {k: 0 < v < threshold for k, v in cells.items()}
    masked = {
        k
        for a, b in (("tp", "fp"), ("fn", "tn"))
        if disclosive[a] or disclosive[b]
        for k in (a, b)
        if cells[k] > 0
    }
    return {k: (f"<{threshold}" if k in masked else v) for k, v in cells.items()}


# ===========================================================================
# Part 2 — Generalized percentile bootstrap for 95% confidence intervals
#
# One resample index is drawn per replicate and reused across every statistic, so
# the returned CIs are mutually coherent (computed on the same resampled cohort).
#
# Resampling unit: ROWS. In the episodic report every row IS one stay, so a row
# bootstrap is already a stay-clustered (patient-level) bootstrap — the honest
# unit for ICU benchmark CIs. Do not reuse this on a row-pooled continuous task,
# where rows are correlated within a stay and a row bootstrap would understate
# the interval.
#
# CALLER CONTRACT: the frame you hand this must have a DETERMINISTIC row order.
#
# The seed fixes the resample *indices*, not the *patients* they land on. A polars
# group_by returns rows in a nondeterministic order, so a continuous task that
# collapses to one-row-per-stay via group_by and then bootstraps was drawing the
# same indices into a differently-ordered frame on every run — i.e. resampling a
# different set of patients each time. The point estimates (AUROC, PPV, ...) are
# order-invariant and looked perfectly stable, which is exactly what made this
# hard to see: only the CIs moved, and the fixed seed made them *look*
# reproducible.
#
# Every collapse feeding a bootstrap therefore ends in .sort(<stay id>) — see
# continuous._collapse_to_peak, operating_points._collapse_stays,
# continuous._median_first_alert. Sorting on the stay id (rather than
# maintain_order=True) also makes the CIs independent of the row order a site
# happened to write its preds parquet in: same data => same intervals, across
# sites and across runs.
# ===========================================================================

BOOT_REPS = 500
BOOT_SEED = 20260612
BOOT_MAX_N = 100_000


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    stat_fns: dict[str, Callable[[np.ndarray, np.ndarray], float | None]],
    reps: int = BOOT_REPS,
    seed: int = BOOT_SEED,
    max_n: int = BOOT_MAX_N,
) -> dict[str, list[float] | None]:
    """Percentile (2.5, 97.5) bootstrap CI for each statistic in `stat_fns`.

    Each stat_fn takes (y_true, y_prob) of a resampled cohort and returns a
    scalar, or None when it is undefined on that resample (e.g. a single-class
    draw for AUROC). A CI is None when too few replicates produced a value.
    """
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    if len(yt) > max_n:
        base = rng.choice(len(yt), max_n, replace=False)
        yt, yp = yt[base], yp[base]
    n = len(yt)
    samples: dict[str, list[float]] = {name: [] for name in stat_fns}
    if n == 0:
        return {name: None for name in stat_fns}

    for _ in range(reps):
        idx = rng.integers(0, n, n)
        bt, bp = yt[idx], yp[idx]
        for name, fn in stat_fns.items():
            try:
                v = fn(bt, bp)
            except Exception:
                v = None
            if v is not None and np.isfinite(v):
                samples[name].append(float(v))

    out: dict[str, list[float] | None] = {}
    for name, vals in samples.items():
        if len(vals) < reps // 2:
            out[name] = None
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[name] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def scalar_ci(values: np.ndarray, stat=np.median) -> list[float] | None:
    """Percentile bootstrap CI of one scalar statistic of a 1-D sample.

    Resamples `values` (rows = stays in every report call site) through the same
    engine as :func:`bootstrap_ci`, so the method — one resample stream, fixed
    seed, (2.5, 97.5) percentiles, reps//2 validity floor — matches the rest of
    the report. Default statistic: the median.
    """
    return bootstrap_ci(
        values, values, {"stat": lambda a, _b: float(stat(a))})["stat"]


# ===========================================================================
# Part 3 — Confusion-matrix arithmetic and small numeric helpers
#
# Single owner for the 2x2-derived rates the report publishes, the bootstrap stat
# factory that recomputes them per resample, and the rounding / percentile / NNE
# helpers.
#
# All flagging here is the ALARM convention (y_prob >= t; see report/_basis.py).
# That is what keeps "sensitivity" meaning "how well do we detect the event"
# across every task and both report modes.
# ===========================================================================


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

    ALARM convention (`yp >= t`). Also serves collapsed fired-as-0/1 arrays:
    `mk_stat(0.5, name)` on a {0.0, 1.0} column recovers the fired mask.
    """
    def _fn(yt: np.ndarray, yp: np.ndarray) -> float | None:
        tp, fp, fn, tn = confusion_counts(yt, alarm_mask(yp, t))
        return confusion_metric(tp, fp, fn, tn, name)
    return _fn


def nne_from_ppv(ppv: float | None,
                 ci: list[float] | None) -> tuple[float | None, list | None]:
    """NNE = 1/PPV (alarms per true positive); CI is the inverted PPV CI.

    1/x is decreasing, so [ppv_lo, ppv_hi] -> [1/ppv_hi, 1/ppv_lo]; a zero lower
    PPV bound means an unbounded NNE upper bound (null).
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
    valid-replicate floor (the same rule `bootstrap_ci` above applies)."""
    if len(vals) < reps // 2:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ===========================================================================
# Part 4 — Calibration summaries
#
# Every summary here is model-derived (a 2-parameter logistic recalibration, or
# scalar means), so none of it carries patient-level information and none of it
# needs cell suppression.
# ===========================================================================

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
