"""The statistical primitives every report section shares.

Four groups of small, widely-used helpers. They live in one file because they are
all "the arithmetic the report agrees on" — the point is that there is exactly ONE
definition of each, so two sections can never disagree about what sensitivity
means or how wide a confidence interval is.

    Part 1  suppression   HIPAA cell-count masking
    Part 2  bootstrap     95% confidence intervals
    Part 3  rounding      report precision, percentiles, NNE

Everything here is either a pure function of its arguments or a fixed constant.

Depends on: numpy only.
Used by: every report section, plus cohort/table1.py for suppression only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
# the interval; landmark.py resamples hospitalizations explicitly for that
# reason.
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
# hospitalization_level._collapse and landmark._partition. Sorting on the stay
# id (rather than
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
# Part 3 — Rounding, percentiles, and NNE
#
# The 2x2 arithmetic itself lives (vectorized, over the whole threshold grid) in
# report/_sweep.py, and the calibration fit in report/landmark.py. What is left
# here is the shared numeric etiquette: how many decimals the report publishes,
# how a bootstrap sample becomes an interval, and the one derived metric whose
# CI has to be inverted rather than resampled.
# ===========================================================================


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
