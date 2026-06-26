"""Generalized percentile bootstrap for 95% confidence intervals.

One resample index is drawn per replicate and reused across every statistic, so
the returned CIs are mutually coherent (computed on the same resampled cohort).

Resampling unit: rows. In the episodic report every row IS one stay, so a row
bootstrap is already a stay-clustered (patient-level) bootstrap — the honest unit
for ICU benchmark CIs. Do not reuse this on a row-pooled continuous task, where
rows are correlated within a stay and a row bootstrap would understate the CI.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
    """Percentile (2.5, 97.5) bootstrap CI for each statistic in ``stat_fns``.

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
