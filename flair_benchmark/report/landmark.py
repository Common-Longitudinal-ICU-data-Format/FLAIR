"""Landmark analysis -> landmark.json.

FORWARD landmarking (van Houwelingen). Landmark ``t`` is the slice
``window_number == t`` — hospitalization DAY t for the daily tasks, HOUR t for
an hourly one (unit from META's ``window_unit``), for ``t = 1 ..
META['landmark_horizon']``.

The risk set at landmark t is every stay still producing predictions at t. Task
builders only generate windows inside ``[pred_start, pred_stop)``, so each slice
is automatically restricted to stays still at risk and event-free — the
per-landmark ``n`` is a true Kaplan-Meier "No. at risk" that falls as events
occur and stays end. Nothing anchors on the event or the outcome: the slice at
day t is exactly the cohort a deployed model would score on day t.

FOUR NUMBERS. That is the whole file:

    auroc              ranking, pair-weighted across landmarks (Heagerty & Zheng
                       2005 — the time-varying / dynamic AUC)
    auprc              ranking under class imbalance, case-weighted
    brier              accuracy of the probabilities themselves
    calibration_slope  do the probabilities mean what they say (1.0 = yes)

Each is published pooled (one headline number with a CI) and per landmark (the
curve behind it, with a CI ribbon). NO fairness, NO thresholds, NO ROC/PR
curves, NO operating point — those are hospitalization_level.json's job.

WHY ONE ROW PER STAY PER LANDMARK MATTERS: it makes the stay-level bootstrap
below an honest patient-level bootstrap. Row-pooling every window of every stay
would understate every interval, because windows within a stay are correlated.

THE POOLING WEIGHTS are not interchangeable, and each is chosen for what the
metric is made of:

    auroc  w = cases x controls   Mann-Whitney comparable pairs; a landmark with
                                  no contrast contributes nothing, which is the
                                  correct behaviour rather than a special case.
    auprc  w = cases              average precision is anchored on the positives.
    brier  w = n                  an n-weighted mean of per-slice mean-squared
                                  error IS the pooled mean squared error.
    slope  ONE logistic fit over all landmark rows — NOT a mean of per-landmark
           slopes. Averaging regression coefficients is a different estimator,
           and a single degenerate landmark would poison the mean.

ONE BOOTSTRAP for all of it: resample hospitalizations, recompute every
landmark's four values and all four pooled values inside that replicate. One
stream means the headline and its ribbon are mutually coherent, and it replaces
the old ``3 pillars x 500 reps x N landmarks`` process pool with a single pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._stats import (
    BOOT_REPS,
    BOOT_SEED,
    MIN_CELL_COUNT,
    pct,
)

BLOCK = "hospitalization_join_id"
MIN_LANDMARK_N = MIN_CELL_COUNT

METRICS = ("auroc", "auprc", "brier", "calibration_slope")

_METHOD = (
    "pooled dynamic AUC per Heagerty & Zheng 2005 (pair-weighted across forward "
    "landmarks); landmarking per van Houwelingen. AUPRC is case-weighted, Brier "
    "is n-weighted, and the calibration slope is a single logistic fit over all "
    "landmark rows. Every CI is one shared stay-level bootstrap: one "
    "hospitalization resample per replicate, all landmark and pooled values "
    "recomputed on it, percentile (2.5, 97.5).")

_EPS = 1e-6


# ===========================================================================
# Part 1 — the public builder
# ===========================================================================


def build_landmark(preds: pl.DataFrame, meta: dict,
                   n_landmarks: int | None = None) -> dict:
    """Four pooled numbers with CIs, plus the per-landmark curve behind them."""
    horizon = _resolve_horizon(meta, n_landmarks)
    label_col = meta["label_column"]

    out: dict = {
        "task_type": meta["task_type"],
        "metadata": {
            "window_unit": meta.get("window_unit"),
            "horizon": horizon,
            "min_cell_count": MIN_LANDMARK_N,
            "boot_reps": BOOT_REPS,
            "method": _METHOD,
        },
    }
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "landmarks require a binary task with y_prob"
        out["metadata"]["n_landmarks_scored"] = 0
        out.update({"prevalence": None, "pooled": _empty_pooled(),
                    "by_landmark": []})
        return out

    parts = _partition(preds, label_col)
    slices, entries = [], []
    for t in range(1, horizon + 1):
        sl = parts.get(t)
        n = 0 if sl is None else sl.height
        if n < MIN_LANDMARK_N:
            entries.append({"hospitalization_time": t, "n": f"<{MIN_LANDMARK_N}",
                            "suppressed": True})
            continue
        y = sl["_y"].to_numpy()
        if len(np.unique(y)) < 2:
            # no contrast: AUROC, AUPRC and the calibration fit are all undefined.
            entries.append({"hospitalization_time": t, "n": n, "undefined": True,
                            "reason": "single label class at this "
                                      "hospitalization time"})
            continue
        p = sl["y_prob"].to_numpy().astype(float)
        entries.append({"hospitalization_time": t, "n": n,
                        "prevalence": round(float(y.mean()), 6),
                        **_round(_metrics(y, p))})
        slices.append((t, sl[BLOCK].to_numpy(), y, p))

    out["metadata"]["n_landmarks_scored"] = len(slices)
    out["prevalence"] = (round(float(preds[label_col].to_numpy().mean()), 6)
                         if preds.height else None)
    out["pooled"] = _round(_pool(slices))
    _attach_cis(out, entries, slices)
    out["by_landmark"] = entries
    return out


def _resolve_horizon(meta: dict, n_landmarks: int | None) -> int:
    """Explicit override wins; otherwise the task's META['landmark_horizon'].

    The horizon is a landmark COUNT in the task's window_unit (30 days for the
    daily tasks, N hours for an hourly one) — it lives in the task's META, never
    hardcoded here.
    """
    if n_landmarks is not None:
        return n_landmarks
    h = meta.get("landmark_horizon")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(
            "META['landmark_horizon'] (positive int, landmarks in the task's "
            f"window_unit) is required for the landmark report; got {h!r} for "
            f"task {meta.get('name')!r}.")
    return h


def _partition(preds: pl.DataFrame, label_col: str) -> dict[int, pl.DataFrame]:
    """{window_number: slice} in ONE pass, each slice sorted by stay id.

    Sorting is load-bearing twice over: the seeded bootstrap resamples by row
    index, and stay codes are built from this order. Without it the same seed
    lands on a different set of patients each run.
    """
    df = preds.with_columns((pl.col(label_col) > 0).cast(pl.Int8).alias("_y"))
    parts = df.partition_by("window_number", as_dict=True, maintain_order=True)
    return {(k[0] if isinstance(k, tuple) else k): v.sort(BLOCK)
            for k, v in parts.items()}


def _empty_pooled() -> dict:
    return {k: None for m in METRICS for k in (m, f"{m}_ci")}


def _round(d: dict) -> dict:
    return {k: (None if v is None else round(float(v), 4)) for k, v in d.items()}


# ===========================================================================
# Part 2 — the four metrics, on one slice
# ===========================================================================


def _metrics(y: np.ndarray, p: np.ndarray,
             w: np.ndarray | None = None) -> dict[str, float | None]:
    """AUROC / AUPRC / Brier / calibration slope for one landmark slice.

    ``w`` are bootstrap multiplicities (a stay drawn k times counts k-fold),
    which is identical to materializing repeated rows but far cheaper.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if w is None:
        cases = float(y.sum())
        n = float(len(y))
    else:
        cases = float(w[y == 1].sum())
        n = float(w.sum())
    if n <= 0 or cases <= 0 or cases >= n:
        return dict.fromkeys(METRICS)

    slope, _ = logistic_slope_intercept(y, p, w)
    return {
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": _brier(y, p, w),
        "calibration_slope": slope,
    }


def _brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> float:
    """Mean squared error of the probabilities. Written out rather than called
    from sklearn so the weighted and unweighted forms are provably the same."""
    sq = (p - y) ** 2
    if w is None:
        return float(sq.mean())
    total = float(w.sum())
    return float((sq * w).sum() / total) if total > 0 else float("nan")


def logistic_slope_intercept(
    y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    """Logistic regression of y on logit(p): the calibration slope + intercept.

    A perfectly calibrated model gives slope 1, intercept 0; an overconfident
    one gives slope < 1. Solved by IRLS (Newton) in plain numpy rather than
    sklearn, because the bootstrap runs this once per landmark per replicate —
    tens of thousands of fits — and a scikit-learn estimator's per-call setup
    dominates at that count. Unpenalized, so it matches
    ``LogisticRegression(C=1e6)`` to well within the reported precision.
    """
    if w is not None and float(w.sum()) <= 0:
        return None, None
    q = np.clip(p, _EPS, 1 - _EPS)
    x = np.log(q / (1 - q))
    if len(np.unique(y)) < 2 or np.std(x) == 0:
        return None, None

    weights = np.ones(len(y), dtype=float) if w is None else w.astype(float)
    design = np.column_stack([np.ones(len(y)), x])
    target = y.astype(float)
    beta = np.zeros(2)
    for _ in range(50):
        eta = np.clip(design @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        var = np.clip(mu * (1 - mu), 1e-10, None) * weights
        grad = design.T @ (weights * (target - mu))
        hess = design.T @ (design * var[:, None])
        # a whisper of ridge keeps a separable slice from blowing up; it is far
        # below the reported 4-dp precision.
        hess[0, 0] += 1e-10
        hess[1, 1] += 1e-10
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None, None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    if not np.all(np.isfinite(beta)):
        return None, None
    return float(beta[1]), float(beta[0])


# ===========================================================================
# Part 3 — pooling across landmarks
# ===========================================================================


def _pool(slices: list[tuple], w_of: dict[int, np.ndarray] | None = None) -> dict:
    """The four pooled numbers. ``w_of`` supplies bootstrap weights per landmark.

    See the module docstring for why each metric takes a different weight.
    """
    auc_num = auc_den = 0.0
    ap_num = ap_den = 0.0
    br_num = br_den = 0.0
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_w: list[np.ndarray] = []

    for t, _codes, y, p in slices:
        w = None if w_of is None else w_of[t]
        met = _metrics(y, p, w)
        if met["auroc"] is None:
            continue                      # no contrast in this (re)sample
        if w is None:
            cases = float(y.sum())
            n = float(len(y))
        else:
            cases = float(w[y == 1].sum())
            n = float(w.sum())
        controls = n - cases

        auc_num += met["auroc"] * cases * controls
        auc_den += cases * controls
        ap_num += met["auprc"] * cases
        ap_den += cases
        br_num += met["brier"] * n
        br_den += n
        all_y.append(y)
        all_p.append(p)
        all_w.append(np.ones(len(y)) if w is None else w)

    if not all_y:
        return dict.fromkeys(METRICS)
    slope, _ = logistic_slope_intercept(
        np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_w))
    return {
        "auroc": auc_num / auc_den if auc_den > 0 else None,
        "auprc": ap_num / ap_den if ap_den > 0 else None,
        "brier": br_num / br_den if br_den > 0 else None,
        "calibration_slope": slope,
    }


# ===========================================================================
# Part 4 — one stay-level bootstrap for every CI in the file
# ===========================================================================


def _attach_cis(out: dict, entries: list[dict], slices: list[tuple],
                reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> None:
    """Fill in every ``*_ci``, pooled and per landmark, in place.

    One resample of HOSPITALIZATIONS per replicate, shared by every landmark, so
    a stay that is drawn twice is drawn twice everywhere it appears. That is
    what respects the within-stay correlation across landmarks — per-landmark
    bootstraps cannot, because they resample each slice independently.
    """
    scored = {e["hospitalization_time"]: e for e in entries
              if not e.get("suppressed") and not e.get("undefined")}
    for entry in scored.values():
        for m in METRICS:
            entry[f"{m}_ci"] = None
    for m in METRICS:
        out["pooled"][f"{m}_ci"] = None
    if not slices:
        return

    # one integer code per hospitalization, shared across every landmark
    stays = sorted({s for _t, codes, _y, _p in slices for s in codes.tolist()})
    code_of = {s: i for i, s in enumerate(stays)}
    n_stays = len(stays)
    coded = [(t, np.array([code_of[s] for s in codes.tolist()]), y, p)
             for t, codes, y, p in slices]

    rng = np.random.default_rng(seed)
    pooled_samples: dict[str, list[float]] = {m: [] for m in METRICS}
    lm_samples: dict[tuple[int, str], list[float]] = {}

    for _ in range(reps):
        mult = np.bincount(rng.integers(0, n_stays, n_stays),
                           minlength=n_stays).astype(float)
        w_of = {t: mult[codes] for t, codes, _y, _p in coded}
        for t, _codes, y, p in coded:
            met = _metrics(y, p, w_of[t])
            for m in METRICS:
                v = met[m]
                if v is not None and np.isfinite(v):
                    lm_samples.setdefault((t, m), []).append(float(v))
        for m, v in _pool(coded, w_of).items():
            if v is not None and np.isfinite(v):
                pooled_samples[m].append(float(v))

    for m in METRICS:
        out["pooled"][f"{m}_ci"] = pct(pooled_samples[m], reps)
    for t, entry in scored.items():
        for m in METRICS:
            entry[f"{m}_ci"] = pct(lm_samples.get((t, m), []), reps)
