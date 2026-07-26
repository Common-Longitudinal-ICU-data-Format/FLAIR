"""Data-driven threshold locators (design doc §3).

A menu of statistical cutpoint-selection rules, each answering "on THIS model /
site / test set, where would this rule stand?" — a LOCATOR, never a graded
headline. Every rule reads the already-computed sweep metric vectors (an
argmax/argmin over 99 points, no data access) and its performance is read
straight off the sweep and stamped ``in_sample: true`` (choosing a cutpoint to
maximise a metric on the rows you then report is upward biased — Leeflang 2008;
the honest object is the full sweep, these are annotations on it). The clinical
policy is appended as a rule named ``clinical`` with ``in_sample: false`` so a
reader lines the shipped threshold up against all ten picks in one list.

Rules operate in the risk-classifier convention (predicted-positive = ``p >= t``,
label = event), independent of ``clinical_direction`` — so Youden/F1/etc. separate
events from non-events the same way for "above" and "below" tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flair_benchmark.report._stats import (
    BOOT_MAX_N,
    BOOT_REPS,
    BOOT_SEED,
    MIN_CELL_COUNT,
    nne_from_ppv,
    pct,
    r1,
)
from flair_benchmark.report._sweep import sweep_counts, sweep_metrics


def _argmax(obj: np.ndarray) -> int | None:
    return None if np.all(np.isnan(obj)) else int(np.nanargmax(obj))


def _argmin(obj: np.ndarray) -> int | None:
    return None if np.all(np.isnan(obj)) else int(np.nanargmin(obj))


# ---- the rules: each returns (index into the grid | None, objective value | None)

def _youden(m, grid, auc, prev, p):
    obj = m["sensitivity"] + m["specificity"] - 1.0
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _closest01(m, grid, auc, prev, p):
    obj = np.sqrt((1 - m["sensitivity"]) ** 2 + (1 - m["specificity"]) ** 2)
    i = _argmin(obj)
    return i, (None if i is None else obj[i])


def _concordance(m, grid, auc, prev, p):
    obj = m["sensitivity"] * m["specificity"]
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _index_of_union(m, grid, auc, prev, p):
    if auc is None:
        return None, None
    obj = np.abs(m["sensitivity"] - auc) + np.abs(m["specificity"] - auc)
    i = _argmin(obj)
    return i, (None if i is None else obj[i])


def _f_beta(m, grid, auc, prev, p):
    beta = p.get("f_beta", 1)
    if beta == 1:
        obj = m["f1"]
    else:
        b2 = beta ** 2
        num = (1 + b2) * m["ppv"] * m["sensitivity"]
        den = b2 * m["ppv"] + m["sensitivity"]
        obj = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _sensitivity_target(m, grid, auc, prev, p):
    s0 = p.get("sens_target", 0.90)
    obj = np.where(m["sensitivity"] >= s0, m["specificity"], np.nan)
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _specificity_target(m, grid, auc, prev, p):
    sp0 = p.get("spec_target", 0.90)
    obj = np.where(m["specificity"] >= sp0, m["sensitivity"], np.nan)
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _cost_based(m, grid, auc, prev, p):
    if not prev or prev <= 0:
        return None, None
    slope = ((1 - prev) / prev) * p.get("cost_ratio", 1)
    obj = m["sensitivity"] - slope * m["fpr"]
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


def _ppv_target(m, grid, auc, prev, p):
    target = p.get("ppv_target", 0.30)
    cand = np.where(m["ppv"] >= target)[0]      # smallest threshold that clears it
    if len(cand) == 0:
        return None, None
    i = int(cand[0])
    return i, m["ppv"][i]


def _alert_rate_target(m, grid, auc, prev, p):
    a0 = p.get("alert_rate_target", 0.10)
    obj = np.abs(m["flag_rate"] - a0)
    i = _argmin(obj)
    return i, (None if i is None else m["flag_rate"][i])


def _max_accuracy(m, grid, auc, prev, p):
    prev = 0.0 if prev is None else prev
    obj = m["sensitivity"] * prev + m["specificity"] * (1 - prev)
    i = _argmax(obj)
    return i, (None if i is None else obj[i])


@dataclass(frozen=True)
class _Rule:
    key: str
    fn: Callable
    assumption: str
    citation: str
    degenerate: bool = False


_LOCATORS: tuple[_Rule, ...] = (
    _Rule("youden_j", _youden, "equal FP/FN cost; prevalence-blind", "Youden 1950"),
    _Rule("closest_01", _closest01, "equal weight, geometric distance to (0,1)",
          "Perkins & Schisterman 2006"),
    _Rule("concordance_cz", _concordance, "equal value; penalises lopsided points",
          "Liu 2012"),
    _Rule("index_of_union", _index_of_union, "sensitivity and specificity both near AUC",
          "Ünal 2017"),
    _Rule("f_beta_max", _f_beta, "ignores TN; beta trades recall vs precision",
          "van Rijsbergen 1979"),
    _Rule("sensitivity_target", _sensitivity_target,
          "a miss is worse than a false alarm (max spec s.t. sens >= target)",
          "cutpointr"),
    _Rule("specificity_target", _specificity_target,
          "a false alarm is worse than a miss (max sens s.t. spec >= target)",
          "cutpointr"),
    _Rule("cost_based", _cost_based, "explicit cost ratio C_FP/C_FN weighted by prevalence",
          "Metz 1978"),
    _Rule("ppv_target", _ppv_target, "a minimum precision to be worth acting",
          "deployment practice"),
    _Rule("alert_rate_target", _alert_rate_target, "a fixed alert-workload budget",
          "deployment practice"),
    _Rule("max_accuracy", _max_accuracy,
          "equal cost, prevalence-weighted; collapses toward never-alert for rare events",
          "Metz 1978", degenerate=True),
)

_MA_KEYS = ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr", "flag_rate")

# the rates a CI is drawn for (== _sweep.RATE_KEYS): flag_rate stays point-only, a
# workload count with no published CI, matching the by_threshold band exactly.
_CI_KEYS = ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr")


def _metrics_at(m: dict[str, np.ndarray], i: int) -> dict:
    """The sweep's rates at grid index ``i`` (rounded), plus derived NNE."""
    out = {k: r1(float(m[k][i])) if not np.isnan(m[k][i]) else None for k in _MA_KEYS}
    out["nne"] = nne_from_ppv(out["ppv"], None)[0]
    return out


def locate_band(label: np.ndarray, prob: np.ndarray, grid: np.ndarray,
                auc: float | None, params: dict, reps: int = BOOT_REPS,
                seed: int = BOOT_SEED, max_n: int = BOOT_MAX_N) -> dict[str, dict]:
    """Per-rule SELECTION-AWARE CI: re-select each cutpoint on every resample.

    Each replicate resamples ``(label, prob)``, recomputes the sweep metric vectors,
    RE-RUNS every rule's argmax/argmin (so the cutpoint itself moves), and records
    that rule's landing threshold + rates. Reduced per rule with ``_confusion.pct``
    into ``{threshold, <rate> ...}`` CIs that INCLUDE cutpoint-selection variance —
    the honest interval for a rule fitted on the rows it is then scored on. Shares
    ``BOOT_SEED`` with ``_sweep.band_ci`` (same per-rep ``rng.integers`` draw, same
    over-``max_n`` subsample) so the resample stream is coherent with the
    fixed-threshold band.

    ``auc`` is held fixed at the point estimate (only ``index_of_union``'s target; a
    second-order approximation that avoids a per-replicate AUROC and keeps this layer
    dependency-light). ``prev`` is recomputed per resample (free from the label draw).
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(label)
    pr = np.asarray(prob)
    n = len(lab)
    keys = [rule.key for rule in _LOCATORS]
    if n == 0:
        return {k: {"threshold": None, **{c: None for c in _CI_KEYS}} for k in keys}
    if n > max_n:
        base = rng.choice(n, max_n, replace=False)
        lab, pr = lab[base], pr[base]
        n = max_n

    acc: dict[str, dict[str, list]] = {
        k: {"threshold": [], **{c: [] for c in _CI_KEYS}} for k in keys}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        lb, pb = lab[idx], pr[idx]
        tp, fp, fn, tn = sweep_counts(lb, pb, grid)
        mm = sweep_metrics(tp, fp, fn, tn)
        prev = float(lb.mean())
        for rule in _LOCATORS:
            i, _obj = rule.fn(mm, grid, auc, prev, params)
            if i is None:                    # rule undefined on this resample -> drops out
                continue
            a = acc[rule.key]
            a["threshold"].append(float(grid[i]))
            for c in _CI_KEYS:
                v = mm[c][i]
                if not np.isnan(v):
                    a[c].append(float(v))

    return {k: {"threshold": pct(acc[k]["threshold"], reps),
                **{c: pct(acc[k][c], reps) for c in _CI_KEYS}}
            for k in keys}


def _attach_ci(entry: dict, i: int, ci_map: dict, threshold_ci: list | None,
               counts: tuple | None) -> None:
    """Attach ``threshold_ci`` + ``metrics_at *_ci`` to one entry, in place.

    ``ci_map`` supplies each rate's CI (selection-aware for a fitted rule, the fixed
    band for the clinical policy). NNE's CI inverts the PPV CI — the ``_metrics_row``
    convention. ``counts`` = (tp, fp) grid vectors: a row flagging fewer than
    ``MIN_CELL_COUNT`` stays gives a disclosive, unstable precision, so its PPV / NNE
    CI is nulled (the point estimates, already shipped, are left untouched).
    """
    ma = entry["metrics_at"]
    entry["threshold_ci"] = threshold_ci
    for c in _CI_KEYS:
        ma[f"{c}_ci"] = ci_map.get(c)
    ma["nne_ci"] = nne_from_ppv(ma.get("ppv"), ma.get("ppv_ci"))[1]
    if counts is not None:
        tp, fp = counts
        if int(tp[i] + fp[i]) < MIN_CELL_COUNT:
            ma["ppv_ci"] = ma["nne_ci"] = None


def locate_all(m: dict[str, np.ndarray], grid: np.ndarray, auc: float | None,
               prev: float | None, t0: float, surface: str, params: dict,
               band: dict | None = None, ci: dict | None = None,
               counts: tuple | None = None) -> list[dict]:
    """Every rule's landing point + the clinical policy, as a flat list (§3.3).

    ``band`` (fixed-threshold sweep band, ``_sweep.band_ci``) and ``ci`` (per-rule
    selection-aware band, ``locate_band``) are optional CI sources. The 10 fitted
    rules take the SELECTION-AWARE interval — their cutpoint is re-chosen on every
    resample, so the CI reflects a data-driven cutpoint's true instability. The
    ``clinical`` policy is never fitted, so it takes the plain fixed-threshold band
    (identical to that threshold's ``by_threshold`` row) with a null ``threshold_ci``.
    ``counts`` = (tp, fp) grid vectors gate a disclosive precision on too-few-flagged
    rows. All three default ``None`` → point-only output, as before.
    """
    out: list[dict] = []
    for rule in _LOCATORS:
        i, obj = rule.fn(m, grid, auc, prev, params)
        if i is None:
            out.append({"rule": rule.key, "threshold": None,
                        "reason": "rule undefined on this sweep",
                        "in_sample": True, "assumption": rule.assumption,
                        "citation": rule.citation})
            continue
        t = round(float(grid[i]), 2)
        entry = {
            "rule": rule.key,
            "threshold": t,
            "objective_value": r1(float(obj)) if obj is not None else None,
            "surface": surface,
            "delta_from_clinical": round(t - t0, 4),
            "metrics_at": _metrics_at(m, i),
            "in_sample": True,
            "assumption": rule.assumption,
            "citation": rule.citation,
        }
        if ci is not None:
            rc = ci.get(rule.key, {})
            _attach_ci(entry, i, rc, rc.get("threshold"), counts)
        if rule.degenerate:
            entry["degenerate"] = True
        out.append(entry)

    # the shipped policy, aligned into the same list — never fitted, so in_sample False.
    cidx = int(np.argmin(np.abs(grid - t0)))
    clin = {
        "rule": "clinical",
        "threshold": round(float(grid[cidx]), 2),
        "objective_value": None,
        "surface": surface,
        "delta_from_clinical": 0.0,
        "metrics_at": _metrics_at(m, cidx),
        "in_sample": False,
        "assumption": "the shipped clinical policy threshold (not fitted)",
        "citation": None,
    }
    if band is not None:
        _attach_ci(clin, cidx, {c: band[c][cidx] for c in _CI_KEYS}, None, counts)
    out.append(clin)
    return out
