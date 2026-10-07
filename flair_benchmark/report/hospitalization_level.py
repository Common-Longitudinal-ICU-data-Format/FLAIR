"""Hospitalization-level decision analysis -> hospitalization_level.json.

ONE file, ONE schema, BOTH report modes. Everything that depends on where the
decision threshold is put lives here, and nowhere else.

The unit of analysis is always ONE HOSPITALIZATION:

  continuous : collapse to the peak — ``score = max(y_prob)`` over the stay,
               ``label = ever positive``. One alert per hospitalization, which is
               how a real-time alerting model is actually deployed.
  episodic   : already one row per stay. Nothing to collapse.

After that the two are the same ``(label, score)`` array pair running the
identical sweep, so a reader needs one parser and the FLAIR website needs one
aggregator.

WHAT IS IN IT

  by_threshold        the fixed 99-point grid (0.01 .. 0.99, step 0.01), every
                      metric at every point, each with a bootstrap CI band. The
                      shipped clinical policy is ONE FLAGGED ROW (is_clinical),
                      not a separate analysis — it is already on the grid.
  selected_thresholds Youden's J (Youden 1950) beside the clinical policy.
                      Youden is stamped ``in_sample``: choosing a cutpoint to
                      maximise a metric on the rows you then score is upward
                      biased (Leeflang 2008), so it annotates the sweep rather
                      than heading it.
  fairness            subgroup performance + parity gaps AT THE CLINICAL
                      THRESHOLD ONLY — one point, not 99.

NET BENEFIT is episodic-only. On the peak (max-of-many) surface a decision curve
is the peak-calibration trap, so continuous rows omit the net_benefit keys and
metadata says why.

TWO BASES (report/_basis.py): the 2x2 and sensitivity/specificity/PPV ride the
ALARM basis (``score >= t``, both directions) so "sensitivity" always means "how
well do we detect the event". Net benefit rides the direction-aware DECISION
basis, which is the only place a "below" task's complement enters.

Privacy: every 2x2 passes ``suppress_confusion`` (masking in PAIRS, since the
published ``n`` makes a lone masked cell recoverable by subtraction), and a rate
built on fewer than ``MIN_CELL_COUNT`` flagged stays is nulled.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from flair_benchmark.report._basis import (
    alarm_mask,
    axis_threshold,
    net_benefit_all,
    to_decision_basis,
)
from flair_benchmark.report._stats import (
    BOOT_MAX_N,
    BOOT_REPS,
    BOOT_SEED,
    MIN_CELL_COUNT,
    bootstrap_ci,
    nne_from_ppv,
    pct,
    r1,
    suppress_confusion,
)
from flair_benchmark.report._sweep import (
    RATE_KEYS,
    SWEEP_GRID,
    nb_curve,
    sweep_counts,
    sweep_metrics,
)

BLOCK = "hospitalization_join_id"

SUBGROUP_COLS = ("sex_category", "race_category", "ethnicity_category", "age_bin")
MIN_SUBGROUP_N = MIN_CELL_COUNT

_GRID_META = {"min": 0.01, "max": 0.99, "step": 0.01, "n": len(SWEEP_GRID)}

_PEAK_NB_OMITTED = (
    "net benefit is not reported on the peak (max-of-many) surface — a decision "
    "curve on max-of-many risk is the peak-calibration trap")

_YOUDEN_NOTE = (
    "youden_j is fitted on these same rows, so its threshold_ci is "
    "SELECTION-AWARE (the cutpoint is re-chosen on every bootstrap resample) and "
    "runs wider than the fixed-threshold band. The clinical policy is not "
    "fitted, so its threshold_ci is null.")


# ===========================================================================
# Part 1 — the public builder
# ===========================================================================


def build_hospitalization_level(preds: pl.DataFrame, meta: dict,
                                collapse: bool) -> dict:
    """The 99-threshold sweep + Youden + fairness, for either mode.

    ``collapse=True`` (continuous) reduces the frame to one row per stay on the
    peak score first; ``collapse=False`` (episodic) takes it as given.
    """
    t0 = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")

    md: dict = {
        "surface": "peak" if collapse else "stay",
        "score_rule": (f"max(y_prob) per {BLOCK}" if collapse
                       else "y_prob (one prediction per hospitalization)"),
        "clinical_threshold": t0,
        "clinical_direction": direction,
        "grid": dict(_GRID_META),
        "min_cell_count": MIN_CELL_COUNT,
        "boot_reps": BOOT_REPS,
        "flag_rule": "score >= threshold (alarm convention)",
        "selection_note": _YOUDEN_NOTE,
    }
    out: dict = {"task_type": meta["task_type"], "metadata": md}

    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t0 is None:
        md["reason"] = ("the hospitalization-level analysis requires a binary "
                        "task with y_prob and a clinical_threshold")
        out.update({"by_threshold": [], "selected_thresholds": [],
                    "fairness": None})
        return out

    if collapse:
        md["net_benefit_omitted_reason"] = _PEAK_NB_OMITTED

    stay = _collapse(preds, meta) if collapse else preds
    label = (stay[meta["label_column"]].to_numpy() > 0).astype(int)
    score = stay["y_prob"].to_numpy().astype(float)
    n = len(label)
    out["prevalence"] = round(float(label.mean()), 6) if n else None
    out["n_hospitalizations"] = n

    tp, fp, fn, tn = sweep_counts(label, score)
    m = sweep_metrics(tp, fp, fn, tn)

    # Episodic only: the decision curve. Labels move to the decision basis; the
    # SCORES stay on the label scale, because that is where the acted arm is cut
    # (report/_basis.py's boundary convention). Deriving the arm from 1-p instead
    # puts a patient sitting exactly on t on the wrong side at 20 of the 99
    # thresholds.
    want_nb = not collapse
    axis_grid = np.array([axis_threshold(float(t), direction) for t in SWEEP_GRID])
    yt_dec, _ = to_decision_basis(label, score, direction)
    nb_point = (nb_curve(yt_dec, score, SWEEP_GRID, axis_grid, direction)
                if want_nb else None)
    prev_dec = float(yt_dec.mean()) if n else None
    nb_all = ([None if prev_dec is None else net_benefit_all(prev_dec, float(t))
               for t in axis_grid] if want_nb else None)

    band, youden_ts = _bootstrap(label, score, direction, axis_grid, want_nb)

    clinical_i = int(np.argmin(np.abs(SWEEP_GRID - round(float(t0), 2))))
    md["clinical_index"] = clinical_i

    out["by_threshold"] = [
        _row(i, i == clinical_i, tp, fp, fn, tn, m, band,
             nb_point, nb_all, want_nb)
        for i in range(len(SWEEP_GRID))
    ]
    out["selected_thresholds"] = _selected(m, tp, fp, youden_ts, clinical_i,
                                          round(float(t0), 2))
    out["fairness"] = _fairness(stay, meta, float(t0))
    return out


def _collapse(preds: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """One row per hospitalization: peak score, ever-positive label.

    The trailing .sort is load-bearing: polars group_by row order is
    nondeterministic and the seeded bootstrap resamples by row INDEX, so without
    it the same seed lands on a different set of patients every run — point
    estimates stay put (they are order-invariant) while every CI silently moves.
    """
    label_col = meta["label_column"]
    carry = [c for c in ("age_at_admission", *SUBGROUP_COLS) if c in preds.columns]
    aggs = [pl.col("y_prob").max().alias("y_prob"),
            (pl.col(label_col).max() > 0).cast(pl.Int8).alias(label_col)]
    aggs += [pl.col(c).first().alias(c) for c in carry]
    return preds.group_by(BLOCK).agg(aggs).sort(BLOCK)


# ===========================================================================
# Part 2 — one bootstrap stream for the whole file
# ===========================================================================


def _bootstrap(label: np.ndarray, score: np.ndarray, direction: str,
               axis_grid: np.ndarray, want_nb: bool,
               reps: int = BOOT_REPS, seed: int = BOOT_SEED,
               max_n: int = BOOT_MAX_N) -> tuple[dict[str, list], list[float]]:
    """CI bands for every swept metric AND Youden's cutpoint, from ONE stream.

    Per replicate: resample hospitalizations, re-sweep the whole grid, and record
    (a) each rate at each threshold and (b) where Youden's J lands. One loop for
    both is what makes the band and the cutpoint CI mutually coherent — and it
    halves the work versus bootstrapping them separately.

    The cutpoint is RE-SELECTED inside each replicate on purpose: that is what
    makes its interval reflect a data-driven threshold's real instability rather
    than the much narrower interval of a threshold held fixed.
    """
    grid_n = len(SWEEP_GRID)
    keys = list(RATE_KEYS) + (["net_benefit"] if want_nb else [])
    if len(label) == 0:
        return {k: [None] * grid_n for k in keys}, []

    rng = np.random.default_rng(seed)
    lab, sc = label, score
    if len(lab) > max_n:
        base = rng.choice(len(lab), max_n, replace=False)
        lab, sc = lab[base], sc[base]
    n = len(lab)

    mats = {k: np.full((reps, grid_n), np.nan) for k in keys}
    youden_ts: list[float] = []
    for r in range(reps):
        idx = rng.integers(0, n, n)
        bl, bs = lab[idx], sc[idx]
        bm = sweep_metrics(*sweep_counts(bl, bs))
        for k in RATE_KEYS:
            mats[k][r] = bm[k]
        if want_nb:
            byd, _ = to_decision_basis(bl, bs, direction)
            mats["net_benefit"][r] = nb_curve(byd, bs, SWEEP_GRID, axis_grid,
                                              direction)
        j = _youden_index(bm)
        if j is not None:
            youden_ts.append(float(SWEEP_GRID[j]))

    floor = reps // 2
    band: dict[str, list] = {}
    for k in keys:
        mat = mats[k]
        valid = (~np.isnan(mat)).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # all-nan column -> nan, guarded below
            lo = np.nanpercentile(mat, 2.5, axis=0)
            hi = np.nanpercentile(mat, 97.5, axis=0)
        band[k] = [None if valid[i] < floor
                   else [round(float(lo[i]), 4), round(float(hi[i]), 4)]
                   for i in range(grid_n)]
    return band, youden_ts


def _youden_index(m: dict[str, np.ndarray]) -> int | None:
    """Grid index maximising Youden's J = sensitivity + specificity - 1."""
    obj = m["sensitivity"] + m["specificity"] - 1.0
    if np.all(np.isnan(obj)):
        return None
    return int(np.nanargmax(obj))


# ===========================================================================
# Part 3 — one row of the sweep
# ===========================================================================


def _row(i: int, is_clinical: bool, tp, fp, fn, tn,
         m: dict[str, np.ndarray], band: dict[str, list],
         nb_point, nb_all, want_nb: bool) -> dict:
    """Everything published at one threshold: the 2x2, the rates, the CIs."""
    n_flagged = int(tp[i] + fp[i])

    def val(key: str) -> float | None:
        v = m[key][i]
        return None if np.isnan(v) else r1(float(v))

    row: dict = {
        "threshold": round(float(SWEEP_GRID[i]), 2),
        "is_clinical": is_clinical,
        # n stays in the clear (it is the same stratum size published elsewhere),
        # which is exactly why the cells need PAIR masking: with n known, masking
        # one cell alone leaves it recoverable by subtracting the other three.
        "n": int(tp[i] + fp[i] + fn[i] + tn[i]),
        **suppress_confusion(tp[i], fp[i], fn[i], tn[i]),
    }

    # a precision built on a handful of flagged stays is unstable AND disclosive.
    ppv = None if n_flagged < MIN_CELL_COUNT else val("ppv")
    ppv_ci = None if n_flagged < MIN_CELL_COUNT else band["ppv"][i]
    for k in RATE_KEYS:
        row[k] = ppv if k == "ppv" else val(k)
        row[f"{k}_ci"] = ppv_ci if k == "ppv" else band[k][i]

    fr = m["flag_rate"][i]
    row["flag_rate"] = None if np.isnan(fr) else round(float(fr), 4)
    row["nne"], row["nne_ci"] = nne_from_ppv(ppv, ppv_ci)

    if want_nb:
        nb = nb_point[i]
        row["net_benefit"] = None if np.isnan(nb) else round(float(nb), 6)
        row["net_benefit_ci"] = band["net_benefit"][i]
        row["net_benefit_treat_all"] = nb_all[i]
    return row


# ===========================================================================
# Part 4 — Youden's J, next to the shipped policy
# ===========================================================================

_AT_KEYS = ("sensitivity", "specificity", "ppv", "npv", "f1", "fpr", "flag_rate")


def _selected(m: dict[str, np.ndarray], tp, fp, youden_ts: list[float],
              clinical_i: int, t0: float) -> list[dict]:
    """Two entries: where Youden lands, and where the shipped policy sits."""
    out: list[dict] = []
    j = _youden_index(m)
    if j is not None:
        out.append({
            "rule": "youden_j",
            "threshold": round(float(SWEEP_GRID[j]), 2),
            "threshold_ci": pct(youden_ts, BOOT_REPS),
            "in_sample": True,
            "citation": "Youden 1950",
            "delta_from_clinical": round(float(SWEEP_GRID[j]) - t0, 4),
            **_metrics_at(m, tp, fp, j),
        })
    out.append({
        "rule": "clinical",
        "threshold": round(float(SWEEP_GRID[clinical_i]), 2),
        "threshold_ci": None,
        "in_sample": False,
        "citation": None,
        "delta_from_clinical": 0.0,
        **_metrics_at(m, tp, fp, clinical_i),
    })
    return out


def _metrics_at(m: dict[str, np.ndarray], tp, fp, i: int) -> dict:
    """The sweep's rates at grid index i, plus derived NNE."""
    out = {k: (None if np.isnan(m[k][i]) else r1(float(m[k][i])))
           for k in _AT_KEYS}
    if int(tp[i] + fp[i]) < MIN_CELL_COUNT:
        out["ppv"] = None
    out["nne"] = nne_from_ppv(out["ppv"], None)[0]
    return out


# ===========================================================================
# Part 5 — fairness, at the clinical threshold only
# ===========================================================================


def _fairness(stay: pl.DataFrame, meta: dict, t0: float) -> dict | None:
    """Subgroup performance + parity gaps at the shipped policy.

    ONE threshold, not 99. Subgroup levels with fewer than MIN_SUBGROUP_N stays
    are dropped entirely rather than merely suppressed: even an AUROC computed on
    <10 patients can reveal individual outcomes.
    """
    df = bin_age(stay)
    cols = [c for c in SUBGROUP_COLS if c in df.columns]
    if not cols:
        return None

    label = (df[meta["label_column"]].to_numpy() > 0).astype(int)
    score = df["y_prob"].to_numpy().astype(float)
    # alarm basis in BOTH directions: subgroup TPR/FPR answer "is the model
    # equally good at DETECTING the event here", which must not flip meaning
    # for an act-on-low-risk task.
    flag = alarm_mask(score, t0)

    subgroups: dict = {}
    parity: dict = {}
    for col in cols:
        values = df[col].to_numpy()
        levels = [lv for lv in dict.fromkeys(v for v in values if v is not None)
                  if int((values == lv).sum()) >= MIN_SUBGROUP_N]
        if not levels:
            continue
        codes = encode_levels(values, levels)
        subgroups[col] = {
            str(lv): _level_block(label[codes == k], score[codes == k],
                                  flag[codes == k])
            for k, lv in enumerate(levels)
        }
        if len(levels) < 2:
            continue
        gaps = _parity(codes, label, flag, len(levels))
        if gaps:
            # largest level is the reference; ties break on the label so the pick
            # is stable across runs (polars unique order is hash-dependent).
            gaps["reference_group"] = str(max(
                levels, key=lambda lv: (int((values == lv).sum()), str(lv))))
            parity[col] = gaps

    if not subgroups:
        return None
    return {"at_threshold": round(float(t0), 4),
            "min_subgroup_n": MIN_SUBGROUP_N,
            "subgroups": subgroups, "parity": parity}


def _level_block(label: np.ndarray, score: np.ndarray, flag: np.ndarray) -> dict:
    """One subgroup level: does the model rank it as well, and flag it as often?"""
    from sklearn.metrics import roc_auc_score

    out: dict = {
        "n": int(len(label)),
        "prevalence": round(float(label.mean()), 4) if len(label) else None,
        "positive_rate": r1(float(flag.mean())) if len(flag) else None,
        "tpr": r1(float(flag[label == 1].mean())) if (label == 1).any() else None,
        "fpr": r1(float(flag[label == 0].mean())) if (label == 0).any() else None,
        "auroc": None, "auroc_ci": None,
    }
    if len(np.unique(label)) > 1:
        out["auroc"] = round(float(roc_auc_score(label, score)), 4)
        out["auroc_ci"] = bootstrap_ci(
            label, score,
            {"auroc": lambda yt, yp: (roc_auc_score(yt, yp)
                                      if len(np.unique(yt)) > 1 else None)},
        )["auroc"]
    return out


def bin_age(df: pl.DataFrame) -> pl.DataFrame:
    """Add an age_bin column. A null age must yield a NULL bin.

    A bare .otherwise("80+") would catch nulls too, silently counting every
    missing-age patient as an 80+ patient in the equity blocks.
    """
    if "age_at_admission" not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col("age_at_admission") < 40).then(pl.lit("18-39"))
        .when(pl.col("age_at_admission") < 65).then(pl.lit("40-64"))
        .when(pl.col("age_at_admission") < 80).then(pl.lit("65-79"))
        .when(pl.col("age_at_admission") >= 80).then(pl.lit("80+"))
        .otherwise(None)
        .alias("age_bin")
    )


def encode_levels(values: np.ndarray, levels: list) -> np.ndarray:
    """Integer codes for a PINNED level list (-1 = not one of the levels).

    Encoding once lets the bootstrap index with numpy instead of re-filtering a
    frame per level per replicate.
    """
    codes = np.full(len(values), -1, dtype=int)
    for k, lv in enumerate(levels):
        codes[values == lv] = k
    return codes


def gap_triplet(codes: np.ndarray, label: np.ndarray, flag: np.ndarray,
                n_levels: int) -> tuple[float | None, float | None, float | None]:
    """(demographic parity, TPR, FPR) max-minus-min gaps over PINNED levels.

    Levels are pinned from the full data; inside a resample we do NOT re-apply
    the n<10 drop, so the gap is always taken over the same group set. If a
    pinned level vanishes in a resample the whole triplet is None and that
    replicate is discarded — otherwise the CI would silently shrink toward zero
    as the comparison set got smaller.
    """
    pos_rates, tprs, fprs = [], [], []
    for k in range(n_levels):
        mask = codes == k
        if not mask.any():
            return None, None, None
        lab_k, flag_k = label[mask], flag[mask]
        pos_rates.append(float(flag_k.mean()))
        if (lab_k == 1).any():
            tprs.append(float(flag_k[lab_k == 1].mean()))
        if (lab_k == 0).any():
            fprs.append(float(flag_k[lab_k == 0].mean()))
    if len(pos_rates) < 2:
        return None, None, None
    return (max(pos_rates) - min(pos_rates),
            max(tprs) - min(tprs) if len(tprs) >= 2 else None,
            max(fprs) - min(fprs) if len(fprs) >= 2 else None)


def _parity(codes: np.ndarray, label: np.ndarray, flag: np.ndarray,
            n_levels: int) -> dict:
    """The three gaps with bootstrap CIs, at the clinical threshold."""
    dp, tpr_gap, fpr_gap = gap_triplet(codes, label, flag, n_levels)
    if dp is None:
        return {}
    rng = np.random.default_rng(BOOT_SEED)
    n = len(label)
    dps: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []
    for _ in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        a, b, c = gap_triplet(codes[idx], label[idx], flag[idx], n_levels)
        if a is not None:
            dps.append(a)
        if b is not None:
            tprs.append(b)
        if c is not None:
            fprs.append(c)
    return {
        "demographic_parity_gap": r1(dp),
        "demographic_parity_gap_ci": pct(dps, BOOT_REPS),
        "equalized_odds_tpr_gap": r1(tpr_gap),
        "equalized_odds_tpr_gap_ci": pct(tprs, BOOT_REPS),
        "equalized_odds_fpr_gap": r1(fpr_gap),
        "equalized_odds_fpr_gap_ci": pct(fprs, BOOT_REPS),
    }
