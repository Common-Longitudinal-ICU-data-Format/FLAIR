"""Per-subgroup performance + parity gaps. PHI-safe.

The frame is split-filtered upstream (episodic: one row per stay). The operating
point is the task's clinical threshold (META `clinical_threshold`): y_pred is
rederived as y_prob >= clinical_threshold so positive_rate / TPR / FPR / parity
gaps describe the actual clinical decision, not a 0.5 default.

Subgroups with n < 10 are DROPPED entirely (not just suppressed) since
even an AUROC computed on <10 patients could reveal individual outcomes.

Per-subgroup AUROC and the parity gaps each carry a 95% bootstrap CI.

Output JSON shape:
  {
    "subgroups": {
      "sex_category": {<level>: {n, auroc, auroc_ci, positive_rate, ...}, ...},
      "race_category": {...}, "ethnicity_category": {...}, "age_bin": {...}
    },
    "parity": {
      "sex_category": {demographic_parity_gap, demographic_parity_gap_ci,
                       equalized_odds_tpr_gap, equalized_odds_fpr_gap, ...}, ...
    },
    "metadata": {min_subgroup_n, operating_threshold, ...}
  }
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._ci import bootstrap_ci
from flair_benchmark.report.discrimination import _operating_point

SUBGROUP_COLS = ["sex_category", "race_category", "ethnicity_category", "age_bin"]
MIN_SUBGROUP_N = 10
_GAP_REPS = 500
_GAP_SEED = 20260612


def build_fairness(preds: pl.DataFrame, meta: dict,
                   threshold_metrics: bool = True) -> dict:
    """Per-subgroup metrics (+ parity gaps at the clinical threshold by default).

    ``threshold_metrics=False`` drops everything that depends on a decision
    threshold — the cohort-wide `overall` block, each subgroup's
    positive_rate / TPR / FPR, and the `parity` gaps — leaving only the
    threshold-free question: does the model RANK and CALIBRATE equally well
    across groups (n, prevalence, AUROC, AUPRC, Brier, calibration slope)?

    The episodic report passes False. Its threshold-dependent equity numbers are
    reported per grid threshold in operating_points.json, so they can be read at
    the clinical threshold AND at ± 0.05 / ± 0.10 around it.
    """
    task_type = meta["task_type"]
    label_col = meta["label_column"]

    df = _bin_age(preds)
    clinical_t = meta.get("clinical_threshold")
    op_t = clinical_t if clinical_t is not None else 0.5
    if (threshold_metrics and task_type == "binary" and clinical_t is not None
            and "y_prob" in df.columns):
        # operate at the task's clinical decision threshold (label space)
        df = df.with_columns(
            (pl.col("y_prob") >= clinical_t).cast(pl.Int32).alias("y_pred"))

    subgroups: dict = {}
    parity: dict = {}
    for col in SUBGROUP_COLS:
        if col not in df.columns:
            continue
        levels = _subgroup_metrics(df, col, label_col, task_type, threshold_metrics)
        subgroups[col] = levels
        if not threshold_metrics:
            continue
        gaps = _parity_gaps(df, col, label_col, task_type)
        if levels:
            # largest subgroup is the reference; break exact-n ties on the label so
            # the pick is stable regardless of group_by iteration order (polars is
            # hash-ordered across process runs — an n-tie would otherwise flip).
            gaps["reference_group"] = max(levels, key=lambda lv: (levels[lv]["n"], lv))
        parity[col] = gaps

    if not threshold_metrics:
        return {
            "subgroups": subgroups,
            "metadata": {
                "min_subgroup_n": MIN_SUBGROUP_N,
                "threshold_free": True,
                "note": ("subgroup ranking + calibration only; the decision-threshold "
                         "view (positive rate, TPR, FPR, parity gaps) is reported at "
                         "every grid threshold in operating_points.json"),
            },
        }

    return {
        # cohort-wide operating point at the clinical threshold — the baseline each
        # subgroup is compared against (populated even when demographics are absent).
        "overall": _overall_metrics(df, label_col, meta),
        "subgroups": subgroups,
        "parity": parity,
        "metadata": {"min_subgroup_n": MIN_SUBGROUP_N,
                     "operating_threshold": op_t,
                     "threshold": clinical_t,
                     "clinical_direction": meta.get("clinical_direction", "above"),
                     "flag_rule": "y_prob >= threshold"},
    }


def _overall_metrics(df: pl.DataFrame, label_col: str, meta: dict) -> dict:
    """Whole-cohort confusion metrics at the clinical threshold (+ accuracy/n/prevalence).

    Reuses discrimination._operating_point (sensitivity/specificity/PPV/NPV/F1 each with a 95%
    CI, suppressed tp/fp/fn/tn, flag_rate, threshold, direction) so the numbers match the
    discrimination report exactly; flagging is y_prob >= threshold.
    """
    t = meta.get("clinical_threshold")
    if meta["task_type"] != "binary" or t is None or "y_prob" not in df.columns:
        return {}
    y_true = df[label_col].to_numpy().astype(int)
    y_prob = df["y_prob"].to_numpy().astype(float)
    n = len(y_true)
    if n == 0:
        return {}
    out = _operating_point(y_true, y_prob, meta)
    pred = y_prob >= t
    correct = int((pred == (y_true == 1)).sum())
    out["n"] = n
    out["prevalence"] = round(float(y_true.mean()), 6)
    out["accuracy"] = round(correct / n, 4)
    return out


def _bin_age(df: pl.DataFrame) -> pl.DataFrame:
    if "age_at_admission" not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col("age_at_admission") < 40).then(pl.lit("18-39"))
        .when(pl.col("age_at_admission") < 65).then(pl.lit("40-64"))
        .when(pl.col("age_at_admission") < 80).then(pl.lit("65-79"))
        .otherwise(pl.lit("80+"))
        .alias("age_bin")
    )


def _subgroup_metrics(df: pl.DataFrame, group_col: str, label_col: str,
                      task_type: str, threshold_metrics: bool = True) -> dict:
    out: dict = {}
    for level in df[group_col].unique().drop_nulls().to_list():
        sub = df.filter(pl.col(group_col) == level)
        if sub.height < MIN_SUBGROUP_N:
            continue
        out[str(level)] = _metrics_one_group(sub, label_col, task_type,
                                             threshold_metrics)
    return out


def _metrics_one_group(sub: pl.DataFrame, label_col: str, task_type: str,
                       threshold_metrics: bool = True) -> dict:
    n = sub.height
    if task_type == "binary":
        from sklearn.metrics import average_precision_score, roc_auc_score
        y_true = sub[label_col].to_numpy().astype(float)
        y_prob = sub["y_prob"].to_numpy() if "y_prob" in sub.columns else None
        out: dict = {"n": n, "prevalence": round(float(y_true.mean()), 4)}
        if threshold_metrics:
            # decision-threshold view: who gets flagged, and at what error rates.
            y_pred = sub["y_pred"].to_numpy()
            positive_rate = float(y_pred.mean()) if len(y_pred) else 0.0
            tpr = (float((y_pred[y_true == 1] == 1).mean())
                   if (y_true == 1).any() else None)
            fpr = (float((y_pred[y_true == 0] == 1).mean())
                   if (y_true == 0).any() else None)
            out.update({"positive_rate": round(positive_rate, 4),
                        "tpr": None if tpr is None else round(tpr, 4),
                        "fpr": None if fpr is None else round(fpr, 4)})
        if y_prob is not None and len(np.unique(y_true)) > 1:
            from sklearn.metrics import brier_score_loss

            from flair_benchmark.report._calib import calibration_slope_intercept
            out["auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            out["auroc_ci"] = bootstrap_ci(
                y_true.astype(int), y_prob,
                {"auroc": lambda yt, yp: (roc_auc_score(yt, yp)
                                          if len(np.unique(yt)) > 1 else None)},
            )["auroc"]
            out["auprc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            out["brier"] = round(float(brier_score_loss(y_true, y_prob)), 4)
            slope, intercept = calibration_slope_intercept(y_true, y_prob)
            out["calibration_slope"] = slope
            out["calibration_intercept"] = intercept
        return out
    elif task_type == "multiclass":
        from sklearn.metrics import accuracy_score, f1_score
        y_true = sub[label_col].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        return {"n": n,
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "macro_f1": round(float(f1_score(y_true, y_pred,
                                                  average="macro", zero_division=0)), 4)}
    else:  # regression
        from sklearn.metrics import mean_absolute_error
        y_true = sub[label_col].to_numpy()
        y_pred = sub["y_pred_value"].to_numpy()
        return {"n": n,
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "mean_y_true": round(float(y_true.mean()), 4),
                "mean_y_pred": round(float(y_pred.mean()), 4)}


def _eligible_levels(df: pl.DataFrame, group_col: str) -> list:
    """Subgroup levels passing the n>=MIN floor in the (full) data."""
    return [lv for lv in df[group_col].unique().drop_nulls().to_list()
            if df.filter(pl.col(group_col) == lv).height >= MIN_SUBGROUP_N]


def _gap_triplet(df: pl.DataFrame, group_col: str, label_col: str,
                 levels: list):
    """(dp_gap, tpr_gap, fpr_gap) over a FIXED set of subgroup levels.

    `levels` is pinned from the full data (n>=MIN there); inside a bootstrap
    resample we do NOT re-apply the n<10 drop, so the gap is always taken over
    the same group set. If a pinned level vanishes in a resample (0 rows), the
    whole triplet is None so that resample is discarded — keeps the CI honest
    instead of silently shrinking the comparison set.
    """
    pos_rates, tprs, fprs = [], [], []
    for level in levels:
        sub = df.filter(pl.col(group_col) == level)
        if sub.height == 0:
            return None, None, None
        y_true = sub[label_col].to_numpy().astype(float)
        y_pred = sub["y_pred"].to_numpy()
        pos_rates.append(float(y_pred.mean()) if len(y_pred) else 0.0)
        if (y_true == 1).any():
            tprs.append(float((y_pred[y_true == 1] == 1).mean()))
        if (y_true == 0).any():
            fprs.append(float((y_pred[y_true == 0] == 1).mean()))
    if len(pos_rates) < 2:
        return None, None, None
    dp = max(pos_rates) - min(pos_rates)
    tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else None
    fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else None
    return dp, tpr_gap, fpr_gap


def _parity_gaps(df: pl.DataFrame, group_col: str, label_col: str,
                 task_type: str) -> dict:
    if task_type != "binary":
        return {}
    levels = _eligible_levels(df, group_col)
    if len(levels) < 2:
        return {}
    dp, tpr_gap, fpr_gap = _gap_triplet(df, group_col, label_col, levels)
    if dp is None:
        return {}
    out = {
        "demographic_parity_gap": round(dp, 4),
        "equalized_odds_tpr_gap": None if tpr_gap is None else round(tpr_gap, 4),
        "equalized_odds_fpr_gap": None if fpr_gap is None else round(fpr_gap, 4),
    }
    out.update(_parity_gap_ci(df, group_col, label_col, levels))
    return out


def _parity_gap_ci(df: pl.DataFrame, group_col: str, label_col: str,
                   levels: list) -> dict:
    """95% CI for each parity gap by resampling rows (= stays in episodic).

    The subgroup set is pinned to `levels` (eligible in the full data); a CI is
    None when too few resamples keep every pinned group populated.
    """
    rng = np.random.default_rng(_GAP_SEED)
    n = df.height
    keep = [c for c in (group_col, label_col, "y_pred") if c in df.columns]
    sub = df.select(keep)
    dps, tprs, fprs = [], [], []
    for _ in range(_GAP_REPS):
        idx = rng.integers(0, n, n)
        boot = sub[idx]
        dp, tpr_gap, fpr_gap = _gap_triplet(boot, group_col, label_col, levels)
        if dp is not None:
            dps.append(dp)
        if tpr_gap is not None:
            tprs.append(tpr_gap)
        if fpr_gap is not None:
            fprs.append(fpr_gap)
    return {
        "demographic_parity_gap_ci": _pct(dps),
        "equalized_odds_tpr_gap_ci": _pct(tprs),
        "equalized_odds_fpr_gap_ci": _pct(fprs),
    }


def _pct(vals: list[float]) -> list[float] | None:
    if len(vals) < _GAP_REPS // 2:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]
