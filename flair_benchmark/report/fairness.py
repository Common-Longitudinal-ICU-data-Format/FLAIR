"""Per-subgroup performance + parity gaps. PHI-safe.

Subgroups with n < 10 are DROPPED entirely (not just suppressed) since
even an AUROC computed on <10 patients could reveal individual outcomes.

Output JSON shape:
  {
    "subgroups": {
      "sex_category": {<level>: {n, auroc, auprc, positive_rate, ...}, ...},
      "race_category": {...}, "ethnicity_category": {...}, "age_bin": {...}
    },
    "parity": {
      "sex_category": {demographic_parity_gap, equalized_odds_tpr_gap,
                       equalized_odds_fpr_gap}, ...
    },
    "metadata": {min_subgroup_n, threshold, ...}
  }
"""

from __future__ import annotations

import numpy as np
import polars as pl

SUBGROUP_COLS = ["sex_category", "race_category", "ethnicity_category", "age_bin"]
MIN_SUBGROUP_N = 10


def build_fairness(preds: pl.DataFrame, meta: dict,
                   threshold: float = 0.5) -> dict:
    """Compute per-subgroup metrics + parity gaps on the test split only."""
    task_type = meta["task_type"]
    label_col = meta["label_column"]

    df = preds.filter(pl.col("split") == "test")
    df = _bin_age(df)

    subgroups: dict = {}
    parity: dict = {}
    for col in SUBGROUP_COLS:
        if col not in df.columns:
            continue
        subgroups[col] = _subgroup_metrics(df, col, label_col, task_type, threshold)
        parity[col] = _parity_gaps(df, col, label_col, task_type, threshold)

    return {
        "subgroups": subgroups,
        "parity": parity,
        "metadata": {"min_subgroup_n": MIN_SUBGROUP_N, "threshold": threshold},
    }


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
                      task_type: str, threshold: float) -> dict:
    out: dict = {}
    for level in df[group_col].unique().drop_nulls().to_list():
        sub = df.filter(pl.col(group_col) == level)
        if sub.height < MIN_SUBGROUP_N:
            continue
        out[str(level)] = _metrics_one_group(sub, label_col, task_type, threshold)
    return out


def _metrics_one_group(sub: pl.DataFrame, label_col: str,
                       task_type: str, threshold: float) -> dict:
    n = sub.height
    if task_type == "binary":
        from sklearn.metrics import (
            average_precision_score, roc_auc_score,
        )
        y_true = sub[label_col].to_numpy().astype(float)
        y_pred = sub["y_pred"].to_numpy()
        y_prob = sub["y_prob"].to_numpy() if "y_prob" in sub.columns else None
        positive_rate = float(y_pred.mean()) if len(y_pred) else 0.0
        tpr = float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else None
        fpr = float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else None
        out: dict = {"n": n, "positive_rate": round(positive_rate, 4),
                     "tpr": None if tpr is None else round(tpr, 4),
                     "fpr": None if fpr is None else round(fpr, 4),
                     "prevalence": round(float(y_true.mean()), 4)}
        if y_prob is not None and len(np.unique(y_true)) > 1:
            out["auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            out["auprc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            slope, intercept = _calibration_slope_intercept(y_true, y_prob)
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


def _calibration_slope_intercept(y_true: np.ndarray,
                                  y_prob: np.ndarray) -> tuple[float | None, float | None]:
    """Logistic regression of y_true on logit(y_prob). Returns (slope, intercept).

    A perfectly calibrated model yields slope=1, intercept=0.
    """
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    if len(np.unique(y_true)) < 2 or np.std(logit) == 0:
        return None, None
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y_true.astype(int))
    return round(float(lr.coef_[0, 0]), 4), round(float(lr.intercept_[0]), 4)


def _parity_gaps(df: pl.DataFrame, group_col: str, label_col: str,
                 task_type: str, threshold: float) -> dict:
    if task_type != "binary":
        return {}
    rows = []
    for level in df[group_col].unique().drop_nulls().to_list():
        sub = df.filter(pl.col(group_col) == level)
        if sub.height < MIN_SUBGROUP_N:
            continue
        y_true = sub[label_col].to_numpy().astype(float)
        y_pred = sub["y_pred"].to_numpy()
        pos_rate = float(y_pred.mean()) if len(y_pred) else 0.0
        tpr = float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else None
        fpr = float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else None
        rows.append((pos_rate, tpr, fpr))
    if len(rows) < 2:
        return {}
    pos_rates = [r[0] for r in rows]
    tprs = [r[1] for r in rows if r[1] is not None]
    fprs = [r[2] for r in rows if r[2] is not None]
    return {
        "demographic_parity_gap": round(max(pos_rates) - min(pos_rates), 4),
        "equalized_odds_tpr_gap": round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None,
        "equalized_odds_fpr_gap": round(max(fprs) - min(fprs), 4) if len(fprs) >= 2 else None,
    }
