"""Binned discrimination over time → temporal.json.

Three sections (binary tasks, test split only):
  by_window             AUROC/AUPRC per in-stay window (window_number
                        1..MAX_WINDOW-1, then a "{MAX_WINDOW}+" bucket). The
                        window unit is grid-dependent and reported as
                        by_window.unit, sourced from META["window_unit"]
                        ("day" for T1/T2, "hour" for T4). Continuous
                        (one-per-window) tasks only.
  by_year               AUROC trend by year of admission. The year is derived
                        from min(prediction_dttm) per stay (exact admission
                        year except for stays straddling Dec 31), unless an
                        admission_dttm column is present.
  trajectory_by_outcome Per-window summary of y_prob split by eventual
                        outcome (e.g. expired vs alive) — two lines showing
                        whether predicted risk diverges over the stay.
                        Continuous tasks only.

Suppression (HIPAA, MIN_CELL_COUNT=10):
  - bins with n < 10 are dropped entirely (fairness.py precedent)
  - AUROC/AUPRC reported only when a bin has >= 10 positives AND
    >= 10 negatives; otherwise null, with n_pos passed through suppress()
  - trajectory cells (window x outcome) with n < 10 are null
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress

MAX_WINDOW = 7
MAX_YEARS = 25            # above this, "by year" is date-shifted noise — skip
BOOTSTRAP_REPS = 200
BOOTSTRAP_MIN_N = 100
BOOTSTRAP_SEED = 20260610


def build_temporal(preds: pl.DataFrame, meta: dict) -> dict:
    """Binned discrimination on the test split. Binary tasks with y_prob only."""
    task_type = meta["task_type"]
    continuous = meta.get("temporal_pattern") == "continuous"
    out: dict = {
        "task_type": task_type,
        "by_window": None,
        "by_year": None,
        "trajectory_by_outcome": None,
        "metadata": {"min_bin_n": MIN_CELL_COUNT, "max_window": MAX_WINDOW,
                     "bootstrap_reps": BOOTSTRAP_REPS},
    }
    if task_type != "binary" or "y_prob" not in preds.columns:
        out["metadata"]["reason"] = "requires binary task with y_prob"
        return out

    label_col = meta["label_column"]
    test = preds.filter(pl.col("split") == "test")

    test = _attach_window_bucket(test, continuous)
    if "window_bucket" in test.columns:
        bins = _binned_discrimination(test, label_col, "window_bucket",
                                      order=_window_order())
        out["by_window"] = {
            "bin_col": "window_number",
            "unit": meta.get("window_unit"),  # "day"/"hour"; grid-dependent per task
            "cap": MAX_WINDOW,
            "bins": bins,
            "totals": _window_totals(test, label_col, bins),
        }
        out["trajectory_by_outcome"] = _trajectory(test, label_col)
    elif not continuous:
        out["by_window"] = {"reason": "one_per_stay_task"}
        out["trajectory_by_outcome"] = {"reason": "one_per_stay_task"}

    test = _attach_admission_year(test)
    if "admission_year" in test.columns:
        yr = test["admission_year"].drop_nulls()
        n_yr = yr.n_unique()
        lo, hi = (int(yr.min()), int(yr.max())) if yr.len() else (None, None)
        # MIMIC and many CLIF sites shift admission dates for de-identification
        # (e.g. years 2105-2214) — "year of admission" is then meaningless and the
        # plot is an unreadable cloud. Only emit when years are real + low-cardinality.
        if n_yr > MAX_YEARS or (lo is not None and (lo < 1990 or hi > 2035)):
            out["by_year"] = {"reason": "dates_shifted_or_high_cardinality",
                              "n_distinct_years": n_yr, "year_range": [lo, hi]}
        else:
            years = _binned_discrimination(test, label_col, "admission_year")
            out["by_year"] = {
                "year_source": ("admission_dttm" if "admission_dttm" in preds.columns
                                else "derived_from_min_prediction_dttm"),
                "bins": sorted(years, key=lambda b: b["bin"]),
            }
    return out


def _attach_window_bucket(test: pl.DataFrame, continuous: bool) -> pl.DataFrame:
    """window_bucket = "1".."6" / "7+" from window_number (continuous tasks)."""
    if not continuous:
        return test
    if "window_number" not in test.columns:
        if "prediction_dttm" not in test.columns:
            return test
        test = test.with_columns(
            pl.col("prediction_dttm").rank("ordinal").over("hospitalization_id")
            .cast(pl.Int32).alias("window_number")
        )
    return test.with_columns(
        pl.when(pl.col("window_number") >= MAX_WINDOW)
        .then(pl.lit(f"{MAX_WINDOW}+"))
        .otherwise(pl.col("window_number").cast(pl.Utf8))
        .alias("window_bucket")
    )


def _attach_admission_year(test: pl.DataFrame) -> pl.DataFrame:
    if "admission_dttm" in test.columns:
        return test.with_columns(
            pl.col("admission_dttm").dt.year().alias("admission_year"))
    if "prediction_dttm" in test.columns:
        # min(prediction_dttm) ≈ admission year, exact except a stay straddling
        # Dec 31 (first prediction in year N, rest in N+1) — flagged via
        # by_year.year_source = "derived_from_min_prediction_dttm".
        return test.with_columns(
            pl.col("prediction_dttm").min().over("hospitalization_id")
            .dt.year().alias("admission_year")
        )
    return test


def _window_order() -> list[str]:
    return [str(k) for k in range(1, MAX_WINDOW)] + [f"{MAX_WINDOW}+"]


def _window_totals(test: pl.DataFrame, label_col: str, bins: list[dict]) -> dict:
    """Roll-up over the bucketed test (pre-suppression) feeding by_window."""
    bucketed = test.filter(pl.col("window_bucket").is_not_null())
    n = bucketed.height
    n_pos = int(bucketed[label_col].cast(pl.Int64).sum() or 0)
    n_stays = bucketed["hospitalization_id"].n_unique()
    return {
        "n": n,
        "n_pos": suppress(n_pos),
        "prevalence": round(n_pos / n, 4) if n else None,
        "n_windows_reported": len(bins),
        "n_stays": n_stays,
    }


def _binned_discrimination(test: pl.DataFrame, label_col: str, bin_col: str,
                           order: list | None = None) -> list[dict]:
    """Per-bin n / prevalence / AUROC / AUPRC. Bins with n<10 dropped."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    levels = test[bin_col].unique().drop_nulls().to_list()
    if order is not None:
        levels = [v for v in order if v in set(map(str, levels))]
    bins: list[dict] = []
    for level in levels:
        sub = test.filter(pl.col(bin_col).cast(pl.Utf8) == str(level))
        n = sub.height
        if n < MIN_CELL_COUNT:
            continue
        y_true = sub[label_col].to_numpy().astype(int)
        y_prob = sub["y_prob"].to_numpy().astype(float)
        n_pos = int(y_true.sum())
        n_neg = n - n_pos
        row: dict = {
            "bin": level if isinstance(level, int) else str(level),
            "n": n,
            "n_pos": suppress(n_pos),
            "prevalence": round(n_pos / n, 4),
            "auroc": None, "auroc_ci": None, "auprc": None,
        }
        if n_pos >= MIN_CELL_COUNT and n_neg >= MIN_CELL_COUNT:
            row["auroc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            row["auprc"] = round(float(average_precision_score(y_true, y_prob)), 4)
            if n >= BOOTSTRAP_MIN_N:
                row["auroc_ci"] = _bootstrap_auroc_ci(y_true, y_prob)
        bins.append(row)
    return bins


def _bootstrap_auroc_ci(y_true: np.ndarray, y_prob: np.ndarray) -> list[float] | None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true)
    aucs = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_prob[idx]))
    if len(aucs) < BOOTSTRAP_REPS // 2:
        return None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _trajectory(test: pl.DataFrame, label_col: str) -> dict:
    """Mean/median y_prob per window, one series per outcome (alive vs expired)."""
    windows = _window_order()
    series = []
    for label_value, outcome in ((1, "positive"), (0, "negative")):
        sub = test.filter(pl.col(label_col) == label_value)
        mean_p, median_p, q25, q75, ns = [], [], [], [], []
        for w in windows:
            cell = sub.filter(pl.col("window_bucket") == w)
            c = cell.height
            ns.append(c)
            if c < MIN_CELL_COUNT:
                mean_p.append(None), median_p.append(None)
                q25.append(None), q75.append(None)
                continue
            p = cell["y_prob"].to_numpy().astype(float)
            mean_p.append(round(float(p.mean()), 4))
            median_p.append(round(float(np.median(p)), 4))
            q25.append(round(float(np.quantile(p, 0.25)), 4))
            q75.append(round(float(np.quantile(p, 0.75)), 4))
        series.append({"outcome": outcome, "label_value": label_value,
                       "mean_prob": mean_p, "median_prob": median_p,
                       "q25": q25, "q75": q75,
                       "n": [suppress(c) if c < MIN_CELL_COUNT else c for c in ns]})
    return {"windows": windows, "series": series}
