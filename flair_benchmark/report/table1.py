"""Table 1: cohort characteristics by split and by label.

Output JSON shape:
  {
    "n_total": int,
    "by_split":    { "train": {n, continuous, categorical}, "test": {...} },
    "by_label":    { <level>: {n, continuous, categorical}, ... },
    "continuous":  { "age_at_admission": {mean, std, median, q25, q75, min, max, n_missing} },
    "categorical": { "sex_category": {level: count_or_"<10"}, ... },
    "metadata":    { ... audit fields filled by report/__init__.py ... }
  }
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report._suppress import suppress, suppress_dict

CONTINUOUS_VARS = ["age_at_admission"]
CATEGORICAL_VARS = ["sex_category", "race_category", "ethnicity_category"]


def build_table1(preds: pl.DataFrame, meta: dict) -> dict:
    """Compute the table1 dict (no I/O)."""
    label_col = meta["label_column"]
    task_type = meta["task_type"]

    out: dict = {
        "n_total": preds.height,
        "continuous": {v: _continuous_stats(preds, v) for v in CONTINUOUS_VARS
                       if v in preds.columns},
        "categorical": {v: _categorical_stats(preds, v) for v in CATEGORICAL_VARS
                        if v in preds.columns},
        "by_split": {},
        "by_label": {},
    }

    for split_val in preds["split"].unique().drop_nulls().to_list():
        sub = preds.filter(pl.col("split") == split_val)
        out["by_split"][split_val] = _stratum(sub)

    if task_type in ("binary", "multiclass") and label_col in preds.columns:
        for level in preds[label_col].unique().drop_nulls().to_list():
            sub = preds.filter(pl.col(label_col) == level)
            out["by_label"][str(level)] = _stratum(sub)
    elif task_type == "regression" and label_col in preds.columns:
        q1, q3 = preds[label_col].quantile(0.25), preds[label_col].quantile(0.75)
        out["by_label"]["low_q1"] = _stratum(preds.filter(pl.col(label_col) <= q1))
        out["by_label"]["high_q4"] = _stratum(preds.filter(pl.col(label_col) >= q3))

    return out


def _stratum(df: pl.DataFrame) -> dict:
    return {
        "n": suppress(df.height),
        "continuous": {v: _continuous_stats(df, v) for v in CONTINUOUS_VARS
                       if v in df.columns},
        "categorical": {v: _categorical_stats(df, v) for v in CATEGORICAL_VARS
                        if v in df.columns},
    }


def _continuous_stats(df: pl.DataFrame, var: str) -> dict:
    series = df[var]
    n_missing = int(series.null_count())
    nn = series.drop_nulls()
    if nn.len() == 0:
        return {"mean": None, "std": None, "median": None, "q25": None,
                "q75": None, "min": None, "max": None, "n_missing": n_missing}
    return {
        "mean": round(float(nn.mean()), 2),
        "std": round(float(nn.std()), 2) if nn.len() > 1 else 0.0,
        "median": round(float(nn.median()), 2),
        "q25": round(float(nn.quantile(0.25)), 2),
        "q75": round(float(nn.quantile(0.75)), 2),
        "min": round(float(nn.min()), 2),
        "max": round(float(nn.max()), 2),
        "n_missing": n_missing,
    }


def _categorical_stats(df: pl.DataFrame, var: str) -> dict:
    counts = (df.group_by(var).agg(pl.len().alias("n"))
              .filter(pl.col(var).is_not_null()))
    return suppress_dict({str(row[var]): row["n"] for row in counts.iter_rows(named=True)})
