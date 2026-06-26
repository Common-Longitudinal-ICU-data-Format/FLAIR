"""Table 1: cohort characteristics, overall and stratified by label and split.

Generated at task-generation time (alongside the cohort parquet) and consumed by
the FLAIR website (``r.table1`` in flair_website/static/js/report.js). The flat
top-level keys are at ENCOUNTER grain (one row per hospitalization_join_id) so the
headline numbers read as patients, not repeated prediction rows; ``by_grain``
carries both the encounter and the per-prediction-row strata explicitly.

Output JSON shape:
  {
    "n_total": int,            # encounters (flat keys are encounter-grain)
    "n_rows":  int,            # prediction rows
    "continuous":  { "age_at_admission": {mean, std, median, q25, q75, min, max, n_missing},
                     "sofa_max": {...}, "charlson_score": {...}, "elix_score": {...} },
    "categorical": { "sex_category": {level: count_or_"<10"}, ... },
    "by_split":  { "train": {n, continuous, categorical}, "test": {...} },
    "by_label":  { "1": {...positive...}, "0": {...negative...} },   # the 2 stratified columns
    "by_grain":  { "encounter": {n_total, continuous, categorical, by_split, by_label},
                   "row":       {... same shape at prediction-row grain ...} },
    "metadata":  { ... audit fields stamped by _table1.generate_table1 ... }
  }

Continuous severity/comorbidity vars (sofa_max, charlson_score, elix_score) appear
only when those columns were joined onto the cohort (see _table1.generate_table1);
build_table1 itself works with or without them.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report._suppress import suppress, suppress_dict

JOIN_KEY = "hospitalization_join_id"
CONTINUOUS_VARS = ["age_at_admission", "sofa_max", "charlson_score", "elix_score"]
CATEGORICAL_VARS = ["sex_category", "race_category", "ethnicity_category"]


def build_table1(cohort: pl.DataFrame, meta: dict) -> dict:
    """Compute the table1 dict (no I/O). Flat keys are encounter-grain; ``by_grain``
    exposes both encounter and prediction-row strata."""
    label_col = meta["label_column"]
    task_type = meta["task_type"]

    enc = _to_encounter(cohort, label_col, task_type)

    out = _block(enc, label_col, task_type)      # flat headline = encounter grain
    out["n_rows"] = cohort.height
    out["by_grain"] = {
        "encounter": _block(enc, label_col, task_type),
        "row": _block(cohort, label_col, task_type),
    }
    return out


def _to_encounter(cohort: pl.DataFrame, label_col: str, task_type: str) -> pl.DataFrame:
    """One row per stitched encounter. The encounter's label is ever-positive
    (max over its rows) for binary tasks; covariates are constant per encounter."""
    if JOIN_KEY not in cohort.columns:
        return cohort
    keep = [c for c in (CONTINUOUS_VARS + CATEGORICAL_VARS + ["split"])
            if c in cohort.columns]
    label_agg = (pl.col(label_col).max() if task_type == "binary"
                 else pl.col(label_col).first())
    return cohort.group_by(JOIN_KEY).agg(
        [label_agg.alias(label_col), *[pl.col(c).first() for c in keep]]
    )


def _block(df: pl.DataFrame, label_col: str, task_type: str) -> dict:
    out: dict = {
        "n_total": df.height,
        "continuous": {v: _continuous_stats(df, v) for v in CONTINUOUS_VARS
                       if v in df.columns},
        "categorical": {v: _categorical_stats(df, v) for v in CATEGORICAL_VARS
                        if v in df.columns},
        "by_split": {},
        "by_label": {},
    }
    if "split" in df.columns:
        for split_val in df["split"].unique().drop_nulls().to_list():
            out["by_split"][str(split_val)] = _stratum(df.filter(pl.col("split") == split_val))

    if label_col not in df.columns:
        return out

    if task_type in ("binary", "multiclass"):
        for level in df[label_col].unique().drop_nulls().to_list():
            out["by_label"][str(level)] = _stratum(df.filter(pl.col(label_col) == level))
    elif task_type == "regression":
        q1, q3 = df[label_col].quantile(0.25), df[label_col].quantile(0.75)
        out["by_label"]["low_q1"] = _stratum(df.filter(pl.col(label_col) <= q1))
        out["by_label"]["high_q4"] = _stratum(df.filter(pl.col(label_col) >= q3))
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
