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
    "prevalence": float|None,  # binary: positive rate (6 dp). None when either the
                               # positive or negative count is a nonzero value < 10
                               # (prevalence x n would recover a suppressed cell);
                               # true extremes 0.0 / 1.0 are shown. Multiclass:
                               # {level: proportion_or_None}; regression: None.
    "continuous":  { "age_at_admission": {mean, std, median, q25, q75, min, max, n_missing},
                     "sofa_max": {...}, "charlson_score": {...}, "elix_score": {...} },
    "categorical": { "sex_category": {level: count_or_"<10"}, ... },
    "by_split":  { "total": {n, date_range, prevalence, continuous, categorical},  # ALWAYS 3 keys
                   "train": {...}, "test": {...} },          # empty split -> n=0, stats null
                   # date_range = {"start": "Jan 2020", "end": "Mar 2021"}  (month-year, no day)
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

from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress, suppress_dict

JOIN_KEY = "hospitalization_join_id"
CONTINUOUS_VARS = ["age_at_admission", "sofa_max", "charlson_score", "elix_score"]
CATEGORICAL_VARS = ["sex_category", "race_category", "ethnicity_category"]
# Date column used for the per-split calendar range (month-year only). First present wins.
DATE_COLS = ["admission_dttm", "prediction_dttm"]


def _first_date_col(df: pl.DataFrame) -> str | None:
    return next((c for c in DATE_COLS if c in df.columns), None)


def _date_range(df: pl.DataFrame) -> dict:
    """Calendar span of a stratum as month-year strings (e.g. 'Jan 2020'), no day."""
    col = _first_date_col(df)
    if col is None or df.height == 0:
        return {"start": None, "end": None}
    s = df[col].drop_nulls()
    if s.len() == 0:
        return {"start": None, "end": None}
    return {"start": s.min().strftime("%b %Y"), "end": s.max().strftime("%b %Y")}


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
    date_col = _first_date_col(cohort)
    keep = [c for c in (CONTINUOUS_VARS + CATEGORICAL_VARS + ["split"]
                        + ([date_col] if date_col else []))
            if c in cohort.columns]
    aggs = [pl.col(c).first() for c in keep]
    if label_col in cohort.columns:
        label_agg = (pl.col(label_col).max() if task_type == "binary"
                     else pl.col(label_col).first())
        aggs = [label_agg.alias(label_col), *aggs]
    return cohort.group_by(JOIN_KEY).agg(aggs)


def _prevalence(df: pl.DataFrame, label_col: str | None, task_type: str | None):
    """Outcome prevalence of a stratum, suppression-aware.

    Disclosable iff the underlying label counts would survive ``suppress()``:
    a published prevalence times the stratum's n recovers the positive count
    (and 1-p the negative), so any nonzero count below MIN_CELL_COUNT forces
    None. True extremes (0.0 / 1.0) are shown — a true zero is not PHI.
    """
    if label_col is None or label_col not in df.columns:
        return None
    s = df[label_col].drop_nulls()
    n = s.len()
    if n == 0:
        return None
    if task_type == "binary":
        n_pos = int((s > 0).sum())
        n_neg = n - n_pos
        if 0 < n_pos < MIN_CELL_COUNT or 0 < n_neg < MIN_CELL_COUNT:
            return None
        return round(n_pos / n, 6)
    if task_type == "multiclass":
        counts = s.value_counts()
        return {
            str(level): (round(cnt / n, 6) if cnt >= MIN_CELL_COUNT else None)
            for level, cnt in counts.iter_rows()
        }
    return None


def _block(df: pl.DataFrame, label_col: str, task_type: str) -> dict:
    out: dict = {
        "n_total": df.height,
        "prevalence": _prevalence(df, label_col, task_type),
        "continuous": {v: _continuous_stats(df, v) for v in CONTINUOUS_VARS
                       if v in df.columns},
        "categorical": {v: _categorical_stats(df, v) for v in CATEGORICAL_VARS
                        if v in df.columns},
        "by_split": {},
        "by_label": {},
    }
    # Always emit the same 3 columns — total / train / test — so every site (incl.
    # inference-only, where only 'test' exists) produces an identical schema for
    # cross-site aggregation. Missing splits are zero-filled (_stratum on empty).
    has_split = "split" in df.columns
    out["by_split"] = {
        "total": _stratum(df, label_col, task_type),
        "train": _stratum(df.filter(pl.col("split") == "train"), label_col, task_type)
                 if has_split else _stratum(df.head(0), label_col, task_type),
        "test":  _stratum(df.filter(pl.col("split") == "test"), label_col, task_type)
                 if has_split else _stratum(df.head(0), label_col, task_type),
    }

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


def _stratum(df: pl.DataFrame, label_col: str | None = None,
             task_type: str | None = None) -> dict:
    # True zero (e.g. no training at an inference-only site) is not PHI, so report it
    # as 0 rather than the '<10' suppression string used for real small counts.
    # by_label strata omit the label args: prevalence conditioned on the label is
    # degenerate, so the key stays present but None for a uniform stratum schema.
    return {
        "n": 0 if df.height == 0 else suppress(df.height),
        "date_range": _date_range(df),
        "prevalence": _prevalence(df, label_col, task_type),
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
