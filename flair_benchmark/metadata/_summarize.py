"""Per-task summarizer used by `flair extract-metadata`.

Takes a task module, its materialized cohort, and per-stay SOFA /
comorbidity tables; returns a JSON-ready dict describing the cohort.
Branches on `task_type` (binary / multiclass / regression) and
`prediction_unit` (one-per-stay / decision-point / timestep) to choose
the right label-block shape.

HIPAA conventions:
  * Cell counts < 10 are replaced with the string "<10" via
    `flair_benchmark.report._suppress.suppress`.
  * Demographic subgroups with n < 10 are dropped from label_by_subgroup
    entirely (matches `flair_benchmark.report.fairness.MIN_SUBGROUP_N`).
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress, suppress_dict

SUBGROUP_COLS = ["sex_category", "race_category", "ethnicity_category", "age_band"]


def summarize_task(task_module,
                   cohort: pl.DataFrame,
                   admission_dttm: pl.DataFrame,
                   sofa_per_stay: pl.DataFrame | None,
                   comorb_per_stay: pl.DataFrame | None) -> dict:
    """Build the per-task block.

    cohort:           parquet emitted by task_module.build()
    admission_dttm:   per-hospitalization [hospitalization_id, admission_dttm]
                      (used for yearly bucketing — cohort itself may not
                       carry admission_dttm, depending on the task)
    sofa_per_stay:    optional [hospitalization_id, sofa_max]
    comorb_per_stay:  optional [hospitalization_id, charlson_score, elix_score]
    """
    meta = task_module.META
    label_col = meta["label_column"]
    task_type = meta["task_type"]
    prediction_unit = meta["prediction_unit"]

    cohort = _attach_age_band(cohort)
    cohort = _attach_admission_year(cohort, admission_dttm)

    n_rows = cohort.height
    n_stays = cohort.select("hospitalization_id").n_unique()

    block: dict = {
        "n_rows": n_rows,
        "n_stays": n_stays,
        "label": _label_block(cohort, label_col, task_type, prediction_unit, meta),
        "demographics": _demographics_block(cohort),
        "label_by_subgroup": _label_by_subgroup(
            cohort, label_col, task_type, prediction_unit
        ),
        "yearly": _yearly_block(cohort, label_col, task_type, prediction_unit),
    }

    if sofa_per_stay is not None and sofa_per_stay.height > 0:
        block["sofa_hospitalization_max"] = _per_stay_score_summary(
            cohort, sofa_per_stay, "sofa_max"
        )
    if comorb_per_stay is not None and comorb_per_stay.height > 0:
        block["charlson_admission"] = _per_stay_score_summary(
            cohort, comorb_per_stay, "charlson_score"
        )
        block["elixhauser_admission"] = _per_stay_score_summary(
            cohort, comorb_per_stay, "elix_score"
        )

    return block


# ---- label blocks -----------------------------------------------------------

def _label_block(cohort: pl.DataFrame, label_col: str, task_type: str,
                 prediction_unit: str, meta: dict) -> dict:
    if task_type == "binary":
        return _binary_label_block(cohort, label_col, prediction_unit)
    if task_type == "multiclass":
        return _multiclass_label_block(cohort, label_col, meta)
    if task_type == "regression":
        return _regression_label_block(cohort, label_col)
    raise ValueError(f"Unknown task_type: {task_type}")


def _binary_label_block(cohort: pl.DataFrame, label_col: str,
                        prediction_unit: str) -> dict:
    pos = int(cohort.filter(pl.col(label_col) == 1).height)
    neg = int(cohort.filter(pl.col(label_col) == 0).height)
    n = pos + neg

    if prediction_unit == "one-per-stay":
        return {
            "type": "binary",
            "positive_count": suppress(pos),
            "negative_count": suppress(neg),
            "prevalence_per_prediction_point": _safe_div(pos, n),
        }

    # decision-point / timestep
    stays_ever_pos = int(
        cohort.filter(pl.col(label_col) == 1)
        .select("hospitalization_id").n_unique()
    )
    n_stays = cohort.select("hospitalization_id").n_unique()
    return {
        "type": "binary",
        "positive_rows": suppress(pos),
        "negative_rows": suppress(neg),
        "stays_ever_positive": suppress(stays_ever_pos),
        "prevalence_per_prediction_point": _safe_div(pos, n),
        "prevalence_per_hospitalization": _safe_div(stays_ever_pos, n_stays),
    }


def _multiclass_label_block(cohort: pl.DataFrame, label_col: str,
                            meta: dict) -> dict:
    class_names = meta.get("class_names")
    counts_raw = (
        cohort.group_by(label_col).agg(pl.len().alias("n"))
        .sort(label_col).to_dicts()
    )
    by_class = {int(r[label_col]): int(r["n"]) for r in counts_raw}
    if class_names is None:
        keys = sorted(by_class.keys())
        class_names = [str(k) for k in keys]
    counts = [by_class.get(i, 0) for i in range(len(class_names))]
    total = sum(counts) or 1
    return {
        "type": "multiclass",
        "classes": class_names,
        "counts": [suppress(c) for c in counts],
        "shares": [round(c / total, 4) for c in counts],
    }


def _regression_label_block(cohort: pl.DataFrame, label_col: str) -> dict:
    s = cohort[label_col].drop_nulls()
    if s.len() == 0:
        return {"type": "regression", "n": 0}
    return {
        "type": "regression",
        "n": suppress(int(s.len())),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "std": round(float(s.std()), 4),
    }


# ---- demographics -----------------------------------------------------------

def _demographics_block(cohort: pl.DataFrame) -> dict:
    out: dict = {}
    # Aggregate to stay-grain so decision-point tasks don't over-count.
    stay_grain = cohort.unique(subset=["hospitalization_id"])
    for col in SUBGROUP_COLS:
        if col not in stay_grain.columns:
            continue
        counts = (
            stay_grain.group_by(col).agg(pl.len().alias("n")).sort(col)
        )
        as_dict = {str(r[col]): int(r["n"]) for r in counts.iter_rows(named=True)
                   if r[col] is not None}
        out[col] = suppress_dict(as_dict)
    return out


def _label_by_subgroup(cohort: pl.DataFrame, label_col: str, task_type: str,
                      prediction_unit: str) -> dict:
    if task_type != "binary":
        return {}

    out: dict = {}
    for col in SUBGROUP_COLS:
        if col not in cohort.columns:
            continue
        per_level: dict = {}
        for level in cohort[col].drop_nulls().unique().to_list():
            sub_rows = cohort.filter(pl.col(col) == level)
            stay_grain = (
                sub_rows.group_by("hospitalization_id")
                .agg(pl.col(label_col).max().alias(label_col))
            )
            n_stays = stay_grain.height
            if n_stays < MIN_CELL_COUNT:
                continue
            stays_ever_pos = int(
                stay_grain.filter(pl.col(label_col) == 1).height
            )
            if prediction_unit == "one-per-stay":
                per_level[str(level)] = {
                    "n_stays": n_stays,
                    "stays_ever_positive": suppress(stays_ever_pos),
                    "prevalence_per_prediction_point": _safe_div(
                        stays_ever_pos, n_stays
                    ),
                }
            else:
                n_rows = sub_rows.height
                positive_rows = int(
                    sub_rows.filter(pl.col(label_col) == 1).height
                )
                per_level[str(level)] = {
                    "n_rows": n_rows,
                    "positive_rows": suppress(positive_rows),
                    "n_stays": n_stays,
                    "stays_ever_positive": suppress(stays_ever_pos),
                    "prevalence_per_prediction_point": _safe_div(
                        positive_rows, n_rows
                    ),
                    "prevalence_per_hospitalization": _safe_div(
                        stays_ever_pos, n_stays
                    ),
                }
        if per_level:
            out[col] = per_level
    return out


# ---- per-stay scores --------------------------------------------------------

def _per_stay_score_summary(cohort: pl.DataFrame, scores: pl.DataFrame,
                            score_col: str) -> dict:
    stay_ids = cohort.select("hospitalization_id").unique()
    joined = stay_ids.join(scores, on="hospitalization_id", how="left")
    s = joined[score_col].drop_nulls()
    n_total = stay_ids.height
    if s.len() == 0:
        return {"n": 0, "n_missing": n_total}
    return {
        "n": suppress(int(s.len())),
        "n_missing": suppress(int(n_total - s.len())),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "p75": round(float(s.quantile(0.75)), 4),
    }


# ---- yearly -----------------------------------------------------------------

def _yearly_block(cohort: pl.DataFrame, label_col: str, task_type: str,
                 prediction_unit: str) -> dict:
    if "_admission_year" not in cohort.columns:
        return {}

    out: dict = {}
    for year in sorted(cohort["_admission_year"].drop_nulls().unique().to_list()):
        sub = cohort.filter(pl.col("_admission_year") == year)
        rows = sub.height
        stays = sub.select("hospitalization_id").n_unique()
        entry: dict = {
            "rows": suppress(rows),
            "stays": suppress(stays),
        }
        if task_type == "binary":
            pos_rows = int(sub.filter(pl.col(label_col) == 1).height)
            if prediction_unit == "one-per-stay":
                entry["positive"] = suppress(pos_rows)
                entry["prevalence_per_prediction_point"] = _safe_div(
                    pos_rows, rows
                )
            else:
                ever_pos = int(
                    sub.filter(pl.col(label_col) == 1)
                    .select("hospitalization_id").n_unique()
                )
                entry["positive_rows"] = suppress(pos_rows)
                entry["stays_ever_positive"] = suppress(ever_pos)
                entry["prevalence_per_prediction_point"] = _safe_div(
                    pos_rows, rows
                )
                entry["prevalence_per_hospitalization"] = _safe_div(
                    ever_pos, stays
                )
        out[str(int(year))] = entry
    return out


# ---- helpers ---------------------------------------------------------------

def _attach_age_band(cohort: pl.DataFrame) -> pl.DataFrame:
    if "age_at_admission" not in cohort.columns:
        return cohort
    return cohort.with_columns(
        pl.when(pl.col("age_at_admission") < 40).then(pl.lit("18-39"))
        .when(pl.col("age_at_admission") < 65).then(pl.lit("40-64"))
        .when(pl.col("age_at_admission") < 80).then(pl.lit("65-79"))
        .otherwise(pl.lit("80+"))
        .alias("age_band")
    )


def _attach_admission_year(cohort: pl.DataFrame,
                           admission_dttm: pl.DataFrame) -> pl.DataFrame:
    if admission_dttm is None or admission_dttm.height == 0:
        return cohort
    yrs = admission_dttm.with_columns(
        pl.col("admission_dttm").dt.year().alias("_admission_year")
    ).select(["hospitalization_id", "_admission_year"])
    return cohort.join(yrs, on="hospitalization_id", how="left")


def _safe_div(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0
