"""KDIGO stage 2 AKI logic + Koyner-style cohort exclusions.

Reference:
  Koyner JL, Carey KA, Edelson DP, Churpek MM.
  The Development of a Machine Learning Inpatient Acute Kidney Injury
  Prediction Model. Crit Care Med 46(7):1070-1077, 2018.
  DOI: 10.1097/CCM.0000000000003123

KDIGO stage 2 definition (creatinine criterion only):
  SCr >= 2.0 * baseline, where baseline = min SCr in the prior 7 days
  at the time of the measurement, falling back to the earliest SCr of
  the hospitalization if no prior reading is in window.

Urine-output KDIGO criteria are NOT implemented: they require continuous
hourly UO recording which is uneven across CLIF sites. Koyner's model
is also creatinine-only — see paper Methods.

Exclusions (verbatim from Koyner Methods):
  1. first SCr >= 3.0 mg/dL                              -> rule_first_scr
  2. ICD code for CKD stage 4 or higher                  -> rule_ckd_icd (NOT IMPLEMENTED;
                                                           CLIF Diagnoses not yet exposed
                                                           via FLAIR loaders)
  3. RRT within 48h of first SCr                         -> rule_early_rrt (skipped if
                                                           rrt_df is None)
"""

from __future__ import annotations

import polars as pl

KDIGO_STAGE2_RATIO = 2.0
BASELINE_WINDOW_HOURS_DEFAULT = 7 * 24  # 7 days
KOYNER_FIRST_SCR_EXCLUSION_MGDL = 3.0
KOYNER_EARLY_RRT_WINDOW_HOURS = 48


def compute_kdigo_stage2(
    creatinine_df: pl.DataFrame,
    baseline_window_hours: int = BASELINE_WINDOW_HOURS_DEFAULT,
) -> pl.DataFrame:
    """Flag every SCr measurement as KDIGO stage 2+ (creatinine criterion).

    Expected input columns:
      hospitalization_id, recorded_dttm, lab_value (SCr in mg/dL)

    Returns the same rows with two added columns:
      baseline_scr           — baseline used for the comparison (mg/dL)
      kdigo_stage2_flag      — bool, True iff lab_value >= 2.0 * baseline_scr

    Baseline rule: min SCr in (recorded_dttm - baseline_window_hours, recorded_dttm).
    If no prior SCr exists in that window, baseline = first SCr observed for the
    hospitalization (the admission/index SCr).
    """
    if creatinine_df.height == 0:
        return creatinine_df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("baseline_scr"),
            pl.lit(None, dtype=pl.Boolean).alias("kdigo_stage2_flag"),
        ])

    scr = (
        creatinine_df.select([
            "hospitalization_id",
            "recorded_dttm",
            pl.col("lab_value").cast(pl.Float64).alias("lab_value"),
        ])
        .filter(pl.col("lab_value").is_not_null())
        .sort(["hospitalization_id", "recorded_dttm"])
    )

    # Index SCr per hospitalization (fallback baseline)
    index_scr = (
        scr.group_by("hospitalization_id")
        .agg(pl.col("lab_value").first().alias("index_scr"))
    )

    # Rolling min over prior 7d, exclusive of the current row's own value:
    # polars rolling_min is left-inclusive at the boundary, so we shift by
    # one row within hospitalization to exclude the current SCr from its own
    # baseline window.
    scr = scr.with_columns(
        pl.col("lab_value").shift(1).over("hospitalization_id").alias("_prev_scr")
    )
    rolling = (
        scr.rolling(
            index_column="recorded_dttm",
            period=f"{baseline_window_hours}h",
            group_by="hospitalization_id",
        )
        .agg(pl.col("_prev_scr").min().alias("rolling_min_prior"))
    )
    scr = scr.join(rolling, on=["hospitalization_id", "recorded_dttm"], how="left")
    scr = scr.join(index_scr, on="hospitalization_id", how="left")

    scr = scr.with_columns(
        pl.coalesce(["rolling_min_prior", "index_scr"]).alias("baseline_scr")
    ).with_columns(
        (pl.col("lab_value") >= KDIGO_STAGE2_RATIO * pl.col("baseline_scr"))
        .alias("kdigo_stage2_flag")
    )

    return scr.select([
        "hospitalization_id", "recorded_dttm", "lab_value",
        "baseline_scr", "kdigo_stage2_flag",
    ])


def apply_koyner_exclusions(
    cohort: pl.DataFrame,
    creatinine_df: pl.DataFrame,
    rrt_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Drop hospitalizations matching any Koyner 2018 exclusion rule.

    cohort         : has at least 'hospitalization_id'.
    creatinine_df  : long table with hospitalization_id, recorded_dttm, lab_value.
    rrt_df         : optional. If provided, expected columns:
                       hospitalization_id, recorded_dttm.
                     Pass None to skip rule_early_rrt (documented deviation).

    Returns the cohort filtered to hospitalizations that survive all rules.
    """
    if creatinine_df.height == 0:
        return cohort

    first_scr = (
        creatinine_df.filter(pl.col("lab_value").is_not_null())
        .sort(["hospitalization_id", "recorded_dttm"])
        .group_by("hospitalization_id")
        .agg([
            pl.col("lab_value").first().alias("first_scr_value"),
            pl.col("recorded_dttm").first().alias("first_scr_dttm"),
        ])
    )

    # Rule 1: exclude if first SCr >= 3.0 mg/dL
    excluded_hi_scr = (
        first_scr.filter(pl.col("first_scr_value") >= KOYNER_FIRST_SCR_EXCLUSION_MGDL)
        .select("hospitalization_id")
    )
    excluded_ids = set(excluded_hi_scr["hospitalization_id"].to_list())

    # Rule 3: exclude if RRT within 48h of first SCr
    if rrt_df is not None and rrt_df.height > 0:
        early_rrt = (
            rrt_df.join(first_scr, on="hospitalization_id", how="inner")
            .filter(
                (pl.col("recorded_dttm") - pl.col("first_scr_dttm"))
                <= pl.duration(hours=KOYNER_EARLY_RRT_WINDOW_HOURS)
            )
            .select("hospitalization_id")
            .unique()
        )
        excluded_ids.update(early_rrt["hospitalization_id"].to_list())

    return cohort.filter(~pl.col("hospitalization_id").is_in(list(excluded_ids)))
