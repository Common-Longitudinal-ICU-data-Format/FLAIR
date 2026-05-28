"""
Task 3: Extubation failure within 24h of first extubation (binary, one-per-stay).

Clinical question:
  After the first confirmed extubation, will the patient accumulate < 1
  ventilator-free day in the next 24h — i.e., die or get reintubated?
  Equivalent to VFD_24h_post_extubation < 1.

Row shape: one row per hospitalization with a confirmed first extubation.
Prediction time = first_extubation_dttm.

Episode detection uses the consensus-window state machine in
_ventilation_episodes.detect_imv_episodes:
  - IMV start counts only if IMV persists >= 15 min
  - Extubation closes the episode only if non-IMV persists >= 30 min
  - A trach sentinel closes any open episode and stops further counting
These windows filter transient mis-classified RespiratorySupport rows.

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort
  2. load_respiratory_support() — all device_categories
  3. detect_imv_episodes()    — confirmed (start, end) intervals
  4. restrict to episode_index == 1 with non-null episode_end_dttm,
     drop pre-existing trach
  5. drop trach-closed episodes (trach row within +/-1min of episode_end)
  6. reintubation_dttm = episode 2's episode_start_dttm (if any)
  7. label = died_in_24h OR reintubated_in_24h
  8. drop ambiguous censoring (discharged alive < 24h post-extubation,
     no death, no reintubation)
  9. assign_split()
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_respiratory_support,
                                    read_clif_config)
from flair_benchmark._split import assign_split
from flair_benchmark._ventilation_episodes import detect_imv_episodes

HORIZON_HOURS = 24
INPUT_WINDOW_HOURS = 24
TRACH_CLOSURE_TOLERANCE_MIN = 1

META = {
    "name": "task3_extubation_failure_24h",
    "display_name": "Extubation failure within 24h (one-per-stay)",
    "task_class": "intervention",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    "label_column": "label_extubation_failure_24h",
    "positive_class": "vfd_lt_1_in_24h_post_extubation",
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": [0, HORIZON_HOURS],
    "cohort_description": "Adults with a first confirmed IMV episode closed by "
                          "extubation (not trach, not end-of-record); one row per "
                          "hospitalization anchored at the first extubation_dttm",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "at least one ICU ADT record",
        "first IMV episode (episode_index == 1) has non-null episode_end_dttm",
    ],
    "exclusion_criteria": [
        "pre-existing tracheostomy",
        "first episode closed by trach sentinel "
        "(trach row within +/-1min of episode_end_dttm)",
        "discharged alive within 24h of extubation with no reintubation and "
        "no death (ambiguous — observability ends before window closes)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
}

OUTPUT_COLS = [
    "hospitalization_id", "prediction_dttm",
    "window_start", "window_end",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)
    cohort, _adt = base_icu_cohort(cfg)

    hosp_ids = cohort["hospitalization_id"].to_list()
    resp = load_respiratory_support(cfg, hospitalization_ids=hosp_ids)

    episodes = detect_imv_episodes(resp)

    first_ext = (
        episodes
        .filter(~pl.col("is_pre_existing_trach"))
        .filter(pl.col("episode_index") == 1)
        .filter(pl.col("episode_end_dttm").is_not_null())
        .select([
            "hospitalization_id",
            pl.col("episode_start_dttm").alias("first_intubation_dttm"),
            pl.col("episode_end_dttm").alias("prediction_dttm"),
        ])
    )

    first_ext = _drop_trach_closed(first_ext, resp)

    reintubation = (
        episodes
        .filter(pl.col("episode_index") == 2)
        .select([
            "hospitalization_id",
            pl.col("episode_start_dttm").alias("reintubation_dttm"),
        ])
    )

    df = (
        first_ext
        .join(reintubation, on="hospitalization_id", how="left")
        .join(cohort, on="hospitalization_id", how="inner")
    )

    horizon = pl.col("prediction_dttm") + pl.duration(hours=HORIZON_HOURS)
    died_in_window = (
        pl.col("death_dttm").is_not_null()
        & (pl.col("death_dttm") > pl.col("prediction_dttm"))
        & (pl.col("death_dttm") <= horizon)
    )
    reintubated_in_window = (
        pl.col("reintubation_dttm").is_not_null()
        & (pl.col("reintubation_dttm") > pl.col("prediction_dttm"))
        & (pl.col("reintubation_dttm") <= horizon)
    )

    df = df.with_columns([
        died_in_window.alias("_died"),
        reintubated_in_window.alias("_reintub"),
        pl.coalesce(["death_dttm", "discharge_dttm"]).alias("_obs_end"),
    ]).with_columns(
        (pl.col("_died") | pl.col("_reintub"))
        .cast(pl.Int32)
        .alias(META["label_column"])
    )

    df = df.filter(
        pl.col("_died") | pl.col("_reintub")
        | (pl.col("_obs_end") >= pl.col("prediction_dttm")
           + pl.duration(hours=HORIZON_HOURS))
    )

    df = df.with_columns([
        (pl.col("prediction_dttm")
         - pl.duration(hours=INPUT_WINDOW_HOURS)).alias("window_start"),
        pl.col("prediction_dttm").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)


def _drop_trach_closed(first_ext: pl.DataFrame,
                       resp: pl.DataFrame) -> pl.DataFrame:
    """Drop first episodes whose end coincides with a trach row.

    The consensus state machine in detect_imv_episodes closes an episode
    at a trach sentinel just like a true extubation. For an
    extubation-failure label we must distinguish: a trach decannulation
    is not the clinical event the model predicts.
    """
    if first_ext.height == 0:
        return first_ext

    has_trach_col = "tracheostomy" in resp.columns
    trach_rows = (
        resp.select([
            "hospitalization_id",
            "recorded_dttm",
            pl.col("device_category").str.to_lowercase().alias("_dev_lc"),
            *([pl.col("tracheostomy").cast(pl.Boolean).fill_null(False)]
              if has_trach_col else [pl.lit(False).alias("tracheostomy")]),
        ])
        .filter(
            (pl.col("_dev_lc") == "trach collar")
            | pl.col("tracheostomy")
        )
        .select(["hospitalization_id", "recorded_dttm"])
    )

    if trach_rows.height == 0:
        return first_ext

    tol = pl.duration(minutes=TRACH_CLOSURE_TOLERANCE_MIN)
    trach_closed_ids = (
        first_ext.select(["hospitalization_id", "prediction_dttm"])
        .join(trach_rows, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("recorded_dttm") >= pl.col("prediction_dttm") - tol)
            & (pl.col("recorded_dttm") <= pl.col("prediction_dttm") + tol)
        )
        .select("hospitalization_id")
        .unique()["hospitalization_id"]
        .to_list()
    )

    return first_ext.filter(
        ~pl.col("hospitalization_id").is_in(trach_closed_ids)
    )
