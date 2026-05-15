"""
Task 9: Reintubation within 72 hours of extubation (binary classification, decision-point).

Clinical question:
  Will this extubation fail — i.e., will the patient require re-initiation
  of invasive mechanical ventilation (IMV) within 72 hours?

Row shape: one row per confirmed extubation event. A patient may
contribute multiple rows if they undergo multiple confirmed
intubate-extubate cycles.

Episode detection uses the consensus-window state machine in
_ventilation_episodes.detect_imv_episodes:
  - IMV start counts only if IMV persists >= 15 min
  - Extubation closes the episode only if non-IMV persists >= 30 min
  - A trach sentinel closes any open episode and stops further counting
These windows filter transient mis-classified RespiratorySupport rows
that would otherwise inflate the extubation count.

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort + ADT
  2. load_respiratory_support() — all device_categories
  3. detect_imv_episodes()    — confirmed (start, end) intervals
  4. extubation events = episodes with non-null episode_end_dttm AND
     episode duration >= 24 h
  5. label = 1 iff any later episode_start_dttm in
              (episode_end, episode_end + 72 h]
  6. drop ambiguous events: patient died within 72 h without
     re-intubation
  7. assign_split()
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import (base_icu_cohort, load_respiratory_support,
                                    read_clif_config)
from flair_benchmark._split import assign_split
from flair_benchmark._ventilation_episodes import detect_imv_episodes

REINTUBATION_HORIZON_HOURS = 72
MIN_PRIOR_IMV_HOURS = 24
INPUT_WINDOW_HOURS = 24

META = {
    "name": "task9_reintubation_72h",
    "display_name": "Reintubation within 72h (decision-point)",
    "task_class": "intervention",
    "temporal_pattern": "fixed_horizon",
    "prediction_unit": "one-per-decision-point",
    "task_type": "binary",
    "label_column": "label_reintubation_72h",
    "positive_class": "reintubated_within_72h",
    "input_window_hours": INPUT_WINDOW_HOURS,
    "prediction_window": [0, REINTUBATION_HORIZON_HOURS],
    "cohort_description": "Adults with a confirmed IMV episode (>= 15 min start "
                          "consensus, >= 30 min extubation consensus) lasting "
                          ">= 24 h that is closed by extubation (not by trach "
                          "or end-of-record)",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "at least one ICU ADT record",
        "confirmed IMV episode with non-null episode_end_dttm",
        "episode duration >= 24 h",
    ],
    "exclusion_criteria": [
        "episode closed by trach sentinel (not a true extubation)",
        "death within 72 h post-extubation without re-intubation "
        "(ambiguous outcome)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
}

OUTPUT_COLS = [
    "hospitalization_id", "extubation_dttm",
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

    closed = (
        episodes.filter(
            pl.col("episode_start_dttm").is_not_null()
            & pl.col("episode_end_dttm").is_not_null()
        )
        .with_columns(
            ((pl.col("episode_end_dttm") - pl.col("episode_start_dttm"))
             .dt.total_seconds() / 3600.0).alias("prior_imv_hours")
        )
        .filter(pl.col("prior_imv_hours") >= MIN_PRIOR_IMV_HOURS)
        .rename({"episode_end_dttm": "extubation_dttm"})
        .select(["hospitalization_id", "extubation_dttm", "prior_imv_hours"])
    )

    later_starts = (
        episodes.filter(pl.col("episode_start_dttm").is_not_null())
        .select(["hospitalization_id",
                 pl.col("episode_start_dttm").alias("next_imv_start")])
    )

    labeled = (
        closed.join(later_starts, on="hospitalization_id", how="left")
        .with_columns(
            (
                pl.col("next_imv_start").is_not_null()
                & (pl.col("next_imv_start") > pl.col("extubation_dttm"))
                & (pl.col("next_imv_start")
                   <= pl.col("extubation_dttm")
                   + pl.duration(hours=REINTUBATION_HORIZON_HOURS))
            ).alias("_hit")
        )
        .group_by(["hospitalization_id", "extubation_dttm"])
        .agg([
            pl.col("_hit").any().alias("_any_hit"),
            pl.col("prior_imv_hours").first(),
        ])
        .with_columns(
            pl.col("_any_hit").fill_null(False)
            .cast(pl.Int32).alias("label_reintubation_72h")
        )
        .select(["hospitalization_id", "extubation_dttm",
                 "prior_imv_hours", "label_reintubation_72h"])
    )

    labeled = _drop_ambiguous_death(labeled, cohort)

    df = labeled.join(cohort, on="hospitalization_id", how="inner").with_columns([
        (pl.col("extubation_dttm")
         - pl.duration(hours=INPUT_WINDOW_HOURS)).alias("window_start"),
        pl.col("extubation_dttm").alias("window_end"),
    ])

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="extubation_dttm")
    df = df.filter(pl.col("split").is_not_null())
    return df.select(OUTPUT_COLS)


def _drop_ambiguous_death(labeled: pl.DataFrame,
                          cohort: pl.DataFrame) -> pl.DataFrame:
    """Drop extubations where the patient dies within 72h without re-intubation."""
    if labeled.height == 0:
        return labeled

    return (
        labeled.join(cohort.select(["hospitalization_id", "death_dttm"]),
                     on="hospitalization_id", how="left")
        .with_columns(
            (
                pl.col("death_dttm").is_not_null()
                & (pl.col("death_dttm") > pl.col("extubation_dttm"))
                & (pl.col("death_dttm")
                   <= pl.col("extubation_dttm")
                   + pl.duration(hours=REINTUBATION_HORIZON_HOURS))
            ).alias("_died_within_72h")
        )
        .filter(~(pl.col("_died_within_72h")
                  & (pl.col("label_reintubation_72h") == 0)))
        .drop(["death_dttm", "_died_within_72h"])
    )
