"""
Extubation failure within 24h of first extubation (binary, one-per-stay).

Clinical question:
  After the first confirmed extubation, will the patient be reintubated
  in the next 24h?

The decision this supports:
  Two-sided — EXTUBATE, or KEEP THE PATIENT INTUBATED. It is anchored at the
  extubation itself, so it fires on LOW risk (clinical_direction "below"): pull
  the tube when P(failure) < 0.10, otherwise leave it in. Nothing else is on the
  table at this timestamp. A tracheostomy in particular is not — it is a decision
  made in a still-intubated patient around vent day 7-14, and you would have to
  reintubate this patient before you could trach them. That decision is a
  different task (icu_daily_ltach).

Threshold — why 0.10:
  The threshold is not a cutoff, it is the exchange rate between the two harms.
  You are indifferent when

      t / (1-t) = harm(one unnecessary continued intubation)
                  / harm(one failed extubation)

  so 0.10 encodes 1:9 — a failed extubation is 9x worse than keeping a ready
  patient intubated. Epstein & Ciubotaru report 43% mortality in reintubated
  patients vs 12% in successfully extubated; discounted heavily for confounding
  by indication, 9x is defensible where 1:4 (t = 0.20) is not.

  Note the horizon. The familiar "acceptable reintubation rate is 10-15%" figure
  is a 48-72h number (Thille HIGH-WEAN, Frutos-Vivar, Epstein); this label is 24h
  and roughly half of reintubations fall outside it, which is why the observed
  rate here is ~5.4%, not ~12%. A threshold anchored to 72h intuition is
  inflated. 0.10 is ~1.85x this cohort's base rate: discriminating, but reaching
  a real fraction of patients.

  Do not confuse the two quantities that land in this range. The population
  reintubation rate is what happens across all extubations; the threshold is the
  individual risk at which the action changes.

LIMITATION — the counterfactual is unobserved:
  100% of this cohort WAS extubated, so "do not extubate" is a recommendation the
  data cannot validate: there are no held-back patients to observe. On the
  decision curve this makes treat-all equal to observed current practice, which
  is the right comparator — but any net benefit above that line assumes the
  failures of the patients the model would have held back would have been avoided
  by waiting, and this dataset cannot test that assumption. Read the DCA as "does
  selective extubation beat extubating everyone, under that assumption", never as
  an estimate of realized benefit.

  Features are cut at prediction_dttm, a fixed lead BEFORE the extubation, for
  the same reason — see PREDICTION_LEAD_MINUTES below.

Row shape: one row per stitched encounter (hospitalization_join_id) with a
confirmed first extubation. Ventilation episodes are detected across the whole
encounter, so a reintubation in a later member hospitalization counts.
Prediction time = first_extubation_dttm - PREDICTION_LEAD_MINUTES.

Two distinct timestamps, do not conflate them:
  extubation_dttm  the clinical event; the label window is measured from here,
                   so "failure within 24h" keeps its usual meaning.
  prediction_dttm  when the model is asked, 30 min earlier; the feature cutoff
                   equals it. This is the decision timestamp — the tube has not
                   been pulled yet, which is the whole point of the task.

Episode detection uses the consensus-window state machine in
cohort.events.detect_imv_episodes:
  - IMV start counts only if IMV persists >= 15 min
  - Extubation closes the episode only if non-IMV persists >= 30 min
  - A trach sentinel closes any open episode and stops further counting
These windows filter transient mis-classified RespiratorySupport rows. They are
artifact filters, NOT a cohort criterion — the cohort floor is MIN_IMV_HOURS.

Pipeline:
  1. base_icu_cohort()        — adult ICU cohort
  2. load_respiratory_support() — all device_categories
  3. detect_imv_episodes()    — confirmed (start, end) intervals
  4. restrict to episode_index == 1 with non-null episode_end_dttm,
     drop pre-existing trach
  4b. drop episodes shorter than MIN_IMV_HOURS (episode_end - episode_start);
     a brief peri-procedural intubation is not the weaning decision modelled
  5. drop trach-closed episodes (trach row within +/-1min of episode_end)
  6. prediction_dttm = extubation_dttm - PREDICTION_LEAD_MINUTES
  7. reintubation_dttm = episode 2's episode_start_dttm (if any)
  8. label = reintubated_in_24h (window measured from extubation_dttm)
  9. drop death in 24h without reintubation (palliative extubation)
 10. drop ambiguous censoring (discharged alive < 24h post-extubation,
     no death, no reintubation)
 11. assign_split()
 12. feature_cutoff_dttm = prediction_dttm (inclusive; the 30-min lead is what
     keeps the extubation itself out, so no extra strictness is needed)
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.cohort.clif import base_icu_cohort, load_respiratory_support, read_clif_config
from flair_benchmark.cohort.encounter import as_encounter_keyed, encounter_attributes
from flair_benchmark.cohort.events import detect_imv_episodes
from flair_benchmark.cohort.stitch import (
    JOIN_ID,
    assign_split,
    load_or_build_encounter_index,
    members_of_joins,
)
from flair_benchmark.tasks._finalize import finalize_cohort

HORIZON_HOURS = 24
TRACH_CLOSURE_TOLERANCE_MIN = 1
# Minimum duration of the first IMV episode: a brief peri-procedural intubation
# is not the weaning decision this task models. 6h is a FLAIR choice, deliberately
# more permissive than the published floors (12h Hernandez 2016 low-risk, 24h
# Thille 2019 HIGH-WEAN and ExPreS 2021, 48h Frutos-Vivar 2006), which keeps the
# cohort broad while still excluding same-day post-op extubations.
MIN_IMV_HOURS = 6
# How far BEFORE the extubation the model is asked. A prediction stamped at the
# extubation itself is not a decision, it is a description: the tube is already
# out and the respiratory_support row that says so is charted at that instant.
# 30 min is the clinical lead time — long enough that the answer can still change
# the action, short enough that the SBT and the last vent settings (the strongest
# signal this task has) are still on the correct side of the cutoff.
PREDICTION_LEAD_MINUTES = 30

META = {
    "name": "extubation_failure_24h",
    "display_name": "Extubation failure within 24h (one-per-stay)",
    "task_class": "intervention",
    "temporal_pattern": "episode",
    "prediction_unit": "one-per-stay",
    "task_type": "binary",
    # Clinical policy — this task owns it; value-validated in tasks/__init__.py.
    # The decision is two-sided and fires on LOW risk: extubate when P(failure) is
    # under 0.10, otherwise leave the tube in. See "Threshold" in the docstring.
    "clinical_threshold": 0.10,
    "clinical_action": "extubate; otherwise keep the patient intubated",
    "clinical_direction": "below",
    "report_mode": "episodic",
    "dca_axis": {"event": "successful extubation",
                 "action": "extubate",
                 "inaction": "keep the patient intubated"},
    "label_column": "label_extubation_failure_24h",
    "positive_class": "reintubation_within_24h_post_extubation",
    "input_window_hours": None,
    # Relative to prediction_dttm, which sits 30 min (0.5h) before the
    # extubation: the label window still runs 0-24h from the extubation itself.
    "prediction_window": [PREDICTION_LEAD_MINUTES / 60,
                          HORIZON_HOURS + PREDICTION_LEAD_MINUTES / 60],
    "cohort_description": "Adults with a first confirmed IMV episode lasting "
                          ">= 6h and closed by extubation (not trach, not "
                          "end-of-record); one row per hospitalization anchored "
                          "30 min before the first extubation_dttm",
    "inclusion_criteria": [
        "age_at_admission >= 18",
        "at least one ICU ADT record",
        "first IMV episode (episode_index == 1) has non-null episode_end_dttm",
        "first IMV episode duration >= 6 hours",
    ],
    "exclusion_criteria": [
        "pre-existing tracheostomy",
        "first episode closed by trach sentinel "
        "(trach row within +/-1min of episode_end_dttm)",
        "death within 24h of extubation without reintubation "
        "(palliative extubation)",
        "discharged alive within 24h of extubation with no reintubation and "
        "no death (ambiguous — observability ends before window closes)",
    ],
    "evaluation_metrics": ["auroc", "auprc", "accuracy", "precision", "recall",
                           "f1", "specificity", "npv"],
}

OUTPUT_COLS = [
    "hospitalization_id", "hospitalization_join_id", "prediction_dttm",
    # Carried so the lead is auditable from the cohort alone: the label window
    # runs from this timestamp, prediction_dttm sits 30 min before it.
    "extubation_dttm",
    "feature_cutoff_dttm",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)
    cohort, _adt = base_icu_cohort(cfg)
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))

    # Load respiratory support for every hospitalization in a cohort ENCOUNTER,
    # not just the cohort's own ids. The relabel below reasons across the whole
    # stitched encounter, so a sibling admission whose rows were never loaded
    # would leave that encounter's ventilation history silently incomplete —
    # a reintubation in the sibling stay simply would not exist.
    hosp_ids = members_of_joins(idx, cohort[JOIN_ID].cast(pl.Utf8).unique().to_list())
    resp = load_respiratory_support(cfg, hospitalization_ids=hosp_ids)

    # Encounter grain: relabel respiratory support to the stitched encounter so
    # ventilation episodes (and a reintubation in a later member hospitalization)
    # are detected across the whole encounter. The join id now lives in the
    # "hospitalization_id" slot until the final join to encounter_attributes.
    resp = as_encounter_keyed(resp, idx)
    episodes = detect_imv_episodes(resp)

    first_ext = (
        episodes
        .filter(~pl.col("is_pre_existing_trach"))
        .filter(pl.col("episode_index") == 1)
        .filter(pl.col("episode_end_dttm").is_not_null())
        .filter(
            (pl.col("episode_end_dttm") - pl.col("episode_start_dttm"))
            >= pl.duration(hours=MIN_IMV_HOURS)
        )
        .select([
            "hospitalization_id",
            pl.col("episode_start_dttm").alias("first_intubation_dttm"),
            pl.col("episode_end_dttm").alias("extubation_dttm"),
        ])
    )

    first_ext = _drop_trach_closed(first_ext, resp)

    # The decision timestamp, PREDICTION_LEAD_MINUTES before the tube comes out.
    # MIN_IMV_HOURS (6h) guarantees it still falls inside the ventilation episode.
    first_ext = first_ext.with_columns(
        (pl.col("extubation_dttm")
         - pl.duration(minutes=PREDICTION_LEAD_MINUTES)).alias("prediction_dttm")
    )

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
        .rename({"hospitalization_id": "hospitalization_join_id"})
        .join(encounter_attributes(cfg, cohort),
              on="hospitalization_join_id", how="inner")
    )

    # The label window is anchored at the EXTUBATION, not at prediction_dttm.
    # "Failure within 24h" is 24h of being off the vent; moving the anchor with
    # the prediction lead would quietly redefine the outcome as 23.5h.
    horizon = pl.col("extubation_dttm") + pl.duration(hours=HORIZON_HOURS)
    died_in_window = (
        pl.col("death_dttm").is_not_null()
        & (pl.col("death_dttm") > pl.col("extubation_dttm"))
        & (pl.col("death_dttm") <= horizon)
    )
    reintubated_in_window = (
        pl.col("reintubation_dttm").is_not_null()
        & (pl.col("reintubation_dttm") > pl.col("extubation_dttm"))
        & (pl.col("reintubation_dttm") <= horizon)
    )

    df = df.with_columns([
        died_in_window.alias("_died"),
        reintubated_in_window.alias("_reintub"),
        # Observability ends at the EARLIER of death and discharge. death_dttm is
        # patient-level and can post-date discharge (registry deaths months out);
        # coalesce(death, discharge) would let such a death mark a stay discharged
        # alive <24h post-extubation as fully observed, defeating the
        # ambiguous-censoring exclusion.
        pl.min_horizontal("death_dttm", "discharge_dttm").alias("_obs_end"),
    ]).with_columns(
        pl.col("_reintub")
        .cast(pl.Int32)
        .alias(META["label_column"])
    )

    df = df.filter(
        pl.col("_reintub")
        | (
            ~pl.col("_died")
            & (pl.col("_obs_end") >= pl.col("extubation_dttm")
               + pl.duration(hours=HORIZON_HOURS))
        )
    )

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())

    # Feature cutoff = prediction_dttm, i.e. PREDICTION_LEAD_MINUTES before the
    # extubation. The lead IS the leakage guard: episode_end_dttm is the first
    # non-IMV respiratory_support row (cohort.events.detect_imv_episodes), the row
    # that ANNOUNCES the extubation, and anyone pulling respiratory_support
    # (device_category / mode_category / fio2_set / peep_set) via clifpy's
    # create_wide_dataset or otherwise gets no automatic guard. Cutting 30 min back
    # puts that row — and the peri-extubation charting around it — safely on the
    # far side, so an inclusive comparison (`time <= feature_cutoff_dttm`) is
    # correct and no per-task strictness is needed. Stamped explicitly rather than
    # left to the _finalize.py default because OUTPUT_COLS names the column.
    df = df.with_columns(
        pl.col("prediction_dttm").alias("feature_cutoff_dttm")
    )
    return finalize_cohort(df, OUTPUT_COLS, META)


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
        first_ext.select(["hospitalization_id", "extubation_dttm"])
        .join(trach_rows, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("recorded_dttm") >= pl.col("extubation_dttm") - tol)
            & (pl.col("recorded_dttm") <= pl.col("extubation_dttm") + tol)
        )
        .select("hospitalization_id")
        .unique()["hospitalization_id"]
        .to_list()
    )

    return first_ext.filter(
        ~pl.col("hospitalization_id").is_in(trach_closed_ids)
    )
