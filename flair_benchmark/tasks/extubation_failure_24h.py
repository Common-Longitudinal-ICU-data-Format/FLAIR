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

  Features are cut STRICTLY before prediction_dttm for the same reason — see
  feature_cutoff_dttm below.

Row shape: one row per stitched encounter (hospitalization_join_id) with a
confirmed first extubation. Ventilation episodes are detected across the whole
encounter, so a reintubation in a later member hospitalization counts.
Prediction time = first_extubation_dttm.

Episode detection uses the consensus-window state machine in
_ventilation_episodes.detect_imv_episodes:
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
  6. reintubation_dttm = episode 2's episode_start_dttm (if any)
  7. label = reintubated_in_24h
  8. drop death in 24h without reintubation (palliative extubation)
  9. drop ambiguous censoring (discharged alive < 24h post-extubation,
     no death, no reintubation)
 10. assign_split()
 11. feature_cutoff_dttm = prediction_dttm - 1us (STRICT: features may not see
     the respiratory_support row that announces the extubation)
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import base_icu_cohort, load_respiratory_support, read_clif_config
from flair_benchmark._encounter import as_encounter_keyed, encounter_attributes
from flair_benchmark._split import assign_split
from flair_benchmark._stitch import load_or_build_encounter_index
from flair_benchmark._ventilation_episodes import detect_imv_episodes
from flair_benchmark.tasks._finalize import finalize_cohort

HORIZON_HOURS = 24
TRACH_CLOSURE_TOLERANCE_MIN = 1
# Minimum duration of the first IMV episode: a brief peri-procedural intubation
# is not the weaning decision this task models. 6h is a FLAIR choice, deliberately
# more permissive than the published floors (12h Hernandez 2016 low-risk, 24h
# Thille 2019 HIGH-WEAN and ExPreS 2021, 48h Frutos-Vivar 2006), which keeps the
# cohort broad while still excluding same-day post-op extubations.
MIN_IMV_HOURS = 6

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
    "prediction_window": [0, HORIZON_HOURS],
    "cohort_description": "Adults with a first confirmed IMV episode lasting "
                          ">= 6h and closed by extubation (not trach, not "
                          "end-of-record); one row per hospitalization anchored "
                          "at the first extubation_dttm",
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
    "feature_cutoff_dttm",
    META["label_column"], "split",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    cfg = read_clif_config(clif_config)
    cohort, _adt = base_icu_cohort(cfg)
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))

    hosp_ids = cohort["hospitalization_id"].to_list()
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
        .rename({"hospitalization_id": "hospitalization_join_id"})
        .join(encounter_attributes(cfg, cohort),
              on="hospitalization_join_id", how="inner")
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
        # Observability ends at the EARLIER of death and discharge. death_dttm is
        # patient-level and can post-date discharge (registry deaths months out);
        # coalesce(death, discharge) would let such a death mark a stay discharged
        # alive <24h post-extubation as fully observed, defeating exclusion 9.
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
            & (pl.col("_obs_end") >= pl.col("prediction_dttm")
               + pl.duration(hours=HORIZON_HOURS))
        )
    )

    df = assign_split(df, cfg["site"], train_end, test_start,
                      admission_col="prediction_dttm")
    df = df.filter(pl.col("split").is_not_null())

    # STRICT feature cutoff — this task only. prediction_dttm is episode_end_dttm,
    # the timestamp of the FIRST non-IMV respiratory_support row
    # (_ventilation_episodes.detect_imv_episodes), i.e. the row that ANNOUNCES the
    # extubation. feature_config.yaml feeds respiratory_support device_category /
    # mode_category / fio2_set / peep_set into the wide extractor, so the default
    # inclusive cutoff would hand a "should we extubate?" model the row saying the
    # patient is already extubated.
    #
    # The 1us is NOT the exact representable predecessor — cohort datetimes arrive
    # as Datetime("ns") through _clif.pd_to_pl. It is a step far below CLIF's
    # charting resolution that reliably excludes the boundary sample, and it is NOT
    # a clinical blanking gap: everything up to the decision stays in, including the
    # SBT (RSI, pressure-support tidal volumes, the last vent settings), which is
    # the most predictive signal this task has.
    df = df.with_columns(
        (pl.col("prediction_dttm") - pl.duration(microseconds=1))
        .alias("feature_cutoff_dttm")
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
