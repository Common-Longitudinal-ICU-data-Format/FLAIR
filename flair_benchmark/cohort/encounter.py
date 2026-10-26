"""Stitched-encounter ICU cohort + first-ICU-stay daily grid.

Shared by the continuous one-per-window ICU tasks (`icu_daily_mortality` and
`icu_daily_ltach`). Those two tasks are identical except for the label they put
on the encounter's final disposition, so all of the cohort, timing and grid logic
lives here and each task adds only its own label expression.

Grain: one entity per ``hospitalization_join_id`` (the stitched encounter), NOT
per ``hospitalization_id``. A patient discharged and re-admitted within the
stitch gap (ED -> inpatient, a facility switch) is ONE encounter; computing the
first ICU stay and the outcome label per hospitalization_id would split that
encounter into two overlapping prediction series with possibly conflicting
labels. We collapse to the encounter and:

  * take the first ICU stay across the whole stitched encounter
    (``compute_readmission_timing`` run at join-id grain — contiguous ICU ADT
    rows, including those spanning the A->B stitch boundary, merge into one
    block);
  * take the label from the encounter's FINAL disposition — the
    ``discharge_category`` of the member hospitalization with the latest
    ``discharge_dttm``. That member may be a NON-ICU hospitalization that the ICU
    base cohort never loads, so we resolve members through the stitch index and
    load their hospitalization rows explicitly.

Prediction grid (one row per local-07:00 day):
  pred_start = (first_icu_start_time truncated to date) + 1 day + 7 hours
  pred_stop  = min( coalesce(first_icu_end_time, discharge_dttm),  # null/terminal ICU end -> discharge
                    discharge_dttm,                                # cap at encounter discharge
                    death_dttm )                                   # earliest in-stay death
  grid = datetime_ranges(pred_start, pred_stop, interval="24h", closed="left")

``pred_stop`` is a true earliest-of (``min_horizontal``), NOT a coalesce of
death over discharge: ``death_dttm`` is patient-level and can post-date this
discharge, so it must be a min term (a future death is dominated by the earlier
discharge/ICU-end; an in-stay death correctly shortens the grid). ``closed=
"left"`` keeps the last row strictly before the stop, so a prediction is never
made at the discharge/death instant. Predictions therefore occur ONLY during the
encounter's first ICU stay.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark.cohort.clif import base_icu_cohort, load_hospitalization
from flair_benchmark.cohort.events import compute_readmission_timing
from flair_benchmark.cohort.stitch import attach_join_id, load_or_build_encounter_index
from flair_benchmark.cohort.strobe import record

PRED_HOUR = 7
GRID_STEP = "24h"

# Columns every consumer (Task 1 / Task 2) needs out of the grid.
GRID_COLS = [
    "hospitalization_join_id", "hospitalization_id", "patient_id",
    "prediction_dttm", "discharge_category",
    "age_at_admission", "sex_category", "race_category", "ethnicity_category",
]


def as_encounter_keyed(df: pl.DataFrame,
                       idx: pl.DataFrame | None = None) -> pl.DataFrame:
    """Relabel a hospitalization-keyed table to ENCOUNTER grain.

    Replaces ``hospitalization_id`` with the row's ``hospitalization_join_id`` so
    any per-hospitalization detector (``compute_readmission_timing``,
    ``detect_imv_episodes``, ``compute_ase``, a vitals min, ...) that groups over
    ``hospitalization_id`` instead aggregates over the stitched encounter — no
    change to the detector itself. Rename the key back afterwards.

    If ``df`` already carries ``hospitalization_join_id`` (e.g. the ADT frame from
    base_icu_cohort) ``idx`` may be omitted; otherwise the stitch index is used to
    attach it (with the self-fallback in ``attach_join_id``).
    """
    if "hospitalization_join_id" not in df.columns:
        if idx is None:
            raise ValueError("as_encounter_keyed: df lacks hospitalization_join_id "
                             "and no stitch index was provided")
        df = attach_join_id(df, idx)
    return (
        df.drop("hospitalization_id")
        .rename({"hospitalization_join_id": "hospitalization_id"})
    )


def _encounter_timing(adt: pl.DataFrame) -> pl.DataFrame:
    """First ICU stay per stitched encounter.

    Reuses ``compute_readmission_timing`` (which groups over ``hospitalization_id``)
    at encounter grain via ``as_encounter_keyed``. Also returns
    ``hospitalization_id`` = the hospitalization that owns the first ICU block
    (earliest ICU ``in_dttm`` in the encounter), used as the representative id in
    the audit trail (prediction_id).
    """
    timing = (
        compute_readmission_timing(as_encounter_keyed(adt))
        .rename({"hospitalization_id": "hospitalization_join_id"})
        .select([
            "hospitalization_join_id",
            "first_icu_start_time",
            "first_icu_end_time",
        ])
    )

    # Representative hospitalization_id: the real hospitalization that owns the
    # earliest ICU ADT row in the encounter (robust to the std_location merge).
    rep = (
        adt.filter(pl.col("location_category").str.to_lowercase() == "icu")
        .group_by("hospitalization_join_id")
        .agg(
            pl.col("hospitalization_id")
            .sort_by("in_dttm")
            .first()
            .alias("hospitalization_id")
        )
    )
    return timing.join(rep, on="hospitalization_join_id", how="left")


def pred_stop_expr() -> pl.Expr:
    """The first-ICU-stay stop boundary: earliest of (ICU end | discharge),
    encounter discharge, and (patient-level) death.

    A true ``min_horizontal`` — NOT a coalesce of death over discharge — so a
    future patient death is dominated by the earlier discharge/ICU-end while an
    in-stay death still shortens the grid. ``coalesce(first_icu_end_time,
    discharge_dttm)`` supplies a fallback when the first ICU stay is terminal
    with a null ``out_dttm``. ``min_horizontal`` skips a null death.
    """
    return pl.min_horizontal(
        pl.coalesce(["first_icu_end_time", "discharge_dttm"]),
        pl.col("discharge_dttm"),
        pl.col("death_dttm"),
    )


def collapse_members(members: pl.DataFrame) -> pl.DataFrame:
    """Collapse join-id-tagged member hospitalizations to one row per encounter:
    encounter admission (min), discharge (max), final disposition (the member
    with the latest discharge_dttm), admitting-member age, and a representative
    ``hospitalization_id`` = the admitting member (earliest admission)."""
    return (
        members.group_by("hospitalization_join_id")
        .agg([
            pl.col("admission_dttm").min().alias("admission_dttm"),
            pl.col("discharge_dttm").max().alias("discharge_dttm"),
            pl.col("discharge_category")
                .sort_by("discharge_dttm").last().alias("discharge_category"),
            pl.col("age_at_admission")
                .sort_by("admission_dttm").first().alias("age_at_admission"),
            pl.col("hospitalization_id")
                .sort_by("admission_dttm").first().alias("hospitalization_id"),
        ])
    )


# Backwards-compatible private alias.
_collapse_members = collapse_members


def collapse_demographics(cohort: pl.DataFrame) -> pl.DataFrame:
    """Patient-level attributes per stitched encounter (constant across members)."""
    return (
        cohort.group_by("hospitalization_join_id")
        .agg([
            pl.col("patient_id").first().alias("patient_id"),
            pl.col("death_dttm").first().alias("death_dttm"),
            pl.col("sex_category").first().alias("sex_category"),
            pl.col("race_category").first().alias("race_category"),
            pl.col("ethnicity_category").first().alias("ethnicity_category"),
        ])
    )


def encounter_attributes(cfg: dict, cohort: pl.DataFrame) -> pl.DataFrame:
    """One row per stitched encounter: patient-level demographics + the
    encounter's admission/discharge window and FINAL disposition.

    ``hospitalization_id`` is the admitting member (representative for the audit
    trail). Tasks anchored inside the ICU stay (T1/T2/T5) typically drop this and
    use the first-ICU owner from the timing step instead.
    """
    idx = load_or_build_encounter_index(cfg, cfg.get("stitch_time_interval_hours", 6))
    demo = collapse_demographics(cohort)
    disp = _encounter_disposition(cfg, cohort, idx)
    return demo.join(disp, on="hospitalization_join_id", how="inner")


def _encounter_disposition(cfg: dict, cohort: pl.DataFrame,
                           idx: pl.DataFrame) -> pl.DataFrame:
    """Final disposition + encounter admission/discharge per stitched encounter.

    The encounter's outcome is the member hospitalization with the latest
    ``discharge_dttm`` — possibly a non-ICU hospitalization absent from the ICU
    base cohort — so we pull all member hospitalizations through the stitch index.
    """
    join_ids = cohort["hospitalization_join_id"].unique().to_list()

    # Member hospitalization_ids of every ICU encounter (from the stitch index),
    # unioned with the cohort's own ids so self-mapped (never-stitched) encounters
    # are always covered even if absent from the index.
    member_ids = (
        idx.filter(pl.col("hospitalization_join_id").is_in(join_ids))
        .select(pl.col("hospitalization_id").cast(pl.Utf8))
    )
    member_ids = (
        pl.concat([
            member_ids,
            cohort.select(pl.col("hospitalization_id").cast(pl.Utf8)),
        ])
        .unique()["hospitalization_id"]
        .to_list()
    )

    members = load_hospitalization(
        cfg,
        hospitalization_ids=member_ids,
        columns=["hospitalization_id", "admission_dttm", "discharge_dttm",
                 "discharge_category", "age_at_admission"],
    )
    members = attach_join_id(members, idx)
    return _collapse_members(members)


def build_icu_encounter_cohort(cfg: dict) -> pl.DataFrame:
    """One row per local-07:00 day of each stitched ICU encounter's first ICU stay.

    Returns a frame with ``GRID_COLS``; the caller adds its own label on
    ``discharge_category`` (the encounter's final disposition), then splits and
    finalizes. Empty if no encounter has a first ICU stay long enough for one
    07:00 grid point after the admission day.
    """
    cohort, adt = base_icu_cohort(cfg)

    # Encounter attributes carry an admitting-member hospitalization_id; for the
    # ICU daily grid we prefer the first-ICU owner from the timing step, so drop
    # the admitting-member id before joining timing.
    enc = encounter_attributes(cfg, cohort).drop("hospitalization_id")
    timing = _encounter_timing(adt)
    enc = enc.join(timing, on="hospitalization_join_id", how="inner").pipe(
        record, "first ICU stay defined")

    enc = enc.with_columns([
        (
            pl.col("first_icu_start_time").dt.truncate("1d")
            + pl.duration(days=1, hours=PRED_HOUR)
        ).alias("pred_start"),
        pred_stop_expr().alias("pred_stop"),
    ])

    grid = (
        enc.with_columns(
            pl.datetime_ranges(
                start=pl.col("pred_start"),
                end=pl.col("pred_stop"),
                interval=GRID_STEP,
                closed="left",
            ).alias("prediction_dttm")
        )
        .explode("prediction_dttm")
        .filter(pl.col("prediction_dttm").is_not_null())
        .pipe(record, "first ICU stay contains >= one 07:00 grid point")
    )

    return grid.select(GRID_COLS)
