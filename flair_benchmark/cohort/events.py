"""Detect clinical events per hospitalization by reading one CLIF table.

Two detectors that share a shape: each takes a raw CLIF table, works out *when
something clinically meaningful happened* to each hospitalization, and returns a
small frame keyed on hospitalization_id. Neither touches the other; they live
together because they are the same kind of thing and are used the same way.

  Part 1  compute_readmission_timing(adt)   — ADT               -> ICU readmission
  Part 2  detect_imv_episodes(resp)         — RespiratorySupport -> IMV episodes

The shared non-obvious point: a CLIF row is a *sample*, not an event. A single
ADT row is a bed assignment and a single RespiratorySupport row is a device
status, so neither can be read literally. Part 1 merges consecutive rows into
stays; Part 2 confirms transitions against a forward consensus window. Both exist
because the naive row-by-row reading produces clinically wrong answers.

Depends on: polars only — callers pass the already-loaded CLIF frame.
Used by: tasks/icu_readmission.py, tasks/extubation_failure_24h.py,
         cohort/encounter.py.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

# ===========================================================================
# Part 1 — Unplanned ICU readmission timing (from the ADT table)
#
# Per hospitalization: the first ICU stay window, and whether the patient
# returns to the ICU during the SAME hospitalization after an intervening
# location that is not an allowed ICU gap.
#
# A planned return through procedural, radiology, or dialysis locations is NOT a
# readmission: the allowed gap is dropped and the two ICU stays merge into one.
# Task 5 additionally requires a Ward segment between the first ICU and a later
# ICU. That changes only which return supplies second_icu_start_time; it does not
# change the first ICU stay or the cohort exclusions derived from it.
#
# Steps:
#   1. Map raw location_category to a coarse std_location via LOCATION_MAPPING.
#   2. Merge consecutive equivalent-location rows into one location group. All
#      allowed ICU-gap locations are equivalent so mixed allowed gaps form one run.
#   3. Drop allowed gap GROUPS sandwiched between two ICU groups (planned), then
#      re-merge so the ICU groups either side become one stay.
#   4. The 1st merged ICU group gives first_icu_start/end_time. By default the 2nd
#      ICU group is the readmission; Task 5 selects the first later ICU with a Ward
#      group between it and the first ICU.
#   5. Flag direct ICU discharge (last location is ICU, so there was no chance
#      to readmit) and intermediate ER (ER after the first location).
#
# Step order matters: merging BEFORE the allowed-gap drop is what lets a trip
# charted as several allowed categories be recognised as one interval. Dropping
# first only ever matches a single-row trip, so a multi-row excursion would read
# as an unplanned ICU readmission.
# ===========================================================================

# Canonical CLIF categories that may occur, in any contiguous combination,
# between two ICU segments without creating an ICU readmission.
ALLOWED_ICU_GAP_LOCATIONS = frozenset({"Procedural", "Radiology", "Dialysis"})

# Coarse location buckets. Anything not explicitly relevant becomes "Other".
# Whether ICU -> Other -> ICU counts as a readmission depends on the caller's
# require_ward_return (icu_readmission passes True, so it does not).
LOCATION_MAPPING = {
    "icu": "ICU",
    "ward": "Ward",
    "ed": "ER",
    "procedural": "Procedural",
    "stepdown": "Other",
    "l&d": "Other",
    "hospice": "Other",
    "psych": "Other",
    "rehab": "Other",
    "radiology": "Radiology",
    "dialysis": "Dialysis",
    "other": "Other",
}

_READMISSION_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "first_icu_start_time": pl.Datetime,
    "first_icu_end_time": pl.Datetime,
    "second_icu_start_time": pl.Datetime,
    "is_direct_icu_discharge": pl.Boolean,
    "has_intermediate_er": pl.Boolean,
}


def _empty_readmission_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_READMISSION_SCHEMA)


def _merge_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse consecutive equivalent-location rows of a hospitalization into one.

    Returns [hospitalization_id, in_dttm, out_dttm, std_location] sorted by time,
    where in/out span the whole run. Allowed ICU-gap locations are equivalent, so
    mixed radiology/dialysis/procedural rows become one run. Runs twice in the
    readmission pipeline: once on raw ADT rows, once after allowed gaps are removed.
    """
    previous_location = pl.col("std_location").shift(1).over("hospitalization_id")
    same_run = (
        (pl.col("std_location") == previous_location)
        | (
            pl.col("std_location").is_in(ALLOWED_ICU_GAP_LOCATIONS)
            & previous_location.is_in(ALLOWED_ICU_GAP_LOCATIONS)
        )
    )
    df = df.sort(["hospitalization_id", "in_dttm"]).with_columns(
        (~same_run)
        .fill_null(True)
        .cum_sum()
        .over("hospitalization_id")
        .alias("_group")
    )
    return (
        df.group_by(["hospitalization_id", "_group"])
        .agg([
            pl.col("in_dttm").min().alias("in_dttm"),
            pl.col("out_dttm").max().alias("out_dttm"),
            pl.col("std_location").first().alias("std_location"),
        ])
        .sort(["hospitalization_id", "in_dttm"])
        .drop("_group")
    )


def compute_readmission_timing(
    adt: pl.DataFrame,
    *,
    require_ward_return: bool = False,
) -> pl.DataFrame:
    """One row per hospitalization_id with unplanned-readmission timing.

    Columns returned:
      hospitalization_id, first_icu_start_time, first_icu_end_time,
      second_icu_start_time (null if no unplanned readmission),
      is_direct_icu_discharge (bool), has_intermediate_er (bool)

    When require_ward_return is true, second_icu_start_time is the first later ICU
    with an intervening Ward. The first ICU timing and all other flags are unchanged.
    """
    if adt.height == 0:
        return _empty_readmission_frame()

    # 1. Standardize location categories. replace_STRICT with a default, not
    #    plain .replace(): the latter leaves an unmapped value UNCHANGED, so a
    #    site-local location string would survive as itself instead of becoming
    #    "Other" — and the docstring above would be a lie the day a site charts
    #    something outside the mCIDE set.
    adt = adt.with_columns(
        pl.col("location_category")
        .str.to_lowercase()
        .replace_strict(LOCATION_MAPPING, default="Other")
        .alias("std_location")
    ).sort(["hospitalization_id", "in_dttm"])

    # 2. Merge consecutive equivalent-location rows into location groups.
    merged = _merge_runs(adt)

    # 3. Drop allowed gap GROUPS sandwiched between two ICU groups (planned), then
    #    re-merge — the two ICU groups either side are now adjacent and must become
    #    one stay, or the second would read as a readmission.
    merged = merged.with_columns([
        pl.col("std_location").shift(1).over("hospitalization_id").alias("_prev"),
        pl.col("std_location").shift(-1).over("hospitalization_id").alias("_next"),
    ])
    merged = merged.filter(
        ~(
            pl.col("std_location").is_in(ALLOWED_ICU_GAP_LOCATIONS)
            & (pl.col("_prev") == "ICU")
            & (pl.col("_next") == "ICU")
        )
    ).drop(["_prev", "_next"])

    if merged.height == 0:
        return _empty_readmission_frame()

    merged = _merge_runs(merged)

    merged = merged.with_columns(
        pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_loc_rank")
    )

    # 4. First and second ICU groups (after merging).
    icu = merged.filter(pl.col("std_location") == "ICU").with_columns(
        pl.col("in_dttm").rank("ordinal").over("hospitalization_id").alias("_icu_rank")
    )

    first_icu = icu.filter(pl.col("_icu_rank") == 1).select([
        "hospitalization_id",
        pl.col("in_dttm").alias("first_icu_start_time"),
        pl.col("out_dttm").alias("first_icu_end_time"),
    ])
    if require_ward_return:
        first_icu_rank = icu.filter(pl.col("_icu_rank") == 1).select([
            "hospitalization_id",
            pl.col("_loc_rank").alias("_first_icu_loc_rank"),
        ])
        first_ward = (
            merged.filter(pl.col("std_location") == "Ward")
            .join(first_icu_rank, on="hospitalization_id", how="inner")
            .filter(pl.col("_loc_rank") > pl.col("_first_icu_loc_rank"))
            .group_by("hospitalization_id")
            .agg(pl.col("_loc_rank").min().alias("_first_ward_loc_rank"))
        )
        second_icu = (
            icu.filter(pl.col("_icu_rank") > 1)
            .join(first_ward, on="hospitalization_id", how="inner")
            .filter(pl.col("_loc_rank") > pl.col("_first_ward_loc_rank"))
            .group_by("hospitalization_id")
            .agg(pl.col("in_dttm").min().alias("second_icu_start_time"))
        )
    else:
        second_icu = icu.filter(pl.col("_icu_rank") == 2).select([
            "hospitalization_id",
            pl.col("in_dttm").alias("second_icu_start_time"),
        ])

    # 5. Direct ICU discharge (last merged location is ICU).
    last_loc = merged.group_by("hospitalization_id").agg(
        pl.col("std_location").sort_by("in_dttm").last().alias("_last")
    ).select([
        "hospitalization_id",
        (pl.col("_last") == "ICU").alias("is_direct_icu_discharge"),
    ])

    # Intermediate ER (ER appears after the first location).
    inter_er = (
        merged.filter((pl.col("_loc_rank") > 1) & (pl.col("std_location") == "ER"))
        .select("hospitalization_id")
        .unique()
        .with_columns(pl.lit(True).alias("has_intermediate_er"))
    )

    result = (
        first_icu.join(second_icu, on="hospitalization_id", how="left")
        .join(last_loc, on="hospitalization_id", how="left")
        .join(inter_er, on="hospitalization_id", how="left")
        .with_columns([
            pl.col("is_direct_icu_discharge").fill_null(False),
            pl.col("has_intermediate_er").fill_null(False),
        ])
    )
    return result


# ===========================================================================
# Part 2 — Confirmed invasive-mechanical-ventilation (IMV) episodes
#
# Pattern from the CLIF SAT/SBT analysis: a RespiratorySupport row is a
# device-status sample, not a clinical event. A single misclassified row (a brief
# sensor disconnect, a vent swap, a transient downgrade) must not be counted as
# an intubation or an extubation. So every transition is confirmed against a
# FORWARD CONSENSUS WINDOW:
#
#   - an IMV start counts only if IMV persists for INTUB_WINDOW_MIN
#   - an extubation closes an episode only if non-IMV persists for EXTUB_WINDOW_MIN
#
# A trach sentinel — device_category == 'trach collar' OR tracheostomy == True —
# closes any open episode and stops further counting for that hospitalization.
# If the trach fires before any IMV row, the patient is flagged as a pre-existing
# trach.
# ===========================================================================

INTUB_WINDOW_MIN_DEFAULT = 15
EXTUB_WINDOW_MIN_DEFAULT = 30

_EPISODE_COLUMNS = {
    "episode_index": pl.Int32,
    "episode_start_dttm": pl.Datetime,
    "episode_end_dttm": pl.Datetime,
    "is_pre_existing_trach": pl.Boolean,
}


def _empty_episodes_frame(resp: pl.DataFrame) -> pl.DataFrame:
    hid_dtype = resp.schema.get("hospitalization_id", pl.Utf8)
    return pl.DataFrame(schema={"hospitalization_id": hid_dtype, **_EPISODE_COLUMNS})


def detect_imv_episodes(
    resp: pl.DataFrame,
    intub_window_min: int = INTUB_WINDOW_MIN_DEFAULT,
    extub_window_min: int = EXTUB_WINDOW_MIN_DEFAULT,
) -> pl.DataFrame:
    """Return one row per confirmed IMV episode per hospitalization.

    Input: a RespiratorySupport-shaped DataFrame with columns
      hospitalization_id, recorded_dttm, device_category
    and optionally `tracheostomy` (bool). Other columns are ignored.

    Output columns:
      hospitalization_id      : str
      episode_index           : int  (1-based per hospitalization)
      episode_start_dttm      : datetime
      episode_end_dttm        : datetime, null only if still on IMV at the end of
                                the record (a trach closes it at the trach time)
      is_pre_existing_trach   : bool, hospitalization-level so it is the same
                                value on every episode of that hospitalization
    """
    if resp.height == 0:
        return _empty_episodes_frame(resp)

    has_trach_col = "tracheostomy" in resp.columns

    base = resp.select([
        "hospitalization_id",
        "recorded_dttm",
        pl.col("device_category").str.to_lowercase().alias("_dev_lc"),
        *([pl.col("tracheostomy").cast(pl.Boolean).fill_null(False)]
          if has_trach_col else [pl.lit(False).alias("tracheostomy")]),
    ])

    base = base.with_columns([
        (pl.col("_dev_lc") == "imv").cast(pl.Int8).alias("_is_imv"),
        (
            (pl.col("_dev_lc") == "trach collar")
            | pl.col("tracheostomy")
        ).alias("_is_trach"),
    ]).filter(
        pl.col("recorded_dttm").is_not_null()
        & (pl.col("_dev_lc").is_not_null() | pl.col("_is_trach"))
    ).sort(["hospitalization_id", "recorded_dttm"])

    intub_win = timedelta(minutes=intub_window_min)
    extub_win = timedelta(minutes=extub_window_min)

    episode_rows: list[dict] = []
    for (hosp_id,), grp in base.group_by("hospitalization_id", maintain_order=True):
        rows = grp.select(["recorded_dttm", "_is_imv", "_is_trach"]).rows()
        times = [r[0] for r in rows]
        imvs = [r[1] for r in rows]
        trachs = [r[2] for r in rows]
        n = len(rows)

        first_imv_idx = next((i for i, v in enumerate(imvs) if v == 1), None)
        first_trach_idx = next((i for i, v in enumerate(trachs) if v), None)
        # <=, not <: one row can be both IMV and trach (a trached patient
        # ventilated through the trach). The loop below breaks on the trach
        # sentinel before it can ever open an episode at that index, so a strict
        # < would call this "not pre-existing" and the hospitalization would
        # return NOTHING — no episode and no sentinel row either.
        pre_existing_trach = (
            first_trach_idx is not None
            and (first_imv_idx is None or first_trach_idx <= first_imv_idx)
        )

        episodes: list[tuple] = []  # (start_dttm, end_dttm | None)
        current_start: object = None

        for i in range(n):
            if trachs[i]:
                # Trach sentinel: close any open episode at this timestamp, then
                # stop counting for this hospitalization.
                if current_start is not None:
                    episodes.append((current_start, times[i]))
                    current_start = None
                break

            t = times[i]
            imv = imvs[i]

            if current_start is None and imv == 1:
                # Candidate intubation — open the episode only if no non-IMV row
                # contradicts it inside the consensus window.
                deadline = t + intub_win
                contradicted = False
                j = i + 1
                while j < n and times[j] <= deadline:
                    if imvs[j] == 0:
                        contradicted = True
                        break
                    j += 1
                if not contradicted:
                    current_start = t

            elif current_start is not None and imv == 0:
                # Candidate extubation — close the episode only if no IMV row
                # contradicts it inside the consensus window.
                deadline = t + extub_win
                contradicted = False
                j = i + 1
                while j < n and times[j] <= deadline:
                    if imvs[j] == 1:
                        contradicted = True
                        break
                    j += 1
                if not contradicted:
                    episodes.append((current_start, t))
                    current_start = None

        if current_start is not None:
            episodes.append((current_start, None))  # still on IMV at end of record

        for idx, (start, end) in enumerate(episodes, start=1):
            episode_rows.append({
                "hospitalization_id": hosp_id,
                "episode_index": idx,
                "episode_start_dttm": start,
                "episode_end_dttm": end,
                "is_pre_existing_trach": pre_existing_trach,
            })

        # Pre-existing trach with no IMV episodes: emit a sentinel row so callers
        # can still filter by hospitalization_id.
        if pre_existing_trach and not episodes:
            episode_rows.append({
                "hospitalization_id": hosp_id,
                "episode_index": 0,
                "episode_start_dttm": None,
                "episode_end_dttm": None,
                "is_pre_existing_trach": True,
            })

    if not episode_rows:
        return _empty_episodes_frame(resp)

    return pl.DataFrame(
        episode_rows,
        schema={
            "hospitalization_id": base.schema["hospitalization_id"],
            **_EPISODE_COLUMNS,
        },
    )
