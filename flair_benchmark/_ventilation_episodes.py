"""Consensus-window detection of confirmed invasive-mechanical-ventilation
(IMV) episodes.

Pattern from the CLIF SAT/SBT analysis: a CLIF RespiratorySupport row is a
device-status sample, not a clinical event. A single misclassified row
(brief sensor disconnect, vent swap, transient downgrade) should not be
counted as intubation or extubation. We confirm transitions with a
forward consensus window: an IMV start counts only if IMV persists for
INTUB_WINDOW_MIN; an extubation closes an episode only if non-IMV
persists for EXTUB_WINDOW_MIN. A trach sentinel — `device_category ==
'trach collar'` OR `tracheostomy == True` — closes any open episode and
stops further counting for that hospitalization. If trach fires before
any IMV row, the patient is flagged as pre-existing trach.

Returned schema:
  hospitalization_id, episode_index, episode_start_dttm,
  episode_end_dttm (nullable iff still on IMV at end of record /
                    closed by trach without an extubation transition),
  is_pre_existing_trach (bool, hospitalization-level — same value for
                         every episode of that hospitalization)
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

INTUB_WINDOW_MIN_DEFAULT = 15
EXTUB_WINDOW_MIN_DEFAULT = 30


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
      episode_end_dttm        : datetime (nullable)
      is_pre_existing_trach   : bool   (hospitalization-level flag)
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
        pre_existing_trach = (
            first_trach_idx is not None
            and (first_imv_idx is None or first_trach_idx < first_imv_idx)
        )

        episodes: list[tuple] = []  # (start_dttm, end_dttm | None)
        current_start: object = None

        for i in range(n):
            if trachs[i]:
                # Trach sentinel: close any open episode at this timestamp,
                # stop further counting for this hospitalization.
                if current_start is not None:
                    episodes.append((current_start, times[i]))
                    current_start = None
                break

            t = times[i]
            imv = imvs[i]

            if current_start is None and imv == 1:
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

        # Pre-existing trach with no IMV episodes: emit a sentinel row so
        # callers can still filter by hospitalization_id.
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
            "episode_index": pl.Int32,
            "episode_start_dttm": pl.Datetime,
            "episode_end_dttm": pl.Datetime,
            "is_pre_existing_trach": pl.Boolean,
        },
    )


def _empty_episodes_frame(resp: pl.DataFrame) -> pl.DataFrame:
    hid_dtype = resp.schema.get("hospitalization_id", pl.Utf8)
    return pl.DataFrame(
        schema={
            "hospitalization_id": hid_dtype,
            "episode_index": pl.Int32,
            "episode_start_dttm": pl.Datetime,
            "episode_end_dttm": pl.Datetime,
            "is_pre_existing_trach": pl.Boolean,
        }
    )
