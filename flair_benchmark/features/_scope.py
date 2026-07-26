"""Work out WHICH hospitalizations to load, and HOW FAR FORWARD to read.

Two questions every extractor must answer before touching CLIF, and getting
either wrong is a silent correctness bug rather than a crash.

WHICH: task cohorts are emitted at stitched-encounter grain. A cohort row carries
one representative raw hospitalization_id, but the encounter can include sibling
hospitalizations that have no prediction row of their own. Loading only the
cohort's own ids would silently drop those siblings' events, so the id set is
expanded back through the stitch index.

HOW FAR: the no-future cutoff is applied at hospitalization_join_id grain, not
per row — see `feature_scope` for why the widest row in a block governs.

Depends on: cohort/stitch.py.
Used by: features/extractors.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from flair_benchmark.cohort.stitch import (
    JOIN_ID,
    attach_join_id,
    load_or_build_encounter_index,
    members_of_joins,
)

HORIZON_COL = "end_time"
CUTOFF_COL = "feature_cutoff_dttm"


@dataclass(frozen=True)
class FeatureScope:
    index: pl.DataFrame
    horizons: pl.DataFrame
    hospitalization_ids: list[str]


def feature_scope(
    task_df: pl.DataFrame,
    clif_cfg: dict,
    *,
    require_horizon: bool = True,
) -> FeatureScope:
    """Return expanded member hospitalizations + join-level horizons.

    ``hospitalization_ids`` is the full raw hospitalization set to load from CLIF.
    ``horizons`` has one row per hospitalization_join_id and, in ``end_time``, the
    last instant whose data may be used — ``max(feature_cutoff_dttm)``, the
    cohort-contract column stamped by ``tasks/_finalize.py`` (see there for why the
    cutoff travels as a value). It equals ``prediction_dttm`` for every task except
    one whose prediction timestamp would itself leak the outcome.

    Callers filter INCLUSIVELY against it (``time <= end_time``); a task wanting a
    strict cutoff moves the value, not the comparison.
    """
    idx = load_or_build_encounter_index(
        clif_cfg, clif_cfg.get("stitch_time_interval_hours", 6)
    )

    df = task_df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    if JOIN_ID not in df.columns:
        df = attach_join_id(df, idx)
    df = df.with_columns(pl.col(JOIN_ID).cast(pl.Utf8))

    join_ids = df.select(JOIN_ID).unique()[JOIN_ID].to_list()
    hosp_ids = members_of_joins(idx, join_ids)

    if "prediction_dttm" in df.columns:
        if CUTOFF_COL not in df.columns:
            if require_horizon:
                raise ValueError(
                    f"cohort is missing {CUTOFF_COL!r} — regenerate it with "
                    f"`flair load-task` on this version of FLAIR. Falling back to "
                    f"prediction_dttm would silently give an INCLUSIVE feature "
                    f"cutoff, which leaks the outcome for tasks anchored on the "
                    f"event itself (e.g. extubation_failure_24h)."
                )
            cutoff = pl.col("prediction_dttm")
        else:
            # Coalesce per ROW before aggregating. A bare .max() over a partly-null
            # column skips nulls, which would let a sibling row's cutoff govern the
            # null row instead of that row's own prediction time.
            cutoff = pl.coalesce([pl.col(CUTOFF_COL), pl.col("prediction_dttm")])
        # max() over the join block: end_time is a LOADING bound for the whole
        # encounter (the per-prediction cut happens downstream in the count
        # featurizer), so the widest row governs. Safe while a task's cutoff is
        # monotone in prediction_dttm, which is true of every cutoff shipped.
        horizons = df.group_by(JOIN_ID).agg(cutoff.max().alias(HORIZON_COL))
    elif require_horizon:
        raise ValueError("feature extraction requires prediction_dttm in task_df")
    else:
        horizons = pl.DataFrame(schema={JOIN_ID: pl.Utf8, HORIZON_COL: pl.Datetime("us")})

    return FeatureScope(index=idx, horizons=horizons, hospitalization_ids=hosp_ids)
