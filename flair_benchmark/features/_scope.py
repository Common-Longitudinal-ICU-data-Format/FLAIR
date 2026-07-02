"""Cohort scope helpers for feature extraction.

Task cohorts are emitted at stitched-encounter grain. A cohort row carries one
representative raw hospitalization_id, but the encounter can include sibling
hospitalizations. Feature extraction must load every raw member hospitalization
and apply the no-future cutoff at the hospitalization_join_id grain.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from flair_benchmark._stitch import (
    JOIN_ID,
    attach_join_id,
    load_or_build_encounter_index,
    members_of_joins,
)

HORIZON_COL = "end_time"


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
    ``horizons`` has one row per hospitalization_join_id and a max
    ``prediction_dttm`` cutoff in ``end_time``.
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
        horizons = (
            df.group_by(JOIN_ID)
            .agg(pl.col("prediction_dttm").max().alias(HORIZON_COL))
        )
    elif require_horizon:
        raise ValueError("feature extraction requires prediction_dttm in task_df")
    else:
        horizons = pl.DataFrame(schema={JOIN_ID: pl.Utf8, HORIZON_COL: pl.Datetime("us")})

    return FeatureScope(index=idx, horizons=horizons, hospitalization_ids=hosp_ids)
