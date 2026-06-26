"""Full-hospitalization SOFA per hospitalization (worst-in-window).

Computed once per site against the full hospitalization window —
`[admission_dttm, discharge_dttm]` — independent of which task is being
summarized. Uses clifpy's polars-native SOFA backend
(`compute_sofa_polars`), which loads raw files directly and handles
vasopressor unit conversion and respiratory waterfall internally.
"""

from __future__ import annotations

import polars as pl


def compute_hospitalization_sofa_max(cfg: dict,
                                     cohort: pl.DataFrame) -> pl.DataFrame:
    """Return [hospitalization_id, sofa_max] across the full hospitalization.

    cohort: pl.DataFrame with [hospitalization_id, admission_dttm,
            discharge_dttm].
    """
    if cohort.height == 0:
        return pl.DataFrame(
            schema={"hospitalization_id": pl.Utf8, "sofa_max": pl.Float64}
        )

    from clifpy.utils.sofa_polars import compute_sofa_polars

    cohort_in = cohort.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("admission_dttm").alias("start_dttm"),
        pl.col("discharge_dttm").alias("end_dttm"),
    ])

    sofa = compute_sofa_polars(
        data_directory=cfg["data_directory"],
        cohort_df=cohort_in,
        filetype=cfg["filetype"],
        id_name="hospitalization_id",
        extremal_type="worst",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        timezone=cfg["timezone"],
    )

    return sofa.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("sofa_total").alias("sofa_max"),
    ])
