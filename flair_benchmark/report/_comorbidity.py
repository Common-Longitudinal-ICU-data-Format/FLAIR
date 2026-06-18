"""Admission-time Charlson + Elixhauser scores from HospitalDiagnosis.

Both scores derive from the ICD codes recorded on the hospitalization
record, so the result is fixed at admission — no temporal leakage.
"""

from __future__ import annotations

import polars as pl

from flair_benchmark._clif import load_hospital_diagnosis


def compute_cci_elix(cfg: dict,
                     hospitalization_ids: list[str]) -> pl.DataFrame:
    """Return one row per hospitalization with charlson_score + elix_score.

    Missing scores (e.g., hospitalization with no ICD codes) come through
    as null and are handled downstream by the summarizer.
    """
    from clifpy.utils.comorbidity import calculate_cci, calculate_elix

    if not hospitalization_ids:
        return _empty_frame()

    dx = load_hospital_diagnosis(cfg, hospitalization_ids=hospitalization_ids)
    if dx.height == 0:
        return _empty_frame()

    cci = calculate_cci(dx, hierarchy=True)
    elix = calculate_elix(dx, hierarchy=True)

    cci_slim = _slim(cci, "cci_score", "charlson_score")
    elix_slim = _slim(elix, "elix_score", "elix_score")

    return cci_slim.join(elix_slim, on="hospitalization_id", how="outer")


def _slim(df, score_col: str, out_name: str) -> pl.DataFrame:
    """Reduce a clifpy comorbidity output to [hospitalization_id, <out_name>].

    Accepts both polars and pandas DataFrames (clifpy returns pandas).
    """
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    if "hospitalization_id" not in df.columns or score_col not in df.columns:
        return _empty_frame()
    return df.select([
        "hospitalization_id",
        pl.col(score_col).cast(pl.Float64).alias(out_name),
    ])


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "hospitalization_id": pl.Utf8,
            "charlson_score": pl.Float64,
            "elix_score": pl.Float64,
        }
    )
