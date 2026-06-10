"""ELF (Event List Format) helpers shared by the wide + meds extractors.

ELF long schema (one row per clinical event), mirroring CLIF_MEDS with one extra
``unit`` column so raw units survive for later conversion::

    subject_id          Utf8   (= patient_id)        required
    time                Datetime (tz-naive)          required
    code                Utf8   "<DOMAIN>//<cat>//.."  required
    numeric_value       Float64                       nullable
    unit                Utf8                           nullable (req. when numeric)
    text_value          Utf8                           nullable
    hospitalization_id  Utf8                           required

Both identifiers are always present (FE keys every row on subject + stay).
"""

from __future__ import annotations

import polars as pl

ELF_COLUMNS = [
    "subject_id", "time", "code",
    "numeric_value", "unit", "text_value",
    "hospitalization_id",
]

ELF_SCHEMA = {
    "subject_id": pl.Utf8,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float64,
    "unit": pl.Utf8,
    "text_value": pl.Utf8,
    "hospitalization_id": pl.Utf8,
}


def build_code(domain: str, *parts: str | None) -> str:
    """Join an ELF code: '<DOMAIN>//<part>//<part>', dropping None/empty parts."""
    out = [domain]
    out.extend(str(p) for p in parts if p is not None and str(p) != "")
    return "//".join(out)


def empty_elf() -> pl.DataFrame:
    """An empty ELF frame with the canonical schema/column order."""
    return pl.DataFrame(schema=ELF_SCHEMA).select(ELF_COLUMNS)


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce to the ELF schema, assert invariants, return column-ordered frame.

    Invariants:
      - subject_id and hospitalization_id non-null on every row,
      - unit non-null wherever numeric_value is non-null (units charted so a
        later step can convert).
    """
    if df.height == 0:
        return empty_elf()

    df = df.with_columns([
        pl.col("subject_id").cast(pl.Utf8),
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("code").cast(pl.Utf8),
        pl.col("numeric_value").cast(pl.Float64),
        pl.col("unit").cast(pl.Utf8),
        pl.col("text_value").cast(pl.Utf8),
    ]).select(ELF_COLUMNS)

    if df.select(
        pl.col("subject_id").is_null().any() | pl.col("hospitalization_id").is_null().any()
    ).item():
        raise ValueError("ELF rows with null subject_id or hospitalization_id")

    bad_unit = df.filter(
        pl.col("numeric_value").is_not_null() & pl.col("unit").is_null()
    ).height
    if bad_unit:
        raise ValueError(
            f"{bad_unit} numeric ELF rows have no unit charted; a unit is "
            "required wherever numeric_value is set (configure features.units "
            "or extend _units.py)."
        )
    return df


def attach_subject_id(df: pl.DataFrame, clif_cfg: dict) -> pl.DataFrame:
    """Add subject_id (= patient_id) by joining the hospitalization table.

    ``df`` must have a string ``hospitalization_id`` column.
    """
    from flair_benchmark._clif import load_hospitalization

    hosp_ids = df.select("hospitalization_id").unique()["hospitalization_id"].to_list()
    hmap = (
        load_hospitalization(clif_cfg, hospitalization_ids=hosp_ids,
                             columns=["hospitalization_id", "patient_id"])
        .with_columns([
            pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col("patient_id").cast(pl.Utf8).alias("subject_id"),
        ])
        .select(["hospitalization_id", "subject_id"])
        .unique(subset=["hospitalization_id"])
    )
    return df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).join(
        hmap, on="hospitalization_id", how="left"
    )
