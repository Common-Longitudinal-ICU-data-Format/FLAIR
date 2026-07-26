"""What to extract, and the shape it comes out in.

Two halves of the same contract:

  Part 1  load_feature_config()  reads feature_config.yaml — WHICH tables and
                                 categories to pull
  Part 2  ELF schema + helpers   the event shape every extractor must emit

They share a file because neither is meaningful alone: the config selects
concepts, the schema fixes how those concepts are written down, and the
extractors in extractors.py are the only consumer of both.

feature_config.yaml shape::

    clif_config: clif_config_mimic.json     # path to existing JSON clif config
    output_directory: ./output/features     # optional default out dir

    meds_stream:                            # lightweight ELF pull (no conversion)
      medication_admin_continuous: {categories: [norepinephrine]}
      medication_admin_intermittent: {categories: [vancomycin, cefepime]}

Config validation WARNS rather than fails. An unsupported table or an unknown
mCIDE category is usually a site-local naming difference, and failing the whole
extraction over one bad category would be worse than skipping it.

Depends on: cohort/clif.py, clifpy (for permissible-value schemas), pyyaml.
Used by: features/extractors.py, cli.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import yaml

from flair_benchmark.cohort.clif import read_clif_config

# CLIF 2.1 tables FLAIR v1 supports.
V1_TABLES = {
    "adt", "hospital_diagnosis", "hospitalization", "labs",
    "medication_admin_continuous", "medication_admin_intermittent",
    "microbiology_culture", "patient", "patient_assessments",
    "respiratory_support", "vitals",
}

# Category column per configurable table (used for permissible-value validation).
_CATEGORY_COLUMN = {
    "medication_admin_continuous": "med_category",
    "medication_admin_intermittent": "med_category",
}


# ===========================================================================
# Part 1 — Feature-config loading and validation
# ===========================================================================


def load_feature_config(path: str) -> tuple[dict, dict]:
    """Load feature_config.yaml; return (feature_cfg, clif_cfg).

    The `clif_config` path is resolved relative to the feature-config file's own
    directory when it is not absolute, so a config pair can be moved together.
    """
    fpath = Path(path)
    with open(fpath) as f:
        feature_cfg = yaml.safe_load(f) or {}

    clif_ref = feature_cfg.get("clif_config")
    if not clif_ref:
        raise ValueError(f"{path}: missing required key 'clif_config'")
    clif_path = Path(clif_ref)
    if not clif_path.is_absolute():
        clif_path = (fpath.parent / clif_path).resolve()
    clif_cfg = read_clif_config(str(clif_path))

    _validate(feature_cfg)
    return feature_cfg, clif_cfg


def _permissible_values(table: str) -> set[str] | None:
    """Permissible category values for a configurable table, from clifpy's schema."""
    col = _CATEGORY_COLUMN.get(table)
    if col is None:
        return None
    import clifpy

    schema = Path(clifpy.__file__).parent / "schemas" / f"{table}_schema.yaml"
    try:
        with open(schema) as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    for field in doc.get("columns", []):
        if field.get("name") == col:
            vals = field.get("permissible_values")
            return {str(v) for v in vals} if vals else None
    return None


def _validate(feature_cfg: dict) -> None:
    """Warn (don't fail) on unsupported tables / unknown mCIDE categories."""
    for block in ("meds_stream",):
        spec = feature_cfg.get(block) or {}
        for table, tspec in spec.items():
            if table not in V1_TABLES:
                warnings.warn(
                    f"feature_config[{block}]: table '{table}' is not in the "
                    f"FLAIR v1 supported set; skipping.", stacklevel=2)
                continue
            cats = (tspec or {}).get("categories")
            if not cats:
                continue
            permissible = _permissible_values(table)
            if permissible is None:
                continue
            unknown = [c for c in cats if c not in permissible]
            if unknown:
                warnings.warn(
                    f"feature_config[{block}][{table}]: categories not in mCIDE "
                    f"permissible values {unknown}", stacklevel=2)


# ===========================================================================
# Part 2 — ELF (Event List Format): the emitted event shape
#
# One row per clinical event, mirroring CLIF_MEDS with one extra `unit` column so
# raw units survive for a later conversion step:
#
#   subject_id              Utf8      (= patient_id)         required
#   time                    Datetime  (tz-naive)             required
#   code                    Utf8      "<DOMAIN>//<cat>//.."  required
#   numeric_value           Float64                          nullable
#   unit                    Utf8                             nullable (req. when numeric)
#   text_value              Utf8                             nullable
#   hospitalization_id      Utf8                             required
#   hospitalization_join_id Utf8      (stitched-encounter)   required
#
# All three identifiers are always present: an extractor keys every row on
# subject + stay + encounter block, so a consumer can join at whichever grain it
# needs without going back to the source.
# ===========================================================================

ELF_COLUMNS = [
    "subject_id", "time", "code",
    "numeric_value", "unit", "text_value",
    "hospitalization_id", "hospitalization_join_id",
]

ELF_SCHEMA = {
    "subject_id": pl.Utf8,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float64,
    "unit": pl.Utf8,
    "text_value": pl.Utf8,
    "hospitalization_id": pl.Utf8,
    "hospitalization_join_id": pl.Utf8,
}


def empty_elf() -> pl.DataFrame:
    """An empty ELF frame with the canonical schema/column order."""
    return pl.DataFrame(schema=ELF_SCHEMA).select(ELF_COLUMNS)


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce to the ELF schema, assert invariants, return a column-ordered frame.

    Invariants, both enforced here rather than trusted:
      - subject_id and hospitalization_id are non-null on every row
      - unit is non-null wherever numeric_value is non-null, so a later step
        always has something to convert from
    """
    if df.height == 0:
        return empty_elf()

    df = df.with_columns([
        pl.col("subject_id").cast(pl.Utf8),
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("hospitalization_join_id").cast(pl.Utf8),
        pl.col("time").cast(pl.Datetime("us")),
        pl.col("code").cast(pl.Utf8),
        pl.col("numeric_value").cast(pl.Float64),
        pl.col("unit").cast(pl.Utf8),
        pl.col("text_value").cast(pl.Utf8),
    ]).select(ELF_COLUMNS)

    if df.select(
        pl.col("subject_id").is_null().any()
        | pl.col("hospitalization_id").is_null().any()
        | pl.col("hospitalization_join_id").is_null().any()
    ).item():
        raise ValueError(
            "ELF rows with null subject_id / hospitalization_id / hospitalization_join_id"
        )

    bad_unit = df.filter(
        pl.col("numeric_value").is_not_null() & pl.col("unit").is_null()
    ).height
    if bad_unit:
        raise ValueError(
            f"{bad_unit} numeric ELF rows have no unit charted; a unit is "
            "required wherever numeric_value is set (meds_stream charts 'UNK' "
            "for a dose whose med_dose_unit is missing)."
        )
    return df


def attach_subject_id(df: pl.DataFrame, clif_cfg: dict) -> pl.DataFrame:
    """Add subject_id (= patient_id) by joining the hospitalization table.

    `df` must have a string `hospitalization_id` column.
    """
    from flair_benchmark.cohort.clif import load_hospitalization

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
