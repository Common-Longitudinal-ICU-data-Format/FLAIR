#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "clifpy",
# ]
# ///
"""
FLAIR Cohort Builder

Builds an ICU cohort from CLIF data with demographics.
Uses clifpy for data loading and polars for ETL.

Usage:
    uv run build_cohort.py [--config clif_config.json] [--output flair_output/cohort.parquet]
"""

import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "clif_config.json") -> dict:
    """Load CLIF configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def build_cohort(
    config_path: str = "clif_config.json",
    output_path: str = "flair_output/cohort.parquet",
    min_age: int = 18,
) -> pl.DataFrame:
    """
    Build ICU cohort from CLIF data.

    Filters:
        - Age >= 18
        - Valid admission/discharge dates
        - Hospitalization LOS > 0
        - Has at least one ICU stay
        - First ICU stay LOS > 0

    Returns:
        Cohort DataFrame with demographics and ICU timing
    """
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Load CLIF tables individually using from_file (preferred pattern)
    logger.info("Loading CLIF tables via clifpy...")
    from clifpy.tables import Hospitalization, Patient, Adt

    data_sources = config.get("data_sources", {})
    data_dir = data_sources.get("base_path", "./data/clif")
    filetype = data_sources.get("format", "parquet")
    timezone = data_sources.get("timezone", "US/Central")

    def pandas_to_polars_with_tz(df, tz: str) -> pl.DataFrame:
        """Convert pandas DataFrame to polars while preserving timezone."""
        pl_df = pl.from_pandas(df)
        # Convert datetime columns to have timezone
        datetime_cols = [
            col for col in pl_df.columns
            if pl_df.schema[col] == pl.Datetime or "dttm" in col.lower()
        ]
        for col in datetime_cols:
            if pl_df.schema[col] == pl.Datetime:
                pl_df = pl_df.with_columns(
                    pl.col(col).dt.replace_time_zone(tz).alias(col)
                )
        return pl_df

    # Load each table using from_file
    logger.info("Loading hospitalization table...")
    hosp_table = Hospitalization.from_file(
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
    )
    hosp = pandas_to_polars_with_tz(hosp_table.df, timezone)

    logger.info("Loading patient table...")
    patient_table = Patient.from_file(
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
    )
    patient = pandas_to_polars_with_tz(patient_table.df, timezone)

    logger.info("Loading adt table...")
    adt_table = Adt.from_file(
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
    )
    adt = pandas_to_polars_with_tz(adt_table.df, timezone)

    initial_count = hosp.height
    logger.info(f"Initial hospitalizations: {initial_count:,}")

    # Step 1: Select required columns from hospitalization
    hosp_cols = [
        "hospitalization_id",
        "patient_id",
        "admission_dttm",
        "discharge_dttm",
        "discharge_category",
        "age_at_admission",
    ]
    hosp = hosp.select([c for c in hosp_cols if c in hosp.columns])

    # Step 2: Select demographics from patient
    patient_cols = [
        "patient_id",
        "sex_category",
        "race_category",
        "ethnicity_category",
    ]
    patient = patient.select([c for c in patient_cols if c in patient.columns])

    # Step 3: Merge hospitalization + patient
    logger.info("Merging hospitalization with patient demographics...")
    cohort = hosp.join(patient, on="patient_id", how="inner")
    logger.info(f"After merge: {cohort.height:,}")

    # Step 4: Filter null dates
    logger.info("Filtering null dates...")
    cohort = cohort.filter(
        pl.col("admission_dttm").is_not_null()
        & pl.col("discharge_dttm").is_not_null()
    )
    logger.info(f"After null date filter: {cohort.height:,}")

    # Step 5: Filter adults (age >= 18)
    logger.info(f"Filtering age >= {min_age}...")
    cohort = cohort.filter(pl.col("age_at_admission") >= min_age)
    logger.info(f"After age filter: {cohort.height:,}")

    # Step 6: Calculate hospitalization LOS
    logger.info("Calculating hospitalization LOS...")
    cohort = cohort.with_columns(
        (
            (pl.col("discharge_dttm") - pl.col("admission_dttm"))
            .dt.total_seconds()
            / 3600
        ).alias("hospitalization_los")
    )

    # Step 7: Filter hospitalization LOS > 0
    cohort = cohort.filter(pl.col("hospitalization_los") > 0)
    logger.info(f"After hospitalization LOS > 0 filter: {cohort.height:,}")

    # Step 8: Identify first ICU stay from ADT
    logger.info("Identifying first ICU stays from ADT...")

    # Get ICU ADT records for cohort hospitalizations
    adt_cohort = adt.filter(
        pl.col("hospitalization_id").is_in(cohort["hospitalization_id"])
    )

    # Filter to ICU locations
    icu_adt = adt_cohort.filter(
        pl.col("location_category").str.to_lowercase() == "icu"
    )

    # Sort by hospitalization and time to get first ICU stay
    icu_adt = icu_adt.sort(["hospitalization_id", "in_dttm"])

    # Calculate out_dttm using next in_dttm or discharge_dttm
    # First, join with hospitalization discharge times
    icu_adt = icu_adt.join(
        cohort.select(["hospitalization_id", "discharge_dttm"]),
        on="hospitalization_id",
        how="left",
    )

    # Get next in_dttm within same hospitalization
    icu_adt = icu_adt.with_columns(
        pl.col("in_dttm")
        .shift(-1)
        .over("hospitalization_id")
        .alias("next_in_dttm")
    )

    # Use out_dttm if available, else next_in_dttm, else discharge_dttm
    if "out_dttm" in icu_adt.columns:
        icu_adt = icu_adt.with_columns(
            pl.coalesce(["out_dttm", "next_in_dttm", "discharge_dttm"]).alias("icu_end")
        )
    else:
        icu_adt = icu_adt.with_columns(
            pl.coalesce(["next_in_dttm", "discharge_dttm"]).alias("icu_end")
        )

    # Get FIRST ICU stay per hospitalization
    first_icu = (
        icu_adt.group_by("hospitalization_id")
        .agg([
            pl.col("in_dttm").first().alias("first_icu_start_time"),
            pl.col("icu_end").first().alias("first_icu_end_time"),
        ])
    )

    # Calculate first ICU LOS
    first_icu = first_icu.with_columns(
        (
            (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
            .dt.total_seconds()
            / 3600
        ).alias("first_icu_los")
    )

    # Step 9: Join first ICU info to cohort
    cohort = cohort.join(first_icu, on="hospitalization_id", how="inner")
    logger.info(f"After ICU join (hospitalizations with ICU): {cohort.height:,}")

    # Step 10: Filter first_icu_los >= 1 hour
    cohort = cohort.filter(pl.col("first_icu_los") >= 1)
    logger.info(f"After first ICU LOS >= 1 hour filter: {cohort.height:,}")

    # Final column selection
    final_columns = [
        "hospitalization_id",
        "patient_id",
        "admission_dttm",
        "discharge_dttm",
        "discharge_category",
        "age_at_admission",
        "sex_category",
        "race_category",
        "ethnicity_category",
        "hospitalization_los",
        "first_icu_start_time",
        "first_icu_end_time",
        "first_icu_los",
    ]

    # Select only columns that exist
    available_cols = [c for c in final_columns if c in cohort.columns]
    cohort = cohort.select(available_cols)

    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort.write_parquet(output_path)
    logger.info(f"Saved cohort to {output_path}")

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("COHORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total hospitalizations: {cohort.height:,}")
    logger.info(f"Unique patients: {cohort['patient_id'].n_unique():,}")
    logger.info(f"Age range: {cohort['age_at_admission'].min():.0f} - {cohort['age_at_admission'].max():.0f}")
    logger.info(f"Hospitalization LOS (median): {cohort['hospitalization_los'].median():.1f} hours")
    logger.info(f"First ICU LOS (median): {cohort['first_icu_los'].median():.1f} hours")
    logger.info("=" * 60)

    return cohort


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FLAIR ICU cohort from CLIF data")
    parser.add_argument(
        "--config",
        default="clif_config.json",
        help="Path to CLIF config JSON (default: clif_config.json)"
    )
    parser.add_argument(
        "--output",
        default="flair_output/cohort.parquet",
        help="Output path for cohort parquet (default: flair_output/cohort.parquet)"
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=18,
        help="Minimum age filter (default: 18)"
    )

    args = parser.parse_args()

    build_cohort(
        config_path=args.config,
        output_path=args.output,
        min_age=args.min_age,
    )
