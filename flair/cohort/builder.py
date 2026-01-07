"""
FLAIR Cohort Builder.

Builds the ICU cohort directly from CLIF data using clifpy,
eliminating the dependency on tokenETL.
"""

import polars as pl
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FLAIRCohortBuilder:
    """
    Build FLAIR benchmark cohort directly from CLIF data using clifpy.

    Uses ADT-first approach: loads ADT table first to identify ICU hospitalizations,
    then loads other tables filtered by those hospitalization IDs.

    The cohort includes all ICU hospitalizations with:
    - Adults (age >= min_age)
    - Length of stay > 0
    - At least one ICU ADT record

    Output schema (9 columns):
    - hospitalization_id, admission_dttm, discharge_dttm
    - age_at_admission, sex_category, race_category, ethnicity_category
    - discharge_category (for mortality task), death_dttm (for readmission exclusions)

    Also returns ADT data for tasks to compute their own ICU timing.
    """

    def __init__(self, clif_config_path: str):
        """
        Initialize cohort builder.

        Args:
            clif_config_path: Path to clifpy configuration JSON file
        """
        self.clif_config_path = clif_config_path
        self._config_cache = None

    def _parse_clif_config(self) -> dict:
        """
        Parse clif_config.json to get data paths.

        Config format:
        {
            "site": "sitename",
            "data_directory": "/path/to/data",
            "filetype": "parquet",
            "timezone": "US/Central"
        }

        Returns:
            dict with site, data_directory, filetype, timezone
        """
        if self._config_cache is not None:
            return self._config_cache

        import json

        with open(self.clif_config_path) as f:
            config = json.load(f)

        self._config_cache = {
            "site": config.get("site", "unknown"),
            "data_directory": config["data_directory"],
            "filetype": config["filetype"],
            "timezone": config.get("timezone", "US/Central"),
        }
        return self._config_cache

    def _load_table(
        self,
        table_name: str,
        filters: dict = None,
        columns: list = None,
    ) -> pl.DataFrame:
        """
        Load a CLIF table using individual table class with filters.

        Args:
            table_name: Name of the CLIF table (hospitalization, patient, adt, etc.)
            filters: Dict of column->values to filter on at load time
            columns: Optional list of columns to select

        Returns:
            Polars DataFrame
        """
        from clifpy.tables import (
            Hospitalization,
            Patient,
            Adt,
        )

        TABLE_CLASSES = {
            "hospitalization": Hospitalization,
            "patient": Patient,
            "adt": Adt,
        }

        if table_name not in TABLE_CLASSES:
            raise ValueError(f"Unknown table: {table_name}. Available: {list(TABLE_CLASSES.keys())}")

        config = self._parse_clif_config()
        table_class = TABLE_CLASSES[table_name]

        logger.info(f"Loading table: {table_name}" + (f" with filters: {list(filters.keys())}" if filters else ""))

        table = table_class.from_file(
            data_directory=config["data_directory"],
            filetype=config["filetype"],
            timezone=config["timezone"],
            filters=filters,
        )

        df = table.df

        # Convert to polars if pandas
        if hasattr(df, "values"):
            df = pl.from_pandas(df)

        # Select columns if specified
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df.select(available)

        logger.info(f"Loaded {table_name}: {df.height:,} rows")
        return df

    def build_cohort(
        self,
        output_path: Optional[str] = None,
        min_age: int = 18,
        min_los_days: float = 0,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, int]]:
        """
        Build minimal ICU cohort from CLIF data using ADT-first approach.

        Steps:
        1. Load ADT table first, filter to ICU locations, get hospitalization_ids
        2. Load hospitalization table filtered by ICU hospitalization_ids
        3. Load patient table filtered by patient_ids
        4. Apply filters (null dates, age, LOS)
        5. Return 7-column cohort + filtered ADT for tasks

        Args:
            output_path: Optional path to save cohort.parquet
            min_age: Minimum age filter (default 18)
            min_los_days: Minimum length of stay in days (default 0)

        Returns:
            Tuple of (cohort DataFrame, ADT DataFrame, exclusion statistics dict)
            - cohort: 9 columns (hospitalization_id, admission_dttm, discharge_dttm,
                      age_at_admission, sex_category, race_category, ethnicity_category,
                      discharge_category, death_dttm)
            - adt_data: ADT DataFrame for tasks to compute ICU timing
            - exclusion_stats: Dict of exclusion counts
        """
        logger.info("=" * 60)
        logger.info("BUILDING FLAIR COHORT (ADT-First Approach)")
        logger.info("=" * 60)

        exclusion_stats = {}

        # ============================================
        # STEP 1: Load ADT and find ICU hospitalizations
        # ============================================
        logger.info("Step 1: Loading ADT table to identify ICU hospitalizations...")
        adt = self._load_table("adt")

        # Filter to ICU locations and get unique hospitalization IDs
        icu_adt = adt.filter(pl.col("location_category").str.to_lowercase() == "icu")
        icu_hosp_ids = icu_adt.select("hospitalization_id").unique()["hospitalization_id"].to_list()

        exclusion_stats["icu_hospitalizations"] = len(icu_hosp_ids)
        logger.info(f"Found {len(icu_hosp_ids):,} hospitalizations with ICU stays")

        # ============================================
        # STEP 2: Load Hospitalization filtered by ICU hosp_ids
        # ============================================
        logger.info("Step 2: Loading hospitalization table (filtered to ICU patients)...")
        hosp = self._load_table(
            "hospitalization",
            filters={"hospitalization_id": icu_hosp_ids},
            columns=[
                "hospitalization_id",
                "patient_id",
                "admission_dttm",
                "discharge_dttm",
                "age_at_admission",
                "discharge_category",  # Needed for mortality task
            ],
        )
        exclusion_stats["initial"] = hosp.height
        logger.info(f"Initial ICU hospitalizations: {hosp.height:,}")

        # ============================================
        # STEP 3: Load Patient filtered by patient_ids
        # ============================================
        patient_ids = hosp.select("patient_id").unique()["patient_id"].to_list()
        logger.info(f"Step 3: Loading patient table for {len(patient_ids):,} patients...")
        patient = self._load_table(
            "patient",
            filters={"patient_id": patient_ids},
            columns=["patient_id", "sex_category", "race_category", "ethnicity_category", "death_dttm"],
        )

        # ============================================
        # STEP 4: Merge and apply filters
        # ============================================
        logger.info("Step 4: Merging tables and applying filters...")
        cohort = hosp.join(patient, on="patient_id", how="inner")

        # Filter null dates
        cohort = cohort.filter(
            pl.col("admission_dttm").is_not_null()
            & pl.col("discharge_dttm").is_not_null()
        )
        exclusion_stats["after_null_filter"] = cohort.height
        exclusion_stats["excluded_null_dates"] = (
            exclusion_stats["initial"] - exclusion_stats["after_null_filter"]
        )
        logger.info(f"After null date filter: {cohort.height:,}")

        # Filter adults
        cohort = cohort.filter(pl.col("age_at_admission") >= min_age)
        exclusion_stats["after_age_filter"] = cohort.height
        exclusion_stats["excluded_age"] = (
            exclusion_stats["after_null_filter"] - exclusion_stats["after_age_filter"]
        )
        logger.info(f"After age filter (>= {min_age}): {cohort.height:,}")

        # Filter by LOS (calculate inline, don't keep column)
        cohort = cohort.filter(
            ((pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 86400)
            > min_los_days
        )
        exclusion_stats["after_los_filter"] = cohort.height
        exclusion_stats["excluded_los"] = (
            exclusion_stats["after_age_filter"] - exclusion_stats["after_los_filter"]
        )
        logger.info(f"After LOS filter (> {min_los_days}): {cohort.height:,}")

        # ============================================
        # STEP 5: Select final columns (9 columns - 7 base + discharge_category + death_dttm)
        # ============================================
        cohort = cohort.select([
            "hospitalization_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
            "sex_category",
            "race_category",
            "ethnicity_category",
            "discharge_category",  # Needed for mortality task labels
            "death_dttm",  # Needed for ICU readmission task exclusions
        ])

        exclusion_stats["final"] = cohort.height

        # ============================================
        # STEP 6: Filter ADT to final cohort for task use
        # ============================================
        final_hosp_ids = cohort.select("hospitalization_id").unique()["hospitalization_id"].to_list()
        adt_filtered = adt.filter(pl.col("hospitalization_id").is_in(final_hosp_ids))
        logger.info(f"ADT records for final cohort: {adt_filtered.height:,}")

        logger.info("=" * 60)
        logger.info("COHORT BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final cohort size: {cohort.height:,} hospitalizations")
        logger.info(f"Columns: {cohort.columns}")

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cohort.write_parquet(output_file)
            logger.info(f"Saved cohort to {output_file}")

        return cohort, adt_filtered, exclusion_stats



def build_cohort(
    clif_config_path: str,
    output_path: Optional[str] = None,
    min_age: int = 18,
    min_los_days: float = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, int]]:
    """
    Build FLAIR cohort from CLIF configuration.

    Convenience function that creates a FLAIRCohortBuilder and builds the cohort.

    Args:
        clif_config_path: Path to clifpy configuration JSON
        output_path: Optional path to save cohort.parquet
        min_age: Minimum age filter (default 18)
        min_los_days: Minimum LOS filter (default 0)

    Returns:
        Tuple of (cohort DataFrame, ADT DataFrame, exclusion statistics)
    """
    builder = FLAIRCohortBuilder(clif_config_path)
    return builder.build_cohort(
        output_path=output_path,
        min_age=min_age,
        min_los_days=min_los_days,
    )
