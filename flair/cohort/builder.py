"""
FLAIR Cohort Builder.

Builds the ICU cohort directly from CLIF data using clifpy,
eliminating the dependency on tokenETL.
"""

import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import timedelta

logger = logging.getLogger(__name__)


class FLAIRCohortBuilder:
    """
    Build FLAIR benchmark cohort directly from CLIF data using clifpy.

    The cohort includes all ICU hospitalizations with:
    - Adults (age >= min_age)
    - Length of stay > 0
    - At least one ICU ADT record

    Output schema (15 columns):
    - hospitalization_id, patient_id, admission_dttm, discharge_dttm
    - age_at_admission, discharge_category, sex_category, race_category
    - ethnicity_category, hospitalization_los, previous_hospitalization_id
    - first_icu_start_time, first_icu_end_time, first_icu_24hr_completion_time
    - second_icu_start_time, imv_at_24hr
    """

    def __init__(self, clif_config_path: str):
        """
        Initialize cohort builder.

        Args:
            clif_config_path: Path to clifpy configuration JSON file
        """
        self.clif_config_path = clif_config_path
        self._clif = None

    @property
    def clif(self):
        """Lazy load CLIF instance."""
        if self._clif is None:
            self._load_clif()
        return self._clif

    def _load_clif(self) -> None:
        """Load CLIF instance from configuration."""
        try:
            from clifpy import CLIF

            self._clif = CLIF(self.clif_config_path)
            self._clif.load_data()
            logger.info(f"Loaded CLIF data from {self.clif_config_path}")
        except ImportError:
            logger.warning("clifpy not installed - cohort building requires clifpy")
            raise ImportError("clifpy is required for cohort building")
        except Exception as e:
            logger.error(f"Failed to load CLIF: {e}")
            raise

    def build_cohort(
        self,
        output_path: Optional[str] = None,
        min_age: int = 18,
        min_los_days: float = 0,
        skip_time_filter: bool = True,
    ) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Build ICU cohort from CLIF data.

        Steps:
        1. Merge hospitalization + patient tables
        2. Filter out null admission/discharge dates
        3. Filter adults (age >= min_age)
        4. Calculate LOS and filter (> min_los_days)
        5. Filter to ICU-only (at least one ADT record with location_category='icu')
        6. Calculate ICU timing metrics
        7. Calculate previous hospitalization
        8. Calculate imv_at_24hr from respiratory_support

        Args:
            output_path: Optional path to save cohort.parquet
            min_age: Minimum age filter (default 18)
            min_los_days: Minimum length of stay in days (default 0)
            skip_time_filter: Skip time period filter (default True for MIMIC compatibility)

        Returns:
            Tuple of (cohort DataFrame, exclusion statistics dict)
        """
        logger.info("=" * 60)
        logger.info("BUILDING FLAIR COHORT")
        logger.info("=" * 60)

        exclusion_stats = {}

        # Step 1: Merge hospitalization + patient
        cohort = self._merge_patient_hospitalization()
        exclusion_stats["initial"] = cohort.height
        logger.info(f"Initial hospitalizations: {cohort.height:,}")

        # Step 2: Filter null dates
        cohort = self._filter_null_dates(cohort)
        exclusion_stats["after_null_filter"] = cohort.height
        exclusion_stats["excluded_null_dates"] = (
            exclusion_stats["initial"] - exclusion_stats["after_null_filter"]
        )
        logger.info(f"After null date filter: {cohort.height:,}")

        # Step 3: Filter adults
        cohort = cohort.filter(pl.col("age_at_admission") >= min_age)
        exclusion_stats["after_age_filter"] = cohort.height
        exclusion_stats["excluded_age"] = (
            exclusion_stats["after_null_filter"] - exclusion_stats["after_age_filter"]
        )
        logger.info(f"After age filter (>= {min_age}): {cohort.height:,}")

        # Step 4: Calculate LOS and filter
        cohort = self._calculate_los(cohort, min_los_days)
        exclusion_stats["after_los_filter"] = cohort.height
        exclusion_stats["excluded_los"] = (
            exclusion_stats["after_age_filter"] - exclusion_stats["after_los_filter"]
        )
        logger.info(f"After LOS filter (> {min_los_days}): {cohort.height:,}")

        # Step 5: Filter to ICU-only
        cohort = self._filter_icu_only(cohort)
        exclusion_stats["after_icu_filter"] = cohort.height
        exclusion_stats["excluded_no_icu"] = (
            exclusion_stats["after_los_filter"] - exclusion_stats["after_icu_filter"]
        )
        logger.info(f"After ICU filter: {cohort.height:,}")

        # Step 6: Calculate ICU timing metrics
        cohort = self._calculate_icu_timing(cohort)
        logger.info("Added ICU timing metrics")

        # Step 7: Calculate previous hospitalization
        cohort = self._calculate_previous_hospitalization(cohort)
        logger.info("Added previous hospitalization IDs")

        # Step 8: Calculate imv_at_24hr
        cohort = self._calculate_imv_at_24hr(cohort)
        logger.info("Added imv_at_24hr flag")

        exclusion_stats["final"] = cohort.height

        # Reorder columns to final schema
        final_columns = [
            "hospitalization_id",
            "patient_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
            "discharge_category",
            "sex_category",
            "race_category",
            "ethnicity_category",
            "hospitalization_los",
            "previous_hospitalization_id",
            "first_icu_start_time",
            "first_icu_end_time",
            "first_icu_24hr_completion_time",
            "second_icu_start_time",
            "imv_at_24hr",
        ]

        # Select only columns that exist
        available_cols = [c for c in final_columns if c in cohort.columns]
        cohort = cohort.select(available_cols)

        logger.info("=" * 60)
        logger.info("COHORT BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final cohort size: {cohort.height:,} hospitalizations")
        logger.info(f"Columns: {len(cohort.columns)}")

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cohort.write_parquet(output_file)
            logger.info(f"Saved cohort to {output_file}")

        return cohort, exclusion_stats

    def _merge_patient_hospitalization(self) -> pl.DataFrame:
        """Merge hospitalization and patient tables."""
        # Get tables from clifpy (as polars DataFrames)
        hosp = self._get_table("hospitalization")
        patient = self._get_table("patient")

        # Select relevant columns from each table
        hosp_cols = [
            "hospitalization_id",
            "patient_id",
            "admission_dttm",
            "discharge_dttm",
            "discharge_category",
        ]
        patient_cols = [
            "patient_id",
            "age_at_admission",
            "sex_category",
            "race_category",
            "ethnicity_category",
        ]

        # Filter to available columns
        hosp_cols = [c for c in hosp_cols if c in hosp.columns]
        patient_cols = [c for c in patient_cols if c in patient.columns]

        hosp = hosp.select(hosp_cols)
        patient = patient.select(patient_cols)

        # Merge on patient_id
        cohort = hosp.join(patient, on="patient_id", how="inner")

        return cohort

    def _get_table(self, table_name: str) -> pl.DataFrame:
        """
        Get a table from clifpy as a Polars DataFrame.

        Args:
            table_name: Name of the CLIF table

        Returns:
            Polars DataFrame
        """
        # clifpy tables are accessible as attributes
        table = getattr(self.clif, table_name, None)

        if table is None:
            raise ValueError(f"Table '{table_name}' not found in CLIF data")

        # Convert to polars if needed (clifpy may return pandas or polars)
        if hasattr(table, "to_pandas"):
            # It's already polars
            return table
        elif hasattr(table, "values"):
            # It's pandas, convert to polars
            return pl.from_pandas(table)
        else:
            return pl.DataFrame(table)

    def _filter_null_dates(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """Filter out rows with null admission or discharge dates."""
        return cohort.filter(
            pl.col("admission_dttm").is_not_null()
            & pl.col("discharge_dttm").is_not_null()
        )

    def _calculate_los(
        self, cohort: pl.DataFrame, min_los_days: float
    ) -> pl.DataFrame:
        """Calculate length of stay and filter."""
        cohort = cohort.with_columns(
            (
                (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds()
                / (24 * 3600)
            ).alias("hospitalization_los")
        )

        return cohort.filter(pl.col("hospitalization_los") > min_los_days)

    def _filter_icu_only(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """Filter to hospitalizations with at least one ICU ADT record."""
        adt = self._get_table("adt")

        # Find hospitalizations with ICU stays
        icu_hosp_ids = (
            adt.filter(pl.col("location_category").str.to_lowercase() == "icu")
            .select("hospitalization_id")
            .unique()
        )

        # Filter cohort to ICU hospitalizations
        return cohort.join(icu_hosp_ids, on="hospitalization_id", how="inner")

    def _calculate_icu_timing(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate ICU stay timing metrics.

        Adds columns:
        - first_icu_start_time
        - first_icu_end_time
        - first_icu_24hr_completion_time
        - second_icu_start_time
        """
        adt = self._get_table("adt")

        # Filter ADT to cohort hospitalizations
        cohort_ids = cohort.select("hospitalization_id").unique()
        adt_cohort = adt.join(cohort_ids, on="hospitalization_id", how="inner")

        # Sort and add ICU flag
        adt_cohort = adt_cohort.sort(["hospitalization_id", "in_dttm"])
        adt_cohort = adt_cohort.with_columns(
            (pl.col("location_category").str.to_lowercase() == "icu").alias("is_icu")
        )

        # Get discharge times from cohort
        discharge_map = cohort.select(["hospitalization_id", "discharge_dttm"])
        adt_cohort = adt_cohort.join(
            discharge_map, on="hospitalization_id", how="left"
        )

        # Calculate next in_dttm for each row (out_dttm approximation)
        adt_cohort = adt_cohort.with_columns(
            pl.col("in_dttm")
            .shift(-1)
            .over("hospitalization_id")
            .alias("next_in_dttm")
        )

        # Use discharge_dttm for last location
        adt_cohort = adt_cohort.with_columns(
            pl.coalesce(["next_in_dttm", "discharge_dttm"]).alias("out_dttm")
        )

        # Detect new ICU stays (previous location was not ICU)
        adt_cohort = adt_cohort.with_columns(
            pl.col("is_icu").shift(1).over("hospitalization_id").alias("prev_is_icu")
        )
        adt_cohort = adt_cohort.with_columns(
            pl.col("prev_is_icu").fill_null(False)
        )
        adt_cohort = adt_cohort.with_columns(
            (pl.col("is_icu") & ~pl.col("prev_is_icu")).alias("is_new_icu_stay")
        )

        # Number ICU stays
        adt_cohort = adt_cohort.with_columns(
            pl.col("is_new_icu_stay")
            .cum_sum()
            .over("hospitalization_id")
            .alias("icu_stay_number")
        )

        # Set non-ICU rows to stay number 0
        adt_cohort = adt_cohort.with_columns(
            pl.when(pl.col("is_icu"))
            .then(pl.col("icu_stay_number"))
            .otherwise(0)
            .alias("icu_stay_number")
        )

        # Filter to ICU events only
        icu_events = adt_cohort.filter(pl.col("is_icu"))

        # Aggregate by ICU stay
        icu_stays = (
            icu_events.group_by(["hospitalization_id", "icu_stay_number"])
            .agg(
                [
                    pl.col("in_dttm").min().alias("icu_start_time"),
                    pl.col("out_dttm").max().alias("icu_end_time"),
                ]
            )
        )

        # Calculate ICU LOS in hours
        icu_stays = icu_stays.with_columns(
            (
                (pl.col("icu_end_time") - pl.col("icu_start_time")).dt.total_seconds()
                / 3600
            ).alias("icu_los_hours")
        )

        # Extract first ICU stay
        first_icu = icu_stays.filter(pl.col("icu_stay_number") == 1).select(
            [
                "hospitalization_id",
                pl.col("icu_start_time").alias("first_icu_start_time"),
                pl.col("icu_end_time").alias("first_icu_end_time"),
                pl.col("icu_los_hours"),
            ]
        )

        # Calculate 24hr completion time (only if >= 24 hours)
        first_icu = first_icu.with_columns(
            pl.when(pl.col("icu_los_hours") >= 24)
            .then(pl.col("first_icu_start_time") + pl.duration(hours=24))
            .otherwise(None)
            .alias("first_icu_24hr_completion_time")
        ).drop("icu_los_hours")

        # Extract second ICU stay
        second_icu = icu_stays.filter(pl.col("icu_stay_number") == 2).select(
            [
                "hospitalization_id",
                pl.col("icu_start_time").alias("second_icu_start_time"),
            ]
        )

        # Merge back to cohort
        cohort = cohort.join(first_icu, on="hospitalization_id", how="left")
        cohort = cohort.join(second_icu, on="hospitalization_id", how="left")

        return cohort

    def _calculate_previous_hospitalization(
        self, cohort: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate previous_hospitalization_id for each patient."""
        # Sort by patient and admission time
        cohort = cohort.sort(["patient_id", "admission_dttm"])

        # Get previous hospitalization_id within each patient
        cohort = cohort.with_columns(
            pl.col("hospitalization_id")
            .shift(1)
            .over("patient_id")
            .alias("previous_hospitalization_id")
        )

        return cohort

    def _calculate_imv_at_24hr(self, cohort: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate whether patient was on invasive mechanical ventilation at 24 hours.

        IMV is defined by device_category containing 'invasive' or 'imv'.
        """
        try:
            resp = self._get_table("respiratory_support")
        except (ValueError, AttributeError):
            logger.warning(
                "respiratory_support table not available - setting imv_at_24hr to null"
            )
            return cohort.with_columns(pl.lit(None).cast(pl.Boolean).alias("imv_at_24hr"))

        # Get admission times from cohort
        admission_times = cohort.select(["hospitalization_id", "admission_dttm"])

        # Filter respiratory support to cohort
        cohort_ids = cohort.select("hospitalization_id").unique()
        resp_cohort = resp.join(cohort_ids, on="hospitalization_id", how="inner")

        # Add admission time
        resp_cohort = resp_cohort.join(
            admission_times, on="hospitalization_id", how="left"
        )

        # Calculate time since admission
        if "recorded_dttm" in resp_cohort.columns:
            time_col = "recorded_dttm"
        elif "start_dttm" in resp_cohort.columns:
            time_col = "start_dttm"
        else:
            logger.warning(
                "No time column found in respiratory_support - setting imv_at_24hr to null"
            )
            return cohort.with_columns(pl.lit(None).cast(pl.Boolean).alias("imv_at_24hr"))

        resp_cohort = resp_cohort.with_columns(
            (
                (pl.col(time_col) - pl.col("admission_dttm")).dt.total_seconds() / 3600
            ).alias("hours_since_admission")
        )

        # Filter to around 24 hours (23-25 hour window)
        resp_at_24hr = resp_cohort.filter(
            (pl.col("hours_since_admission") >= 23)
            & (pl.col("hours_since_admission") <= 25)
        )

        # Check for IMV (invasive mechanical ventilation)
        if "device_category" in resp_at_24hr.columns:
            imv_at_24hr = (
                resp_at_24hr.filter(
                    pl.col("device_category").str.to_lowercase().str.contains("invasive")
                    | pl.col("device_category").str.to_lowercase().str.contains("imv")
                    | pl.col("device_category").str.to_lowercase().str.contains("mechanical")
                )
                .select("hospitalization_id")
                .unique()
                .with_columns(pl.lit(True).alias("imv_at_24hr"))
            )
        else:
            logger.warning(
                "device_category column not found - setting imv_at_24hr to null"
            )
            return cohort.with_columns(pl.lit(None).cast(pl.Boolean).alias("imv_at_24hr"))

        # Merge back to cohort
        cohort = cohort.join(imv_at_24hr, on="hospitalization_id", how="left")
        cohort = cohort.with_columns(
            pl.col("imv_at_24hr").fill_null(False)
        )

        imv_count = cohort.filter(pl.col("imv_at_24hr")).height
        logger.info(f"Patients on IMV at 24hr: {imv_count:,} ({imv_count/cohort.height*100:.1f}%)")

        return cohort


def build_cohort(
    clif_config_path: str,
    output_path: Optional[str] = None,
    min_age: int = 18,
    min_los_days: float = 0,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """
    Build FLAIR cohort from CLIF configuration.

    Convenience function that creates a FLAIRCohortBuilder and builds the cohort.

    Args:
        clif_config_path: Path to clifpy configuration JSON
        output_path: Optional path to save cohort.parquet
        min_age: Minimum age filter (default 18)
        min_los_days: Minimum LOS filter (default 0)

    Returns:
        Tuple of (cohort DataFrame, exclusion statistics)
    """
    builder = FLAIRCohortBuilder(clif_config_path)
    return builder.build_cohort(
        output_path=output_path,
        min_age=min_age,
        min_los_days=min_los_days,
    )
