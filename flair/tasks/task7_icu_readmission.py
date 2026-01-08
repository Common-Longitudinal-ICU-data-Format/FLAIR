"""
Task 7: Unplanned ICU Readmission Prediction

Binary classification task to predict whether a patient will have an
UNPLANNED readmission to the ICU during the same hospitalization.

Uses entire first ICU stay data to predict if there will be an unplanned
return to ICU. Planned readmissions (ICU -> Procedural -> ICU) are excluded.

Exclusions applied:
- Deaths during first ICU stay
- Direct ICU discharges (last location is ICU)
- Short ICU stays (<1 hour)
- Intermediate ER visits (ER after first location)
"""

import polars as pl
import logging
from typing import Dict

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)

# Location category mapping based on actual data
# Available: ed, ward, stepdown, icu, procedural, l&d, hospice, psych, rehab, radiology, dialysis, other
LOCATION_MAPPING = {
    # Relevant for analysis:
    "icu": "ICU",
    "ward": "Ward",
    "ed": "ER",
    "procedural": "Procedural",
    # Excluded from analysis (become "Other"):
    "stepdown": "Other",
    "l&d": "Other",
    "hospice": "Other",
    "psych": "Other",
    "rehab": "Other",
    "radiology": "Other",
    "dialysis": "Other",
    "other": "Other",
}


class Task7ICUReadmission(BaseTask):
    """
    Predict UNPLANNED ICU readmission (return to ICU within same hospitalization).

    Input: Entire first ICU stay (in_dttm to out_dttm of 1st ICU)
    Output: Binary (1 = unplanned readmission, 0 = no readmission)
    Cohort: Patients with ICU stay, excluding deaths, direct discharges, short stays

    Key logic:
    - Consecutive ICU entries are merged (ICU -> ICU = single stay)
    - ICU -> Procedural -> ICU = planned (procedural removed, ICUs merged, no readmission)
    - ICU -> Ward -> ICU = unplanned readmission (label = 1)
    - ICU -> Other -> ICU = unplanned readmission (label = 1)
      (Other = stepdown, hospice, psych, rehab, radiology, dialysis, etc.)
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._exclusion_stats: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "task7_icu_readmission"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task7_icu_readmission",
            display_name="Unplanned ICU Readmission",
            description="Predict whether patient will have unplanned readmission to ICU",
            task_type=TaskType.BINARY_CLASSIFICATION,
            input_window_hours=None,  # Variable window (entire 1st ICU stay)
            prediction_window=None,
            cohort_filter="icu_any",
            label_column="label_icu_readmission",
            positive_class="readmitted",
            evaluation_metrics=[
                "auroc",
                "auprc",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "npv",
            ],
        )

    def _compute_unplanned_icu_timing(
        self, adt: pl.DataFrame, cohort_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Compute ICU timing with unplanned readmission logic.

        Handles:
        - ICU stay merging (consecutive ICU entries -> single stay)
        - Procedural removal (ICU -> Procedural -> ICU -> merged single stay)
        - Direct ICU discharge detection
        - Intermediate ER detection

        Args:
            adt: ADT DataFrame with location data
            cohort_df: Cohort DataFrame with death_dttm column

        Returns:
            DataFrame with columns:
            - hospitalization_id
            - first_icu_start_time, first_icu_end_time
            - second_icu_start_time (null if no unplanned readmission)
            - is_direct_icu_discharge (bool)
            - has_intermediate_er (bool)
            - hospital_id, hospital_type
        """
        if adt.height == 0:
            logger.warning("No ADT data provided")
            return self._empty_icu_timing()

        # 1. Map location categories to standardized names
        adt = adt.with_columns(
            pl.col("location_category")
            .str.to_lowercase()
            .replace(LOCATION_MAPPING)
            .alias("std_location")
        )

        # 2. Keep all locations (including "Other") so ICU → Other → ICU counts as readmission
        # Only procedural sandwiches (ICU → Procedural → ICU) are merged into single stays
        adt_filtered = adt

        if adt_filtered.height == 0:
            logger.warning("No relevant locations found after filtering")
            return self._empty_icu_timing()

        # 3. Sort by hospitalization and time
        adt_filtered = adt_filtered.sort(["hospitalization_id", "in_dttm"])

        # 4. Remove procedural entries sandwiched between ICU entries
        # This handles ICU -> Procedural -> ICU pattern (planned readmission)
        adt_with_neighbors = adt_filtered.with_columns([
            pl.col("std_location")
            .shift(1)
            .over("hospitalization_id")
            .alias("prev_location"),
            pl.col("std_location")
            .shift(-1)
            .over("hospitalization_id")
            .alias("next_location"),
        ])

        # Keep rows that are NOT procedural between ICUs
        adt_no_sandwiched = adt_with_neighbors.filter(
            ~(
                (pl.col("std_location") == "Procedural")
                & (pl.col("prev_location") == "ICU")
                & (pl.col("next_location") == "ICU")
            )
        )

        # 5. Merge consecutive same-location entries (ICU -> ICU = single stay)
        adt_no_sandwiched = adt_no_sandwiched.sort(["hospitalization_id", "in_dttm"])

        # Step 5a: Compute boolean for location change (can't nest .over() calls)
        adt_no_sandwiched = adt_no_sandwiched.with_columns(
            (
                pl.col("std_location")
                != pl.col("std_location").shift(1).over("hospitalization_id")
            )
            .fill_null(True)
            .alias("location_changed")
        )

        # Step 5b: Compute cumulative sum to create location groups
        adt_with_groups = adt_no_sandwiched.with_columns(
            pl.col("location_changed")
            .cum_sum()
            .over("hospitalization_id")
            .alias("location_group")
        )

        # Aggregate: min(in_dttm), max(out_dttm) per location_group
        # Also keep hospital_id and hospital_type from first record
        agg_cols = [
            pl.col("in_dttm").min().alias("in_dttm"),
            pl.col("out_dttm").max().alias("out_dttm"),
            pl.col("std_location").first().alias("std_location"),
        ]
        if "hospital_id" in adt_with_groups.columns:
            agg_cols.append(pl.col("hospital_id").first().alias("hospital_id"))
        if "hospital_type" in adt_with_groups.columns:
            agg_cols.append(pl.col("hospital_type").first().alias("hospital_type"))

        adt_merged = adt_with_groups.group_by(
            ["hospitalization_id", "location_group"]
        ).agg(agg_cols)

        # 6. Re-rank after merging
        adt_merged = adt_merged.sort(["hospitalization_id", "in_dttm"])
        adt_merged = adt_merged.with_columns(
            pl.col("in_dttm")
            .rank("ordinal")
            .over("hospitalization_id")
            .alias("location_rank")
        )

        # 7. Get first and second ICU (after merging)
        icu_entries = adt_merged.filter(pl.col("std_location") == "ICU")
        icu_entries = icu_entries.with_columns(
            pl.col("in_dttm")
            .rank("ordinal")
            .over("hospitalization_id")
            .alias("icu_rank")
        )

        # First ICU with hospital info
        first_icu_cols = [
            "hospitalization_id",
            pl.col("in_dttm").alias("first_icu_start_time"),
            pl.col("out_dttm").alias("first_icu_end_time"),
        ]
        if "hospital_id" in icu_entries.columns:
            first_icu_cols.append("hospital_id")
        if "hospital_type" in icu_entries.columns:
            first_icu_cols.append("hospital_type")

        first_icu = icu_entries.filter(pl.col("icu_rank") == 1).select(first_icu_cols)

        # Second ICU (if exists)
        second_icu = icu_entries.filter(pl.col("icu_rank") == 2).select([
            "hospitalization_id",
            pl.col("in_dttm").alias("second_icu_start_time"),
        ])

        # 8. Detect direct ICU discharge (last location is ICU)
        last_location = adt_merged.group_by("hospitalization_id").agg(
            pl.col("std_location").sort_by("in_dttm").last().alias("last_location")
        )
        last_location = last_location.with_columns(
            (pl.col("last_location") == "ICU").alias("is_direct_icu_discharge")
        )

        # 9. Detect intermediate ER (ER appears after first location)
        has_intermediate_er = (
            adt_merged.filter(
                (pl.col("location_rank") > 1) & (pl.col("std_location") == "ER")
            )
            .select("hospitalization_id")
            .unique()
            .with_columns(pl.lit(True).alias("has_intermediate_er"))
        )

        # 10. Join all components
        result = first_icu.join(second_icu, on="hospitalization_id", how="left")
        result = result.join(
            last_location.select(["hospitalization_id", "is_direct_icu_discharge"]),
            on="hospitalization_id",
            how="left",
        )
        result = result.join(has_intermediate_er, on="hospitalization_id", how="left")

        # Fill nulls for boolean columns
        result = result.with_columns([
            pl.col("is_direct_icu_discharge").fill_null(False),
            pl.col("has_intermediate_er").fill_null(False),
        ])

        # Ensure hospital columns exist
        if "hospital_id" not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias("hospital_id"))
        if "hospital_type" not in result.columns:
            result = result.with_columns(
                pl.lit(None).cast(pl.Utf8).alias("hospital_type")
            )

        logger.info(
            f"Computed unplanned ICU timing: {result.height} hospitalizations, "
            f"{second_icu.height} with potential readmissions"
        )

        return result

    def _empty_icu_timing(self) -> pl.DataFrame:
        """Return empty ICU timing DataFrame with correct schema."""
        return pl.DataFrame({
            "hospitalization_id": [],
            "first_icu_start_time": [],
            "first_icu_end_time": [],
            "second_icu_start_time": [],
            "is_direct_icu_discharge": [],
            "has_intermediate_er": [],
            "hospital_id": [],
            "hospital_type": [],
        }).cast({
            "hospitalization_id": pl.Utf8,
            "first_icu_start_time": pl.Datetime,
            "first_icu_end_time": pl.Datetime,
            "second_icu_start_time": pl.Datetime,
            "is_direct_icu_discharge": pl.Boolean,
            "has_intermediate_er": pl.Boolean,
            "hospital_id": pl.Utf8,
            "hospital_type": pl.Utf8,
        })

    def filter_cohort(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Filter cohort applying unplanned readmission exclusions.

        Excludes:
        1. Patients without ICU data
        2. Deaths during first ICU stay
        3. Direct ICU discharges (last location is ICU)
        4. Short ICU stays (<1 hour)
        5. Intermediate ER visits

        Args:
            cohort_df: Base cohort DataFrame (with death_dttm column)
            icu_timing: ICU timing DataFrame from _compute_unplanned_icu_timing

        Returns:
            Filtered cohort
        """
        initial_count = cohort_df.height
        self._exclusion_stats["initial"] = initial_count

        # 1. Keep only patients with ICU data
        filtered = cohort_df.join(
            icu_timing.select([
                "hospitalization_id",
                "first_icu_start_time",
                "first_icu_end_time",
                "is_direct_icu_discharge",
                "has_intermediate_er",
            ]),
            on="hospitalization_id",
            how="inner",
        )
        self._exclusion_stats["after_icu_filter"] = filtered.height
        self._exclusion_stats["excluded_no_icu"] = initial_count - filtered.height

        # 2. Exclude deaths during first ICU stay
        if "death_dttm" in filtered.columns:
            before_death = filtered.height
            filtered = filtered.filter(
                ~(
                    pl.col("death_dttm").is_not_null()
                    & (pl.col("death_dttm") >= pl.col("first_icu_start_time"))
                    & (pl.col("death_dttm") <= pl.col("first_icu_end_time"))
                )
            )
            self._exclusion_stats["excluded_death_during_icu"] = (
                before_death - filtered.height
            )
        else:
            self._exclusion_stats["excluded_death_during_icu"] = 0
            logger.warning("death_dttm column not found, skipping death exclusion")

        self._exclusion_stats["after_death_filter"] = filtered.height

        # 3. Exclude direct ICU discharges
        before_direct = filtered.height
        filtered = filtered.filter(~pl.col("is_direct_icu_discharge"))
        self._exclusion_stats["excluded_direct_icu_discharge"] = (
            before_direct - filtered.height
        )
        self._exclusion_stats["after_direct_discharge_filter"] = filtered.height

        # 4. Exclude short ICU stays (<1 hour)
        before_short = filtered.height
        filtered = filtered.filter(
            (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
            >= pl.duration(hours=1)
        )
        self._exclusion_stats["excluded_short_stay"] = before_short - filtered.height
        self._exclusion_stats["after_short_stay_filter"] = filtered.height

        # 5. Exclude intermediate ER visits
        before_er = filtered.height
        filtered = filtered.filter(~pl.col("has_intermediate_er"))
        self._exclusion_stats["excluded_intermediate_er"] = before_er - filtered.height
        self._exclusion_stats["final"] = filtered.height

        # Log exclusion statistics
        self._log_exclusion_stats()

        # Drop helper columns before returning
        drop_cols = [
            "first_icu_start_time",
            "first_icu_end_time",
            "is_direct_icu_discharge",
            "has_intermediate_er",
        ]
        filtered = filtered.drop([c for c in drop_cols if c in filtered.columns])

        return filtered

    def _log_exclusion_stats(self) -> None:
        """Log detailed exclusion statistics."""
        logger.info("=" * 50)
        logger.info("ICU READMISSION COHORT EXCLUSIONS")
        logger.info("=" * 50)
        logger.info(f"Initial cohort: {self._exclusion_stats.get('initial', 0):,}")
        logger.info(
            f"  - Excluded (no ICU data): {self._exclusion_stats.get('excluded_no_icu', 0):,}"
        )
        logger.info(
            f"  After ICU filter: {self._exclusion_stats.get('after_icu_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (death during ICU): {self._exclusion_stats.get('excluded_death_during_icu', 0):,}"
        )
        logger.info(
            f"  After death filter: {self._exclusion_stats.get('after_death_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (direct ICU discharge): {self._exclusion_stats.get('excluded_direct_icu_discharge', 0):,}"
        )
        logger.info(
            f"  After direct discharge filter: {self._exclusion_stats.get('after_direct_discharge_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (short stay <1hr): {self._exclusion_stats.get('excluded_short_stay', 0):,}"
        )
        logger.info(
            f"  After short stay filter: {self._exclusion_stats.get('after_short_stay_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (intermediate ER): {self._exclusion_stats.get('excluded_intermediate_er', 0):,}"
        )
        logger.info(f"Final cohort: {self._exclusion_stats.get('final', 0):,}")
        logger.info("=" * 50)

    def build_time_windows(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build time windows for Task 7.

        Uses entire first ICU stay (variable length):
        - window_start = first_icu_start_time
        - window_end = first_icu_end_time (end of first ICU stay = prediction time)

        Args:
            cohort_df: Filtered cohort DataFrame
            icu_timing: ICU timing DataFrame

        Returns:
            DataFrame with hospitalization_id, window_start, window_end
        """
        return (
            cohort_df.select("hospitalization_id")
            .join(
                icu_timing.select([
                    "hospitalization_id",
                    "first_icu_start_time",
                    "first_icu_end_time",
                ]),
                on="hospitalization_id",
                how="inner",
            )
            .select([
                pl.col("hospitalization_id"),
                pl.col("first_icu_start_time").alias("window_start"),
                pl.col("first_icu_end_time").alias("window_end"),
            ])
        )

    def build_labels(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build UNPLANNED ICU readmission labels.

        Label = 1 if second_icu_start_time exists (after all merging/filtering)
        Since we already:
        - Removed procedural between ICUs (making them single stays)
        - Merged consecutive ICUs
        Any remaining second ICU is an unplanned readmission.

        Args:
            cohort_df: Cohort DataFrame
            icu_timing: ICU timing DataFrame with second_icu_start_time column

        Returns:
            DataFrame with [hospitalization_id, label_icu_readmission]
        """
        labels = (
            cohort_df.select("hospitalization_id")
            .join(
                icu_timing.select([
                    "hospitalization_id",
                    "second_icu_start_time",
                ]),
                on="hospitalization_id",
                how="inner",
            )
            .select([
                pl.col("hospitalization_id"),
                pl.col("second_icu_start_time")
                .is_not_null()
                .cast(pl.Int32)
                .alias("label_icu_readmission"),
            ])
        )

        self._log_readmission_stats(labels)
        return labels

    def _log_readmission_stats(self, labels: pl.DataFrame) -> None:
        """Log statistics about the unplanned ICU readmission labels."""
        pos_count = labels.filter(pl.col("label_icu_readmission") == 1).height
        total = labels.height
        readmission_rate = (pos_count / total * 100) if total > 0 else 0

        logger.info(
            f"Unplanned ICU readmission labels: "
            f"{pos_count} positive ({readmission_rate:.1f}%), "
            f"{total - pos_count} negative"
        )

    def build_task_dataset(
        self,
        cohort_df: pl.DataFrame,
        adt_data: pl.DataFrame,
        train_start: str = None,
        train_end: str = None,
        test_start: str = None,
        test_end: str = None,
        site: str = None,
    ) -> pl.DataFrame:
        """
        Build complete task-specific dataset with temporal split.

        Overrides base class to use _compute_unplanned_icu_timing instead
        of the base _compute_icu_timing.

        For MIMIC (de-identified dates): Uses 70/30 temporal split by
        sorting admission_dttm. Date parameters are ignored.

        For other sites: Uses date-based split. Date parameters are required.

        Args:
            cohort_df: Base cohort DataFrame
            adt_data: ADT DataFrame for computing ICU timing
            train_start: Train period start date (YYYY-MM-DD). Optional for MIMIC.
            train_end: Train period end date (YYYY-MM-DD). Optional for MIMIC.
            test_start: Test period start date (YYYY-MM-DD). Optional for MIMIC.
            test_end: Test period end date (YYYY-MM-DD). Optional for MIMIC.
            site: Site name from config (e.g., "mimic"). Used to determine split method.

        Returns:
            DataFrame with 13 columns
        """
        from datetime import date

        # 1. Compute ICU timing using UNPLANNED logic (overriding base)
        icu_timing = self._compute_unplanned_icu_timing(adt_data, cohort_df)

        # 2. Filter cohort for this task (with all exclusions)
        task_cohort = self.filter_cohort(cohort_df, icu_timing)
        logger.info(f"Task {self.name}: {task_cohort.height} patients after filter")

        # 3. Build labels (task-specific)
        labels = self.build_labels(task_cohort, icu_timing)

        # 4. Build time windows (task-specific)
        windows = self.build_time_windows(task_cohort, icu_timing)

        # 5. Assign split based on site
        if site and site.lower() == "mimic":
            # MIMIC: Use 70/30 temporal split (de-identified dates)
            split_df = self._assign_temporal_split(task_cohort)
            logger.info("Using 70/30 temporal split for MIMIC (de-identified dates)")
        else:
            # Non-MIMIC: Use date-based split (require dates)
            if not all([train_start, train_end, test_start, test_end]):
                raise ValueError(
                    "Date parameters required for non-MIMIC sites. "
                    "Provide train_start, train_end, test_start, test_end."
                )
            train_start_dt = date.fromisoformat(train_start)
            train_end_dt = date.fromisoformat(train_end)
            test_start_dt = date.fromisoformat(test_start)
            test_end_dt = date.fromisoformat(test_end)

            split_df = task_cohort.select([
                pl.col("hospitalization_id"),
                pl.when(
                    (pl.col("admission_dttm").dt.date() >= train_start_dt)
                    & (pl.col("admission_dttm").dt.date() <= train_end_dt)
                )
                .then(pl.lit("train"))
                .when(
                    (pl.col("admission_dttm").dt.date() >= test_start_dt)
                    & (pl.col("admission_dttm").dt.date() <= test_end_dt)
                )
                .then(pl.lit("test"))
                .otherwise(pl.lit(None))
                .alias("split"),
            ])

        # 6. Get hospital info from icu_timing
        hospital_info = icu_timing.select([
            "hospitalization_id",
            "hospital_id",
            "hospital_type",
        ])

        # 7. Join all components
        core_cols = [
            "hospitalization_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
            "sex_category",
            "race_category",
            "ethnicity_category",
        ]
        available_core_cols = [c for c in core_cols if c in task_cohort.columns]

        result = (
            task_cohort.select(available_core_cols)
            .join(windows, on="hospitalization_id")
            .join(labels, on="hospitalization_id")
            .join(split_df, on="hospitalization_id")
            .join(hospital_info, on="hospitalization_id", how="left")
        )

        # 8. Filter to only train/test
        result = result.filter(pl.col("split").is_not_null())

        # 9. Reorder to final 13-column schema
        label_col = self._task_config.label_column
        result = result.select([
            "hospitalization_id",
            "admission_dttm",
            "discharge_dttm",
            "window_start",
            "window_end",
            label_col,
            "split",
            "age_at_admission",
            "sex_category",
            "race_category",
            "ethnicity_category",
            "hospital_id",
            "hospital_type",
        ])

        # Log statistics
        train_count = result.filter(pl.col("split") == "train").height
        test_count = result.filter(pl.col("split") == "test").height
        logger.info(
            f"Task {self.name}: {result.height} total, "
            f"{train_count} train, {test_count} test"
        )

        return result
