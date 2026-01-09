"""
Task 5: ICU Length of Stay Prediction

Regression task to predict remaining ICU length of stay using the first
24 hours of ICU data.

Uses first 24 hours of ICU data to predict remaining ICU LOS (total - 24hr).
Only includes patients with at least 24 hours ICU stay.
Target is a continuous value in hours (remaining LOS after first 24 hours).
"""

import polars as pl
import logging
from typing import Dict

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task5ICULoS(BaseTask):
    """
    Predict remaining ICU length of stay (first ICU stay).

    Input: First 24 hours of ICU data
    Output: Continuous value (hours) - remaining LOS after first 24 hours
    Cohort: ICU patients with >= 24hr first ICU stay
    """

    # Default outlier bounds for remaining LOS (in hours)
    DEFAULT_MIN_REMAINING_LOS = 2
    DEFAULT_MAX_REMAINING_LOS = 1000

    def __init__(self, config=None):
        super().__init__(config)
        self._exclusion_stats: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "task5_icu_los"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task5_icu_los",
            display_name="ICU Length of Stay",
            description="Predict remaining ICU length of stay from first 24 hours",
            task_type=TaskType.REGRESSION,
            input_window_hours=24,
            prediction_window=None,
            cohort_filter="icu_24hr_complete",
            label_column="icu_los_hours",
            evaluation_metrics=[
                "mse",
                "rmse",
                "mae",
                "r2",
                "explained_variance",
            ],
        )

    def filter_cohort(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Filter to patients with:
        1. At least 24 hours ICU stay
        2. Remaining LOS within valid range (2 < hours < 1000)

        Args:
            cohort_df: Base cohort DataFrame (7 columns)
            icu_timing: ICU timing DataFrame with first_icu_start_time, first_icu_end_time

        Returns:
            Filtered cohort (only patients meeting both criteria)
        """
        initial_count = cohort_df.height
        self._exclusion_stats["initial"] = initial_count

        # Join cohort with ICU timing
        cohort_with_timing = cohort_df.join(
            icu_timing.select([
                "hospitalization_id",
                "first_icu_start_time",
                "first_icu_end_time",
            ]),
            on="hospitalization_id",
            how="inner",
        )

        # Compute remaining LOS (total ICU stay - 24 hours)
        cohort_with_timing = cohort_with_timing.with_columns(
            (
                (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
                .dt.total_seconds()
                / 3600
                - 24
            ).alias("_remaining_los_hours")
        )

        # Filter 1: >= 24 hours ICU stay (remaining_los >= 0)
        before_24hr = cohort_with_timing.height
        filtered = cohort_with_timing.filter(pl.col("_remaining_los_hours") >= 0)
        self._exclusion_stats["excluded_insufficient_stay"] = before_24hr - filtered.height
        self._exclusion_stats["after_24hr_filter"] = filtered.height

        # Filter 2: Exclude very short remaining LOS (<= min threshold)
        before_short = filtered.height
        filtered = filtered.filter(
            pl.col("_remaining_los_hours") > self.DEFAULT_MIN_REMAINING_LOS
        )
        self._exclusion_stats["excluded_short_remaining_los"] = before_short - filtered.height
        self._exclusion_stats["after_short_filter"] = filtered.height

        # Filter 3: Exclude very long remaining LOS (>= max threshold)
        before_long = filtered.height
        filtered = filtered.filter(
            pl.col("_remaining_los_hours") < self.DEFAULT_MAX_REMAINING_LOS
        )
        self._exclusion_stats["excluded_long_remaining_los"] = before_long - filtered.height
        self._exclusion_stats["final"] = filtered.height

        # Log exclusion statistics
        self._log_exclusion_stats()

        # Drop helper columns (base cohort should stay 7 columns)
        filtered = filtered.drop([
            "first_icu_start_time",
            "first_icu_end_time",
            "_remaining_los_hours",
        ])

        return filtered

    def build_time_windows(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build time windows for Task 5.

        window_start = first_icu_start_time
        window_end = first_icu_start_time + 24 hours

        Args:
            cohort_df: Filtered cohort DataFrame
            icu_timing: ICU timing DataFrame

        Returns:
            DataFrame with hospitalization_id, window_start, window_end
        """
        return (
            cohort_df.select("hospitalization_id")
            .join(
                icu_timing.select(["hospitalization_id", "first_icu_start_time"]),
                on="hospitalization_id",
                how="inner",
            )
            .select([
                pl.col("hospitalization_id"),
                pl.col("first_icu_start_time").alias("window_start"),
                (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("window_end"),
            ])
        )

    def build_labels(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build ICU LOS labels.

        Label = (first_icu_end_time - first_icu_start_time) - 24 hours
        We predict remaining LOS after the first 24 hours of ICU data.

        Args:
            cohort_df: Cohort DataFrame (7 columns)
            icu_timing: ICU timing DataFrame with first_icu_start_time, first_icu_end_time

        Returns:
            DataFrame with [hospitalization_id, icu_los_hours]
        """
        labels = (
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
                (
                    (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
                    .dt.total_seconds()
                    / 3600
                    - 24  # Subtract 24 hours - predicting remaining LOS after first 24hr
                ).alias("icu_los_hours"),
            ])
        )

        self._log_los_stats(labels)
        return labels

    def _log_los_stats(self, labels: pl.DataFrame) -> None:
        """Log statistics about the remaining LOS values (after first 24hr)."""
        stats = labels.select([
            pl.col("icu_los_hours").mean().alias("mean"),
            pl.col("icu_los_hours").std().alias("std"),
            pl.col("icu_los_hours").min().alias("min"),
            pl.col("icu_los_hours").max().alias("max"),
            pl.col("icu_los_hours").median().alias("median"),
        ]).row(0, named=True)

        logger.info(
            f"Remaining ICU LOS stats (hours after first 24hr): "
            f"mean={stats['mean']:.1f}, "
            f"median={stats['median']:.1f}, "
            f"std={stats['std']:.1f}, "
            f"range=[{stats['min']:.1f}, {stats['max']:.1f}]"
        )

    def _log_exclusion_stats(self) -> None:
        """Log detailed exclusion statistics for ICU LOS cohort."""
        logger.info("=" * 50)
        logger.info("ICU LOS COHORT EXCLUSIONS")
        logger.info("=" * 50)
        logger.info(f"Initial cohort: {self._exclusion_stats.get('initial', 0):,}")
        logger.info(
            f"  - Excluded (<24hr ICU stay): "
            f"{self._exclusion_stats.get('excluded_insufficient_stay', 0):,}"
        )
        logger.info(
            f"  After 24hr filter: {self._exclusion_stats.get('after_24hr_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (remaining LOS <= {self.DEFAULT_MIN_REMAINING_LOS}hr): "
            f"{self._exclusion_stats.get('excluded_short_remaining_los', 0):,}"
        )
        logger.info(
            f"  After short stay filter: {self._exclusion_stats.get('after_short_filter', 0):,}"
        )
        logger.info(
            f"  - Excluded (remaining LOS >= {self.DEFAULT_MAX_REMAINING_LOS}hr): "
            f"{self._exclusion_stats.get('excluded_long_remaining_los', 0):,}"
        )
        logger.info(f"Final cohort: {self._exclusion_stats.get('final', 0):,}")
        logger.info(
            f"Outlier bounds: {self.DEFAULT_MIN_REMAINING_LOS} < hours < "
            f"{self.DEFAULT_MAX_REMAINING_LOS}"
        )
        logger.info("=" * 50)
