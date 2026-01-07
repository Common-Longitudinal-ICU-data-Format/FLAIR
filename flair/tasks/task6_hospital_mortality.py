"""
Task 6: In-Hospital Mortality Prediction

Binary classification task to predict whether a patient will die
during the hospitalization.

Uses first 24 hours of ICU data to predict mortality at discharge.
Only includes patients with at least 24 hours ICU stay.
Expected prevalence: ~5-15% mortality.
"""

import polars as pl
import logging

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task6HospitalMortality(BaseTask):
    """
    Predict in-hospital mortality.

    Input: First 24 hours of ICU data
    Output: Binary (1 = expired, 0 = survived)
    Cohort: ICU patients with >= 24 hours ICU stay
    Expected prevalence: ~5-15%
    """

    @property
    def name(self) -> str:
        return "task6_hospital_mortality"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task6_hospital_mortality",
            display_name="In-Hospital Mortality",
            description="Predict whether patient will die during hospitalization",
            task_type=TaskType.BINARY_CLASSIFICATION,
            input_window_hours=24,
            prediction_window=None,
            cohort_filter="icu_24hr_complete",
            label_column="label_mortality",
            positive_class="expired",
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

    def filter_cohort(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Filter to patients with at least 24 hours ICU stay.

        Args:
            cohort_df: Base cohort DataFrame (8 columns including discharge_category)
            icu_timing: ICU timing DataFrame

        Returns:
            Filtered cohort (only patients with >= 24hr first ICU stay)
        """
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

        # Filter to >= 24 hours ICU stay
        filtered = cohort_with_timing.filter(
            (
                (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
                .dt.total_seconds()
                / 3600
            )
            >= 24
        )

        # Drop the timing columns
        filtered = filtered.drop(["first_icu_start_time", "first_icu_end_time"])

        logger.info(
            f"Filtered to patients with >= 24hr ICU stay: "
            f"{filtered.height} of {cohort_df.height}"
        )

        return filtered

    def build_time_windows(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Build time windows for Task 6.

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
        Build mortality labels from discharge_category.

        Label = 1 if discharge_category in ['expired', 'hospice'], else 0

        Args:
            cohort_df: Cohort DataFrame with discharge_category column
            icu_timing: ICU timing DataFrame (not used for mortality)

        Returns:
            DataFrame with [hospitalization_id, label_mortality]
        """
        if "discharge_category" not in cohort_df.columns:
            raise ValueError(
                "Cohort must have discharge_category column for mortality labels"
            )

        labels = cohort_df.select([
            pl.col("hospitalization_id"),
            (
                pl.col("discharge_category").str.to_lowercase().is_in(["expired", "hospice"])
            )
            .cast(pl.Int32)
            .alias("label_mortality"),
        ])

        self._log_mortality_stats(labels)
        return labels

    def _log_mortality_stats(self, labels: pl.DataFrame) -> None:
        """Log statistics about the mortality labels."""
        pos_count = labels.filter(pl.col("label_mortality") == 1).height
        total = labels.height
        mortality_rate = (pos_count / total * 100) if total > 0 else 0

        logger.info(
            f"Mortality labels: "
            f"{pos_count} positive ({mortality_rate:.1f}%), "
            f"{total - pos_count} negative"
        )
