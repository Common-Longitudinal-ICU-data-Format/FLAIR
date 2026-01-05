"""
Task 5: ICU Length of Stay Prediction

Regression task to predict total ICU length of stay using the first
24 hours of ICU data.

Uses first 24 hours of ICU data to predict total ICU LOS.
Only includes patients with at least 24 hours ICU stay.
Target is a continuous value in hours.
"""

import polars as pl
import logging
from typing import Optional

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task5ICULoS(BaseTask):
    """
    Predict total ICU length of stay (first ICU stay).

    Input: First 24 hours of ICU data
    Output: Continuous value (hours)
    Cohort: ICU patients with first_icu_24hr_completion_time not null (>= 24hr ICU stay)
    """

    @property
    def name(self) -> str:
        return "task5_icu_los"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task5_icu_los",
            display_name="ICU Length of Stay",
            description="Predict total ICU length of stay from first 24 hours",
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

    def filter_cohort(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter to patients with at least 24 hours ICU stay.

        Requires first_icu_24hr_completion_time to be not null,
        indicating the patient had at least 24 hours of ICU data.
        """
        if "first_icu_24hr_completion_time" in cohort_df.columns:
            filtered = cohort_df.filter(
                pl.col("first_icu_24hr_completion_time").is_not_null()
            )
            logger.info(
                f"Filtered to patients with >= 24hr ICU stay: "
                f"{filtered.height} of {cohort_df.height}"
            )
            return filtered

        # Fallback: calculate from first_icu_start_time and first_icu_end_time
        if (
            "first_icu_start_time" in cohort_df.columns
            and "first_icu_end_time" in cohort_df.columns
        ):
            filtered = cohort_df.filter(
                (
                    (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
                    .dt.total_seconds()
                    / 3600
                )
                >= 24
            )
            logger.info(
                f"Filtered to patients with >= 24hr ICU stay: "
                f"{filtered.height} of {cohort_df.height}"
            )
            return filtered

        logger.warning("No ICU timing columns found - returning full cohort")
        return cohort_df

    def build_labels(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Build ICU LOS labels.

        Label = first_icu_end_time - first_icu_start_time (in hours)

        Args:
            cohort_df: Cohort DataFrame with ICU timing columns
            narratives_dir: Not used for this task

        Returns:
            DataFrame with [hospitalization_id, icu_los_hours]
        """
        # Check if labels are pre-computed
        if "icu_los_hours" in cohort_df.columns:
            labels = cohort_df.select(["hospitalization_id", "icu_los_hours"])
            self._log_los_stats(labels)
            return labels

        # Calculate from ICU timing columns
        if (
            "first_icu_start_time" not in cohort_df.columns
            or "first_icu_end_time" not in cohort_df.columns
        ):
            raise ValueError(
                "Cohort must have first_icu_start_time and first_icu_end_time columns"
            )

        labels = cohort_df.select(
            [
                pl.col("hospitalization_id"),
                (
                    (pl.col("first_icu_end_time") - pl.col("first_icu_start_time"))
                    .dt.total_seconds()
                    / 3600
                ).alias("icu_los_hours"),
            ]
        )

        self._log_los_stats(labels)
        return labels

    def _log_los_stats(self, labels: pl.DataFrame) -> None:
        """Log statistics about the LOS values."""
        stats = labels.select(
            [
                pl.col("icu_los_hours").mean().alias("mean"),
                pl.col("icu_los_hours").std().alias("std"),
                pl.col("icu_los_hours").min().alias("min"),
                pl.col("icu_los_hours").max().alias("max"),
                pl.col("icu_los_hours").median().alias("median"),
            ]
        ).row(0, named=True)

        logger.info(
            f"ICU LOS stats (hours): "
            f"mean={stats['mean']:.1f}, "
            f"median={stats['median']:.1f}, "
            f"std={stats['std']:.1f}, "
            f"range=[{stats['min']:.1f}, {stats['max']:.1f}]"
        )
