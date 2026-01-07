"""
Task 7: ICU Readmission Prediction

Binary classification task to predict whether a patient will be
readmitted to the ICU during the same hospitalization.

Uses entire first ICU stay data to predict ICU readmission.
Only includes patients with at least 24 hours first ICU stay.
"""

import polars as pl
import logging

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task7ICUReadmission(BaseTask):
    """
    Predict ICU readmission (return to ICU within same hospitalization).

    Input: Entire first ICU stay (variable length, from start to end of 1st ICU)
    Output: Binary (1 = readmitted to ICU, 0 = not readmitted)
    Cohort: ICU patients with >= 24 hours first ICU stay
    Prediction time: End of first ICU stay

    Note: This predicts ICU readmission within the same hospitalization,
    NOT hospital readmission after discharge.
    """

    @property
    def name(self) -> str:
        return "task7_icu_readmission"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task7_icu_readmission",
            display_name="ICU Readmission",
            description="Predict whether patient will be readmitted to ICU during hospitalization",
            task_type=TaskType.BINARY_CLASSIFICATION,
            input_window_hours=None,  # Variable window (entire 1st ICU stay)
            prediction_window=None,
            cohort_filter="icu_24hr_complete",
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

    def filter_cohort(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Filter to patients with at least 24 hours first ICU stay.

        Args:
            cohort_df: Base cohort DataFrame
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
            f"Filtered to patients with >= 24hr first ICU stay: "
            f"{filtered.height} of {cohort_df.height}"
        )

        return filtered

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
        Build ICU readmission labels.

        Label = 1 if second_icu_start_time exists (patient returned to ICU)

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
        """Log statistics about the ICU readmission labels."""
        pos_count = labels.filter(pl.col("label_icu_readmission") == 1).height
        total = labels.height
        readmission_rate = (pos_count / total * 100) if total > 0 else 0

        logger.info(
            f"ICU readmission labels: "
            f"{pos_count} positive ({readmission_rate:.1f}%), "
            f"{total - pos_count} negative"
        )
