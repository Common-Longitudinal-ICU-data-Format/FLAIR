"""
Task 7: ICU Readmission Prediction

Binary classification task to predict whether a patient will be
readmitted to the ICU during the same hospitalization.

Uses first 24 hours of ICU data to predict ICU readmission.
Only includes patients with at least 24 hours first ICU stay.
"""

import polars as pl
import logging
from typing import Optional

from flair.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task7ICUReadmission(BaseTask):
    """
    Predict ICU readmission (return to ICU within same hospitalization).

    Input: First 24 hours of ICU data
    Output: Binary (1 = readmitted to ICU, 0 = not readmitted)
    Cohort: ICU patients with >= 24 hours first ICU stay

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
            input_window_hours=24,
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

    def filter_cohort(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter to patients with at least 24 hours first ICU stay.

        Requires first_icu_24hr_completion_time to be not null.
        """
        if "first_icu_24hr_completion_time" in cohort_df.columns:
            filtered = cohort_df.filter(
                pl.col("first_icu_24hr_completion_time").is_not_null()
            )
            logger.info(
                f"Filtered to patients with >= 24hr first ICU stay: "
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
                f"Filtered to patients with >= 24hr first ICU stay: "
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
        Build ICU readmission labels.

        Label = 1 if patient returned to ICU (second_icu_start_time exists)

        The cohort builder already computes second_icu_start_time for patients
        who had multiple ICU stays within the same hospitalization.

        Args:
            cohort_df: Cohort DataFrame with second_icu_start_time column
            narratives_dir: Not used for this task

        Returns:
            DataFrame with [hospitalization_id, label_icu_readmission]
        """
        # Check if labels are pre-computed
        if "label_icu_readmission" in cohort_df.columns:
            labels = cohort_df.select(["hospitalization_id", "label_icu_readmission"])
            self._log_readmission_stats(labels)
            return labels

        # Extract from second_icu_start_time
        if "second_icu_start_time" in cohort_df.columns:
            labels = cohort_df.select(
                [
                    pl.col("hospitalization_id"),
                    pl.col("second_icu_start_time")
                    .is_not_null()
                    .cast(pl.Int32)
                    .alias("label_icu_readmission"),
                ]
            )
            self._log_readmission_stats(labels)
            return labels

        raise ValueError(
            "Cannot build labels: cohort has no second_icu_start_time column. "
            "Ensure cohort builder calculates ICU readmission timing."
        )

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
