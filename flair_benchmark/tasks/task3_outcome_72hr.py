"""
Task 3: 72-Hour Respiratory Outcome Prediction

Multi-class classification task to predict the respiratory status
at 72 hours post-ICU admission for patients on invasive mechanical
ventilation (IMV).

Classes:
- imv_on: Still on mechanical ventilation at 72 hours
- imv_off: Successfully weaned from ventilation
- expired: Deceased by 72 hours

Uses first 24 hours of ICU data to predict 24-72 hour window outcomes.
Only includes patients on IMV at the 24-hour mark.
"""

import polars as pl
import logging
from typing import Dict, Any, Optional

from flair_benchmark.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task3Outcome72hr(BaseTask):
    """
    Predict respiratory outcome at 72 hours for IMV patients.

    Input: First 24 hours of ICU data
    Output: Multi-class (imv_on=0, imv_off=1, expired=2)
    Cohort: Only patients on IMV at 24 hours

    This task evaluates the ability to predict ventilator weaning
    success or mortality within the 24-72 hour window.
    """

    @property
    def name(self) -> str:
        return "task3_outcome_72hr"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task3_outcome_72hr",
            display_name="72-Hour Respiratory Outcome",
            description="Predict respiratory status at 72 hours for IMV patients",
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            input_window_hours=24,
            prediction_window=(24, 72),  # Predict outcomes in 24-72hr window
            cohort_filter="imv_only",  # Only IMV patients at 24hr
            label_column="task3_label",
            class_names=["imv_on", "imv_off", "expired"],
            evaluation_metrics=[
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "per_class_precision",
                "per_class_recall",
                "per_class_f1",
                "confusion_matrix",
            ],
        )

    def filter_cohort(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter to patients on IMV at 24 hours.

        This task only includes patients who are on invasive mechanical
        ventilation at the 24-hour mark.
        """
        # Check for IMV indicator column
        if "imv_at_24hr" in cohort_df.columns:
            filtered = cohort_df.filter(pl.col("imv_at_24hr") == True)
            logger.info(
                f"Filtered to IMV patients: {filtered.height} of {cohort_df.height}"
            )
            return filtered

        # If not available, return all (will need to filter during label extraction)
        logger.warning(
            "No imv_at_24hr column found - cohort filtering will happen during label extraction"
        )
        return cohort_df

    def build_labels(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Build multi-class labels for respiratory outcome.

        Labels are extracted from the 24-72 hour window:
        - 0 (imv_on): Patient still on IMV at 72 hours
        - 1 (imv_off): Patient weaned from IMV by 72 hours
        - 2 (expired): Patient deceased by 72 hours

        Args:
            cohort_df: Cohort DataFrame (should be pre-filtered to IMV patients)
            narratives_dir: Path to narratives for outcome extraction

        Returns:
            DataFrame with [hospitalization_id, task3_label]
        """
        # Check if labels are pre-computed
        if "task3_label" in cohort_df.columns:
            labels = cohort_df.select(["hospitalization_id", "task3_label"])
            self._log_class_distribution(labels)
            return labels

        # Extract from narratives
        if narratives_dir:
            return self._extract_labels_from_narratives(cohort_df, narratives_dir)

        raise ValueError(
            "Cannot build labels: no task3_label column and no narratives_dir provided"
        )

    def _extract_labels_from_narratives(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: str,
    ) -> pl.DataFrame:
        """
        Extract respiratory outcome labels from narrative sequences.

        Analyzes the 24-72 hour window for:
        - Device tokens (resp_device_imv -> still on IMV)
        - Disposition tokens (disposition_expired -> deceased)
        - Absence of IMV tokens (-> weaned)
        """
        from pathlib import Path

        narratives_path = Path(narratives_dir)
        hosp_ids = cohort_df["hospitalization_id"].unique().to_list()

        # This is a simplified extraction - real implementation would
        # analyze the 24-72hr time window more carefully
        narratives = pl.scan_parquet(narratives_path / "*.parquet")

        # Get events in 24-72hr window
        # Note: This requires event_time column in narratives
        labels = (
            narratives.filter(pl.col("hospitalization_id").is_in(hosp_ids))
            .group_by("hospitalization_id")
            .agg(
                [
                    pl.col("clif_sentence")
                    .filter(pl.col("clif_sentence") == "disposition_expired")
                    .count()
                    .alias("expired_count"),
                    pl.col("clif_sentence")
                    .filter(pl.col("clif_sentence").str.contains("resp_device_imv"))
                    .count()
                    .alias("imv_count"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("expired_count") > 0)
                    .then(pl.lit(2))  # expired
                    .when(pl.col("imv_count") > 0)
                    .then(pl.lit(0))  # imv_on
                    .otherwise(pl.lit(1))  # imv_off
                    .alias("task3_label"),
                ]
            )
            .select(["hospitalization_id", "task3_label"])
            .collect()
        )

        self._log_class_distribution(labels)
        return labels

    def _log_class_distribution(self, labels: pl.DataFrame) -> None:
        """Log the class distribution for this task."""
        class_counts = labels.group_by("task3_label").count().sort("task3_label")
        logger.info("Task 3 label distribution:")
        for row in class_counts.iter_rows(named=True):
            class_name = self._get_default_config().class_names[row["task3_label"]]
            logger.info(f"  {class_name} ({row['task3_label']}): {row['count']}")
