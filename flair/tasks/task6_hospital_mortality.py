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
from typing import Optional

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

    def filter_cohort(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter to patients with at least 24 hours ICU stay.

        Requires first_icu_24hr_completion_time to be not null.
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
        Build mortality labels from discharge_category.

        Label = 1 if discharge_category == "expired" (case insensitive), else 0

        Args:
            cohort_df: Cohort DataFrame with discharge_category column
            narratives_dir: Path to narratives (for token-based extraction)

        Returns:
            DataFrame with [hospitalization_id, label_mortality]
        """
        # Check if labels are pre-computed
        if "label_mortality" in cohort_df.columns:
            labels = cohort_df.select(["hospitalization_id", "label_mortality"])
            self._log_mortality_stats(labels)
            return labels

        # Extract from discharge_category
        if "discharge_category" in cohort_df.columns:
            labels = cohort_df.select(
                [
                    pl.col("hospitalization_id"),
                    (pl.col("discharge_category").str.to_lowercase() == "expired")
                    .cast(pl.Int32)
                    .alias("label_mortality"),
                ]
            )
            self._log_mortality_stats(labels)
            return labels

        # Try narratives if available
        if narratives_dir:
            return self._extract_labels_from_narratives(cohort_df, narratives_dir)

        raise ValueError(
            "Cannot build labels: cohort has no discharge_category and no narratives_dir provided"
        )

    def _extract_labels_from_narratives(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: str,
    ) -> pl.DataFrame:
        """
        Extract mortality labels from narrative token sequences.

        Looks for 'disposition_expired' token in the sequence.
        """
        from pathlib import Path

        narratives_path = Path(narratives_dir)
        hosp_ids = cohort_df["hospitalization_id"].unique().to_list()

        # Load narratives and check for disposition tokens
        narratives = pl.scan_parquet(narratives_path / "*.parquet")

        labels = (
            narratives.filter(pl.col("hospitalization_id").is_in(hosp_ids))
            .filter(pl.col("clif_sentence").str.starts_with("disposition_"))
            .group_by("hospitalization_id")
            .agg(
                [
                    pl.col("clif_sentence")
                    .filter(pl.col("clif_sentence") == "disposition_expired")
                    .count()
                    .alias("expired_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("expired_count") > 0).cast(pl.Int32).alias("label_mortality"),
                ]
            )
            .select(["hospitalization_id", "label_mortality"])
            .collect()
        )

        # Add missing hospitalizations with 0 label
        all_hosp = cohort_df.select("hospitalization_id")
        labels = all_hosp.join(labels, on="hospitalization_id", how="left").with_columns(
            pl.col("label_mortality").fill_null(0)
        )

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
