"""
Task 2: Discharged to LTACH Prediction

Binary classification task to predict whether a patient will be
discharged to Long-Term Acute Care Hospital (LTACH).

Uses first 24 hours of ICU data to predict disposition at discharge.
Prevalence: ~3.6% positive (highly imbalanced).
"""

import polars as pl
import logging
from typing import Dict, Any, Optional

from flair_benchmark.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task2DischargedLTACH(BaseTask):
    """
    Predict whether patient will be discharged to LTACH.

    Input: First 24 hours of ICU data
    Output: Binary (1 = discharged to LTACH, 0 = other disposition)
    Prevalence: ~3.6% positive (LTACH) - highly imbalanced

    Note: This is an imbalanced classification task. Consider using
    AUPRC as the primary metric rather than AUROC.
    """

    @property
    def name(self) -> str:
        return "task2_discharged_ltach"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task2_discharged_ltach",
            display_name="Discharged to LTACH",
            description="Predict whether patient will be discharged to Long-Term Acute Care",
            task_type=TaskType.BINARY_CLASSIFICATION,
            input_window_hours=24,
            prediction_window=None,  # Predicts at discharge
            cohort_filter=None,  # All ICU patients
            label_column="label_ltach",
            positive_class="disposition_ltach",
            evaluation_metrics=[
                "auroc",
                "auprc",  # Primary metric for imbalanced data
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "npv",
            ],
        )

    def build_labels(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Build labels from disposition data.

        The disposition is extracted from the cohort or narratives.
        disposition_ltach = 1, all other dispositions = 0.

        Args:
            cohort_df: Cohort DataFrame
            narratives_dir: Path to narratives (for token-based extraction)

        Returns:
            DataFrame with [hospitalization_id, label_ltach]
        """
        # Check if disposition is already in cohort
        if "disposition_category" in cohort_df.columns:
            labels = cohort_df.select(
                [
                    pl.col("hospitalization_id"),
                    (
                        pl.col("disposition_category").str.to_lowercase().str.contains("ltach")
                        | pl.col("disposition_category")
                        .str.to_lowercase()
                        .str.contains("long.?term")
                    )
                    .cast(pl.Int32)
                    .alias("label_ltach"),
                ]
            )
            logger.info(
                f"Built labels from disposition_category: "
                f"{labels.filter(pl.col('label_ltach') == 1).height} positive, "
                f"{labels.filter(pl.col('label_ltach') == 0).height} negative"
            )
            return labels

        # If narratives provided, extract from token sequences
        if narratives_dir:
            return self._extract_labels_from_narratives(cohort_df, narratives_dir)

        raise ValueError(
            "Cannot build labels: cohort has no disposition_category and no narratives_dir provided"
        )

    def _extract_labels_from_narratives(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: str,
    ) -> pl.DataFrame:
        """
        Extract disposition labels from narrative token sequences.

        Looks for 'disposition_ltach' or similar tokens in the sequence.
        """
        from pathlib import Path

        narratives_path = Path(narratives_dir)

        # Load narratives
        narratives = pl.scan_parquet(narratives_path / "*.parquet")

        # Filter to cohort hospitalization IDs
        hosp_ids = cohort_df["hospitalization_id"].unique().to_list()

        labels = (
            narratives.filter(pl.col("hospitalization_id").is_in(hosp_ids))
            .filter(pl.col("clif_sentence").str.starts_with("disposition_"))
            .group_by("hospitalization_id")
            .agg(
                [
                    pl.col("clif_sentence")
                    .filter(
                        pl.col("clif_sentence").str.contains("ltach")
                        | pl.col("clif_sentence").str.contains("long_term")
                    )
                    .count()
                    .alias("ltach_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("ltach_count") > 0).cast(pl.Int32).alias("label_ltach"),
                ]
            )
            .select(["hospitalization_id", "label_ltach"])
            .collect()
        )

        # Add missing hospitalizations with 0 label
        all_hosp = cohort_df.select("hospitalization_id")
        labels = all_hosp.join(labels, on="hospitalization_id", how="left").with_columns(
            pl.col("label_ltach").fill_null(0)
        )

        logger.info(
            f"Extracted labels from narratives: "
            f"{labels.filter(pl.col('label_ltach') == 1).height} positive (LTACH)"
        )

        return labels
