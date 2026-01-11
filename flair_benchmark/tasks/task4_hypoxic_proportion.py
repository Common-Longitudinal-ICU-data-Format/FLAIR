"""
Task 4: Hypoxic Failure Proportion Prediction

Regression task to predict the proportion of hours between 24-72
that a patient will experience hypoxic respiratory failure.

Uses first 24 hours of ICU data to predict outcomes in 24-72 hour window.
Only includes patients on IMV at the 24-hour mark.
Target is a continuous value between 0.0 and 1.0.
"""

import polars as pl
import logging
from typing import Dict, Any, Optional

from flair_benchmark.tasks.base import BaseTask, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class Task4HypoxicProportion(BaseTask):
    """
    Predict proportion of hypoxic failure hours (24-72hr window).

    Input: First 24 hours of ICU data
    Output: Continuous proportion (0.0 - 1.0)
    Cohort: Only patients on IMV at 24 hours

    Hypoxic failure is defined as SpO2 < 90% or PaO2/FiO2 < 200.
    The target is the fraction of hours in the 24-72hr window
    that meet this criterion.
    """

    @property
    def name(self) -> str:
        return "task4_hypoxic_proportion"

    def _get_default_config(self) -> TaskConfig:
        return TaskConfig(
            name="task4_hypoxic_proportion",
            display_name="Hypoxic Failure Proportion",
            description="Predict proportion of hypoxic hours in 24-72hr window",
            task_type=TaskType.REGRESSION,
            input_window_hours=24,
            prediction_window=(24, 72),  # Predict in 24-72hr window
            cohort_filter="imv_only",  # Only IMV patients at 24hr
            label_column="task4_proportion",
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
        Filter to patients on IMV at 24 hours.

        This task only includes patients who are on invasive mechanical
        ventilation at the 24-hour mark (same cohort as Task 3).
        """
        # Check for IMV indicator column
        if "imv_at_24hr" in cohort_df.columns:
            filtered = cohort_df.filter(pl.col("imv_at_24hr") == True)
            logger.info(
                f"Filtered to IMV patients: {filtered.height} of {cohort_df.height}"
            )
            return filtered

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
        Build regression labels for hypoxic proportion.

        The proportion is calculated as:
        (number of hours with hypoxic failure) / (total hours in 24-72hr window)

        Hypoxic failure criteria:
        - SpO2 < 90%
        - OR PaO2/FiO2 ratio < 200

        Args:
            cohort_df: Cohort DataFrame (should be pre-filtered to IMV patients)
            narratives_dir: Path to narratives for proportion calculation

        Returns:
            DataFrame with [hospitalization_id, task4_proportion]
        """
        # Check if labels are pre-computed
        if "task4_proportion" in cohort_df.columns:
            labels = cohort_df.select(["hospitalization_id", "task4_proportion"])
            self._log_proportion_stats(labels)
            return labels

        # Extract from narratives/wide dataset
        if narratives_dir:
            return self._extract_labels_from_narratives(cohort_df, narratives_dir)

        raise ValueError(
            "Cannot build labels: no task4_proportion column and no narratives_dir provided"
        )

    def _extract_labels_from_narratives(
        self,
        cohort_df: pl.DataFrame,
        narratives_dir: str,
    ) -> pl.DataFrame:
        """
        Extract hypoxic proportion from narrative sequences.

        Analyzes the 24-72 hour window for hypoxic tokens.
        """
        from pathlib import Path

        narratives_path = Path(narratives_dir)
        hosp_ids = cohort_df["hospitalization_id"].unique().to_list()

        # This is a simplified extraction
        # Real implementation would calculate hourly SpO2/PF ratios
        narratives = pl.scan_parquet(narratives_path / "*.parquet")

        # Count hypoxic indicators in 24-72hr window
        # Note: This is a placeholder - real implementation needs time filtering
        labels = (
            narratives.filter(pl.col("hospitalization_id").is_in(hosp_ids))
            .group_by("hospitalization_id")
            .agg(
                [
                    # Count low SpO2 tokens
                    pl.col("clif_sentence")
                    .filter(
                        pl.col("clif_sentence").str.contains("spo2")
                        & (
                            pl.col("clif_sentence").str.contains("_80_")
                            | pl.col("clif_sentence").str.contains("_85_")
                            | pl.col("clif_sentence").str.contains("_88_")
                        )
                    )
                    .count()
                    .alias("hypoxic_tokens"),
                    # Count all respiratory tokens
                    pl.col("clif_sentence")
                    .filter(pl.col("clif_sentence").str.contains("spo2"))
                    .count()
                    .alias("total_spo2_tokens"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("total_spo2_tokens") > 0)
                    .then(pl.col("hypoxic_tokens") / pl.col("total_spo2_tokens"))
                    .otherwise(pl.lit(0.0))
                    .clip(0.0, 1.0)
                    .alias("task4_proportion"),
                ]
            )
            .select(["hospitalization_id", "task4_proportion"])
            .collect()
        )

        # Add missing hospitalizations with 0 proportion
        all_hosp = cohort_df.select("hospitalization_id")
        labels = all_hosp.join(labels, on="hospitalization_id", how="left").with_columns(
            pl.col("task4_proportion").fill_null(0.0)
        )

        self._log_proportion_stats(labels)
        return labels

    def _log_proportion_stats(self, labels: pl.DataFrame) -> None:
        """Log statistics about the proportion values."""
        stats = labels.select(
            [
                pl.col("task4_proportion").mean().alias("mean"),
                pl.col("task4_proportion").std().alias("std"),
                pl.col("task4_proportion").min().alias("min"),
                pl.col("task4_proportion").max().alias("max"),
                pl.col("task4_proportion").median().alias("median"),
            ]
        ).row(0, named=True)

        logger.info(
            f"Task 4 proportion stats: "
            f"mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}, "
            f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
        )
