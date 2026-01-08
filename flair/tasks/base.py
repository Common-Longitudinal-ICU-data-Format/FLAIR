"""
Base task class for FLAIR benchmark.

All 7 tasks inherit from this base class. Each task has its own cohort
based on task-specific filters (all from base ICU hospitalizations).

Each task receives:
- cohort: 7 columns (hospitalization_id, admission_dttm, discharge_dttm,
          age_at_admission, sex_category, race_category, ethnicity_category)
- adt_data: ADT DataFrame to compute ICU timing

Each task outputs:
- 11 columns (hospitalization_id, admission_dttm, discharge_dttm,
             window_start, window_end, {label}, split,
             age_at_admission, sex_category, race_category, ethnicity_category)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import polars as pl
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Type of prediction task."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass"
    REGRESSION = "regression"


@dataclass
class TaskConfig:
    """Configuration for a FLAIR task."""

    name: str
    display_name: str
    description: str
    task_type: TaskType
    input_window_hours: int = 24
    prediction_window: Optional[Tuple[int, int]] = None
    cohort_filter: Optional[str] = None
    label_column: str = "label"
    positive_class: Optional[str] = None
    class_names: Optional[List[str]] = None
    evaluation_metrics: List[str] = None

    def __post_init__(self):
        if self.evaluation_metrics is None:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self.evaluation_metrics = [
                    "auroc",
                    "auprc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                ]
            elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                self.evaluation_metrics = [
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "per_class_f1",
                ]
            else:  # Regression
                self.evaluation_metrics = ["mse", "rmse", "mae", "r2"]


class BaseTask(ABC):
    """
    Abstract base class for FLAIR benchmark tasks.

    Each task receives a minimal 7-column cohort + ADT data, and outputs
    a standardized 11-column dataset with task-specific windows and labels.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize task.

        Args:
            config: Task-specific configuration overrides
        """
        self._config_overrides = config or {}
        self._task_config = self._get_default_config()

        # Apply overrides
        task_settings = self._config_overrides.get("tasks", {}).get(self.name, {})
        if task_settings:
            if "input_window_hours" in task_settings:
                self._task_config.input_window_hours = task_settings["input_window_hours"]
            if "prediction_window" in task_settings:
                self._task_config.prediction_window = tuple(task_settings["prediction_window"])

    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier."""
        pass

    @abstractmethod
    def _get_default_config(self) -> TaskConfig:
        """Get default task configuration."""
        pass

    @property
    def config(self) -> TaskConfig:
        """Get task configuration."""
        return self._task_config

    @property
    def task_type(self) -> TaskType:
        """Get task type."""
        return self._task_config.task_type

    def filter_cohort(
        self, cohort_df: pl.DataFrame, icu_timing: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Apply task-specific cohort filters.

        Args:
            cohort_df: Full cohort DataFrame (7 columns)
            icu_timing: ICU timing DataFrame from _compute_icu_timing

        Returns:
            Filtered cohort for this task
        """
        # Base implementation: no filtering
        # Override in subclasses for task-specific filters
        return cohort_df

    def _compute_icu_timing(self, adt: pl.DataFrame) -> pl.DataFrame:
        """
        Compute ICU timing and hospital info from ADT data.

        Uses rank-based approach: each ADT record with location_category="icu"
        is a distinct ICU stay, ranked by in_dttm within each hospitalization.

        Args:
            adt: ADT DataFrame with location data

        Returns:
            DataFrame with columns:
            - hospitalization_id
            - first_icu_start_time
            - first_icu_end_time
            - second_icu_start_time (null if no readmission)
            - hospital_id (from first ICU location)
            - hospital_type (from first ICU location)
        """
        # Filter to ICU locations
        icu_adt = adt.filter(pl.col("location_category").str.to_lowercase() == "icu")

        if icu_adt.height == 0:
            logger.warning("No ICU locations found in ADT data")
            return pl.DataFrame({
                "hospitalization_id": [],
                "first_icu_start_time": [],
                "first_icu_end_time": [],
                "second_icu_start_time": [],
                "hospital_id": [],
                "hospital_type": [],
            })

        # Rank ICU stays by in_dttm within each hospitalization
        icu_ranked = icu_adt.with_columns(
            pl.col("in_dttm")
            .rank("ordinal")
            .over("hospitalization_id")
            .alias("icu_stay_num")
        )

        # Build first ICU stay columns
        first_icu_cols = [
            "hospitalization_id",
            pl.col("in_dttm").alias("first_icu_start_time"),
            pl.col("out_dttm").alias("first_icu_end_time"),
        ]
        if "hospital_id" in icu_ranked.columns:
            first_icu_cols.append("hospital_id")
        if "hospital_type" in icu_ranked.columns:
            first_icu_cols.append("hospital_type")

        # Get first ICU stay (rank 1)
        first_icu = icu_ranked.filter(pl.col("icu_stay_num") == 1).select(first_icu_cols)

        # Get second ICU stay (rank 2) for readmission label
        second_icu = icu_ranked.filter(pl.col("icu_stay_num") == 2).select([
            "hospitalization_id",
            pl.col("in_dttm").alias("second_icu_start_time"),
        ])

        # Join first and second
        result = first_icu.join(second_icu, on="hospitalization_id", how="left")

        # Ensure hospital_id and hospital_type columns exist (fill with null if missing)
        if "hospital_id" not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias("hospital_id"))
        if "hospital_type" not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias("hospital_type"))

        logger.info(
            f"Computed ICU timing: {result.height} hospitalizations, "
            f"{second_icu.height} with readmissions"
        )

        return result

    @abstractmethod
    def build_labels(
        self,
        cohort_df: pl.DataFrame,
        icu_timing: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Build labels DataFrame for this task.

        Args:
            cohort_df: Cohort DataFrame (7 columns)
            icu_timing: ICU timing DataFrame from _compute_icu_timing

        Returns:
            DataFrame with columns: [hospitalization_id, {label_column}]
        """
        pass

    @abstractmethod
    def build_time_windows(
        self,
        cohort_df: pl.DataFrame,
        icu_timing: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Build time windows for data extraction.

        Must be implemented by each task to define its specific window.

        Args:
            cohort_df: Cohort DataFrame (7 columns)
            icu_timing: ICU timing DataFrame from _compute_icu_timing

        Returns:
            DataFrame with: hospitalization_id, window_start, window_end
        """
        pass

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

        Each task outputs a standardized 13-column parquet.

        For MIMIC (de-identified dates): Uses 70/30 temporal split by
        sorting admission_dttm. Date parameters are ignored.

        For other sites: Uses date-based split. Date parameters are required.

        Args:
            cohort_df: Base cohort DataFrame (7 columns)
            adt_data: ADT DataFrame for computing ICU timing
            train_start: Train period start date (YYYY-MM-DD). Optional for MIMIC.
            train_end: Train period end date (YYYY-MM-DD). Optional for MIMIC.
            test_start: Test period start date (YYYY-MM-DD). Optional for MIMIC.
            test_end: Test period end date (YYYY-MM-DD). Optional for MIMIC.
            site: Site name from config (e.g., "mimic"). Used to determine split method.

        Returns:
            DataFrame with 13 columns:
            - hospitalization_id, admission_dttm, discharge_dttm
            - window_start, window_end
            - {task_label}
            - split
            - age_at_admission, sex_category, race_category, ethnicity_category
            - hospital_id, hospital_type
        """
        # 1. Compute ICU timing from ADT data
        icu_timing = self._compute_icu_timing(adt_data)

        # 2. Filter cohort for this task
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

        # 7. Join all components (select only the 7 core columns, exclude task-specific like discharge_category)
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

    def _assign_temporal_split(
        self,
        task_cohort: pl.DataFrame,
        train_ratio: float = 0.7
    ) -> pl.DataFrame:
        """
        Assign train/test split by sorting admission_dttm temporally.

        Used for MIMIC (de-identified dates) where date-based splits
        are not meaningful. First 70% by admission time = train,
        last 30% = test.

        Args:
            task_cohort: Cohort DataFrame with admission_dttm column
            train_ratio: Fraction of data for training (default 0.7)

        Returns:
            DataFrame with [hospitalization_id, split]
        """
        # Sort by admission_dttm and add row index
        sorted_df = task_cohort.sort("admission_dttm").with_row_index("_row_num")
        total = sorted_df.height
        cutoff = int(total * train_ratio)

        return sorted_df.select([
            pl.col("hospitalization_id"),
            pl.when(pl.col("_row_num") < cutoff)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("split"),
        ])

    def get_evaluation_metrics(self) -> List[str]:
        """Get list of evaluation metrics for this task."""
        return self._task_config.evaluation_metrics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.task_type.value})"
