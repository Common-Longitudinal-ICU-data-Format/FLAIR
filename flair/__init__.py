"""
FLAIR: Federated Learning Assessment for ICU Research

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction
tasks using the CLIF (Common Longitudinal ICU Format) data standard.

Key Features:
- Privacy-first: Network requests blocked, PHI detection, review process
- CLIF-native: Uses clifpy's create_wide_dataset() for feature generation
- Federated: Methods developed on MIMIC-CLIF, evaluated at 17+ sites
- 7 Tasks: Discharged home, LTACH, 72hr outcome, hypoxic proportion,
           ICU LOS, hospital mortality, ICU readmission

Usage:
    from flair import generate_task_dataset, TASK_REGISTRY

    # Generate dataset for ICU LOS task with temporal split
    df = generate_task_dataset(
        config_path="clif_config.json",
        task_name="task5_icu_los",
        train_start="2020-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2023-12-31",
    )
"""

import polars as pl

__version__ = "1.0.0"
__author__ = "CLIF Consortium"
__email__ = "clif_consortium@uchicago.edu"

# Core library exports
from flair.tasks import get_task, TASK_REGISTRY, BaseTask, TaskConfig, TaskType
from flair.cohort.builder import FLAIRCohortBuilder
from flair.datasets.builder import FLAIRDatasetBuilder

# Helper exports
from flair.helpers.table1 import get_table1
from flair.helpers.tripod_ai import generate_tripod_ai_report


def generate_task_dataset(
    config_path: str,
    task_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    output_path: str,
) -> pl.DataFrame:
    """
    Generate a complete task-specific dataset with temporal split.

    Each task has its own cohort size (N) based on task-specific filters.
    All tasks share the same base: hospitalizations with at least 1 ICU stay.

    Args:
        config_path: Path to CLIF config JSON file
        task_name: One of TASK_REGISTRY.keys() (e.g., "task5_icu_los")
        train_start: Train period start date (YYYY-MM-DD)
        train_end: Train period end date (YYYY-MM-DD)
        test_start: Test period start date (YYYY-MM-DD)
        test_end: Test period end date (YYYY-MM-DD)
        output_path: Path to save parquet file

    Returns:
        DataFrame with columns:
        - hospitalization_id, admission_dttm, discharge_dttm
        - window_start, window_end
        - task-specific label column
        - split (train/test)
        - age_at_admission, sex_category, race_category, ethnicity_category

    Example:
        >>> df = generate_task_dataset(
        ...     config_path="clif_config.json",
        ...     task_name="task5_icu_los",
        ...     train_start="2020-01-01",
        ...     train_end="2022-12-31",
        ...     test_start="2023-01-01",
        ...     test_end="2023-12-31",
        ... )
        >>> print(f"N={len(df)}, Train={len(df.filter(pl.col('split')=='train'))}")
    """
    # Build base cohort (all ICU hospitalizations)
    cohort_builder = FLAIRCohortBuilder(config_path)
    cohort = cohort_builder.build_cohort()

    # Get task instance
    task = get_task(task_name)

    # Build task-specific dataset with temporal split
    dataset = task.build_task_dataset(
        cohort,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    # Save to parquet
    dataset.write_parquet(output_path)

    return dataset


__all__ = [
    "__version__",
    # Main API
    "generate_task_dataset",
    # Core components
    "get_task",
    "TASK_REGISTRY",
    "BaseTask",
    "TaskConfig",
    "TaskType",
    "FLAIRCohortBuilder",
    "FLAIRDatasetBuilder",
    # Helpers
    "get_table1",
    "generate_tripod_ai_report",
]
