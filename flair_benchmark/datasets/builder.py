"""
Dataset builder for FLAIR benchmark.

Uses clifpy's create_wide_dataset() for feature generation.
All tasks share the same ICU cohort (built with 'flair build-cohort').
"""

import polars as pl
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from flair_benchmark.tasks import get_task, BaseTask
from flair_benchmark.privacy.network_guard import NetworkBlocker
from flair_benchmark.privacy.phi_detector import PHIDetector, PHIViolationError
from flair_benchmark.privacy.audit_log import log_operation

logger = logging.getLogger(__name__)


class FLAIRDatasetBuilder:
    """
    Build FLAIR benchmark datasets using clifpy.

    This class generates wide-format datasets for each task using
    clifpy's create_wide_dataset() function. All tasks share the
    same underlying cohort from tokenETL.

    Usage:
        builder = FLAIRDatasetBuilder(config)
        builder.build_all_tasks()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset builder.

        Args:
            config: FLAIR configuration dictionary or FLAIRConfig object
        """
        self.config = config
        self.phi_detector = PHIDetector()
        self._clif = None
        self._cohort = None

        # Ensure network is blocked
        if config.get("privacy", {}).get("enable_network_blocking", True):
            NetworkBlocker.block()

    @property
    def clif(self):
        """Lazy load CLIF instance."""
        if self._clif is None:
            self._load_clif()
        return self._clif

    @property
    def cohort(self) -> pl.DataFrame:
        """Lazy load cohort."""
        if self._cohort is None:
            self._load_cohort()
        return self._cohort

    def _load_clif(self) -> None:
        """Load CLIF instance from configuration."""
        try:
            from clifpy import CLIF

            clif_config_path = self.config["data"]["clif_config_path"]
            self._clif = CLIF(clif_config_path)
            self._clif.load_data()
            logger.info(f"Loaded CLIF data from {clif_config_path}")
        except ImportError:
            logger.warning("clifpy not installed - using mock CLIF for testing")
            self._clif = None
        except Exception as e:
            logger.error(f"Failed to load CLIF: {e}")
            raise

    def _load_cohort(self) -> None:
        """Load cohort from FLAIR or external source.

        Checks for cohort at:
        1. data.cohort_path (if specified in config)
        2. cohort.output_path (FLAIR's built cohort)
        """
        # Try data.cohort_path first (external cohort)
        cohort_path = self.config.get("data", {}).get("cohort_path")

        # Fall back to cohort.output_path (FLAIR-built cohort)
        if not cohort_path:
            cohort_path = self.config.get("cohort", {}).get(
                "output_path", "flair_output/cohort.parquet"
            )

        if not Path(cohort_path).exists():
            raise FileNotFoundError(
                f"Cohort file not found: {cohort_path}\n"
                "Run 'flair build-cohort' to generate the cohort from CLIF data."
            )

        self._cohort = pl.read_parquet(cohort_path)
        logger.info(f"Loaded cohort: {self._cohort.height} hospitalizations from {cohort_path}")

    def build_all_tasks(self) -> Dict[str, Path]:
        """
        Build datasets for all enabled tasks.

        Returns:
            Dictionary mapping task names to output directories
        """
        output_paths = {}
        enabled_tasks = self.config.get("tasks", {}).get("enabled", [])

        for task_name in enabled_tasks:
            try:
                output_path = self.build_task(task_name)
                output_paths[task_name] = output_path
                log_operation(
                    "build_task_dataset",
                    "success",
                    {"task": task_name, "output_path": str(output_path)},
                )
            except Exception as e:
                logger.error(f"Failed to build {task_name}: {e}")
                log_operation(
                    "build_task_dataset",
                    "failure",
                    {"task": task_name, "error": str(e)},
                )
                raise

        return output_paths

    def build_task(self, task_name: str) -> Path:
        """
        Build dataset for a single task.

        Args:
            task_name: Name of the task to build

        Returns:
            Path to output directory
        """
        logger.info(f"Building dataset for {task_name}")

        # Get task instance
        task = get_task(task_name, self.config)

        # Filter cohort for this task
        task_cohort = task.filter_cohort(self.cohort)
        hosp_ids = task_cohort["hospitalization_id"].unique().to_list()

        logger.info(f"Task cohort: {len(hosp_ids)} hospitalizations")

        # Build wide dataset using clifpy
        wide_df = self._build_wide_dataset(task, hosp_ids)

        # Build labels
        narratives_dir = self.config.get("data", {}).get("narratives_dir")
        labels = task.build_labels(task_cohort, narratives_dir)

        # Build demographics
        demographics = task.build_demographics(task_cohort)

        # Validate no PHI in outputs
        self._validate_no_phi(wide_df, "wide_dataset")
        self._validate_no_phi(labels, "labels")
        self._validate_no_phi(demographics, "demographics")

        # Save outputs
        output_dir = self._get_output_dir(task_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to pandas for parquet writing if needed
        if isinstance(wide_df, pd.DataFrame):
            wide_df.to_parquet(output_dir / "wide_dataset.parquet", index=False)
        else:
            wide_df.write_parquet(output_dir / "wide_dataset.parquet")

        labels.write_parquet(output_dir / "labels.parquet")
        demographics.write_parquet(output_dir / "demographics.parquet")

        logger.info(f"Saved {task_name} datasets to {output_dir}")

        return output_dir

    def _build_wide_dataset(
        self,
        task: BaseTask,
        hospitalization_ids: List[str],
    ) -> pd.DataFrame:
        """
        Build wide dataset using clifpy.

        Args:
            task: Task instance
            hospitalization_ids: List of hospitalization IDs to include

        Returns:
            Wide-format DataFrame
        """
        if self._clif is None:
            # Return mock data for testing without clifpy
            return self._build_mock_wide_dataset(hospitalization_ids)

        try:
            from clifpy.utils.wide_dataset import create_wide_dataset

            # Create time-windowed cohort for clifpy
            cohort_df = task.create_time_windowed_cohort(self.cohort)

            # Get category filters from config
            category_filters = self.config.get("wide_dataset", {}).get(
                "category_filters",
                {
                    "vitals": ["heart_rate", "sbp", "dbp", "spo2", "temp_c", "respiratory_rate"],
                    "labs": ["hemoglobin", "wbc", "sodium", "potassium", "creatinine", "glucose_serum"],
                    "respiratory_support": ["device_category", "fio2_set", "peep_set"],
                    "patient_assessments": ["gcs_total", "rass"],
                },
            )

            # Call clifpy create_wide_dataset
            wide_df = create_wide_dataset(
                self._clif,
                category_filters=category_filters,
                cohort_df=cohort_df,
                hospitalization_ids=hospitalization_ids,
                output_format="dataframe",
                show_progress=True,
            )

            logger.info(f"Created wide dataset: {len(wide_df)} rows, {len(wide_df.columns)} columns")

            return wide_df

        except Exception as e:
            logger.error(f"Failed to create wide dataset: {e}")
            raise

    def _build_mock_wide_dataset(self, hospitalization_ids: List[str]) -> pd.DataFrame:
        """
        Build mock wide dataset for testing.

        Returns a simple DataFrame with hospitalization_id and placeholder features.
        """
        import numpy as np

        logger.warning("Building mock wide dataset (clifpy not available)")

        n = len(hospitalization_ids)
        return pd.DataFrame(
            {
                "hospitalization_id": hospitalization_ids,
                "heart_rate_mean": np.random.normal(80, 15, n),
                "sbp_mean": np.random.normal(120, 20, n),
                "dbp_mean": np.random.normal(80, 10, n),
                "spo2_mean": np.random.normal(96, 3, n),
                "temp_c_mean": np.random.normal(37, 0.5, n),
                "respiratory_rate_mean": np.random.normal(18, 4, n),
            }
        )

    def _validate_no_phi(self, df: Any, name: str) -> None:
        """
        Validate DataFrame contains no PHI.

        Args:
            df: DataFrame to validate
            name: Name for error messages

        Raises:
            PHIViolationError: If PHI is detected
        """
        if not self.config.get("privacy", {}).get("enable_phi_detection", True):
            return

        violations = self.phi_detector.scan_dataframe(df)

        if violations:
            high_severity = [v for v in violations if v.severity == "high"]
            if high_severity:
                log_operation(
                    "phi_detection",
                    "blocked",
                    {"dataset": name, "violations": [v.to_dict() for v in high_severity]},
                )
                raise PHIViolationError(high_severity)
            else:
                logger.warning(
                    f"Low/medium severity PHI patterns in {name}: "
                    f"{[v.pattern_name for v in violations]}"
                )

    def _get_output_dir(self, task_name: str) -> Path:
        """Get output directory for a task."""
        datasets_dir = self.config.get("output", {}).get("datasets_dir", "flair_output/datasets")
        return Path(datasets_dir) / task_name


def build_all_datasets(config_path: str) -> Dict[str, Path]:
    """
    Build all FLAIR datasets from configuration file.

    Args:
        config_path: Path to flair_config.yaml

    Returns:
        Dictionary mapping task names to output directories
    """
    from flair_benchmark.config.loader import load_config

    config = load_config(config_path)
    builder = FLAIRDatasetBuilder(config.model_dump())
    return builder.build_all_tasks()
