"""
Pydantic configuration schema for FLAIR.

Defines the structure and validation rules for FLAIR configuration files.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path


class SiteConfig(BaseModel):
    """Site-specific configuration."""

    name: str = Field(..., description="Site identifier (e.g., 'uchicago', 'mimic_clif')")
    description: str = Field(default="", description="Human-readable site description")
    timezone: str = Field(default="US/Central", description="Timezone for datetime operations")


class DataConfig(BaseModel):
    """Data paths configuration."""

    clif_config_path: str = Field(..., description="Path to clifpy configuration file")
    cohort_path: Optional[str] = Field(
        default=None,
        description="Path to cohort parquet file (use 'flair build-cohort' to generate)"
    )
    narratives_dir: str = Field(
        default="OutputTokens/narratives", description="Path to narratives directory"
    )
    filetype: str = Field(default="parquet", description="File type for CLIF data")


class CohortConfig(BaseModel):
    """Cohort building configuration."""

    output_path: str = Field(
        default="flair_output/cohort.parquet",
        description="Path to save built cohort"
    )
    min_age: int = Field(default=18, ge=0, le=120, description="Minimum patient age")
    min_los_days: float = Field(
        default=0, ge=0, description="Minimum length of stay in days"
    )
    skip_time_filter: bool = Field(
        default=True, description="Skip time period filter (for MIMIC compatibility)"
    )


class OutputConfig(BaseModel):
    """Output paths configuration."""

    base_dir: str = Field(default="flair_output", description="Base output directory")
    datasets_dir: str = Field(default="flair_output/datasets", description="Datasets directory")
    results_dir: str = Field(default="flair_output/results", description="Results directory")
    submissions_dir: str = Field(
        default="flair_output/submissions", description="Submissions directory"
    )


class TaskSettings(BaseModel):
    """Settings for individual task."""

    input_window_hours: int = Field(default=24, description="Hours of input data to use")
    min_sequence_length: int = Field(default=10, description="Minimum tokens required")
    max_sequence_length: int = Field(default=8192, description="Maximum tokens to include")
    prediction_window: Optional[List[int]] = Field(
        default=None, description="Prediction window [start_hour, end_hour]"
    )
    cohort_filter: Optional[str] = Field(
        default=None, description="Cohort filter (e.g., 'imv_only')"
    )


class TasksConfig(BaseModel):
    """Tasks configuration."""

    enabled: List[str] = Field(
        default=[
            "task1_discharged_home",
            "task2_discharged_ltach",
            "task3_outcome_72hr",
            "task4_hypoxic_proportion",
        ],
        description="List of enabled tasks",
    )
    task1_discharged_home: TaskSettings = Field(default_factory=TaskSettings)
    task2_discharged_ltach: TaskSettings = Field(default_factory=TaskSettings)
    task3_outcome_72hr: TaskSettings = Field(
        default_factory=lambda: TaskSettings(
            prediction_window=[24, 72], cohort_filter="imv_only"
        )
    )
    task4_hypoxic_proportion: TaskSettings = Field(
        default_factory=lambda: TaskSettings(
            prediction_window=[24, 72], cohort_filter="imv_only"
        )
    )

    @field_validator("enabled")
    @classmethod
    def validate_enabled_tasks(cls, v: List[str]) -> List[str]:
        valid_tasks = {
            "task1_discharged_home",
            "task2_discharged_ltach",
            "task3_outcome_72hr",
            "task4_hypoxic_proportion",
        }
        for task in v:
            if task not in valid_tasks:
                raise ValueError(f"Unknown task: {task}. Valid tasks: {valid_tasks}")
        return v


class WideDatasetCategoryFilters(BaseModel):
    """Category filters for wide dataset generation."""

    vitals: List[str] = Field(
        default=[
            "heart_rate",
            "sbp",
            "dbp",
            "spo2",
            "temp_c",
            "respiratory_rate",
            "map",
        ]
    )
    labs: List[str] = Field(
        default=[
            "hemoglobin",
            "wbc",
            "sodium",
            "potassium",
            "creatinine",
            "glucose_serum",
            "lactate",
            "platelet_count",
            "bilirubin_total",
        ]
    )
    respiratory_support: List[str] = Field(
        default=[
            "device_category",
            "mode_category",
            "fio2_set",
            "peep_set",
            "tidal_volume_set",
        ]
    )
    patient_assessments: List[str] = Field(default=["gcs_total", "rass"])


class WideDatasetAggregation(BaseModel):
    """Aggregation settings for wide dataset."""

    hourly_window: int = Field(default=1, ge=1, le=72, description="Aggregation window in hours")
    fill_gaps: bool = Field(default=False, description="Whether to fill gaps in time series")


class WideDatasetConfig(BaseModel):
    """Wide dataset configuration (passed to clifpy)."""

    category_filters: WideDatasetCategoryFilters = Field(
        default_factory=WideDatasetCategoryFilters
    )
    aggregation: WideDatasetAggregation = Field(default_factory=WideDatasetAggregation)


class PrivacyConfig(BaseModel):
    """Privacy enforcement configuration."""

    enable_network_blocking: bool = Field(
        default=True, description="Block all network access at Python socket level"
    )
    enable_phi_detection: bool = Field(
        default=True, description="Scan outputs for PHI patterns"
    )
    min_cell_count: int = Field(
        default=10,
        ge=1,
        description="Minimum cell count (suppress smaller for HIPAA compliance)",
    )
    audit_log_path: str = Field(
        default="flair_output/audit.log", description="Path to audit log file"
    )
    banned_packages: List[str] = Field(
        default=[
            "requests",
            "urllib3",
            "httpx",
            "aiohttp",
            "socket",
            "paramiko",
            "ftplib",
            "smtplib",
            "telnetlib",
            "websocket",
        ],
        description="Packages banned in submitted methods",
    )


class SplitsConfig(BaseModel):
    """Train/validation/test split configuration."""

    method: Literal["temporal", "random"] = Field(
        default="temporal", description="Split method"
    )
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.0, le=0.5)
    test_ratio: float = Field(default=0.15, ge=0.1, le=0.5)
    temporal_cutoff: Optional[str] = Field(
        default="2023-01-01", description="Cutoff date for temporal splits"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    @field_validator("test_ratio")
    @classmethod
    def validate_ratios_sum(cls, v: float, info) -> float:
        train = info.data.get("train_ratio", 0.7)
        val = info.data.get("val_ratio", 0.15)
        total = train + val + v
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        return v


class FLAIRConfig(BaseModel):
    """Main FLAIR configuration."""

    site: SiteConfig
    data: DataConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    cohort: CohortConfig = Field(default_factory=CohortConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    wide_dataset: WideDatasetConfig = Field(default_factory=WideDatasetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)

    def get_task_settings(self, task_name: str) -> TaskSettings:
        """Get settings for a specific task."""
        if not hasattr(self.tasks, task_name):
            raise ValueError(f"Unknown task: {task_name}")
        return getattr(self.tasks, task_name)

    def get_cohort_path(self) -> str:
        """Get cohort path (from data config or cohort config)."""
        if self.data.cohort_path:
            return self.data.cohort_path
        return self.cohort.output_path

    def validate_paths(self) -> List[str]:
        """Validate that required paths exist."""
        errors = []

        if not Path(self.data.clif_config_path).exists():
            errors.append(f"CLIF config not found: {self.data.clif_config_path}")

        # Cohort path is optional - can be built with 'flair build-cohort'
        cohort_path = self.get_cohort_path()
        if cohort_path and not Path(cohort_path).exists():
            errors.append(
                f"Cohort file not found: {cohort_path}. "
                "Run 'flair build-cohort' to generate it."
            )

        return errors


class TaskConfig(BaseModel):
    """Configuration for a single task (used in task YAML files)."""

    name: str = Field(..., description="Task identifier")
    display_name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    task_type: Literal["binary_classification", "multiclass", "regression"] = Field(
        ..., description="Type of prediction task"
    )
    input_window_hours: int = Field(default=24, description="Hours of input data")
    prediction_window: Optional[List[int]] = Field(
        default=None, description="Prediction window [start, end]"
    )
    cohort_filter: Optional[str] = Field(default=None, description="Cohort filter")
    label_column: str = Field(..., description="Name of label column")
    positive_class: Optional[str] = Field(
        default=None, description="Positive class name (for binary)"
    )
    class_names: Optional[List[str]] = Field(
        default=None, description="Class names (for multiclass)"
    )
    evaluation_metrics: List[str] = Field(
        default=["auroc", "auprc", "accuracy"],
        description="Metrics to compute",
    )
