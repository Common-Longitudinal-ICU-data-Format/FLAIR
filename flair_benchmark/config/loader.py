"""
Configuration loader for FLAIR.

Loads and validates YAML configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from flair_benchmark.config.schema import FLAIRConfig, TaskConfig

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: Union[str, Path]) -> FLAIRConfig:
    """
    Load and validate FLAIR configuration.

    Args:
        path: Path to flair_config.yaml

    Returns:
        Validated FLAIRConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        pydantic.ValidationError: If config is invalid
    """
    config_dict = load_yaml(path)
    config = FLAIRConfig(**config_dict)

    logger.info(f"Loaded FLAIR config from {path}")
    logger.info(f"Site: {config.site.name}")
    logger.info(f"Enabled tasks: {config.tasks.enabled}")

    return config


def validate_config(config: FLAIRConfig) -> bool:
    """
    Validate configuration including path checks.

    Args:
        config: FLAIRConfig to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    errors = config.validate_paths()

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return True


def load_task_config(path: Union[str, Path]) -> TaskConfig:
    """
    Load and validate a task configuration file.

    Args:
        path: Path to task config YAML

    Returns:
        Validated TaskConfig object
    """
    config_dict = load_yaml(path)

    # Handle nested 'task' key if present
    if "task" in config_dict:
        task_dict = config_dict["task"]
    else:
        task_dict = config_dict

    # Merge in other sections if present
    for key in ["input", "cohort", "labels", "evaluation", "output"]:
        if key in config_dict:
            task_dict[key] = config_dict[key]

    # Flatten nested config
    flat_config = {}

    if "task" in config_dict:
        flat_config.update(config_dict["task"])

    if "input" in config_dict:
        flat_config["input_window_hours"] = config_dict["input"].get("window_hours", 24)

    if "cohort" in config_dict:
        flat_config["cohort_filter"] = config_dict["cohort"].get("filters")

    if "labels" in config_dict:
        flat_config["label_column"] = config_dict["labels"].get("column")
        flat_config["positive_class"] = config_dict["labels"].get("positive_class")

    if "evaluation" in config_dict:
        flat_config["evaluation_metrics"] = config_dict["evaluation"].get("metrics", [])

    return TaskConfig(**flat_config)


def create_default_config(output_path: Union[str, Path], site_name: str = "my_site") -> Path:
    """
    Create a default configuration file.

    Args:
        output_path: Path to write config file
        site_name: Site identifier

    Returns:
        Path to created file
    """
    output_path = Path(output_path)

    default_config = {
        "site": {
            "name": site_name,
            "description": "FLAIR benchmark site",
            "timezone": "US/Central",
        },
        "data": {
            "clif_config_path": "clif_config.json",
            "cohort_path": "OutputTokens/tokentables/cohort.parquet",
            "narratives_dir": "OutputTokens/narratives",
            "filetype": "parquet",
        },
        "output": {
            "base_dir": "flair_output",
            "datasets_dir": "flair_output/datasets",
            "results_dir": "flair_output/results",
            "submissions_dir": "flair_output/submissions",
        },
        "tasks": {
            "enabled": [
                "task1_discharged_home",
                "task2_discharged_ltach",
                "task3_outcome_72hr",
                "task4_hypoxic_proportion",
            ],
        },
        "privacy": {
            "enable_network_blocking": True,
            "enable_phi_detection": True,
            "min_cell_count": 10,
            "audit_log_path": "flair_output/audit.log",
        },
        "splits": {
            "method": "temporal",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "temporal_cutoff": "2023-01-01",
            "random_seed": 42,
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created default config at {output_path}")
    return output_path


def get_config_path(search_paths: Optional[list] = None) -> Optional[Path]:
    """
    Find FLAIR configuration file.

    Searches in order:
    1. Provided search_paths
    2. Current directory
    3. Parent directories

    Args:
        search_paths: Additional paths to search

    Returns:
        Path to config file if found, None otherwise
    """
    config_names = ["flair_config.yaml", "flair_config.yml", "flair.yaml"]

    # Build search order
    paths_to_search = []

    if search_paths:
        paths_to_search.extend([Path(p) for p in search_paths])

    # Current directory
    paths_to_search.append(Path.cwd())

    # Parent directories (up to 3 levels)
    current = Path.cwd()
    for _ in range(3):
        parent = current.parent
        if parent != current:
            paths_to_search.append(parent)
            current = parent

    # Search for config
    for search_dir in paths_to_search:
        for config_name in config_names:
            config_path = search_dir / config_name
            if config_path.exists():
                return config_path

    return None
