"""
TRIPOD-AI compliant reporting for FLAIR benchmark.

Generates standardized reporting following the TRIPOD-AI checklist
for transparent reporting of prediction model studies.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TRIPODAIReport:
    """
    TRIPOD-AI compliant reporting structure.

    Based on the Transparent Reporting of a Multivariable Prediction Model
    for Individual Prognosis or Diagnosis - Artificial Intelligence checklist.
    """

    # Title and Abstract (Items 1-2)
    title: str
    abstract_summary: str

    # Introduction (Items 3-4)
    clinical_setting: str
    prediction_task: str
    intended_use: str

    # Methods - Data (Items 5-8)
    data_source: str
    study_dates: str
    inclusion_criteria: str
    exclusion_criteria: str
    sample_size: int

    # Methods - Model (Items 9-11)
    model_type: str
    features_used: str
    preprocessing: str
    handling_missing: str

    # Methods - Evaluation (Items 12-16)
    validation_strategy: str
    performance_metrics: List[str]
    calibration_assessment: bool
    subgroup_analysis: List[str]

    # Results (Items 17-19)
    participant_flow: Dict[str, Any]
    model_performance: Dict[str, float]
    calibration_results: Optional[Dict[str, Any]] = None

    # Discussion (Items 20-22)
    limitations: List[str] = field(default_factory=list)
    generalizability: str = ""

    # Metadata
    flair_version: str = "1.0.0"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "abstract_summary": self.abstract_summary,
            "introduction": {
                "clinical_setting": self.clinical_setting,
                "prediction_task": self.prediction_task,
                "intended_use": self.intended_use,
            },
            "methods": {
                "data": {
                    "source": self.data_source,
                    "dates": self.study_dates,
                    "inclusion": self.inclusion_criteria,
                    "exclusion": self.exclusion_criteria,
                    "sample_size": self.sample_size,
                },
                "model": {
                    "type": self.model_type,
                    "features": self.features_used,
                    "preprocessing": self.preprocessing,
                    "missing_data": self.handling_missing,
                },
                "evaluation": {
                    "validation": self.validation_strategy,
                    "metrics": self.performance_metrics,
                    "calibration": self.calibration_assessment,
                    "subgroups": self.subgroup_analysis,
                },
            },
            "results": {
                "participant_flow": self.participant_flow,
                "performance": self.model_performance,
                "calibration": self.calibration_results,
            },
            "discussion": {
                "limitations": self.limitations,
                "generalizability": self.generalizability,
            },
            "metadata": {
                "flair_version": self.flair_version,
                "generated_at": self.generated_at,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        return format_tripod_ai_markdown(self)


def generate_tripod_ai_report(
    task_config: Dict[str, Any],
    model_info: Dict[str, Any],
    results: Dict[str, Any],
    table1: Dict[str, Any],
) -> TRIPODAIReport:
    """
    Generate TRIPOD-AI compliant report from FLAIR results.

    Auto-populates many fields from task configuration and results.

    Args:
        task_config: Task configuration dictionary
        model_info: Information about the model used
        results: Model performance results
        table1: Table 1 summary statistics

    Returns:
        TRIPODAIReport object
    """
    task_name = task_config.get("name", "Unknown Task")
    task_type = task_config.get("task_type", "classification")

    # Determine metrics based on task type
    if task_type == "binary_classification":
        default_metrics = ["auroc", "auprc", "sensitivity", "specificity"]
    elif task_type == "multiclass":
        default_metrics = ["accuracy", "macro_f1", "weighted_f1"]
    else:
        default_metrics = ["mse", "rmse", "r2"]

    return TRIPODAIReport(
        # Title and Abstract
        title=f"FLAIR Benchmark: {task_config.get('display_name', task_name)}",
        abstract_summary=task_config.get(
            "description",
            f"Prediction model for {task_name}",
        ),
        # Introduction
        clinical_setting="Intensive Care Unit (ICU)",
        prediction_task=task_config.get("description", task_name),
        intended_use=(
            "Research benchmark - not for clinical use. "
            "Models are developed on MIMIC-CLIF and evaluated across federated sites."
        ),
        # Methods - Data
        data_source="CLIF-formatted ICU data",
        study_dates=task_config.get("study_dates", "2018-2024"),
        inclusion_criteria=(
            "Adult ICU patients (age >= 18), Length of stay > 0 days, "
            "Valid admission and discharge times"
        ),
        exclusion_criteria=(
            "Pediatric patients, non-ICU encounters, "
            "incomplete records (missing admission/discharge times)"
        ),
        sample_size=table1.get("n_total", 0),
        # Methods - Model
        model_type=model_info.get("model_type", "Unknown"),
        features_used=model_info.get(
            "features",
            "First 24 hours of ICU data (vitals, labs, respiratory support, assessments)",
        ),
        preprocessing=model_info.get(
            "preprocessing",
            "clifpy wide dataset generation with hourly aggregation",
        ),
        handling_missing=model_info.get(
            "handling_missing",
            "Carry-forward imputation via tokenization, missing indicators",
        ),
        # Methods - Evaluation
        validation_strategy=model_info.get(
            "validation", "Temporal train/test split"
        ),
        performance_metrics=results.get("metrics_computed", default_metrics),
        calibration_assessment=results.get("calibration_computed", True),
        subgroup_analysis=results.get("subgroups", ["age_group", "sex"]),
        # Results
        participant_flow=table1.get("outcomes", {}),
        model_performance=results.get("metrics", {}),
        calibration_results=results.get("calibration", None),
        # Discussion
        limitations=[
            "Single-site development data (MIMIC-CLIF)",
            "Temporal validation only for development (no external validation until federated)",
            "Limited to variables available in CLIF format",
            "Model generalizability pending multi-site evaluation",
        ],
        generalizability=(
            "Intended for multi-site evaluation via federated learning. "
            "Final generalizability will be assessed after evaluation across 17+ CLIF sites."
        ),
    )


def format_tripod_ai_markdown(report: TRIPODAIReport) -> str:
    """
    Format TRIPOD-AI report as markdown.

    Args:
        report: TRIPODAIReport object

    Returns:
        Markdown-formatted string
    """
    metrics_str = ", ".join(report.performance_metrics)
    limitations_str = "\n".join(f"- {lim}" for lim in report.limitations)

    # Format performance metrics
    perf_lines = []
    for metric, value in report.model_performance.items():
        if isinstance(value, float):
            perf_lines.append(f"| {metric} | {value:.4f} |")
        else:
            perf_lines.append(f"| {metric} | {value} |")
    perf_table = "\n".join(perf_lines)

    return f"""# {report.title}

## Abstract

{report.abstract_summary}

---

## 1. Introduction

**Clinical Setting:** {report.clinical_setting}

**Prediction Task:** {report.prediction_task}

**Intended Use:** {report.intended_use}

---

## 2. Methods

### 2.1 Data

- **Source:** {report.data_source}
- **Study Period:** {report.study_dates}
- **Sample Size:** {report.sample_size:,}

**Inclusion Criteria:** {report.inclusion_criteria}

**Exclusion Criteria:** {report.exclusion_criteria}

### 2.2 Model Development

- **Model Type:** {report.model_type}
- **Features:** {report.features_used}
- **Preprocessing:** {report.preprocessing}
- **Missing Data:** {report.handling_missing}

### 2.3 Evaluation

- **Validation Strategy:** {report.validation_strategy}
- **Performance Metrics:** {metrics_str}
- **Calibration Assessment:** {"Yes" if report.calibration_assessment else "No"}
- **Subgroup Analysis:** {", ".join(report.subgroup_analysis)}

---

## 3. Results

### 3.1 Model Performance

| Metric | Value |
|--------|-------|
{perf_table}

---

## 4. Discussion

### 4.1 Limitations

{limitations_str}

### 4.2 Generalizability

{report.generalizability}

---

## Metadata

- **FLAIR Version:** {report.flair_version}
- **Generated:** {report.generated_at}

---

*This report follows the TRIPOD-AI guidelines for transparent reporting of prediction model studies.*
"""
