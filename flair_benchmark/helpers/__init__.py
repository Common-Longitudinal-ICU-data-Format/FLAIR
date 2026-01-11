"""PHI-safe helper functions for FLAIR benchmark.

These functions return ONLY aggregated statistics - no individual-level data.
All cell counts < 10 are suppressed to prevent re-identification.
"""

from flair_benchmark.helpers.table1 import get_table1
from flair_benchmark.helpers.tripod_ai import generate_tripod_ai_report, TRIPODAIReport
from flair_benchmark.helpers.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)

__all__ = [
    "get_table1",
    "generate_tripod_ai_report",
    "TRIPODAIReport",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "compute_regression_metrics",
]
