"""Dataset generation for FLAIR benchmark.

Uses clifpy's create_wide_dataset() for feature generation.
"""

from flair_benchmark.datasets.builder import FLAIRDatasetBuilder
from flair_benchmark.datasets.splitter import DataSplitter, SplitMethod

__all__ = [
    "FLAIRDatasetBuilder",
    "DataSplitter",
    "SplitMethod",
]
