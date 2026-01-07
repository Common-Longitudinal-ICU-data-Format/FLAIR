"""Data splitting utilities for FLAIR benchmark.

Provides temporal train/test splits for clinical data.
"""

from enum import Enum
from typing import Tuple, Optional
import polars as pl


class SplitMethod(Enum):
    """Methods for splitting data into train/test sets."""
    TEMPORAL = "temporal"  # Split by admission date
    RANDOM = "random"      # Random split (not recommended for clinical data)


class DataSplitter:
    """
    Split datasets into train/test sets.

    For clinical data, temporal splits are recommended to avoid data leakage
    and simulate real-world deployment scenarios.
    """

    def __init__(
        self,
        method: SplitMethod = SplitMethod.TEMPORAL,
        date_column: str = "admission_dttm"
    ):
        """
        Initialize data splitter.

        Args:
            method: Split method (TEMPORAL recommended)
            date_column: Column name for temporal splitting
        """
        self.method = method
        self.date_column = date_column

    def split(
        self,
        df: pl.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str
    ) -> pl.DataFrame:
        """
        Add split column to DataFrame based on date ranges.

        Args:
            df: Input DataFrame with date_column
            train_start: Train period start (YYYY-MM-DD)
            train_end: Train period end (YYYY-MM-DD)
            test_start: Test period start (YYYY-MM-DD)
            test_end: Test period end (YYYY-MM-DD)

        Returns:
            DataFrame with 'split' column added ('train', 'test', or null)
        """
        if self.method != SplitMethod.TEMPORAL:
            raise NotImplementedError("Only TEMPORAL split method is currently supported")

        # Parse dates
        train_start_dt = pl.lit(train_start).str.to_datetime()
        train_end_dt = pl.lit(train_end).str.to_datetime()
        test_start_dt = pl.lit(test_start).str.to_datetime()
        test_end_dt = pl.lit(test_end).str.to_datetime()

        # Add split column
        return df.with_columns([
            pl.when(
                (pl.col(self.date_column) >= pl.lit(train_start).str.to_datetime()) &
                (pl.col(self.date_column) <= pl.lit(train_end).str.to_datetime())
            ).then(pl.lit("train"))
            .when(
                (pl.col(self.date_column) >= pl.lit(test_start).str.to_datetime()) &
                (pl.col(self.date_column) <= pl.lit(test_end).str.to_datetime())
            ).then(pl.lit("test"))
            .otherwise(pl.lit(None))
            .alias("split")
        ])

    def get_split_stats(self, df: pl.DataFrame) -> dict:
        """
        Get statistics about the split.

        Args:
            df: DataFrame with 'split' column

        Returns:
            Dictionary with split statistics
        """
        if "split" not in df.columns:
            raise ValueError("DataFrame must have 'split' column")

        total = len(df)
        train_count = len(df.filter(pl.col("split") == "train"))
        test_count = len(df.filter(pl.col("split") == "test"))
        null_count = len(df.filter(pl.col("split").is_null()))

        return {
            "total": total,
            "train": train_count,
            "test": test_count,
            "excluded": null_count,
            "train_pct": train_count / total * 100 if total > 0 else 0,
            "test_pct": test_count / total * 100 if total > 0 else 0
        }
