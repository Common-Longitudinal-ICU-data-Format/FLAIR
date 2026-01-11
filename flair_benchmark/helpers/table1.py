"""
Table 1 generation for FLAIR benchmark.

Generates summary statistics tables with PHI-safe cell suppression.
All counts < minimum threshold are suppressed to prevent re-identification.
"""

import polars as pl
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# HIPAA Safe Harbor minimum cell count
DEFAULT_MIN_CELL_COUNT = 10


def get_table1(
    cohort_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    stratify_by: Optional[str] = "label",
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
    min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
) -> Dict[str, Any]:
    """
    Generate Table 1 summary statistics.

    This function returns ONLY aggregated statistics - no individual-level data.
    All cell counts below the minimum threshold are suppressed with '<N' to
    prevent re-identification of patients.

    Args:
        cohort_df: Cohort DataFrame with demographics
        labels_df: Labels DataFrame with outcomes
        stratify_by: Column to stratify by (e.g., 'label')
        continuous_vars: List of continuous variables to summarize
        categorical_vars: List of categorical variables to summarize
        min_cell_count: Minimum count to display (smaller suppressed)

    Returns:
        Dictionary with aggregated statistics:
        {
            'n_total': int,
            'continuous': {var: {'mean': float, 'std': float, 'median': float, ...}},
            'categorical': {var: {category: count_or_suppressed}},
            'outcomes': {label: count_or_suppressed}
        }
    """
    # Default variable lists
    if continuous_vars is None:
        continuous_vars = ["age_at_admission"]

    if categorical_vars is None:
        categorical_vars = ["sex_category", "race_category", "ethnicity_category"]

    # Join cohort with labels
    df = cohort_df.join(labels_df, on="hospitalization_id", how="inner")

    result = {
        "n_total": df.height,
        "continuous": {},
        "categorical": {},
        "outcomes": {},
        "metadata": {
            "min_cell_count": min_cell_count,
            "suppression_note": f"Cells with count < {min_cell_count} suppressed",
        },
    }

    # Process continuous variables
    for var in continuous_vars:
        if var in df.columns:
            result["continuous"][var] = _compute_continuous_stats(df, var)

    # Process categorical variables with cell suppression
    for var in categorical_vars:
        if var in df.columns:
            result["categorical"][var] = _compute_categorical_stats(
                df, var, min_cell_count
            )

    # Process outcomes/labels
    label_col = stratify_by
    if label_col and label_col in df.columns:
        result["outcomes"] = _compute_categorical_stats(df, label_col, min_cell_count)

    # Add stratified statistics if requested
    if stratify_by and stratify_by in df.columns:
        result["stratified"] = _compute_stratified_stats(
            df, stratify_by, continuous_vars, categorical_vars, min_cell_count
        )

    logger.info(f"Generated Table 1: n={result['n_total']}")

    return result


def _compute_continuous_stats(df: pl.DataFrame, var: str) -> Dict[str, float]:
    """Compute summary statistics for a continuous variable."""
    col = df[var].drop_nulls()

    if len(col) == 0:
        return {"mean": None, "std": None, "median": None, "q25": None, "q75": None}

    return {
        "mean": round(float(col.mean()), 2),
        "std": round(float(col.std()), 2),
        "median": round(float(col.median()), 2),
        "q25": round(float(col.quantile(0.25)), 2),
        "q75": round(float(col.quantile(0.75)), 2),
        "min": round(float(col.min()), 2),
        "max": round(float(col.max()), 2),
        "n_missing": df.height - len(col),
    }


def _compute_categorical_stats(
    df: pl.DataFrame,
    var: str,
    min_cell_count: int,
) -> Dict[str, Union[int, str]]:
    """
    Compute counts for a categorical variable with cell suppression.

    Counts below min_cell_count are replaced with '<N' string.
    """
    counts = df.group_by(var).agg(pl.count().alias("count"))

    result = {}
    for row in counts.iter_rows(named=True):
        category = str(row[var]) if row[var] is not None else "missing"
        count = row["count"]

        if count < min_cell_count:
            result[category] = f"<{min_cell_count}"
        else:
            result[category] = count

    return result


def _compute_stratified_stats(
    df: pl.DataFrame,
    stratify_by: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
    min_cell_count: int,
) -> Dict[str, Dict[str, Any]]:
    """Compute statistics stratified by a grouping variable."""
    result = {}

    strata = df[stratify_by].unique().to_list()

    for stratum in strata:
        stratum_key = str(stratum)
        stratum_df = df.filter(pl.col(stratify_by) == stratum)

        stratum_stats = {
            "n": stratum_df.height,
            "continuous": {},
            "categorical": {},
        }

        # Continuous stats per stratum
        for var in continuous_vars:
            if var in stratum_df.columns:
                stratum_stats["continuous"][var] = _compute_continuous_stats(
                    stratum_df, var
                )

        # Categorical stats per stratum with suppression
        for var in categorical_vars:
            if var in stratum_df.columns:
                stratum_stats["categorical"][var] = _compute_categorical_stats(
                    stratum_df, var, min_cell_count
                )

        result[stratum_key] = stratum_stats

    return result


def format_table1_markdown(table1: Dict[str, Any]) -> str:
    """
    Format Table 1 as markdown.

    Args:
        table1: Output from get_table1()

    Returns:
        Markdown-formatted string
    """
    lines = [
        "# Table 1: Cohort Characteristics",
        "",
        f"**Total N = {table1['n_total']}**",
        "",
        f"*Note: {table1['metadata']['suppression_note']}*",
        "",
        "## Continuous Variables",
        "",
        "| Variable | Mean (SD) | Median [IQR] |",
        "|----------|-----------|--------------|",
    ]

    for var, stats in table1.get("continuous", {}).items():
        if stats["mean"] is not None:
            mean_sd = f"{stats['mean']} ({stats['std']})"
            median_iqr = f"{stats['median']} [{stats['q25']}-{stats['q75']}]"
        else:
            mean_sd = "N/A"
            median_iqr = "N/A"
        lines.append(f"| {var} | {mean_sd} | {median_iqr} |")

    lines.extend(["", "## Categorical Variables", ""])

    for var, categories in table1.get("categorical", {}).items():
        lines.append(f"### {var}")
        lines.append("")
        lines.append("| Category | N (%) |")
        lines.append("|----------|-------|")

        for category, count in categories.items():
            if isinstance(count, int):
                pct = round(100 * count / table1["n_total"], 1)
                lines.append(f"| {category} | {count} ({pct}%) |")
            else:
                lines.append(f"| {category} | {count} |")

        lines.append("")

    return "\n".join(lines)
