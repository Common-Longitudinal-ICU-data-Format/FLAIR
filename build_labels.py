#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "clifpy",
# ]
# ///
"""
FLAIR Label Builder

Generates label parquet files for each benchmark task.
Each task gets its own parquet with: hospitalization_id, time_to_predict, label

Usage:
    uv run build_labels.py [--config clif_config.json] [--cohort flair_output/cohort.parquet]
"""

import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "clif_config.json") -> dict:
    """Load CLIF configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def build_icu_los_labels(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Build ICU LOS labels (regression).

    time_to_predict: first_icu_start_time + 24 hours
    label: first_icu_los (hours)
    """
    # Filter to patients with at least 24 hours ICU stay
    cohort_filtered = cohort.filter(pl.col("first_icu_los") >= 24)

    labels = cohort_filtered.select([
        pl.col("hospitalization_id"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("time_to_predict"),
        pl.col("first_icu_los").alias("label"),
    ])

    return labels


def build_icu_readmission_labels(cohort: pl.DataFrame, adt: pl.DataFrame) -> pl.DataFrame:
    """
    Build ICU readmission labels (binary).

    time_to_predict: first_icu_end_time (at first ICU discharge)
    label: 1 if second ICU stay exists, else 0
    """
    # Get hospitalization IDs from cohort
    hosp_ids = cohort["hospitalization_id"].to_list()

    # Filter ADT to cohort hospitalizations
    adt_cohort = adt.filter(pl.col("hospitalization_id").is_in(hosp_ids))

    # Filter to ICU locations and sort
    icu_adt = adt_cohort.filter(
        pl.col("location_category").str.to_lowercase() == "icu"
    ).sort(["hospitalization_id", "in_dttm"])

    # Detect ICU stay transitions
    icu_adt = icu_adt.with_columns(
        pl.col("in_dttm").shift(1).over("hospitalization_id").alias("prev_in_dttm")
    )

    # Mark new ICU stays (gap > 1 hour from previous ICU record indicates new stay)
    icu_adt = icu_adt.with_columns(
        pl.when(pl.col("prev_in_dttm").is_null())
        .then(True)
        .otherwise(
            (pl.col("in_dttm") - pl.col("prev_in_dttm")).dt.total_hours() > 1
        )
        .alias("is_new_icu_stay")
    )

    # Count ICU stays per hospitalization
    icu_stay_counts = (
        icu_adt.filter(pl.col("is_new_icu_stay"))
        .group_by("hospitalization_id")
        .agg(pl.len().alias("icu_stay_count"))
    )

    # Join with cohort and create labels
    labels = cohort.join(icu_stay_counts, on="hospitalization_id", how="left")
    labels = labels.with_columns(
        pl.col("icu_stay_count").fill_null(1)
    )

    labels = labels.select([
        pl.col("hospitalization_id"),
        pl.col("first_icu_end_time").alias("time_to_predict"),
        pl.when(pl.col("icu_stay_count") > 1)
        .then(1)
        .otherwise(0)
        .alias("label"),
    ])

    # Filter out null time_to_predict
    labels = labels.filter(pl.col("time_to_predict").is_not_null())

    return labels


def build_hospital_mortality_labels(cohort: pl.DataFrame) -> pl.DataFrame:
    """
    Build hospital mortality labels (binary).

    time_to_predict: first_icu_start_time + 24 hours
    label: 1 if discharge_category == "Expired", else 0
    """
    # Filter to patients with at least 24 hours ICU stay
    cohort_filtered = cohort.filter(pl.col("first_icu_los") >= 24)

    labels = cohort_filtered.select([
        pl.col("hospitalization_id"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("time_to_predict"),
        pl.when(pl.col("discharge_category").str.to_lowercase() == "expired")
        .then(1)
        .otherwise(0)
        .alias("label"),
    ])

    return labels


def build_icu_mortality_labels(cohort: pl.DataFrame, patient: pl.DataFrame) -> pl.DataFrame:
    """
    Build ICU mortality labels (binary, once per stay, after 24H).

    time_to_predict: first_icu_start_time + 24 hours
    label: 1 if death_dttm between first_icu_start and first_icu_end, else 0
    Filter: Only include patients with first_icu_los >= 24 hours
    """
    # Filter to patients with at least 24 hours ICU stay
    cohort_filtered = cohort.filter(pl.col("first_icu_los") >= 24)

    # Join with patient table to get death_dttm
    cohort_with_death = cohort_filtered.join(
        patient.select(["patient_id", "death_dttm"]),
        on="patient_id",
        how="left"
    )

    # Determine if death occurred during first ICU stay
    labels = cohort_with_death.select([
        pl.col("hospitalization_id"),
        (pl.col("first_icu_start_time") + pl.duration(hours=24)).alias("time_to_predict"),
        pl.when(
            pl.col("death_dttm").is_not_null()
            & (pl.col("death_dttm") >= pl.col("first_icu_start_time"))
            & (pl.col("death_dttm") <= pl.col("first_icu_end_time"))
        )
        .then(1)
        .otherwise(0)
        .alias("label"),
    ])

    return labels


def build_labels(
    config_path: str = "clif_config.json",
    cohort_path: str = "flair_output/cohort.parquet",
    output_dir: str = "flair_output/labels",
) -> dict:
    """
    Build labels for all enabled tasks.

    Returns dict of task_name -> label DataFrame
    """
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    logger.info(f"Loading cohort from {cohort_path}")
    cohort = pl.read_parquet(cohort_path)
    logger.info(f"Cohort size: {cohort.height:,} hospitalizations")

    # Get data sources config
    data_sources = config.get("data_sources", {})
    data_dir = data_sources.get("base_path", "./data/clif")
    filetype = data_sources.get("format", "parquet")
    timezone = data_sources.get("timezone", "US/Central")

    # Load additional tables as needed
    from clifpy.tables import Adt, Patient

    logger.info("Loading ADT table...")
    adt_table = Adt.from_file(
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
    )
    adt = pl.from_pandas(adt_table.df)

    logger.info("Loading patient table...")
    patient_table = Patient.from_file(
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
    )
    patient = pl.from_pandas(patient_table.df)

    # Get enabled tasks
    benchmark_outcomes = config.get("benchmark_outcomes", {})

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Build labels for each enabled task
    for task_name, task_config in benchmark_outcomes.items():
        if not task_config.get("generate_labels", True):
            logger.info(f"Skipping {task_name} (generate_labels=false)")
            continue

        logger.info(f"Building labels for {task_name}...")

        if task_name == "icu_los":
            labels = build_icu_los_labels(cohort)
        elif task_name == "icu_readmission":
            labels = build_icu_readmission_labels(cohort, adt)
        elif task_name == "hospital_mortality":
            labels = build_hospital_mortality_labels(cohort)
        elif task_name == "icu_mortality":
            labels = build_icu_mortality_labels(cohort, patient)
        else:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue

        # Save labels
        label_path = output_path / f"{task_name}.parquet"
        labels.write_parquet(label_path)
        logger.info(f"Saved {labels.height:,} labels to {label_path}")

        # Print stats
        if task_name == "icu_los":
            logger.info(f"  Label stats: min={labels['label'].min():.1f}, max={labels['label'].max():.1f}, median={labels['label'].median():.1f} hours")
        else:
            pos_count = labels.filter(pl.col("label") == 1).height
            neg_count = labels.filter(pl.col("label") == 0).height
            pos_rate = pos_count / labels.height * 100 if labels.height > 0 else 0
            logger.info(f"  Positive: {pos_count:,} ({pos_rate:.1f}%), Negative: {neg_count:,}")

        results[task_name] = labels

    logger.info("=" * 60)
    logger.info("LABEL GENERATION COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FLAIR task labels from cohort")
    parser.add_argument(
        "--config",
        default="clif_config.json",
        help="Path to CLIF config JSON (default: clif_config.json)"
    )
    parser.add_argument(
        "--cohort",
        default="flair_output/cohort.parquet",
        help="Path to cohort parquet (default: flair_output/cohort.parquet)"
    )
    parser.add_argument(
        "--output",
        default="flair_output/labels",
        help="Output directory for label files (default: flair_output/labels)"
    )

    args = parser.parse_args()

    build_labels(
        config_path=args.config,
        cohort_path=args.cohort,
        output_dir=args.output,
    )
