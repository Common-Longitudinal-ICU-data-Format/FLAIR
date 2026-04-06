# FLAIR Example Notebooks

Interactive [Marimo](https://marimo.io) notebooks demonstrating each FLAIR benchmark task.

## Setup

```bash
# Install flair-benchmark with examples dependencies
uv pip install -e ".[examples]"
```

## Running

```bash
# Open the overview notebook
marimo edit examples/00_overview.py

# Or open a specific task
marimo edit examples/task1_discharged_home.py
```

## Notebooks

| Notebook | Task | Type |
|----------|------|------|
| `00_overview.py` | All tasks overview | Index |
| `task1_discharged_home.py` | Discharged Home | Binary Classification |
| `task2_discharged_ltach.py` | Discharged to LTACH | Binary Classification (imbalanced) |
| `task3_72hr_respiratory_outcome.py` | 72-Hour Respiratory Outcome | Multiclass Classification |
| `task4_hypoxic_failure_proportion.py` | Hypoxic Failure Proportion | Regression |
| `task5_icu_length_of_stay.py` | ICU Length of Stay | Regression |
| `task6_hospital_mortality.py` | In-Hospital Mortality | Binary Classification |
| `task7_unplanned_icu_readmission.py` | Unplanned ICU Readmission | Binary Classification |

## Prerequisites

To run the dataset generation cells, you need:
1. CLIF-formatted data (e.g., MIMIC-CLIF)
2. A `clif_config.json` file pointing to your data directory

The task inspection cells (config, metrics list) work without any data.
