# 🏥 FLAIR - Federated Learning Assessment for ICU Research (WIP 🚧)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![CLIF](https://img.shields.io/badge/data-CLIF_format-orange.svg)](https://clif-icu.com)

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction tasks across 17+ hospital sites using the CLIF (Common Longitudinal ICU Format) data standard.

In the ICU, clinical decisions happen in real-time and directly impact patient survival. Rich data—diagnoses, laboratory values, vital signs, and outcomes—can drive better care, and AI offers new ways to support clinical reasoning in these high-stakes environments. Yet building trustworthy, generalizable AI requires diverse datasets that reflect varied patient populations and clinical practices.

Public ICU datasets like MIMIC and eICU have enabled significant research progress, but they represent only a handful of institutions. Meanwhile, the vast majority of ICU data remains locked in private hospital silos—inaccessible for multi-site validation due to privacy regulations. This fragmentation limits our ability to develop AI that generalizes beyond the institutions where it was trained.

**CLIF and FLAIR bridge this gap.** The Common Longitudinal ICU Format (CLIF) provides a shared data standard that harmonizes ICU data across institutions. FLAIR builds on this foundation, enabling federated model evaluation on real-world private data from 17+ hospitals—without patient information ever leaving each site.

------------------------------------------------------------------------

## 🎯 Why FLAIR?

### The Problem: Public Benchmarks Aren't Enough

Existing ICU benchmarks like [YAIB](https://github.com/rvandewater/YAIB) and [HiRID-ICU-Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark) have done excellent work harmonizing **public datasets** (MIMIC, eICU, HiRID, AUMCdb). However, models validated solely on public data face a critical limitation:

> **Models that perform well on public benchmarks may not generalize to real-world clinical settings.**

Public datasets represent a handful of institutions with specific patient populations, workflows, and documentation practices. Real-world deployment requires validation across diverse hospitals.

### The Challenge: Patient Data Cannot Leave Hospitals

Multi-site validation is essential, but patient data is protected by strict privacy regulations (HIPAA, IRB). **Raw clinical data cannot be shared or centralized** for traditional benchmark evaluation.

### The Solution: Federated Evaluation

FLAIR enables researchers to validate their methods on **private ICU datasets from 17+ US hospitals** without the data ever leaving each site:

![FLAIR Workflow](assets/FlairWorkflow.gif)

**Result**: Your model gets evaluated on real-world private clinical data from diverse institutions, enabling robust generalization assessment.

------------------------------------------------------------------------

## 🏛️ The CLIF Consortium

FLAIR is built on the **Common Longitudinal ICU Format (CLIF)** data standard, maintained by a consortium of **17+ academic medical centers** across the United States.

| Metric              | Value           |
|---------------------|-----------------|
| Participating Sites | 17+             |
| Geographic Coverage | Nationwide (US) |
| Combined ICU Beds   | 2,000+          |
| Data Standard       | CLIF v2.1       |

The consortium includes major academic medical centers, community hospitals, and health systems—providing diverse patient populations, clinical workflows, and documentation practices for robust model validation.

### 🚪 MIMIC-CLIF: Your Entry Point

Since CLIF consortium data is private, how do you develop your method? **MIMIC-CLIF** is the answer:

```         
┌─────────────────┐         ┌─────────────────-┐
│   MIMIC-CLIF    │         │  CLIF Consortium │
│   (Public)      │         │  (Private)       │
├─────────────────┤         ├─────────────────-┤
│ • PhysioNet     │   Same  │ • 17+ Sites      │
│ • ~70k ICU stays│ ══════► │ • 500k+ ICU stays│
│ • Single site   │  Schema │ • Diverse pops   │
│ • Development   │         │ • Evaluation     │
└─────────────────┘         └─────────────────-┘
```

This approach ensures your code will work on consortium data without modification.

------------------------------------------------------------------------

## 📊 Benchmark Tasks

FLAIR provides **7 clinically relevant prediction tasks**:

### Binary Classification Tasks

| Task | Name | Description | Cohort Filter |
|---------------|---------------|--------------------|-----------------------|
| 1 | Discharged Home | Predict if patient will be discharged directly home | All ICU patients |
| 2 | Discharged to LTACH | Predict if patient will go to long-term acute care | All ICU patients |
| 6 | Hospital Mortality | Predict in-hospital death (first 24hr ICU data) | 1st ICU stay ≥ 24hr |
| 7 | Unplanned ICU Readmission | Predict unplanned return to ICU (entire 1st ICU stay data) | 1st ICU stay ≥ 24hr |

### Multiclass Classification Tasks

| Task | Name | Description | Cohort Filter |
|---------------|---------------|--------------------|-----------------------|
| 3 | 72-Hour Respiratory Outcome | Predict ventilator status at 72hr (on/off/expired) | IMV at 24hr only |

### Regression Tasks

| Task | Name | Description | Cohort Filter |
|---------------|---------------|--------------------|-----------------------|
| 4 | Hypoxic Proportion | Predict fraction of hypoxic hours (24-72hr window) | IMV at 24hr only |
| 5 | ICU Length of Stay | Predict 1st ICU stay duration (first 24hr data) | 1st ICU stay ≥ 24hr |

### Understanding Time Windows

Each task defines a **time window** for data extraction:

- **`window_start`**: Beginning of the data collection period
- **`window_end`**: The **prediction time** - this is when the model makes its prediction

**Critical**: You can use ALL data points within the window (`window_start` to `window_end`), but you **CANNOT** use any data after `window_end`. Using data beyond the prediction time would be data leakage.

The window definition varies by task:
- **Tasks 1-6**: Window is first 24 hours from ICU admission (`first_icu_start_time` to `+24hr`)
- **Task 7**: Window is the entire first ICU stay (`first_icu_start_time` to `first_icu_end_time`)

The window is task-specific and not always aligned with ICU admission/discharge times.

> **Community-Driven**: These tasks are driven by community needs. Have a cool prediction task idea? Open a [PR](https://github.com/clif-consortium/FLAIR/pulls) or [Issue](https://github.com/clif-consortium/FLAIR/issues)! We're actively working on adding more tasks.

**Note**: Each task has its own cohort size (N) based on task-specific filters. All tasks share the same base criteria: hospitalizations with at least 1 ICU stay.

------------------------------------------------------------------------

## 🎁 What FLAIR Provides

FLAIR is a Python library that generates task-specific datasets for ICU prediction benchmarks:

| FLAIR Provides                    | You Provide                 |
|-----------------------------------|-----------------------------|
| ✅ Task-specific cohort filtering | 🔧 Your feature engineering |
| ✅ Consistent label extraction    | 🔧 Your model architecture  |
| ✅ Temporal train/test splits     | 🔧 Your training pipeline   |
| ✅ Demographics & time windows    | 🔧 Your evaluation          |

### Output Format

Each task outputs a **single parquet file** with all required columns:

| Column               | Type      | Description                            |
|----------------------|-----------|----------------------------------------|
| `hospitalization_id` | str       | Unique identifier                      |
| `admission_dttm`     | datetime  | Hospital admission time                |
| `discharge_dttm`     | datetime  | Hospital discharge time                |
| `window_start`       | datetime  | ICU start time (input window start)    |
| `window_end`         | datetime  | Prediction time (+24hr from ICU start) |
| `{task_label}`       | int/float | Task-specific label                    |
| `split`              | str       | "train" or "test"                      |
| `age_at_admission`   | int       | Patient age                            |
| `sex_category`       | str       | Patient sex                            |
| `race_category`      | str       | Patient race                           |
| `ethnicity_category` | str       | Patient ethnicity                      |

------------------------------------------------------------------------

## 💿 Installation

``` bash
# Install with pip
pip install flair-benchmark

# Or from source
git clone https://github.com/clif-consortium/FLAIR.git
cd FLAIR
pip install -e .
```

**Requirements**: Python 3.10+, clifpy

------------------------------------------------------------------------

## 🚀 Quick Start

### 1. Configure CLIF Data Source

``` bash
cp clif_config_template.json clif_config.json
```

Edit `clif_config.json` to set your data path and timezone.

### 2. Generate Task Dataset

``` python
from flair_benchmark import generate_task_dataset, TASK_REGISTRY

# View available tasks
print(TASK_REGISTRY.keys())
# ['task1_discharged_home', 'task2_discharged_ltach', 'task3_outcome_72hr',
#  'task4_hypoxic_proportion', 'task5_icu_los', 'task6_hospital_mortality',
#  'task7_icu_readmission']

# Generate dataset for ICU LOS task with temporal split
df = generate_task_dataset(
    config_path="clif_config.json",
    task_name="task5_icu_los",
    train_start="2020-01-01",
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2023-12-31",
    output_path="task5_icu_los.parquet"
)

print(f"Total N: {len(df)}")
print(f"Train: {len(df.filter(df['split'] == 'train'))}")
print(f"Test: {len(df.filter(df['split'] == 'test'))}")
```

### 3. Use the Dataset

``` python
import polars as pl

# Load dataset
df = pl.read_parquet("task5_icu_los.parquet")

# Split into train/test
train = df.filter(pl.col("split") == "train")
test = df.filter(pl.col("split") == "test")

# Access labels
y_train = train["icu_los_hours"]
y_test = test["icu_los_hours"]

# Access demographics for subgroup analysis
demographics = train.select(["age_at_admission", "sex_category", "race_category"])
```

------------------------------------------------------------------------

## 🔒 Privacy Policy

```         
╔════════════════════════════════════════════════════════════════╗
║                     FLAIR PRIVACY POLICY                       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  1. NO NETWORK REQUESTS                                        ║
║     • All network access blocked at Python socket level        ║
║     • Packages like requests, urllib3, httpx are banned        ║
║     • Violation = immediate submission rejection               ║
║                                                                ║
║  2. PHI PROTECTION                                             ║
║     • All outputs scanned for PHI patterns                     ║
║     • Cell counts < 10 are suppressed (HIPAA safe harbor)      ║
║     • Individual-level data never leaves the site              ║
║                                                                ║
║  3. REVIEW PROCESS                                             ║
║     • PIs at each site review code before execution            ║
║     • PIs have final say — they are not required to run        ║
║     • Code inspection for data exfiltration attempts           ║
║                                                                ║
║  4. CONSEQUENCES                                               ║
║     • If data exfiltration found during review:                ║
║       → Submitter BANNED from FLAIR                            ║
║       → Incident reported to submitter's institution           ║
║                                                                ║
║  PIs are doing you a favor by running your code on their data. ║
║  Respect their trust and protect patient privacy.              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

------------------------------------------------------------------------

## 🏗️ Architecture

```         
FLAIR/
├── flair/                      # Main package
│   ├── __init__.py             # Main API: generate_task_dataset()
│   ├── cohort/                 # Cohort builder (clifpy integration)
│   ├── config/                 # Configuration management
│   ├── datasets/               # Dataset builder
│   ├── helpers/                # table1, metrics, tripod_ai
│   └── tasks/                  # Task definitions (7 tasks)
│       ├── base.py             # BaseTask with build_task_dataset()
│       ├── task1_discharged_home.py
│       ├── task2_discharged_ltach.py
│       ├── task3_outcome_72hr.py
│       ├── task4_hypoxic_proportion.py
│       ├── task5_icu_los.py
│       ├── task6_hospital_mortality.py
│       └── task7_icu_readmission.py
├── clif_config_template.json   # CLIF data configuration template
├── pyproject.toml              # Package configuration
└── tests/                      # Test suite
```

------------------------------------------------------------------------

## 🔗 Related Projects

| Project | Description |
|------------------------------|------------------------------------------|
| [clifpy](https://github.com/clif-consortium/clifpy) | Python library for CLIF data manipulation |
| [MIMIC-CLIF](https://physionet.org/) | CLIF-formatted MIMIC-IV (development entry point) |
| [CLIF Consortium](https://clif-icu.com) | Official CLIF consortium website |

------------------------------------------------------------------------

## 📖 Citation

If you use FLAIR in your research, please cite:

``` bibtex
@software{flair2024,
  title = {FLAIR: Federated Learning Assessment for ICU Research},
  author = {CLIF Consortium},
  year = {2024},
  url = {https://github.com/clif-consortium/FLAIR}
}
```

------------------------------------------------------------------------

## 📜 License

This source code is released under the APACHE 2.0 license. See [LICENSE](LICENSE) for details.

We do not own any of the clinical datasets used with this benchmark. Access to CLIF consortium data requires approval from each participating institution.

------------------------------------------------------------------------

## 📬 Contact

-   **Website**: [clif-icu.com](https://clif-icu.com)
-   **Email**: clif_consortium\@uchicago.edu
-   **Issues**: [GitHub Issues](https://github.com/clif-consortium/FLAIR/issues)
