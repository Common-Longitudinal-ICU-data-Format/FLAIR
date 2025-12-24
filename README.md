# 🏥 FLAIR - Federated Learning Assessment for ICU Research

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![CLIF](https://img.shields.io/badge/data-CLIF_format-orange.svg)](https://clif-icu.com)

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction tasks across 17+ hospital sites using the CLIF (Common Longitudinal ICU Format) data standard.

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

FLAIR provides four clinically relevant prediction tasks, all using the first 24 hours of ICU data:

All tasks share the same underlying ICU cohort, making cross-task comparison straightforward.

------------------------------------------------------------------------

## 🎁 What FLAIR Provides

FLAIR handles the hard, error-prone parts of clinical ML benchmarking so you can focus on your method:

| FLAIR Provides                         | You Provide                 |
|----------------------------------------|-----------------------------|
| ✅ Standardized ICU cohort definitions | 🔧 Your feature engineering |
| ✅ Consistent label extraction         | 🔧 Your model architecture  |
| ✅ Wide-format datasets (ready to use) | 🔧 Your training pipeline   |
| ✅ Train/val/test splits               | 🔧 Your hyperparameters     |
| ✅ Evaluation metrics                  |                             |
| ✅ TRIPOD-AI reporting                 |                             |

### Why This Matters

**Without standardization**, comparing methods is impossible: - Different cohort inclusion criteria → different patient populations - Different label definitions → different prediction targets - Different time windows → different features available

**With FLAIR**, every submission uses: - 📋 **Same cohort**: Identical patient inclusion/exclusion criteria - 🏷️ **Same labels**: Identical outcome definitions and extraction logic - ⏱️ **Same time windows**: First 24 hours of ICU data for all tasks - 📊 **Same evaluation**: Identical metrics computed the same way

This makes results **directly comparable** across methods and sites.

### Your Freedom

FLAIR provides the `wide_dataset.parquet` with aggregated clinical features, but **you decide how to use it**:

-   Extract your own features from the wide format
-   Apply your own preprocessing and normalization
-   Use any model architecture (XGBoost, LSTM, Transformer, etc.)
-   Implement your own training strategy

The standardized inputs and outputs ensure fair comparison while giving you complete freedom in methodology.

------------------------------------------------------------------------

## 💿 Installation

``` bash
# Clone the repository
git clone https://github.com/clif-consortium/FLAIR.git
cd FLAIR

# Install from source
pip install -e .

# Or with uv
uv pip install -e .
```

**Requirements**: Python 3.10+, clifpy

------------------------------------------------------------------------

## 🚀 Quick Start

### 1. Initialize Configuration

``` bash
flair init --site my_hospital --output flair_config.yaml
```

Edit `flair_config.yaml` to point to your CLIF data.

### 2. Build Cohort

``` bash
flair build-cohort
```

Creates the shared ICU cohort from your CLIF data with: - Adults (age \>= 18) - At least one ICU ADT record - Length of stay \> 0 - `imv_at_24hr` flag for Tasks 3 & 4

### 3. Build Task Datasets

``` bash
# Build all tasks
flair build-datasets

# Or specific task
flair build-datasets --task task1_discharged_home
```

### 4. Generate Table 1

``` bash
flair table1 task1_discharged_home --format markdown
```

### 5. Validate Your Submission

``` bash
flair validate /path/to/my_method
```

------------------------------------------------------------------------

## 📝 Method Submission

### Required Files

```         
my_method/
├── README.md           # Describe your method
├── requirements.txt    # Dependencies (see banned packages)
├── train.py           # Training script
└── predict.py         # Prediction script
```

### train.py Interface

``` python
def train(
    wide_dataset_path: str,      # Path to wide_dataset.parquet
    labels_path: str,            # Path to labels.parquet
    output_dir: str,             # Directory to save trained model
    **kwargs
) -> None:
    """Train your model and save to output_dir."""
    pass
```

### predict.py Interface

``` python
def predict(
    wide_dataset_path: str,      # Path to test wide_dataset.parquet
    model_path: str,             # Path to trained model
    output_path: str,            # Path to save predictions
    **kwargs
) -> None:
    """Load model and generate predictions."""
    pass
```

### Banned Packages ⛔

The following packages will cause **immediate rejection**:

| Category     | Banned Packages                           |
|--------------|-------------------------------------------|
| HTTP Clients | `requests`, `urllib3`, `httpx`, `aiohttp` |
| Network      | `socket`, `websocket`, `paramiko`         |
| Protocols    | `ftplib`, `smtplib`, `telnetlib`          |
| Cloud SDKs   | `boto3`, `google.cloud`, `azure`          |

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

## 📁 Dataset Output Format

Each task produces three parquet files:

| File                   | Description                               |
|------------------------|-------------------------------------------|
| `wide_dataset.parquet` | Features from first 24 hours (via clifpy) |
| `labels.parquet`       | Task-specific labels                      |
| `demographics.parquet` | Age, sex, race, ethnicity                 |

------------------------------------------------------------------------

## 🖥️ CLI Commands

``` bash
flair --help                          # Show all commands
flair build-cohort                    # Build ICU cohort from CLIF data
flair build-datasets                  # Build all task datasets
flair validate /path/to/submission    # Validate method submission
flair table1 task_name                # Generate Table 1 statistics
flair package-results                 # Package results for submission
flair privacy-warning                 # Display privacy policy
flair version                         # Show version
```

------------------------------------------------------------------------

## ⚙️ Configuration

See `flair_config.yaml.template` for all options:

``` yaml
site:
  name: "your_site_name"
  timezone: "US/Central"

data:
  clif_config_path: "clif_config.json"

cohort:
  output_path: "flair_output/cohort.parquet"
  min_age: 18
  min_los_days: 0

tasks:
  enabled:
    - task1_discharged_home
    - task2_discharged_ltach
    - task3_outcome_72hr
    - task4_hypoxic_proportion

privacy:
  enable_network_blocking: true
  enable_phi_detection: true
  min_cell_count: 10

# NOTE: No wandb_api_key - all tracking must be offline only
```

------------------------------------------------------------------------

## 🏗️ Architecture

```         
FLAIR/
├── flair/                      # Main package
│   ├── cohort/                 # Cohort builder (clifpy integration)
│   ├── config/                 # Configuration management
│   ├── privacy/                # Network blocking, PHI detection
│   ├── tasks/                  # Task definitions (4 tasks)
│   ├── datasets/               # Dataset builder (clifpy wide format)
│   ├── helpers/                # table1, tripod_ai, metrics
│   ├── submission/             # Validation, packaging
│   └── cli.py                  # Command-line interface
├── configs/tasks/              # Task configuration YAMLs
├── templates/method_submission/ # Submission template
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