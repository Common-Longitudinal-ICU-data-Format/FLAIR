# 🏥 FLAIR - Federated Learning Assessment for ICU Research (WIP 🚧)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![CLIF](https://img.shields.io/badge/data-CLIF_format-orange.svg)](https://clif-icu.com)

A privacy-first benchmark framework for evaluating ML/AI methods on ICU prediction tasks across 12+ hospital sites using the CLIF (Common Longitudinal ICU Format) data standard.

In the ICU, clinical decisions happen in real-time and directly impact patient survival. Rich data—diagnoses, laboratory values, vital signs, and outcomes—can drive better care, and AI offers new ways to support clinical reasoning in these high-stakes environments. Yet building trustworthy, generalizable AI requires diverse datasets that reflect varied patient populations and clinical practices.

Public ICU datasets like MIMIC and eICU have enabled significant research progress, but they represent only a handful of institutions. Meanwhile, the vast majority of ICU data remains locked in private hospital silos—inaccessible for multi-site validation due to privacy regulations. This fragmentation limits our ability to develop AI that generalizes beyond the institutions where it was trained.

**CLIF and FLAIR bridge this gap.** The Common Longitudinal ICU Format (CLIF) provides a shared data standard that harmonizes ICU data across institutions. FLAIR builds on this foundation, enabling federated model evaluation on real-world private data from 12+ hospitals—without patient information ever leaving each site.

------------------------------------------------------------------------

## 🎯 Why FLAIR?

### The Problem: Public Benchmarks Aren't Enough

Existing ICU benchmarks like [YAIB](https://github.com/rvandewater/YAIB) and [HiRID-ICU-Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark) have done excellent work harmonizing **public datasets** (MIMIC, eICU, HiRID, AUMCdb). However, models validated solely on public data face a critical limitation:

> **Models that perform well on public benchmarks may not generalize to real-world clinical settings.**

Public datasets represent a handful of institutions with specific patient populations, workflows, and documentation practices. Real-world deployment requires validation across diverse hospitals.

### The Challenge: Patient Data Cannot Leave Hospitals

Multi-site validation is essential, but patient data is protected by strict privacy regulations (HIPAA, IRB). **Raw clinical data cannot be shared or centralized** for traditional benchmark evaluation.

### The Solution: Federated Evaluation

FLAIR enables researchers to validate their methods on **private ICU datasets from 12+ US hospitals** without the data ever leaving each site:

![FLAIR Workflow](assets/FlairWorkflow.gif)

**Result**: Your model gets evaluated on real-world private clinical data from diverse institutions, enabling robust generalization assessment.

------------------------------------------------------------------------

## 🏛️ The CLIF Consortium

FLAIR is built on the **Common Longitudinal ICU Format (CLIF)** data standard, maintained by a consortium of **12+ academic medical centers** across the United States.

| Metric              | Value           |
|---------------------|-----------------|
| Participating Sites | 12+             |
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
│ • PhysioNet     │   Same  │ • 12+ Sites      │
│ • ~70k ICU stays│ ══════► │ • 500k+ ICU stays│
│ • Single site   │  Schema │ • Diverse pops   │
│ • Development   │         │ • Evaluation     │
└─────────────────┘         └─────────────────-┘
```

This approach ensures your code will work on consortium data without modification.

------------------------------------------------------------------------

## 📊 Benchmark Tasks

FLAIR provides **5 clinically relevant prediction tasks** organised by a two-axis taxonomy: *task class* (what is predicted) × *temporal pattern* (when the prediction is made). A third practical axis — *prediction unit* — determines the shape of the prediction file each submission ships. The taxonomy is declared once on every task's `META` dict and mirrored in `paper/supplement.md §S0`.

### Task design space

|   | episode | continuous |
|------------------|------------------|----------------------------|
| **outcome**      | — | T1 (daily mortality), T2 (daily LTACH) |
| **deterioration**| — | T5 (AKI, Koyner-style) |
| **intervention** | T3 (extubation failure) | T4 (sepsis, antibiotic-anchored) |

### v1 task list

| \# | Name | task_class | temporal_pattern | prediction_unit | Type | Cohort filter |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1 | ICU Daily In-Hospital Mortality (7AM) | outcome | continuous | one-per-timestep | Binary | First ICU stay, discharge ∈ {LTACH, mortality} |
| 2 | ICU Daily LTACH Discharge (7AM) | outcome | continuous | one-per-timestep | Binary | First ICU stay, discharge ∈ {LTACH, mortality} |
| 3 | Extubation Failure within 24h | intervention | episode | one-per-stay | Binary | Confirmed first IMV extubation |
| 4 | Sepsis (antibiotic-anchored) within 6h | intervention | continuous | one-per-timestep | Binary | Adults, no psych/rehab, not admitted-on-treatment |
| 5 | AKI KDIGO-2 within 24h ([Koyner 2018](https://doi.org/10.1097/CCM.0000000000003123)) | deterioration | continuous | one-per-timestep | Binary | ≥ 1 SCr, first SCr \< 3.0 mg/dL, no RRT within 48h |

### Understanding Time Windows

Each task defines a **time window** for data extraction:

- **`window_start`**: Beginning of the data collection period
- **`window_end`**: The **prediction time** - this is when the model makes its prediction

**Critical**: You can use ALL data points within the window (`window_start` to `window_end`), but you **CANNOT** use any data after `window_end`. Using data beyond the prediction time would be data leakage.

Window shape varies by `prediction_unit`: - **`one-per-stay`** (T3): one window per hospitalization, anchored at the first confirmed extubation; `window_end = first_extubation_dttm` and `window_start = first_extubation_dttm − 24h`. - **`one-per-timestep`** (T1, T2, T4, T5): each row carries its own `prediction_dttm`; `window_end = prediction_dttm` and `window_start = prediction_dttm − 24h`. T1/T2 anchor on local-07:00 daily; T4 is hourly until first qualifying antibiotic; T5 is hourly until first KDIGO-2 onset.

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

### Cohort parquet (what `flair load-task` emits)

| Column               | Type      | Description                                |
|--------------------|------------------|-----------------------------------|
| `hospitalization_id` | str       | Unique identifier                          |
| `window_start`       | datetime  | Start of the input window                  |
| `window_end`         | datetime  | Prediction time — no data beyond this      |
| `{task_label}`       | int/float | Task-specific label (see each task's META) |
| `split`              | str       | "train" or "test"                          |
| `age_at_admission`   | int       | Patient age                                |
| `sex_category`       | str       | Patient sex                                |
| `race_category`      | str       | Patient race                               |
| `ethnicity_category` | str       | Patient ethnicity                          |

### Predictions parquet (what you write back, strict schema)

Append to the cohort parquet:

| Column | When required | Type |
|-------------------|------------------------------|------------------------|
| `y_true` | always | int/float |
| `y_pred` | binary, multiclass | int |
| `y_prob` | binary (recommended) | float in \[0, 1\] |
| `class_probs_<class>` | multiclass | float in \[0, 1\] (one column per class) |
| `y_pred_value` | regression | float |

### Report bundle (what `flair build-report` emits)

Four JSON files written to `--out`, each carrying audit metadata (`task_script_sha256`, `flair_version`, `generated_at`):

- `table1.json` — cohort characteristics by split and by label (HIPAA cell suppression: counts \< 10 → `"<10"`).
- `tripod_ai.json` — TRIPOD-AI report auto-populated with discrimination / calibration metrics.
- `fairness.json` — per-subgroup metrics (sex, race, ethnicity, age bin) plus parity gaps. Subgroups with n \< 10 are dropped entirely.
- `curves.json` — ROC / PR / calibration as downsampled raw points (no images). The central dashboard renders.

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

FLAIR is a CLI. Two commands cover the whole workflow.

### 1. Configure CLIF Data Source

``` bash
cp clif_config_template.json clif_config.json
```

Edit `clif_config.json` to set your data path, filetype, and timezone.

### 2. Build a task cohort

``` bash
flair list-tasks
# Prints name + task_class + temporal_pattern + prediction_unit + task_type
# for every discovered task (T1–T5).

flair load-task task1_icu_daily_mortality \
    --clif-config clif_config.json \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --out cohort.parquet
# (omit --train-end / --test-start for MIMIC: 70/30 temporal split is used)
```

### 3. Build features and train a model (your code)

``` python
import polars as pl
from clifpy import ClifOrchestrator

cohort = pl.read_parquet("cohort.parquet")
ids = cohort["hospitalization_id"].to_list()

co = ClifOrchestrator(data_directory="...", filetype="parquet", timezone="US/Central")
wide = co.create_wide_dataset(
    hospitalization_ids=ids,
    # ... vitals/labs/respiratory_support categories you want ...
)
# join wide → train your model → produce y_pred / y_prob.
```

Or query CLIF tables directly:

``` python
from clifpy.tables import Vitals
vitals = Vitals.from_file(
    data_directory="...", filetype="parquet", timezone="US/Central",
    filters={"hospitalization_id": ids,
             "vital_category": ["heart_rate", "sbp", "spo2"]},
)
```

### 4. Append predictions and run the report

``` python
preds = cohort.with_columns([
    pl.Series("y_true", y_true),
    pl.Series("y_pred", y_pred),
    pl.Series("y_prob", y_prob),  # binary tasks
])
preds.write_parquet("preds.parquet")
```

``` bash
flair build-report preds.parquet \
    --task task1_icu_daily_mortality \
    --out ./report/
# → report/table1.json table1.json fairness.json curves.json
```

See [examples/end_to_end.py](examples/end_to_end.py) for a runnable demo.

------------------------------------------------------------------------

## 🔒 Privacy model

FLAIR-Bench is one component of a larger federated stack. The library itself is intentionally minimal — it does not enforce network isolation or PHI runtime checks. Those guarantees come from two other layers:

1.  **CLIF-WorkBench container** — every site runs FLAIR inside a Singularity container with no internet, no root, immutable read-only image. That is where network egress is blocked at the kernel level.
2.  **Aggregate-only outputs** — patient-level data never leaves the site. `flair build-report` emits four JSON files of aggregated statistics; that is the only artifact a site shares.

What the library *does* enforce:

- HIPAA cell suppression in `table1.json` (counts \< 10 → `"<10"`).
- Subgroup drop in `fairness.json` (any group with n \< 10 is omitted entirely).
- Calibration bin suppression in `curves.json` (bins with count \< 10 report `null` for the true rate).
- Per-task script SHA-256 stamped into every output JSON so the central aggregator can verify two sites ran byte-identical task code.

PIs at each site still audit code before running it and have final approval. The simplicity of the new layout — each task = one polars file — makes that audit tractable.

------------------------------------------------------------------------

## 🏗️ Architecture

```         
flair_benchmark/
├── __init__.py                 # tiny: list_tasks, get_task, __version__
├── cli.py                      # the `flair` CLI: load-task, build-report, list-tasks
├── _clif.py                    # clifpy → polars loaders (Hospitalization, Adt, Patient,
│                               #   Vitals, Labs, RespiratorySupport)
├── _split.py                   # temporal date split / MIMIC 70-30 fallback
├── _icu_timing.py              # rank-based first/second ICU stay (shared default)
├── _hash.py                    # SHA-256 of a task .py for audit metadata
├── _aki.py                     # KDIGO stage-2 logic + Koyner exclusions (used by task5)
├── _ventilation_episodes.py    # consensus-window IMV episode detector (used by task3)
├── tasks/                      # one polars script per task — add a new task = drop a .py
│   ├── __init__.py             # importlib auto-discovery (no registry dict)
│   ├── task1_icu_daily_mortality.py
│   ├── task2_icu_daily_ltach.py
│   ├── task3_extubation_failure_24h.py
│   ├── task4_sepsis_abx_6h.py
│   └── task5_aki_kdigo2_24h.py
└── report/
    ├── __init__.py             # build_report() orchestrator
    ├── _suppress.py            # HIPAA cell suppression helper
    ├── table1.py               # cohort characteristics
    ├── tripod_ai.py            # TRIPOD-AI auto-populated
    ├── fairness.py             # per-subgroup metrics + parity gaps
    └── curves.py               # ROC / PR / calibration as raw points
```

### Adding a new task

Drop a new module under `flair_benchmark/tasks/` exposing:

``` python
META = { "name": ...,
         "task_class":       "outcome"|"deterioration"|"intervention"|"operational",
         "temporal_pattern": "episode"|"fixed_horizon"|"continuous",
         "prediction_unit":  "one-per-stay"|"one-per-decision-point"|"one-per-timestep",
         "task_type": "binary"|"multiclass"|"regression",
         "label_column": ..., "input_window_hours": ..., ... }

def build(clif_config: str,
          train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    """Return columns: hospitalization_id, window_start, window_end,
    <META['label_column']>, split, age_at_admission, sex_category,
    race_category, ethnicity_category."""
```

That is the entire contract. `flair list-tasks` auto-discovers it. See `task1_icu_daily_mortality.py` as the reference template.

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

- **Website**: [clif-icu.com](https://clif-icu.com)
- **Email**: clif_consortium\@uchicago.edu
- **Issues**: [GitHub Issues](https://github.com/clif-consortium/FLAIR/issues)