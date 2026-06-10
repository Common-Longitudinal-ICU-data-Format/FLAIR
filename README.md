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
| **outcome**      | T5 (unplanned ICU readmission) | T1 (daily mortality), T2 (daily LTACH) |
| **deterioration**| — | — |
| **intervention** | T3 (extubation failure) | T4 (sepsis, antibiotic-anchored) |

### v1 task list

| \# | Name | task_class | temporal_pattern | prediction_unit | Type | Cohort filter |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1 | ICU Daily In-Hospital Mortality (7AM) | outcome | continuous | one-per-timestep | Binary | First ICU stay, discharge ∈ {LTACH, mortality} |
| 2 | ICU Daily LTACH Discharge (7AM) | outcome | continuous | one-per-timestep | Binary | First ICU stay, discharge ∈ {LTACH, mortality} |
| 3 | Extubation Failure within 24h | intervention | episode | one-per-stay | Binary | Confirmed first IMV extubation |
| 4 | Sepsis (antibiotic-anchored) within 6h | intervention | continuous | one-per-timestep | Binary | Adults, no psych/rehab, not admitted-on-treatment |
| 5 | Unplanned ICU Readmission (end of first ICU stay) | outcome | episode | one-per-stay | Binary | First ICU stay ≥ 1h, survives first stay, not direct-discharged from ICU, no intermediate ER |

### Understanding the Prediction Time

Each row carries a single **`prediction_dttm`** — the moment the model makes its prediction.

**Critical**: You may use **ALL** data with `time ≤ prediction_dttm` (the entire available history for that stay — there is no fixed lookback window). You **CANNOT** use any data after `prediction_dttm`. Using data beyond the prediction time would be data leakage.

The prediction anchor varies by `prediction_unit`: - **`one-per-stay`** (T3, T5): one prediction per hospitalization. T3 anchors at the first confirmed extubation (`prediction_dttm = first_extubation_dttm`); T5 anchors at the end of the first ICU stay (`prediction_dttm = first_icu_end_time`). - **`one-per-timestep`** (T1, T2, T4): each row carries its own `prediction_dttm`. T1/T2 anchor on local-07:00 daily; T4 is hourly until first qualifying antibiotic.

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
| ✅ Demographics & prediction times | 🔧 Your evaluation          |

### Cohort parquet (what `flair load-task` emits)

| Column               | Type      | Description                                |
|--------------------|------------------|-----------------------------------|
| `hospitalization_id` | str       | Unique identifier                          |
| `prediction_dttm`    | datetime  | Prediction time — no data beyond this      |
| `{task_label}`       | int/float | Task-specific label (see each task's META) |
| `split`              | str       | "train" or "test"                          |
| `age_at_admission`   | int       | Patient age                                |
| `sex_category`       | str       | Patient sex                                |
| `race_category`      | str       | Patient race                               |
| `ethnicity_category` | str       | Patient ethnicity                          |
| `window_number`      | int       | 1-based k-th prediction timestep per stay (1 for one-per-stay tasks) |
| `prediction_id`      | str       | Stable row key `{patient_id}-{hospitalization_id}[-{window_number}]` — carry it through your predictions parquet |

### Add your predictions to the cohort file

There is **no separate predictions file**. The cohort parquet already carries the
label, `split`, demographics, `window_number`, and `prediction_id` — so you just add
your model output **as a new column on that same file** (one `with_columns` call).
This keeps every prediction aligned to its row; nothing can drift out of sync.

| Column | When required | Type |
|-------------------|------------------------------|------------------------|
| `y_prob` | binary (the one column to add) | float in \[0, 1\] |
| `y_pred` | binary: optional — derived from `y_prob` at 0.5 when absent; multiclass: required | int |
| `class_probs_<class>` | multiclass | float in \[0, 1\] (one column per class) |
| `y_pred_value` | regression | float |

The label is already in the file (no need to re-supply `y_true`).

### Report bundle (what `flair build-report` emits)

Seven JSON files written to `--out`, each carrying audit metadata (`task_script_sha256`, `flair_version`, `generated_at`):

- `table1.json` — cohort characteristics by split and by label (HIPAA cell suppression: counts \< 10 → `"<10"`).
- `tripod_ai.json` — TRIPOD-AI report auto-populated with discrimination / calibration metrics (incl. calibration slope/intercept, O/E ratio, Brier).
- `fairness.json` — per-subgroup metrics (sex, race, ethnicity, age bin) plus parity gaps (equalized odds) and a reference group. Subgroups with n \< 10 are dropped entirely.
- `curves.json` — ROC / PR / calibration as downsampled raw points, plus a smoothed logistic-recalibration calibration curve (no images). The central dashboard renders.
- `temporal.json` — binned discrimination: AUROC per day-in-stay window (1..7+, one-per-timestep tasks), AUROC trend by year of admission, and the predicted-risk trajectory by eventual outcome.
- `dca.json` — decision curve analysis: net benefit of the model vs treat-all/treat-none across thresholds 0.01–0.99, with the threshold range where the model is net-beneficial and the net benefit at the task's clinical threshold.
- `threshold_analysis.json` — fixed clinical-threshold analysis at each task's consulted decision point (task1 mortality > 0.95 → withdraw life support; task2 LTACH > 0.50 → early trach; task3 extubation failure < 0.15 → extubate; task4 sepsis > 0.10 → antibiotics; task5 readmission > 0.20 → keep in ICU): sensitivity/specificity/PPV/NPV + confusion matrix at the threshold, action metrics (direction-aware), partial AUROC within ±0.05 of the threshold, and a ±0.02/±0.05 robustness sweep. `fairness.json` additionally reports equalized-odds parity gaps at this clinical threshold.

Pass `--viz` to also write local sanity-check PNGs rendered from the same JSONs (`pip install flair-benchmark[viz]`). Pass `--cohort cohort.parquet` if your predictions parquet does not carry `window_number` / `prediction_dttm` / `prediction_id` through from the task cohort.

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

**Requirements**: Python 3.10+, clifpy, polars, typer, pyyaml

------------------------------------------------------------------------

## 🚀 Quick Start

FLAIR is a CLI. The whole workflow is **Configure → ① Make a task → ② Get data →
③ Train → ④ Report.** Commands:

| Command | Stage | What it does |
|---|---|---|
| `flair init-config` | — | Scaffold `clif_config.json` + `feature_config.yaml` templates |
| `flair list-tasks` | ① | List discovered tasks + taxonomy |
| `flair load-task` | ① | Build a task cohort parquet |
| `flair fe-meds` | ② | Assist: CLIF → MEDS/ELF med event stream |
| `flair fe-wide` | ② | Assist: CLIF → wide feature set as ELF events |
| `flair build-report` | ④ | Aggregate report bundle (7 JSONs, optional `--viz` PNGs) |
| `flair extract-metadata` | — | Per-site cohort metadata for the leaderboard |

### Configure the CLIF data source

``` bash
flair init-config            # writes clif_config.json + feature_config.yaml here
```

Edit `clif_config.json` to set your data path, filetype, and timezone. Edit
`feature_config.yaml` to pick the mCIDE categories for `fe-wide` / `fe-meds`.

------------------------------------------------------------------------

### ① Task making — `flair load-task`

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

The cohort parquet (schema in [Cohort parquet](#cohort-parquet-what-flair-load-task-emits))
carries `hospitalization_id`, `prediction_dttm` (the no-data-after time),
the label, `split`, and demographics. Every downstream stage keys off it.

------------------------------------------------------------------------

### ② Getting data — three options

From the cohort, turn CLIF tables into model inputs. Pick one:

| Option | How | Output |
|---|---|---|
| **A · Direct FE** | your own code over cohort + `clifpy` | whatever you build |
| **B · Assist: CLIF as MEDS** | `flair fe-meds` / `get_meds_stream` | ELF long med events |
| **C · Assist: CLIF as wide dataset** | `flair fe-wide` / `get_wide_feature_dataset` | ELF long feature events |

Both assist functions are **config-driven** (`feature_config.yaml`), **always
load via clifpy**, **write parquet**, strip timezone to wall-clock, and emit the
**same ELF long schema** (see [ELF output schema](#elf-output-schema)). They
guarantee **no future-data leakage** (no event after a stay's prediction point)
and **chart the unit** on every numeric event so a downstream step can convert.

#### Option A — Direct FE (bring your own)

Full control: read the cohort, pull whatever CLIF data you want, engineer
features your way.

``` python
import polars as pl
from clifpy import ClifOrchestrator

cohort = pl.read_parquet("cohort.parquet")
ids = cohort["hospitalization_id"].to_list()

co = ClifOrchestrator(data_directory="...", filetype="parquet", timezone="US/Central")
co.create_wide_dataset(
    tables_to_load=["vitals", "labs"],
    category_filters={"vitals": ["heart_rate", "sbp"], "labs": ["creatinine"]},
    hospitalization_ids=ids,
)
wide = co.wide_df  # join → engineer → train → y_pred / y_prob
```

Or query a single CLIF table directly:

``` python
from clifpy.tables import Vitals
vitals = Vitals.from_file(
    data_directory="...", filetype="parquet", timezone="US/Central",
    filters={"hospitalization_id": ids,
             "vital_category": ["heart_rate", "sbp", "spo2"]},
)
```

#### Option B — Assist: CLIF as MEDS (`flair fe-meds`)

Lightweight pull of the configured med categories into an ELF event stream —
**raw dose + raw unit, no unit conversion / weight join / dose normalization**
(the raw unit is charted so you can convert later).

``` bash
flair fe-meds \
    --feature-config feature_config.yaml \
    --task cohort.parquet \
    --out features/meds.parquet
```

``` text
subject_id | time             | code                        | numeric_value | unit
P101       | 2150-03-01 04:00 | MED_INT//vancomycin//mg     | 1000          | mg
P101       | 2150-03-01 06:30 | MED_CON//norepinephrine//.. | 0.04          | mcg/kg/min
```

#### Option C — Assist: CLIF as wide dataset (`flair fe-wide`)

Pulls **all configured time-series features** (vitals, labs, meds,
patient_assessments, respiratory_support) via clifpy's wide dataset, truncates
each stay at its prediction point, and emits the same ELF long schema.

``` bash
flair fe-wide \
    --feature-config feature_config.yaml \
    --task cohort.parquet \
    --out features/wide.parquet
```

``` text
subject_id | time             | code                   | numeric_value | unit
P101       | 2150-03-01 06:10 | VITAL//heart_rate//bpm | 88            | bpm
P101       | 2150-03-01 06:30 | LAB//creatinine//mg/dl | 1.8           | mg/dl
```

> ⚠️ **Scale note:** clifpy loads full CLIF tables before filtering, so `fe-wide`
> over a full-MIMIC cohort is memory/time-heavy. For large cohorts prefer Option
> A with pre-sharded tables, or run `fe-wide` on a cohort subset.

#### `feature_config.yaml`

Both assist functions read one YAML (example at the repo root). It points at the
`clif_config.json`, lists the mCIDE categories per table, and optionally
overrides units. Unknown categories warn (not fail), validated against clifpy
schemas.

``` yaml
clif_config: clif_config_mimic.json        # the JSON clif config (site/data_dir/tz)
wide:                                        # -> fe-wide
  vitals:  {categories: [heart_rate, sbp, spo2, map]}
  labs:    {categories: [creatinine, lactate, wbc]}
  medication_admin_continuous: {categories: [norepinephrine, propofol]}
  respiratory_support: {columns: [device_category, fio2_set, peep_set]}
meds_stream:                                 # -> fe-meds
  medication_admin_continuous:   {categories: [norepinephrine, propofol]}
  medication_admin_intermittent: {categories: [vancomycin, cefepime]}
# units: {labs: {creatinine: mg/dl}}         # optional overrides
```

#### ELF output schema

Both `fe-meds` and `fe-wide` emit the same MEDS/ELF long event table:

| Column | Type | Notes |
|---|---|---|
| `subject_id` | str | = `patient_id`; always present |
| `time` | datetime | tz-naive wall-clock; ≤ stay's prediction point |
| `code` | str | `"<DOMAIN>//<category>//<unit>"` (e.g. `LAB//creatinine//mg/dl`) |
| `numeric_value` | float | null for categorical events |
| `unit` | str | charted wherever `numeric_value` is set |
| `text_value` | str | for categorical events (e.g. device category) |
| `hospitalization_id` | str | always present |

------------------------------------------------------------------------

### ③ Train a model (your code)

Train on your features → add a single `y_prob` column to the cohort file:

``` python
import polars as pl
cohort = pl.read_parquet("cohort.parquet")
preds = cohort.with_columns(pl.Series("y_prob", y_prob))  # binary tasks
# y_pred is derived from y_prob at 0.5; add your own pl.Series("y_pred", ...) to override.
preds.write_parquet("preds.parquet")
```

------------------------------------------------------------------------

### ④ Reporting tools — `flair build-report`

``` bash
flair build-report preds.parquet \
    --task task1_icu_daily_mortality \
    --out ./report/ \
    --viz                       # optional sanity-check PNGs (flair-benchmark[viz])
# → report/table1.json  tripod_ai.json  fairness.json  curves.json
#   temporal.json  dca.json  [viz/*.png]
```

Emits the aggregate-only [report bundle](#report-bundle-what-flair-build-report-emits)
— seven JSONs of cohort, TRIPOD-AI, fairness, curve, temporal-discrimination, threshold, and
decision-curve statistics, with HIPAA cell suppression. Patient-level data never
leaves the site.

See [examples/end_to_end.py](examples/end_to_end.py) for a runnable demo.

------------------------------------------------------------------------

## 🔒 Privacy model

FLAIR-Bench is one component of a larger federated stack. The library itself is intentionally minimal — it does not enforce network isolation or PHI runtime checks. Those guarantees come from two other layers:

1.  **CLIF-WorkBench container** — every site runs FLAIR inside a Singularity container with no internet, no root, immutable read-only image. That is where network egress is blocked at the kernel level.
2.  **Aggregate-only outputs** — patient-level data never leaves the site. `flair build-report` emits seven JSON files of aggregated statistics; that is the only artifact a site shares.

What the library *does* enforce:

- HIPAA cell suppression in `table1.json` (counts \< 10 → `"<10"`).
- Subgroup drop in `fairness.json` (any group with n \< 10 is omitted entirely).
- Calibration bin suppression in `curves.json` (bins with count \< 10 report `null` for the true rate).
- Temporal bin drop in `temporal.json` (window/year bins with n \< 10 omitted; AUROC requires ≥ 10 of each class) and DCA point suppression in `dca.json` (thresholds flagging \< 10 patients report `null`).
- Per-task script SHA-256 stamped into every output JSON so the central aggregator can verify two sites ran byte-identical task code.

PIs at each site still audit code before running it and have final approval. The simplicity of the new layout — each task = one polars file — makes that audit tractable.

------------------------------------------------------------------------

## 🏗️ Architecture

```         
flair_benchmark/
├── __init__.py                 # tiny: list_tasks, get_task, __version__
├── cli.py                      # the `flair` CLI: init-config, load-task, fe-wide,
│                               #   fe-meds, build-report, extract-metadata, list-tasks
├── _templates.py               # config templates shipped in the wheel (flair init-config)
├── _clif.py                    # clifpy → polars loaders (Hospitalization, Adt, Patient,
│                               #   Vitals, Labs, RespiratorySupport, meds) + pd_to_pl (tz strip)
├── _split.py                   # temporal date split / MIMIC 70-30 fallback
├── _icu_timing.py              # rank-based first/second ICU stay (shared default)
├── _hash.py                    # SHA-256 of a task .py for audit metadata
├── _readmission.py             # ICU/Ward/ER/Procedural ADT collapse + sandwich-merge first/second ICU stay (used by task5)
├── _ventilation_episodes.py    # consensus-window IMV episode detector (used by task3)
├── tasks/                      # one polars script per task — add a new task = drop a .py
│   ├── __init__.py             # importlib auto-discovery (no registry dict)
│   ├── task1_icu_daily_mortality.py
│   ├── task2_icu_daily_ltach.py
│   ├── task3_extubation_failure_24h.py
│   ├── task4_sepsis_abx_6h.py
│   └── task5_icu_readmission.py
├── features/                   # CLIF → MEDS/ELF feature extractors (config-driven)
│   ├── __init__.py             # get_wide_feature_dataset, get_meds_stream
│   ├── _config.py              # load feature_config.yaml + mCIDE category validation
│   ├── _elf.py                 # ELF long schema + invariant checks
│   ├── _units.py               # per-category unit lookup (labs schema, vitals, resp)
│   ├── wide.py                 # fe-wide: clifpy wide dataset → ELF long
│   └── meds_stream.py          # fe-meds: raw med pull (no conversion) → ELF long
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
    """Return columns: hospitalization_id, prediction_dttm,
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