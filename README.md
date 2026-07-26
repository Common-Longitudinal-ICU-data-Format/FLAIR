# FLAIR

[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CLIF](https://img.shields.io/badge/data-CLIF_format-orange.svg)](https://clif-icu.com)

**FLAIR** is a CLIF-based benchmark toolkit for ICU prediction research.

It helps teams build the same prediction tasks at every site, run models where
the clinical data already lives, and share only aggregate evaluation reports.
That makes it practical to study whether an ICU model works beyond one dataset,
one hospital, or one documentation workflow.

FLAIR is designed for a collaborative loop:

1. Develop and test code on public **MIMIC-CLIF** data.
2. Review the code with the clinical and site teams.
3. Run the same code locally at each CLIF site.
4. Return aggregate JSON reports, not patient-level rows.

Public ICU datasets and benchmarks are important foundations for clinical ML.
FLAIR builds on that work by adding a path for private, multi-site evaluation
using the same CLIF schema.

![FLAIR workflow](assets/FlairWorkflow.gif)

## What FLAIR Gives You

FLAIR is a Python package and CLI for benchmark construction and evaluation.

| FLAIR provides | You provide |
|---|---|
| Task definitions and cohort builders | Your model or method |
| Consistent labels and prediction times | Your feature engineering choices |
| Train/test split support | Your training and inference code |
| Optional CLIF-to-event feature extractors | Your modeling pipeline |
| Aggregate report generation | Your interpretation and validation plan |

The core idea is simple: FLAIR defines **what should be predicted**, **when the
prediction is made**, and **how results are reported**. You decide how to build
the model.

## The Workflow

Most projects follow this path:

```text
Configure -> Build a cohort -> Get features -> Train -> Report
```

| Stage | Command | What happens |
|---|---|---|
| Configure | `flair init-config` | Write editable CLIF and feature config templates |
| Build a cohort | `flair load-task` | Create one benchmark cohort parquet and a Table 1 JSON |
| Get features | `flair fe-meds` | Optionally extract a leak-guarded ELF medication event table |
| Train | your code | Fit a model and add predictions to the cohort rows |
| Report | `flair build-report` | Write aggregate JSON metrics, with optional local PNGs |

The most important rule is the prediction time:

> For each row, a model may use data with `time <= feature_cutoff_dttm` and
> nothing after it.

That timestamp is the no-future-data boundary for the row. Every cohort carries it,
and it equals `prediction_dttm` for all but one task. A task sets it *earlier* when
its own prediction timestamp would leak the outcome — `extubation_failure_24h`
anchors on the respiratory-support row that announces the extubation, so it cuts
strictly before, and a model asked "should we extubate?" never sees the extubation.
FLAIR task builders create the cutoff, feature extractors honor it, and reports
assume predictions respect it.

## Installation

```bash
pip install flair-benchmark
```

For local sanity-check plots during report building:

```bash
pip install "flair-benchmark[viz]"
```

Requirements:

- Python 3.10+
- CLIF-formatted ICU data
- `clifpy`, `polars`, `pandas`, `numpy`, `scikit-learn`, `typer`, and `pyyaml`

Install from source when developing the package:

```bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
cd FLAIR
pip install -e ".[dev,viz]"
```

## Quick Start

Create config files:

```bash
flair init-config --out .
```

Edit `clif_config.json` so FLAIR knows where your CLIF files live:

```json
{
  "site": "sitename",
  "data_directory": "/path/to/clif/data",
  "filetype": "parquet",
  "timezone": "US/Central",
  "stitch_time_interval_hours": 6,
  "cache_directory": "./output"
}
```

List the available tasks:

```bash
flair list-tasks
```

Build a benchmark cohort:

```bash
flair load-task icu_daily_mortality \
  --clif-config clif_config.json \
  --out cohort.parquet
```

This writes:

- `cohort.parquet`: one row per prediction point
- `cohort.table1.json`: aggregate cohort characteristics

Optionally extract CLIF medication features into an ELF-long event table:

```bash
flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out features.parquet
```

Train your model in your own code. For binary tasks, the simplest valid output
is the cohort with one added probability column:

```python
import polars as pl

cohort = pl.read_parquet("cohort.parquet")

# y_prob must align row-for-row with the cohort.
preds = cohort.with_columns(pl.Series("y_prob", y_prob))
preds.write_parquet("preds.parquet")
```

Build an aggregate report:

```bash
flair build-report preds.parquet \
  --task icu_daily_mortality \
  --cohort cohort.parquet \
  --out ./report/
```

The mode is inferred from the task: continuous tasks use `--mode continuous`
(peak section + per-landmark pillars + leadtime + an `operating_points` section
— a per-threshold confusion matrix plus optional intervention sub-analyses — in
one bundle), episode tasks use `--mode episodic`.

## Benchmark Tasks

FLAIR currently includes four binary ICU prediction tasks.

| Task | Clinical question | Pattern | Unit | Recommended report |
|---|---|---|---|---|
| `icu_daily_mortality` | Daily in-hospital mortality risk at 07:00 | continuous | one row per window | `continuous` (30-day landmark horizon) |
| `icu_daily_ltach` | Daily LTACH discharge risk at 07:00 | continuous | one row per window | `continuous` (30-day landmark horizon) |
| `extubation_failure_24h` | Extubation failure within 24 hours | episode | one row per stay | `episodic` |
| `icu_readmission` | Unplanned ICU readmission after first ICU stay | episode | one row per stay | `episodic` |

Task metadata records the task class, temporal pattern, prediction unit, label
column, clinical threshold, threshold direction, and recommended report mode.
Each task's clinical policy (threshold, direction, report mode) is declared inline in its `META` in `flair_benchmark/tasks/<name>.py`.

For more detail, see [docs/101/tasks.md](docs/101/tasks.md).

## Cohort Contract

`flair load-task` writes a cohort parquet. Every row is a prediction opportunity.

Core columns:

| Column | Meaning |
|---|---|
| `hospitalization_id` | Source hospitalization key |
| `hospitalization_join_id` | Stitched encounter key used for stay-level grouping |
| `prediction_dttm` | The no-data-after timestamp |
| `<label_column>` | Task-specific outcome label |
| `split` | `train` or `test` |
| `age_at_admission` | Age used for reporting and subgroup analysis |
| `sex_category` | Sex category used for reporting and subgroup analysis |
| `race_category` | Race category used for reporting and subgroup analysis |
| `ethnicity_category` | Ethnicity category used for reporting and subgroup analysis |
| `window_number` | One-based prediction window within the stay |
| `prediction_id` | Auditable prediction-level key |

`prediction_id` is built from the stitched encounter, hospitalization, patient,
and window number. Carry it through your prediction file. It is the safest way
to keep model outputs aligned with the exact row being scored.

For MIMIC-style development, omitting `--train-end` and `--test-start` uses the
deterministic MIMIC split when `site` is set to `"mimic"` in `clif_config.json`.
For site data, pass explicit temporal boundaries when you need a fixed train/test
split:

```bash
flair load-task icu_daily_mortality \
  --clif-config clif_config.json \
  --train-end 2022-12-31 \
  --test-start 2023-01-01 \
  --out cohort.parquet
```

## Feature Options

FLAIR supports two ways to turn CLIF data into model inputs.

| Mode | How it works | Best for |
|---|---|---|
| Direct | Use `cohort.parquet` with `clifpy` or your own CLIF loaders. Wide time-series (vitals, labs, assessments, respiratory support) goes through clifpy's `ClifOrchestrator.create_wide_dataset` — **you** apply the `feature_cutoff_dttm` filter | Maximum control; everything except meds |
| `fe-meds` | Extract configured medication events into ELF-long parquet | Lightweight, leak-guarded medication streams |

The assisted extractor:

- reads `feature_config.yaml`
- loads CLIF data through `clifpy`
- truncates events at each row's prediction point
- writes parquet
- emits the ELF-long schema

Example:

```bash
flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out meds.parquet
```

ELF-long output columns:

| Column | Meaning |
|---|---|
| `subject_id` | Patient identifier |
| `time` | Event time, truncated to the prediction boundary |
| `code` | Event code such as `LAB//creatinine//mg/dl` |
| `numeric_value` | Numeric value when available |
| `unit` | Unit for numeric values |
| `text_value` | Text value for categorical events |
| `hospitalization_id` | Source hospitalization key |
| `hospitalization_join_id` | Stitched encounter key |

The assisted extractors are conveniences, not requirements. If your method needs
custom features, use the cohort and CLIF tables directly.

For more detail, see [docs/101/features.md](docs/101/features.md).

## Prediction Contract

The report builder validates prediction files before scoring.

All prediction files must include:

- `hospitalization_id`
- `prediction_id`
- `split`
- the task label column, such as `label_mortality`

Binary tasks also require:

- `y_prob`: probability in `[0, 1]`

`y_pred` is optional for binary tasks. If it is absent, FLAIR derives it from
`y_prob >= 0.5`. For multiclass and regression tasks, use the task-specific
prediction columns required by the report builder.

The recommended pattern is to keep the cohort rows intact and add model output
columns to that same table. That avoids row drift and keeps reporting auditable.

## Reports

`flair build-report` turns predictions into aggregate JSON files.

```bash
flair build-report preds.parquet \
  --task extubation_failure_24h \
  --cohort cohort.parquet \
  --mode episodic \
  --out ./report/ \
  --no-viz
```

Report modes:

| Mode | Use for | Output focus |
|---|---|---|
| `episodic` | One prediction per stay | Four **threshold-free** pillars (discrimination, calibration, decision curves, fairness) + an `operating_points` scorecard holding every threshold-dependent metric — confusion matrix, sensitivity/specificity/PPV/NPV/F1/FPR/NNE/alert-rate, the act taken and its net benefit, and subgroup parity gaps — swept over the clinical threshold and ± 0.05 / ± 0.10 |
| `continuous` | Multi-row-per-stay tasks | One bundle: stay-peak screening section + forward landmarks (time-varying AUC/calibration per hospitalization day/hour over `META['landmark_horizon']` times, KM at-risk counts) + leadtime + an `operating_points` section (per-threshold alert×label confusion matrix — clinical ± 0.05 — plus optional intervention sub-analyses) + a `kpi.json` scoreboard (pooled dynamic AUROC, NNE, calibration, lead time) |

Reports are computed on the held-out `test` split. JSON outputs include audit
metadata such as the task name, FLAIR version, generation time, report mode, and
task script SHA-256.

Depending on mode, the report directory can include JSON files for:

- discrimination
- calibration
- decision curve analysis
- fairness
- lead time
- threshold or operating-point behavior

For a task whose action fires on LOW risk (`clinical_direction: "below"`), the
decision curve is built on the **complement** of the label. `extubation_failure_24h`
is the live example: it predicts extubation *failure*, but the decision is whether
to *extubate*, which fires when failure risk is low. Its DCA event is therefore
*successful extubation* and its x-axis is the probability of success a clinician
would require before extubating — 0 (extubate everyone) to 1 (keep everyone
intubated), placing its 0.10 policy at x = 0.90. Note that only the frame is
complemented: the model still emits P(failure), and AUROC / AUPRC / calibration /
the confusion matrix all stay on the failure label.

For an `"above"` task (e.g. `icu_readmission` "keep in ICU when readmission risk
≥ 0.20") the DCA event is the label event itself and the policy threshold appears
directly on the x-axis. `dca.json`'s `metadata.x_axis` names the axis, its
`metadata.action` / `metadata.inaction` name the two arms of the decision, and its
`metrics.verdict` states whether the model beats acting on everyone / no one.

When visualization support is installed, `--viz` writes local PNGs from the same
JSON data. These PNGs are for site review; the portable artifact is the JSON
bundle.

For more detail, see [docs/101/reports.md](docs/101/reports.md).

## Privacy Model

FLAIR is built for a model-to-data workflow. The library helps standardize tasks,
features, and reports, but it is not a complete security boundary by itself.

In a federated run:

- Code is reviewed before execution.
- Sites run the code locally against their CLIF data.
- Patient-level rows stay at the site.
- Shared outputs are aggregate JSON reports.

The library also applies reporting safeguards, including:

- small-cell suppression for Table 1 counts
- subgroup omission for fairness groups below the minimum size
- suppressed or omitted metric bins when counts are too small
- task script hashes in report metadata for auditability

Network isolation, PHI runtime controls, and site approval live in the broader
deployment and governance process. FLAIR keeps its own role narrow: build the same
benchmark everywhere and report it consistently.

## Project Layout

```text
flair_benchmark/
  cli.py                  # the `flair` command
  _constants.py           # package version and task policy
  _clif.py                # CLIF loading helpers
  _split.py               # train/test split helpers
  tasks/                  # one Python module per benchmark task
  features/               # CLIF-to-ELF feature extractors
  meds_etl/               # MEDS/ELF concept and transform helpers
  report/                 # aggregate report builders
```

The public user surface is the CLI. Python helpers are available for advanced
users, but the stable workflow is command-first.

## Adding A Task

A task is a Python module under `flair_benchmark/tasks/` with:

- `META`: task metadata and benchmark policy
- `build(clif_config, train_end, test_start)`: a function that returns the cohort

The task catalog is auto-discovered, so adding a task does not require editing a
central registry. New tasks should define the cohort, label, prediction time,
prediction unit, and reporting policy clearly enough for both clinical review and
site audit.

Start with:

- `flair_benchmark/tasks/icu_daily_mortality.py` for a continuous task
- `flair_benchmark/tasks/extubation_failure_24h.py` for an episode task

For the task authoring guide, see [docs/101/tasks.md](docs/101/tasks.md).

## Documentation

The README is the project manual. The 101 docs go deeper by topic:

| Topic | Link |
|---|---|
| End-to-end overview | [docs/101.md](docs/101.md) |
| CLIF data layer | [docs/101/clif.md](docs/101/clif.md) |
| Tasks and task authoring | [docs/101/tasks.md](docs/101/tasks.md) |
| Feature engineering | [docs/101/features.md](docs/101/features.md) |
| Reports and report modes | [docs/101/reports.md](docs/101/reports.md) |

Task schematics are available under [docs/task_schematics](docs/task_schematics/).

## Related Projects

| Project | Role |
|---|---|
| [CLIF](https://clif-icu.com) | Common Longitudinal ICU Format |
| [clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) | Python tooling for CLIF data |
| [MIMIC-CLIF](https://physionet.org/) | Public CLIF-formatted development dataset |

## Citation

If you use FLAIR in research, cite the project:

```bibtex
@software{flair2024,
  title = {FLAIR: Federated Learning Assessment for ICU Research},
  author = {CLIF Consortium},
  year = {2024},
  url = {https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).

FLAIR does not own or distribute the clinical datasets used with the benchmark.
Access to private CLIF consortium data requires approval from the participating
institutions.

## Contact

- Website: [clif-icu.com](https://clif-icu.com)
- Email: clif_consortium@uchicago.edu
- Issues: [GitHub Issues](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/issues)
