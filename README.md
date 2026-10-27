<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/FLAIR_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/FLAIR_logo_light.png">
    <img alt="FLAIR" src="assets/FLAIR_logo_light.png" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green.svg"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://clif-icu.com"><img alt="CLIF" src="https://img.shields.io/badge/data-CLIF_format-orange.svg"></a>
</p>

<h3 align="center">Build the same ICU prediction benchmark at every hospital. Share results, not patients.</h3>

FLAIR is a Python package and command-line tool for ICU prediction research on [CLIF](https://clif-icu.com)-formatted data. It defines the prediction tasks, builds the cohorts, and writes the evaluation reports — so several hospitals can run the *same* benchmark on their own data and compare evals.

![FLAIR workflow](assets/FlairWorkflow.gif)

## Why FLAIR

A model that works great on one dataset often falls apart at the next hospital. Testing that properly is hard, because patient data usually cannot leave the building.

FLAIR takes the other route: **send the code to the data.**

1.  Write and test your code on public **MIMIC-CLIF** data.
2.  Have the clinical and site teams review it.
3.  Each CLIF site runs the same code on its own data, locally.
4.  Sites send back aggregate JSON reports — never patient-level rows.

## What FLAIR Does and Doesn't Do

| FLAIR handles                        | You handle                         |
|-------------------------------------|-----------------------------------|
| Task definitions and cohort building | Your model or method               |
| Labels and prediction timestamps     | Your feature engineering           |
| Train/test splits                    | Your training and inference code   |
| Optional CLIF feature extractors     | Your modeling pipeline             |
| Aggregate report generation          | Your interpretation and validation |

In short: FLAIR fixes **what** is predicted, **when** it is predicted, and **how** results are reported. Everything about the model is yours.

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [How It Works](#how-it-works)
- [The Prediction Time Rule](#the-prediction-time-rule)
- [Benchmark Tasks](#benchmark-tasks)
- [The Cohort File](#the-cohort-file)
- [Features](#features)
- [Predictions](#predictions)
- [Reports](#reports)
- [Privacy](#privacy)
- [Project Layout](#project-layout)
- [Adding a Task](#adding-a-task)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## Install

``` bash
pip install flair-benchmark
```

Submitting to the leaderboard? Install the pinned `clifpy` / `polars` versions so every site runs the same library code (both versions are recorded in every Table 1 and report audit):

``` bash
pip install "flair-benchmark[benchmark]"
```

You need:

- Python 3.10+
- ICU data in CLIF format
- `clifpy`, `polars`, `pandas`, `numpy`, `scikit-learn`, `threadpoolctl`, `typer`, `pyyaml`, `matplotlib` (installed for you)

Developing on FLAIR itself:

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
cd FLAIR
pip install -e ".[dev]"
```

## Quick Start

**1. Create the config files**

``` bash
flair init-config --out .
```

This writes two templates into the current directory: `clif_config.json` and `feature_config.yaml`. Existing files are left alone; pass `--force` to overwrite them.

**2. Tell FLAIR where your CLIF data lives** — edit `clif_config.json`:

``` json
{
    "site": "sitename",
    "data_directory": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central",
    "stitch_time_interval_hours": 6,
    "cache_directory": "./output/_flair_cache"
}
```

At minimum set `site`, `data_directory`, and `timezone` to your own values. Every key is explained in [Configuration](#configuration).

**3. See what tasks exist**

``` bash
flair list-tasks
```

**4. Build a cohort**

``` bash
flair load-task icu_daily_mortality \
  --clif-config clif_config.json \
  --out output
```

`--out` is a root folder (relative to where you run `flair`). It gets two site-prefixed folders:

- `output/<site>_flair_phi/icu_daily_mortality/cohort.parquet` — one row per prediction point. **PHI: stays at the site.**
- `output/<site>_flair_non_phi_for_upload/icu_daily_mortality/` — safe to share:
  - `table1.json` — aggregate cohort characteristics; `attrition` (STROBE counts after each inclusion/exclusion step); and `metadata.cohort_sha256` / `train_cohort_sha256` / `test_cohort_sha256` (SHA-256 over `hospitalization_join_id, prediction_dttm, feature_cutoff_dttm, label, split`, sorted by join id + prediction time)
  - `strobe.png` / `strobe_attrition.png` — the STROBE flow diagram and per-step attrition chart

**5. Train your model, your way**

The simplest valid output is the cohort with one extra probability column:

``` python
import polars as pl

cohort = pl.read_parquet("output/<site>_flair_phi/icu_daily_mortality/cohort.parquet")

# y_prob must line up row-for-row with the cohort.
preds = cohort.with_columns(pl.Series("y_prob", y_prob))
preds.write_parquet("preds.parquet")
```

**6. Build the report**

``` bash
flair build-report preds.parquet \
  --task icu_daily_mortality \
  --cohort cohort.parquet \
  --out ./report/
```

That's the whole loop.

## Configuration

FLAIR reads two config files. Both come from `flair init-config` and live wherever you like; you pass their paths on the command line.

### `clif_config.json`

Used by `flair load-task` (`--clif-config`) and referenced by `feature_config.yaml`.

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `site` | yes | — | Your site name. Use `mimic` for MIMIC-CLIF: it switches on the random MIMIC split and skips the real-date benchmark window, because MIMIC dates are shifted. |
| `data_directory` | yes | — | Folder holding your CLIF tables. FLAIR only reads from it. |
| `filetype` | yes | — | File format of the CLIF tables, e.g. `parquet`. |
| `timezone` | yes | — | Your site's IANA timezone, e.g. `US/Eastern`. There is no default: a wrong zone shifts every timestamp, including the 07:00 daily prediction time. |
| `stitch_time_interval_hours` | no | `6` | Gap, in hours, under which back-to-back hospitalizations are stitched into one encounter. |
| `cache_directory` | no | `./output/_flair_cache` | Where FLAIR caches the encounter index and cohort scores. Relative paths resolve against the directory you run `flair` from. |

FLAIR never writes into `data_directory`.

### `feature_config.yaml`

Used by `flair fe-meds` (`--feature-config`).

| Key | Meaning |
|-----|---------|
| `clif_config` | Path to your `clif_config.json`. A relative path resolves against the YAML file's own folder. |
| `output_directory` | Conventional output folder; `--out` decides where the file is written. |
| `meds_stream` | Medication categories to extract, per table (`medication_admin_continuous`, `medication_admin_intermittent`). Values are CLIF mCIDE categories; unknown ones warn but don't fail. |

### What is not configurable

Task definitions, clinical thresholds, report settings, and the global benchmark window (encounters whose every admission and discharge falls within **2017-12-31 – 2025-12-31**, fixed in `constants.py`; at every non-MIMIC site an encounter with any leg outside it is dropped whole) are fixed in code on purpose. They are what makes two sites' results comparable, and the task definitions are folded into the SHA-256 that every report carries.

### Environment variables

Optional performance settings for the vendored CLIF→MEDS ETL (`flair_benchmark.meds_etl`, the Python `build_meds_tables` path). `flair fe-meds` reads neither, and a `max_workers` key in the ETL config overrides `FLAIR_MEDS_WORKERS`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLAIR_MEDS_WORKERS` | `min(4, number of domains)` | Threads used to extract domains in parallel. |
| `FLAIR_PUSHDOWN_MAX_IDS` | `50000` | Above this many cohort ids, FLAIR loads the table and filters in memory instead of pushing the id filter down to the file scan. |

## CLI Reference

| Command | What it does |
|---------|--------------|
| `flair init-config [--out DIR] [--force]` | Write the `clif_config.json` and `feature_config.yaml` templates. |
| `flair list-tasks` | List the benchmark tasks. |
| `flair load-task TASK --clif-config PATH --out DIR` | Build a task cohort into `DIR/<site>_flair_phi/TASK/`, and `table1.json` (STROBE `attrition`, cohort hashes) + STROBE PNGs into `DIR/<site>_flair_non_phi_for_upload/TASK/`. |
| `flair fe-meds --feature-config PATH --task cohort.parquet --out meds.parquet` | Extract a medication event table, cut per encounter at its latest `feature_cutoff_dttm`. |
| `flair build-report PREDS --task TASK --out DIR [--cohort cohort.parquet] [--mode episodic\|continuous]` | Score predictions into the aggregate JSON report bundle. |

Run `flair COMMAND --help` for every option.

## How It Works

``` text
Configure -> Build a cohort -> Get features -> Train -> Report
```

| Stage | Command | What happens |
|------------------------|------------------------|------------------------|
| Configure | `flair init-config` | Writes editable CLIF and feature config templates |
| Build a cohort | `flair load-task` | Writes one cohort parquet + a Table 1 JSON |
| Get features | `flair fe-meds` | Optional: pulls a medication event table (you still apply the per-row cutoff) |
| Train | your code | Fit a model, add predictions to the cohort rows |
| Report | `flair build-report` | Writes aggregate JSON metrics (+ optional local PNGs) |

## The Prediction Time Rule

This is the one rule that matters most:

> For each row, a model may use data where `time <= feature_cutoff_dttm`, and nothing after it.

That timestamp is the **no-future-data line** for that row. Every cohort row carries one. Task builders create it, feature extractors respect it, and reports assume your predictions did too.

For every shipped task, `feature_cutoff_dttm` equals `prediction_dttm`. The contract allows a task to set the cutoff *earlier* if its own prediction time would leak the answer.

`extubation_failure_24h` solves the leak a different way — it moves the prediction itself. The prediction is stamped **30 minutes before** the extubation, so a model asked "should we extubate?" never sees the respiratory-support row that announces the extubation happened.

## Benchmark Tasks

Four binary ICU prediction tasks ship with FLAIR:

| Task | Clinical question | Pattern | Unit | Report mode |
|---------------|---------------|---------------|---------------|---------------|
| `icu_daily_mortality` | Daily in-hospital mortality risk at 07:00 | continuous | one row per window | `continuous` (30-day landmark) |
| `icu_daily_ltach` | Daily LTACH discharge risk at 07:00 | continuous | one row per window | `continuous` (30-day landmark) |
| `extubation_failure_24h` | Extubation failure within 24 hours | episode | one row per stay | `episodic` |
| `icu_readmission` | Unplanned ICU readmission after first ICU stay | episode | one row per stay | `episodic` |

Each task records its own metadata: task class, temporal pattern, prediction unit, label column, clinical threshold, threshold direction, and recommended report mode (`report_mode` is validated and folded into the audit hash; `build-report` itself picks the mode from `--mode` or `temporal_pattern`). The clinical policy lives inline in that task's `META`, in `flair_benchmark/tasks/<name>.py`.

More detail: [docs/101/tasks.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/tasks.md).

## The Cohort File

`flair load-task` writes a cohort parquet where **every row is one chance to predict**.

| Column | Meaning |
|------------------------------------|------------------------------------|
| `hospitalization_id` | Source hospitalization key |
| `hospitalization_join_id` | Stitched encounter key, used for stay-level grouping |
| `patient_id` | Patient key |
| `prediction_dttm` | When the prediction is made |
| `feature_cutoff_dttm` | The no-data-after timestamp (equals `prediction_dttm` for every shipped task) |
| `extubation_dttm` | `extubation_failure_24h` only: the extubation, 30 min after `prediction_dttm` |
| `<label_column>` | The task's outcome label |
| `split` | `train` or `test` |
| `age_at_admission` | Age, for reporting and subgroup analysis |
| `sex_category` | Sex, for reporting and subgroup analysis |
| `race_category` | Race, for reporting and subgroup analysis |
| `ethnicity_category` | Ethnicity, for reporting and subgroup analysis |
| `window_number` | Which prediction window this is within the stay (starts at 1) |
| `prediction_id` | Auditable prediction-level key |

**Carry `prediction_id` through to your prediction file.** It is built from the stitched encounter, hospitalization, patient, and window number, and it is the safest way to keep model outputs glued to the exact row being scored.

### Train/test splits

Every cohort is split 75% train / 25% test by stitched encounter, at every site. There are no split options; the `site` in `clif_config.json` picks only the order:

- `"site": "mimic"` — encounters ordered by the sha256 of their id. The MIMIC split is identical wherever it runs.
- any other site — encounters ordered by their earliest time: the oldest 75% train, the newest 25% test.

## Features

Two ways to turn CLIF data into model inputs. Pick either.

| Mode | How it works | Best for |
|------------------------|------------------------|------------------------|
| **Direct** | Use `cohort.parquet` with `clifpy` or your own loaders. Wide time-series (vitals, labs, assessments, respiratory support) go through clifpy's `ClifOrchestrator.create_wide_dataset` — **you** apply the `feature_cutoff_dttm` filter | Full control; everything except meds |
| **`fe-meds`** | FLAIR pulls configured medication events into an ELF-long parquet | Quick medication streams (per-row cutoff still yours to apply) |

The `fe-meds` extractor reads `feature_config.yaml`, loads CLIF through `clifpy`, cuts every event at its encounter's **latest** `feature_cutoff_dttm`, and writes ELF-long parquet. For a daily task that means a day-2 row can still see day-5 events — before training, join on `hospitalization_join_id` and keep `time <= feature_cutoff_dttm` per row:

``` bash
flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out meds.parquet
```

ELF-long output columns:

| Column | Meaning |
|----|----|
| `subject_id` | Patient identifier |
| `time` | Event time, at or before the encounter's latest `feature_cutoff_dttm` |
| `code` | Event code, e.g. `MED_CON//norepinephrine//mcg/kg/min` |
| `numeric_value` | Numeric value, when there is one |
| `unit` | Unit for numeric values |
| `text_value` | Text value for categorical events |
| `hospitalization_id` | Source hospitalization key |
| `hospitalization_join_id` | Stitched encounter key |

The extractors are a convenience, not a requirement. Custom features? Use the cohort and CLIF tables directly.

More detail: [docs/101/features.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/features.md).

## Predictions

The report builder checks your prediction file before scoring it.

Every prediction file needs:

- `hospitalization_join_id`
- `prediction_id`
- `split`
- the task's label column (e.g. `label_mortality`)

`hospitalization_id` may be carried for traceability, but it is not the evaluation unit.

Binary tasks also need:

- `y_prob` — a probability between 0 and 1

`y_pred` is optional; without it FLAIR uses `y_prob >= 0.5`. Multiclass and regression tasks use their own prediction columns.

**Recommended:** keep the cohort rows exactly as they are and just add your output columns. No row drift, and the report stays auditable.

## Reports

`flair build-report` turns predictions into aggregate JSON.

``` bash
flair build-report preds.parquet \
  --task extubation_failure_24h \
  --cohort cohort.parquet \
  --mode episodic \
  --out ./report/
```

Every run writes **exactly two JSON files**. Without `--mode`, the mode is inferred from the task's `temporal_pattern`: `continuous` tasks get `continuous`, everything else gets `episodic` (episode tasks print a warning — pass `--mode episodic` to silence it).

| Mode | Use for | Files |
|------------------------|------------------------|------------------------|
| `episodic` | One prediction per stay | `overall.json` + `hospitalization_level.json` |
| `continuous` | Many rows per stay | `landmark.json` + `hospitalization_level.json` |

**`overall.json`** (episodic) is threshold-free: AUROC and AUPRC with curves on a fixed grid, plus Brier score, calibration slope and intercept, and a quantile reliability curve.

**`landmark.json`** (continuous) gives four pooled numbers — Heagerty-pooled dynamic AUROC, AUPRC, Brier, and calibration slope — each with a confidence interval and the per-landmark curve behind it, across `META['landmark_horizon']` forward landmarks with Kaplan-Meier at-risk counts.

**`hospitalization_level.json`** is the same file in both modes: a fixed 99-point threshold sweep (0.01 … 0.99) with every rate metric and its CI band (flag rate has none; NNE's band is the PPV band inverted), the shipped clinical policy flagged as one row, Youden's J next to it, and subgroup fairness at the clinical threshold. In continuous mode, each hospitalization is first collapsed to its peak score — one alert per stay, the way it would actually be deployed.

Reports are always computed on the held-out `test` split. Each JSON carries audit metadata: task name, FLAIR version, generation time, report mode, and the task script's SHA-256.

### Decision curves and the direction of the alert

Most tasks alert when risk goes **up**. For an `"above"` task like `icu_readmission` ("keep in ICU when readmission risk ≥ 0.20"), the decision-curve event is the label event itself and the policy threshold is weighed directly.

`extubation_failure_24h` runs the other way (`clinical_direction: "below"`): it predicts *failure*, but the action — extubate — fires when failure risk is **low**. So its decision curve is built on the **complement** of the label. The event becomes *successful extubation*, and the x-axis is how confident of success a clinician would want before extubating: 0 means extubate everyone, 1 means keep everyone intubated. That puts its 0.10 policy at x = 0.90.

Only the decision frame flips. The model still emits P(failure), and AUROC, AUPRC, calibration, and the confusion matrix all stay on the failure label.

Net benefit is reported per threshold inside `hospitalization_level.json` alongside `net_benefit_treat_all`, so the decision curve and the operating-point table are literally the same array and can't drift apart. Decision curves are episodic-only — on the continuous peak surface they'd be the peak-calibration trap.

### Plots

`build-report` also writes local PNGs from the same JSON data into `viz/`. Those PNGs are for site review only. **The JSON bundle is the artifact you share.**

More detail: [docs/101/reports.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/reports.md).

## Privacy

FLAIR is built for a model-to-data workflow. It standardizes tasks, features, and reports — it is **not** a complete security boundary on its own.

In a federated run:

- Code is reviewed before it runs.
- Each site runs it locally against its own CLIF data.
- Patient-level rows never leave the site.
- Only aggregate JSON reports are shared.

FLAIR's own safeguards:

- small-cell suppression for Table 1 counts
- fairness subgroups omitted below a minimum size
- metric bins suppressed or omitted when counts are too small
- task script hashes in report metadata, for auditability

Network isolation, PHI runtime controls, and site approval belong to your deployment and governance process. FLAIR keeps its job narrow: build the same benchmark everywhere and report it the same way.

## Project Layout

``` text
flair_benchmark/
  cli.py                  # the `flair` command
  constants.py            # package version, shared constants, task audit hash
  cohort/                 # CLIF loading, encounter stitching, events, table 1
  tasks/                  # one Python module per benchmark task
  features/               # CLIF-to-ELF feature extractors
  meds_etl/               # MEDS/ELF concept and transform helpers
  report/                 # aggregate report builders
  templates/              # config templates written by `flair init-config`
```

The CLI is the public surface. Python helpers exist for advanced users, but the stable workflow is command-first.

## Adding a Task

A task is one Python module in `flair_benchmark/tasks/` with two things:

- `META` — task metadata and benchmark policy
- `build(clif_config)` — returns the cohort

The catalog is auto-discovered, so you don't edit a central registry. A new task should state its cohort, label, prediction time, prediction unit, and reporting policy clearly enough for both clinical review and site audit.

Copy from:

- `flair_benchmark/tasks/icu_daily_mortality.py` — a continuous task
- `flair_benchmark/tasks/extubation_failure_24h.py` — an episode task

Full guide: [docs/101/tasks.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/tasks.md).

## Documentation

This README is the manual. The 101 docs go deeper:

| Topic                    | Link                                         |
|--------------------------|----------------------------------------------|
| End-to-end overview      | [docs/101.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101.md)                   |
| CLIF data layer          | [docs/101/clif.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/clif.md)         |
| Tasks and task authoring | [docs/101/tasks.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/tasks.md)       |
| Feature engineering      | [docs/101/features.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/features.md) |
| Reports and report modes | [docs/101/reports.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/101/reports.md)   |
| Every key in the report bundle | [docs/eval_modes/episodic.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/eval_modes/episodic.md) · [docs/eval_modes/continuous.md](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/docs/eval_modes/continuous.md) |

Task schematics live in [docs/task_schematics](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/tree/main/docs/task_schematics/).

## Related Projects

| Project | Role |
|------------------------------------|------------------------------------|
| [CLIF](https://clif-icu.com) | Common Longitudinal ICU Format |
| [clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) | Python tooling for CLIF data |
| [MIMIC-CLIF](https://physionet.org/) | Public CLIF-formatted development dataset |

## Citation

``` bibtex
@software{flair2024,
  title = {FLAIR: Federated Learning Assessment for ICU Research},
  author = {CLIF Consortium},
  year = {2024},
  url = {https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR}
}
```

## License

Apache License 2.0 — see [LICENSE](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/blob/main/LICENSE).

FLAIR does not own or distribute the clinical datasets used with the benchmark. Access to private CLIF consortium data requires approval from the participating institutions.

## Contact

- Website: [clif-icu.com](https://clif-icu.com)
- Email: clif_consortium@uchicago.edu
- Issues: [GitHub Issues](https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR/issues)