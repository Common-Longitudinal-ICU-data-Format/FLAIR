# FLAIR end-to-end flow (baseline-model perspective)

This document traces a **baseline model run** (a site PI installs the wheel,
points it at their CLIF data, trains a simple model, and submits the four/six
report JSONs to the central aggregator). It also audits each step against the
code so we can see what is wired up, what is logical, and what is missing or
stale.

The flow has exactly four user-facing stages: **Configure → ① Make a task →
② Get data → ③ Train → ④ Report.** Everything else (`list-tasks`,
`extract-metadata`) is optional bookkeeping.

---

## 0. One-time configure

| Command | Code path | Writes |
|---|---|---|
| `flair init-config [--out .]` | `cli.init_config_cmd` → `_templates.TEMPLATES` | `clif_config.json` + `feature_config.yaml` |

After scaffolding, the site PI edits **`clif_config.json`** with:

```json
{ "site": "uchicago",
  "data_directory": "/data/clif/",
  "filetype": "parquet",
  "timezone": "US/Central" }
```

`site` matters: when `site == "mimic"` (or `train_end`/`test_start` are
omitted), `_split.assign_split` falls back to a deterministic 70/30 temporal
split because MIMIC dates are de-identified. Any other site requires explicit
`--train-end` / `--test-start` calendar cutoffs.

`feature_config.yaml` is only needed if the site uses the assist extractors
(`fe-wide` / `fe-meds`). It lists mCIDE categories per CLIF table and points
back at the `clif_config.json` via `clif_config:`.

---

## 1. ① Make a task — `flair load-task`

```bash
flair list-tasks                                   # discovery
flair load-task task1_icu_daily_mortality \
    --clif-config clif_config.json \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --out cohort.parquet
```

**Code path:** `cli.load_task` → `tasks.get_task(name)` (importlib auto-
discovery, with META taxonomy enum validation in `tasks/__init__._validate_taxonomy`) → `task_module.build(clif_config, train_end, test_start)` → `df.write_parquet(out)`.

Every task ends with `tasks/_finalize.finalize_cohort(df, OUTPUT_COLS, META)`,
which produces a stable schema:

| Column | Type | Source | Purpose |
|---|---|---|---|
| `patient_id` | str | clifpy `patient` | optional, for cross-stay joins |
| `hospitalization_id` | str | clifpy `hospitalization` | required key |
| `prediction_dttm` | datetime (tz-naive) | task-specific anchor | prediction time — hard "no data after" line; all prior history is usable |
| `<label_column>` | int/float | task | y_true |
| `split` | str | `_split.assign_split` | `"train"` / `"test"` |
| `age_at_admission` | int | clifpy `hospitalization` | demographic |
| `sex_category` | str | clifpy `patient` | demographic |
| `race_category` | str | clifpy `patient` | demographic |
| `ethnicity_category` | str | clifpy `patient` | demographic |
| `window_number` | Int32 | `_finalize` | k-th timestep per stay (1 for one-per-stay tasks) |
| `prediction_id` | str | `_finalize` | `{hospitalization_join_id}-{hospitalization_id}-{patient_id}-{window_number}` (always 4 parts; window=1 when none) — auditable per-row key; join id = stay-level key |

> **Audit note (resolved):** the README "Cohort parquet" table now lists
> `prediction_dttm` (the single no-data-after cutoff) alongside
> `window_number` and `prediction_id`. The legacy `window_start` /
> `window_end` data-collection-window columns were removed from every task —
> sites may use all history up to `prediction_dttm`.

### Per-task pipeline summary

| Task | Cohort builder | Label rule | Row shape |
|---|---|---|---|
| `task1_icu_daily_mortality` | `base_icu_cohort` + `compute_icu_timing` | `discharge_category ∈ {expired, hospice}` (null → 0) | 07:00-local daily grid during 1st ICU stay |
| `task2_icu_daily_ltach` | same as T1 | regex `ltach|long.?term` on discharge_category | same grid as T1 |
| `task3_extubation_failure_24h` | `base_icu_cohort` + `_ventilation_episodes.detect_imv_episodes` (15/30 min consensus windows) | death OR reintubation within 24h of first extubation; ambiguous censoring dropped | one-per-stay anchored at first extubation |
| `task4_sepsis_abx_6h` | `_hospital_wide_cohort` + `_exclude_psych_rehab` + `clifpy.utils.ase.compute_ase` | ASE onset (`ase_onset_w_lactate_dttm`) ≤ prediction_dttm + 6h | hourly grid from `max(first_vital, admit+1h)` to `min(t_zero, discharge, death)` |
| `task5_icu_readmission` | `base_icu_cohort` + `_readmission.compute_readmission_timing` (sandwiched-procedural merge) | exists a 2nd ICU group after exclusions | one-per-stay anchored at first ICU end |

> **Audit note (resolved):** README now describes T5 as "Unplanned ICU
> Readmission" throughout (task design matrix, v1 task list, and architecture
> tree all point at `task5_icu_readmission.py`). The legacy "AKI KDIGO-2"
> naming and the `task5_aki_kdigo2_24h.py` / `_aki.py` files never existed in
> the tree and are gone from the docs. The add-a-new-task example points at
> `task1_icu_daily_mortality.py`.

---

## 2. ② Get data — three options

The CLI ships two **assist extractors** that turn a task cohort parquet into a
MEDS/ELF long event table; sites may also write features directly with clifpy.

| Option | CLI | Code path | Output |
|---|---|---|---|
| A. Direct FE (BYO) | — | user code over `clifpy.ClifOrchestrator` or table classes | site-defined |
| B. Meds stream | `flair fe-meds` | `features.meds_stream.get_meds_stream` | ELF long, raw dose + raw unit |
| C. Wide features | `flair fe-wide` | `features.wide.get_wide_feature_dataset` | ELF long, all configured vitals/labs/meds/PA/resp |

Both assist extractors share `cli._run_fe`:

```text
load_feature_config(yaml) → (feature_cfg, clif_cfg)
read_parquet(task)        → task_df  # the cohort
extractor(task_df, feature_cfg, clif_cfg) → pl.DataFrame (ELF schema)
df.write_parquet(out)
```

**ELF schema** (enforced by `features/_elf.finalize`):

| Column | Required | Notes |
|---|---|---|
| `subject_id` | yes | `= patient_id` |
| `time` | yes | tz-naive Datetime("us"), ≤ stay's `prediction_dttm` |
| `code` | yes | `<DOMAIN>//<category>//<unit>`, e.g. `LAB//creatinine//mg/dl` |
| `numeric_value` | nullable | for numeric events |
| `unit` | required wherever `numeric_value` is set | charted so a downstream step can convert |
| `text_value` | nullable | for categorical (device_category, etc.) |
| `hospitalization_id` | yes | for keying back to the cohort |

`fe-wide` enforces a **no-future-data guard** in `wide._cohort_windows`:
events kept only when `time ≤ max(prediction_dttm)` per stay (no lower
bound — all history up to the prediction time is usable). `fe-meds` doesn't
need its own guard — `meds_stream` truncates inside the extractor.

> **Scale caveat (already in README):** `fe-wide` calls
> `ClifOrchestrator.create_wide_dataset`, which loads full CLIF tables before
> filtering. For full-MIMIC cohorts this is memory-heavy. Sites with sharded
> exports should prefer Option A (direct clifpy table filters).

---

## 3. ③ Train (your code, baseline example)

The baseline contract is intentionally minimal: produce a parquet with the
required columns. `examples/end_to_end.py` is the reference; the core is:

```python
cohort = pl.read_parquet("cohort.parquet")
# ... your feature build (Option A/B/C) ...
# ... your model fit on cohort[split=="train"] ...
preds = cohort.with_columns(pl.Series("y_prob", y_prob))  # binary — the one column to add
preds.write_parquet("preds.parquet")
```

**Schema required by `flair build-report`** (validated by `report.__init__._validate_preds_schema`):

| Column | When | Type |
|---|---|---|
| `hospitalization_id` | always | str |
| `split` | always | str |
| `<label_column>` | always — already in the cohort file | int/float |
| `y_prob` | binary: the one column to add (ROC/PR/cal/DCA/temporal) | float in [0,1] |
| `y_pred` | binary: optional, derived from `y_prob` at 0.5 when absent; multiclass: required | int |
| `class_probs_<class>` | multiclass | float per class |
| `y_pred_value` | regression | float |

**Strongly recommended (not validated, but enables sections of the report):**
`prediction_id`, `prediction_dttm`, `window_number`, `admission_dttm`.
The baseline workflow that calls `cohort.with_columns([...])` keeps them
automatically (the cohort schema already carries them); custom write paths
must remember to.

> **Note — `y_true` / `y_pred`.** Sites add only `y_prob` to the cohort file.
> The report reads `<label_column>` directly (no `y_true` needed), and derives
> `y_pred` from `y_prob` at 0.5 when it isn't supplied (`_ensure_y_pred`). Both
> are honored if present. `examples/end_to_end.py` may still write a redundant
> `y_true` — harmless.

---

## 4. ④ Report — `flair build-report`

```bash
flair build-report preds.parquet \
    --task task1_icu_daily_mortality \
    --cohort cohort.parquet \
    --out ./report/ \
    [--viz]
```

**Code path:** `cli.build_report_cmd` → `report.build_report(preds_path,
task_module, out_dir, cohort_path, viz)`. The function:

1. Reads the predictions parquet.
2. Calls `_validate_preds_schema(preds, meta)`.
3. Calls `_attach_cohort_context(preds, cohort_path, meta)` — left-joins
   `window_number`, `prediction_dttm`, `admission_dttm` from the cohort
   parquet using, in order of preference, `prediction_id`,
   `(hospitalization_id, prediction_dttm)`, or `hospitalization_id` (only
   safe for one-per-stay tasks; warns otherwise).
4. Builds the **audit block** (`task`, `task_script_sha256`, `flair_version`,
   `generated_at`) and stamps it into every JSON.
5. Runs six builders concurrently in code order and writes the six JSON
   payloads.

### Report bundle — what each JSON contains

| File | Builder | Required preds cols | Suppression / privacy |
|---|---|---|---|
| `table1.json` | `report/table1.py` | label, split, demographics | counts < 10 → `"<10"` via `_suppress.suppress` |
| `tripod_ai.json` | `report/tripod_ai.py` | label, split, `y_pred` (+`y_prob` for binary) | aggregate metrics only |
| `fairness.json` | `report/fairness.py` | label, split, `y_pred`, `y_prob`, demographics | subgroups with n<10 **dropped entirely** |
| `curves.json` | `report/curves.py` | binary: label, `y_prob`; multiclass: `class_probs_*`; regression: `y_pred_value` | calibration bins with count<10 → `bin_true_prob=null`; ROC/PR downsampled to 500 points |
| `temporal.json` | `report/temporal.py` | binary only: label, `y_prob`, `window_number` (or `prediction_dttm`), `admission_dttm` (or derived) | bins with n<10 dropped; AUROC reported only when ≥10 positives AND ≥10 negatives; bootstrap CI only when n≥100 |
| `dca.json` | `report/dca.py` | binary only: label, `y_prob` | thresholds flagging <10 patients → `null`, recorded in `suppressed_thresholds` |

> **Audit note — README says "four JSON files" (lines 148-156).** That is
> stale: `build_report` returns six JSONs (the README pre-dated the
> `temporal` and `dca` modules). The PR-paragraph that lists the report
> bundle should be regenerated.

### Section-by-section logic check (baseline-model perspective)

**`tripod_ai.results.performance`** — uses `preds.filter(split=="test")`.
- Binary: AUROC, AUPRC, accuracy/precision/recall/F1, TP/FP/TN/FN, specificity, NPV. Needs `y_prob` for AUROC/AUPRC; degrades gracefully without it.
- Multiclass: macro/weighted F1, per-class precision/recall/F1, confusion matrix; `class_names` from META used when present.
- Regression: MSE/RMSE/MAE/R²/explained_variance via `y_pred_value`.

**`tripod_ai.results.calibration`** (binary only) — ECE (10 bins) + extended (slope/intercept/Brier/O:E) via `report/_calib.calibration_extended`. Full reliability curve lives in `curves.json`.

**`fairness.subgroups`** — by `sex_category`, `race_category`, `ethnicity_category`, and an `age_bin` derived in-builder (`18-39 / 40-64 / 65-79 / 80+`). Computes per-group AUROC/AUPRC/Brier/calibration slope+intercept for binary. Drops groups with `n < 10`.

**`fairness.parity`** — demographic parity gap, equalized-odds TPR/FPR gaps. Reference group = largest surviving subgroup. Returns `{}` for multiclass/regression.

**`curves.calibration`** — 10 equal-width bins; suppresses `bin_true_prob` when count<10. `calibration_extended.smoothed` is a fitted-logistic recalibration curve evaluated on a 50-point grid (2 parameters → no PHI).

**`temporal.by_window`** — bins predictions by `window_number` (cap at 7+). Requires a continuous (`one-per-timestep`) task; for `one-per-stay` tasks it returns `{"reason": "one_per_stay_task"}`. Uses `_attach_window_bucket` which falls back to deriving `window_number` from `prediction_dttm` if needed.

**`temporal.by_year`** — bins by admission year; derives the year from `min(prediction_dttm) per stay` if `admission_dttm` is absent.

**`temporal.trajectory_by_outcome`** — per-window mean/median/IQR of `y_prob`, split by label. Visualizes whether predicted risk diverges over the stay.

**`dca`** — sweeps thresholds 0.01–0.99 step 0.01, computes net benefit vs treat-all / treat-none, reports widest contiguous "useful interval" (model strictly better than both baselines).

### When sections silently drop

The report degrades gracefully — a baseline model that forgets `y_prob` still
gets a valid `tripod_ai.json` (no AUROC) and gets `temporal.json` /
`dca.json` with `metadata.reason = "requires binary task with y_prob"`.
**The CLI does not warn about this.** A site PI could submit a report bundle
that silently omits DCA and temporal because they forgot `y_prob`. Consider
adding a CLI-level "predictions missing y_prob — temporal/DCA will be empty"
warning at the top of `build_report_cmd`.

---

## 5. Optional — `flair extract-metadata`

```bash
flair extract-metadata --clif-config clif_config.json --out ./metadata/
```

**Code path:** `metadata.__init__.extract_site_metadata` runs `base_icu_cohort`
once, computes SOFA (`compute_hospitalization_sofa_max`) and Charlson/
Elixhauser (`compute_cci_elix`) once, then loops every task and writes
`<task_name>.json` carrying label distribution, demographics, severity,
comorbidity, and yearly counts — all HIPAA-suppressed and audit-stamped. This
feeds the leaderboard's "site cohort overview" panel and does **not** require
predictions.

`--skip-sofa` / `--skip-comorb` are escape hatches for sites without
`HospitalDiagnosis` or with very large cohorts.

---

## 6. End-to-end command sequence for a baseline run

```bash
# 0. configure
flair init-config --out .
$EDITOR clif_config.json feature_config.yaml

# 1. (optional) inventory
flair list-tasks

# 2. (optional) federation overview
flair extract-metadata --clif-config clif_config.json --out ./metadata/

# 3. make the cohort for ONE task
flair load-task task1_icu_daily_mortality \
    --clif-config clif_config.json \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --out artifacts/task1/cohort.parquet

# 4. get features — choose ONE
#    A. direct (write your own script over clifpy)
#    B. lightweight med stream
flair fe-meds --feature-config feature_config.yaml \
              --task artifacts/task1/cohort.parquet \
              --out artifacts/task1/meds.parquet
#    C. wide time-series feature pull
flair fe-wide --feature-config feature_config.yaml \
              --task artifacts/task1/cohort.parquet \
              --out artifacts/task1/wide.parquet

# 5. train a baseline (your code) → artifacts/task1/preds.parquet
python my_baseline.py \
    --cohort artifacts/task1/cohort.parquet \
    --features artifacts/task1/wide.parquet \
    --out artifacts/task1/preds.parquet

# 6. emit the aggregate-only report bundle
flair build-report artifacts/task1/preds.parquet \
    --task task1_icu_daily_mortality \
    --cohort artifacts/task1/cohort.parquet \
    --out artifacts/task1/report/
# → table1.json tripod_ai.json fairness.json curves.json temporal.json dca.json
```

Repeat steps 3–6 per task. Submit `artifacts/task*/report/*.json` (and the
optional `metadata/*.json`) to the central aggregator. **Patient-level data
never leaves the site.**

---

## 7. Audit findings

### A. Doc-sync bugs (no code change needed, README updates only)

1. **T5 description (resolved).** README now lists T5 as "Unplanned ICU
   Readmission" across §"Benchmark Tasks", the v1 task list table, the task
   design matrix, and the architecture tree, pointing at
   `task5_icu_readmission` (commit `e57c430`). The "AKI KDIGO-2" naming and
   the `task5_aki_kdigo2_24h.py` / `_aki.py` files are gone from the docs.

2. **Report bundle is six JSONs, not four.** README lines 148–156 list
   `table1`, `tripod_ai`, `fairness`, `curves`. The current
   `report/__init__.py` also writes `temporal.json` and `dca.json`. Update
   the "Report bundle" subsection and the §④ description.

3. **Cohort parquet schema (resolved).** The README "Cohort parquet" table
   now documents `prediction_dttm`, `window_number`, and `prediction_id`
   (the columns `build-report` joins on for `temporal.json`). The legacy
   `window_start` / `window_end` columns were dropped from the schema and
   every task — `prediction_dttm` is the sole no-data-after cutoff.

4. **"Adding a new task" example (resolved).** The README add-a-new-task
   example now points at `task1_icu_daily_mortality.py` as the reference
   template (the missing `task5_aki_kdigo2_24h.py` reference is gone).

5. **Single-file predictions contract (resolved).** Sites now add only
   `y_prob` to the cohort file; `y_true` is gone from the docs (the report
   reads `<label_column>` directly) and `y_pred` is derived from `y_prob` at
   0.5 when absent (`report.__init__._ensure_y_pred`). One file, no row drift.

### B. Functional gaps worth considering (small adds)

6. **No CLI warning when predictions lack `y_prob`.** `temporal.json` and
   `dca.json` silently degrade to `{"metadata":{"reason":"requires binary
   task with y_prob"}}`. A `typer.echo` warning at the top of
   `build_report_cmd` would surface this loud and early.

7. **No CLI warning when continuous-task predictions lack `prediction_id` /
   `prediction_dttm` / `window_number` and no `--cohort` is supplied.**
   The code emits a `warnings.warn` from `_attach_cohort_context`, but
   warnings are easy to miss in batch runs. A CLI-level pre-flight check
   would be friendlier.

8. **No `flair train-baseline` command.** Each site re-implements the
   "logistic-regression-on-demographics" baseline from `examples/end_to_end.py`.
   A canned `flair train-baseline --task --features` command (writing a
   `preds.parquet`) would give every site a frictionless smoke-test and a
   reference point on the leaderboard. This is **explicitly excluded from
   the current design** ("You provide your model"), so this is a design
   call, not a missing piece — flagging it for the team to weigh.

9. **No `--dry-run` or schema-validator for `preds.parquet`.** A user could
   point `build-report` at the wrong file and only learn about column-name
   typos halfway through report generation. A `flair check-preds preds.parquet
   --task <name>` would run `_validate_preds_schema` alone and exit 0/1,
   useful in CI / pre-submission.

10. **No `flair pack-report` to ZIP and hash the bundle.** The aggregator
    receives six JSONs per task plus the metadata JSONs; collecting and
    integrity-checking them at the site is currently user-scripted.

### C. Things that ARE wired up correctly

- **Federation invariants.** Every JSON carries `task_script_sha256`,
  `flair_version`, `generated_at` — the central aggregator can reject
  mismatched-task-code submissions without any extra plumbing.
- **HIPAA suppression.** Three layers — `MIN_CELL_COUNT=10` in
  `_suppress.py`, subgroup drop in `fairness.py`, calibration bin
  suppression in `curves.py`, DCA threshold suppression in `dca.py`.
  Different mechanism per file (suppress / drop / null) but all keyed off
  the same `MIN_CELL_COUNT=10` constant.
- **No-future-data guard.** `wide._cohort_windows` enforces a per-stay
  `time ≤ max(prediction_dttm)` filter on every emitted event (no lower
  bound — all history up to prediction is usable); task `build()` functions
  filter `prediction_dttm < pred_stop` (closed on-left); `temporal.py` joins
  on a derived window number that never sees post-prediction data.
- **Task auto-discovery + taxonomy enforcement.** Drop a `.py` in
  `tasks/`, expose `META` + `build`, and `flair list-tasks` /
  `flair load-task` find it; META values are validated against the three
  taxonomy enums in `tasks/__init__._validate_taxonomy`.
- **MIMIC vs site split.** `_split.assign_split` detects MIMIC (or
  unspecified cutoffs) and falls back to 70/30 by rank — no flag fiddling
  needed for the dev path.
- **`prediction_id` is the right key.** Every task adds a stable
  per-row id with a deterministic format. As long as sites carry it
  through to `preds.parquet`, the cohort join in `_attach_cohort_context`
  is exact and `temporal.json` works without ambiguity.

---

## 8. Quick mental model

```
 clif_config.json + feature_config.yaml
            │
            ▼
   ┌────────────────────┐
   │  flair load-task   │  per task
   └─────────┬──────────┘
             │ cohort.parquet (with prediction_id, window_number, label, demo)
             ├──────────────────────────┐
             ▼                          ▼
   ┌────────────────────┐    ┌────────────────────┐
   │ flair fe-wide /    │    │ your direct FE     │
   │ flair fe-meds      │    │ (clifpy)           │
   └─────────┬──────────┘    └─────────┬──────────┘
             │ features.parquet (ELF long)        │
             └────────────┬──────────────────────-┘
                          ▼
                ┌────────────────────┐
                │ your model.fit /   │
                │ .predict           │
                └─────────┬──────────┘
                          │ preds.parquet (hosp_id, split, label, y_pred, y_prob)
                          ▼
                ┌────────────────────┐
                │ flair build-report │  task_script_sha256 + suppression
                └─────────┬──────────┘
                          │ table1 / tripod_ai / fairness / curves /
                          │ temporal / dca  +  optional metadata/*.json
                          ▼
            central FLAIR aggregator (verifies SHA-256)
```

The whole pipeline is **eight CLI calls or fewer** per task. The federation
guarantee is the audit metadata in the JSONs — there is no networked
plumbing, just deterministic outputs.
