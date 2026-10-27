# Tasks: the four benchmarks & authoring your own

[← back to FLAIR 101](../101.md)

## The four v1 tasks

List them at any time:

```bash
flair list-tasks
```

| task | class | temporal pattern | prediction unit | type | threshold (dir) | clinical action |
|---|---|---|---|---|---|---|
| `icu_daily_mortality` | outcome | continuous | one-per-window (daily 07:00) | binary | 0.50 (above) | order a palliative care consult |
| `icu_daily_ltach` | outcome | continuous | one-per-window (daily 07:00) | binary | 0.50 (above) | perform early tracheostomy |
| `extubation_failure_24h` | intervention | episode | one-per-stay | binary | 0.10 (**below**) | extubate; otherwise keep the patient intubated |
| `icu_readmission` | outcome | episode | one-per-stay | binary | 0.20 (above) | keep in ICU |

The **report mode** follows from the temporal pattern: episode tasks → `episodic`;
continuous tasks (T1, T2) → `continuous`, which emits `landmark.json` (four pooled
numbers over `META['landmark_horizon']` forward landmarks — 30 days for T1/T2) and
`hospitalization_level.json`. See [reports.md](reports.md).

> The threshold, direction, and report mode for every task are declared inline in that
> task's `META` in `flair_benchmark/tasks/<name>.py` — value-validated in `tasks/__init__.py`
> and folded into the audit hash. Change them in the task file.

## Loading a task → cohort

```bash
flair load-task icu_daily_mortality \
  --clif-config clif_config.json \
  --out output
```

This runs the task's `build()` and writes two site-prefixed folders under `--out`
(the same convention as flair_baseline; only the non-PHI one leaves the site):
- `<site>_flair_phi/<task>/cohort.parquet` — the cohort (one row per `prediction_id`)
- `<site>_flair_non_phi_for_upload/<task>/table1.json` — cohort characteristics; its `attrition` key is the STROBE flow (counts after each inclusion/exclusion step, cells < 10 suppressed), and `metadata` carries `cohort_sha256`, `train_cohort_sha256`, `test_cohort_sha256` — SHA-256 over `hospitalization_join_id, prediction_dttm, feature_cutoff_dttm, label, split`, rows sorted by `(hospitalization_join_id, prediction_dttm)`, values as strings — plus `clifpy_version` / `polars_version` and `task_script_sha256` (task source + META policy + all of `constants.py`: benchmark window, stitch gap, ...)
- `<site>_flair_non_phi_for_upload/<task>/strobe.png`, `strobe_attrition.png` — flow diagram + attrition chart drawn from Table 1's `attrition`

### The train/test split

Made in `cohort/stitch.py::assign_split`: always **75% train / 25% test**, always at
**stitched-encounter grain** (`hospitalization_join_id`) — one encounter is never torn
across the two arms. There are no split options; the site picks only the order:

| site | order of encounters |
|---|---|
| `mimic` | sha256 of the encounter id — MIMIC's dates are de-identified/shifted, so calendar order means nothing. Identical wherever it runs. |
| anything else | earliest timestamp — oldest 75% train, newest 25% test |

At every site except `mimic` the global benchmark window (`BENCHMARK_WINDOW_START` /
`BENCHMARK_WINDOW_END` in `constants.py`, 2017-12-31 – 2025-12-31) is applied first,
in `base_icu_cohort`: a stitched encounter is kept only if every member admission and
discharge falls inside it, so no encounter straddles either edge and partially-ETL'd
current-year data never enters the cohort. It is skipped for `mimic`, whose shifted
dates would make a real-date window empty the cohort.

An encounter spans days to weeks, so the temporal arms are separated in calendar
time. A patient with several encounters can appear in both arms; that is the
accepted cost of a test set that is genuinely later than training.

Expect a small overlap even on a correct split: a stay open across the boundary
belongs wholly to one arm, so its later rows land past the start of test legitimately.
Days to a few weeks is normal; **months means the split was wrong**.
`table1.json` publishes `temporal_overlap` (`overlap_days`,
`train_rows_after_test_start`, `train_frac_after_test_start`) so this is checkable
without touching PHI, and `flair build-report` warns past 60 days.

> The split is always per stitched encounter (`hospitalization_join_id`); there is
> no other grain, and `assign_split` raises on a cohort without the join id.

### The cohort contract

Every cohort carries a fixed set of columns (`flair_benchmark/tasks/_finalize.py`):

| column | role |
|---|---|
| `hospitalization_id` | raw hospitalization key |
| `hospitalization_join_id` | **stitched-encounter (stay) key** |
| `prediction_dttm` | when the prediction is made |
| `feature_cutoff_dttm` | the last instant whose data may be used. Defaults to `prediction_dttm`; set **earlier** by a task whose prediction timestamp would itself leak the outcome |
| `<label_column>` | the task label (e.g. `label_mortality`) |
| `split` | `train` / `test` |
| `age_at_admission`, `sex_category`, `race_category`, `ethnicity_category` | demographics (drive fairness) |
| `window_number` | 1-based k-th prediction per stay (always `1` for one-per-stay) |
| `prediction_id` | auditable per-row key |

`prediction_id = {hospitalization_join_id}-{hospitalization_id}-{patient_id}-{window_number}` —
all four segments always present, so any prediction row traces back to its stitched stay,
the specific hospitalization, the patient, and the window.

> **Leakage rule.** A row may use **all** data with `time ≤ feature_cutoff_dttm` and
> **nothing** after it. Every `build()` respects this, `fe-meds` enforces it, and any
> direct extraction you write must enforce it too.

The cutoff travels as a *timestamp*, not a flag, and that is deliberate: `flair
fe-meds --task` takes a **cohort parquet path** and never imports the task module,
so a per-task rule expressed any other way would silently fall back to the default.
Carrying a value instead means `features/extractors.py` keeps a single comparison
(`time <= end_time`) and is correct without knowing that per-task strictness exists —
and it means the rule is legible to code FLAIR never wrote. A cohort predating the
column is rejected loudly by `features/_scope.py` rather than defaulted.

No shipped task sets it away from `prediction_dttm`.
`extubation_failure_24h` anchors on the first non-IMV respiratory-support row —
the row that *announces* the extubation — and moves the **prediction** instead: `prediction_dttm = extubation_dttm − 30 min`, and the
cutoff sits on it. The 30-min lead is the leakage guard and also makes the row a
real decision (the tube is still in when the model is asked). The label window is
still measured from `extubation_dttm`, which the cohort carries alongside it, so
"failure within 24h" keeps its usual meaning.

---

## Authoring a new task (the "task maker")

A task is **a single Python file** dropped into `flair_benchmark/tasks/<name>.py` that
exposes two things — a `META` dict and a `build()` function. The catalog auto-discovers it
(`flair_benchmark/tasks/__init__.py`); no registry edits.

### `META` — required fields

The **taxonomy triple** (validated; reviewers read this before any code):

| field | allowed values |
|---|---|
| `task_class` | `outcome`, `deterioration`, `intervention`, `operational` |
| `temporal_pattern` | `episode`, `fixed_horizon`, `continuous` |
| `prediction_unit` | `one-per-stay`, `one-per-window` |

For **binary** tasks, also required:
- `clinical_threshold` — float in `(0, 1)`, the operating point. It must be a
  **multiple of `0.01`** so it lands on the 99-point operating-point sweep grid;
  off-grid values are rejected, because the shipped policy is one flagged row of
  that grid rather than a separate analysis.
- `clinical_direction` — `"above"` or `"below"` (act on high vs. low risk)

Also validated:
- `report_mode` — `"episodic"` or `"continuous"`. Validated and covered by the
  audit hash, but the report builder does not read it: with `--mode` omitted, the
  mode is inferred from `temporal_pattern` (`continuous` → continuous, anything
  else → episodic, with a warning for `episode`).
- `landmark_horizon` — **required when `report_mode` is `"continuous"`**: a
  positive int, the number of forward landmarks in the task's `window_unit`.
  Omitting it raises at task load.

Carried by convention, not validated: `dca_axis` — `{event, action[, inaction]}`,
the decision-curve framing in words, with a `"below"` task naming its `inaction`
arm too. All four shipped tasks set it, but nothing reads it; the decision basis
is derived from `clinical_direction` alone (`report/_basis.py`). It is there for
the clinical reviewer, not for the code.

Plus the descriptive fields: `name` (must equal the module filename),
`display_name`, `task_type` (`binary`/`multiclass`/`regression`), `label_column`,
`positive_class`, `clinical_action`, `input_window_hours`, `prediction_window`,
`window_unit` (continuous / one-per-window tasks only), `cohort_description`, `inclusion_criteria`, `exclusion_criteria`,
`evaluation_metrics`.

### `build()` — signature & contract

```python
def build(clif_config: str) -> pl.DataFrame:
    ...
    return finalize_cohort(df, OUTPUT_COLS, META)
```

It must **end with `finalize_cohort(df, OUTPUT_COLS, META)`** (`tasks/_finalize.py`), which
selects your `OUTPUT_COLS` and stamps on `window_number` + `prediction_id`. Typical body:
read config → build base cohort (helpers in `flair_benchmark/cohort/encounter.py`) → engineer the
label → `assign_split(df, cfg["site"], admission_col="prediction_dttm")` →
`finalize_cohort(...)`. Every row gets `train` or `test`.

### Author decision checklist

1. **Cohort scope** — all-comers ICU? hospital-wide? specific inclusion/exclusion?
2. **Label anchor** — what clinical event defines the positive label?
3. **Prediction timing** — one-per-stay (episode) or one-per-window (continuous)?
4. **Threshold + direction** — at what risk do you act, and above or below? The
   threshold is not a cutoff; it is the exchange rate between the harm of acting
   unnecessarily and the harm of the event (`t/(1-t)`). Name the action first — get
   that wrong and the threshold encodes the wrong trade.
5. **`prediction_dttm`** — when the prediction is made.
6. **`feature_cutoff_dttm`** — does your `prediction_dttm` row *itself* leak the
   outcome? If the anchor is the event (an extubation, a transfer, a procedure),
   set the cutoff strictly earlier. Otherwise leave it to default.

### Templates to copy

- `tasks/icu_daily_mortality.py` — simplest **continuous** (one-per-window) task.
- `tasks/icu_readmission.py` — simplest **episode** (one-per-stay) task.

### Acceptance

A new task enters the federated suite after: reproducibility against MIMIC-CLIF, a
clinician-led face-validity review, and site-governance acceptance at a minimum of two
consortium sites.
