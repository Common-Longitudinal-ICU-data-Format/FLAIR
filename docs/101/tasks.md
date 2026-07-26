# Tasks: the five benchmarks & authoring your own

[← back to FLAIR 101](../101.md)

## The five v1 tasks

List them at any time:

```bash
flair list-tasks
```

| task | class | temporal pattern | prediction unit | type | threshold (dir) | clinical action |
|---|---|---|---|---|---|---|
| `icu_daily_mortality` | outcome | continuous | one-per-window (daily 07:00) | binary | 0.95 (above) | order a palliative care consult |
| `icu_daily_ltach` | outcome | continuous | one-per-window (daily 07:00) | binary | 0.50 (above) | early tracheostomy |
| `extubation_failure_24h` | intervention | episode | one-per-stay | binary | 0.10 (**below**) | extubate; otherwise keep the patient intubated |
| `icu_readmission` | outcome | episode | one-per-stay | binary | 0.20 (above) | keep in ICU |

The **report mode** follows from the temporal pattern: episode tasks → `episodic`;
continuous tasks (T1, T2) → `continuous`, one bundle holding the stay-peak section and
the per-landmark pillars over `META['landmark_horizon']` lead times (30 days for T1/T2)
plus leadtime/alert timing. See [reports.md](reports.md).

> The threshold, direction, and report mode for every task are declared inline in that
> task's `META` in `flair_benchmark/tasks/<name>.py` — value-validated in `tasks/__init__.py`
> and folded into the audit hash. Change them in the task file.

## Loading a task → cohort

```bash
flair load-task icu_daily_mortality \
  --clif-config clif_config.json \
  --out cohort.parquet \
  [--train-end YYYY-MM-DD] [--test-start YYYY-MM-DD]
```

This runs the task's `build()` and writes:
- `cohort.parquet` — the cohort (one row per `prediction_id`)
- `cohort.table1.json` — cohort characteristics (auto-generated alongside)

Omit `--train-end`/`--test-start` to use MIMIC's deterministic 70/30 split (when
`site="mimic"`).

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
> **nothing** after it. Every `build()` respects this, and the feature extractors
> enforce it.

The cutoff travels as a *timestamp*, not a flag, and that is deliberate: `flair
fe-wide --task` takes a **cohort parquet path** and never imports the task module,
so a per-task rule expressed any other way would silently fall back to the default.
Carrying a value instead means `features/wide.py` and `features/meds_stream.py` keep
a single comparison (`time <= end_time`) and are correct without knowing that
per-task strictness exists. A cohort predating the column is rejected loudly by
`features/_scope.py` rather than defaulted.

`extubation_failure_24h` is the one task that sets it: its `prediction_dttm` is the
first non-IMV respiratory-support row — the row that *announces* the extubation — so
it cuts 1 µs earlier. Everything up to the decision stays in, including the SBT.

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
- `clinical_threshold` — float in `(0, 1)`, the operating point
- `clinical_direction` — `"above"` or `"below"` (act on high vs. low risk)

Plus the descriptive fields: `name`, `display_name`, `task_type`
(`binary`/`multiclass`/`regression`), `label_column`, `positive_class`, `clinical_action`,
`input_window_hours`, `prediction_window`, `window_unit`, `cohort_description`,
`inclusion_criteria`, `exclusion_criteria`, `evaluation_metrics`.

### `build()` — signature & contract

```python
def build(clif_config: str, train_end: str | None,
          test_start: str | None) -> pl.DataFrame:
    ...
    return finalize_cohort(df, OUTPUT_COLS, META)
```

It must **end with `finalize_cohort(df, OUTPUT_COLS, META)`** (`tasks/_finalize.py`), which
selects your `OUTPUT_COLS` and stamps on `window_number` + `prediction_id`. Typical body:
read config → build base cohort (helpers in `flair_benchmark/_encounter.py`) → engineer the
label → `assign_split(...)` → `finalize_cohort(...)`.

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
- `tasks/extubation_failure_24h.py` — simplest **episode** (one-per-stay) task.

### Acceptance

A new task enters the federated suite after: reproducibility against MIMIC-CLIF, a
clinician-led face-validity review, and site-governance acceptance at a minimum of two
consortium sites.
