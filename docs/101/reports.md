# Building a report: episodic, peak, landmark

[← back to FLAIR 101](../101.md)

A report turns a predictions parquet into the aggregate-only JSON bundle that a site
uploads. The key choice is **`--mode`** — how a task's predictions are scored.

## The command

```bash
flair build-report preds.parquet \
  --task task1_icu_daily_mortality \
  --out ./report/ \
  --mode landmark \
  --cohort cohort.parquet \
  --split test \
  --viz            # or --no-viz
```

| flag | default | purpose |
|---|---|---|
| `--task` | (required) | task name |
| `--out` | (required) | output directory for the JSON bundle |
| `--mode` | inferred | `episodic` \| `peak` \| `landmark` (see below) |
| `--split` | `test` | which split to analyze |
| `--cohort` | — | task cohort parquet; joins demographics + `window_number` when they don't flow through `preds` |
| `--viz/--no-viz` | on | render local sanity-check PNGs (needs `flair-benchmark[viz]`); never uploaded |

## The predictions parquet

Required columns (validated by the report builder):
- `hospitalization_id`, `split`, `<label_column>`
- **binary** → `y_pred` and/or `y_prob` (recommended — enables calibration, DCA, fairness AUROC)
- **multiclass** → `y_pred` + `class_probs_<class>`
- **regression** → `y_pred_value`

Carry `hospitalization_join_id`, `window_number`, and demographics through too (or pass
`--cohort` and they're joined in). `y_prob` must be a valid probability in `[0, 1]` with no
nulls.

## Choosing `--mode`

| mode | use for | pillars produced |
|---|---|---|
| **episodic** | one-per-stay tasks (T3, T5) | discrimination, calibration, dca, fairness |
| **landmark** | continuous tasks, per-lead-time honesty (T1, T2) | the four pillars **per landmark** (12 risk-set lead times) + `leadtime` |
| **peak** | continuous screening (T4) | discrimination + fairness only |

**Inference & the rule that bites:** if you omit `--mode`, it's inferred from META's
`temporal_pattern` — `episode` → `episodic`. But a **continuous task with no `--mode` is an
error**: you must explicitly pick `peak` or `landmark`. This is deliberate; the two answer
different questions:

- **landmark** freezes the clock *L* windows before each stay's endpoint, scores one row
  per stay at that lead time, and repeats for `L = 0..11` → an honest skill-and-calibration
  curve over time. It also emits `leadtime`: per-window detection sensitivity, median
  lead-time before the first alert, and mean risk trajectory by eventual outcome.
- **peak** collapses each stay to its worst (peak) risk — deployment-like for screening
  discrimination, but calibration/DCA are omitted because peak-selection breaks calibration
  (the "peak-calibration trap").

See [`../eval_continuous_three_ways.md`](../eval_continuous_three_ways.md) for the full
row-pooled vs. landmark vs. stay-peak rationale.

## What each pillar reports

| pillar | contents |
|---|---|
| **discrimination** | AUROC, AUPRC, ROC/PR curves, operating point at the clinical threshold (sensitivity/specificity/PPV/NPV/F1), NNE |
| **calibration** | ECE, Brier, slope, intercept, reliability + smoothed curves |
| **dca** | decision-curve net benefit across a threshold grid; net benefit at the clinical threshold |
| **fairness** | per-subgroup (sex, race, ethnicity, age-bin) AUROC + parity gaps (demographic parity, equalized-odds TPR/FPR) |

Every metric carries a **500-replicate bootstrap 95% CI** (deterministic seed). Cell
suppression (`<10`) and subgroup dropping (`n < 10`) are applied so nothing identifying
leaves the site. Output is one JSON per pillar under `--out` (landmark adds `leadtime.json`);
`--viz` writes matching PNGs under `report/viz/` for local validation only.
