# Building a report: episodic vs continuous

[← back to FLAIR 101](../101.md)

A report turns a predictions parquet into the aggregate-only JSON bundle that a site
uploads. The key choice is **`--mode`** — how a task's predictions are scored.

## The command

```bash
flair build-report preds.parquet \
  --task task1_icu_daily_mortality \
  --out ./report/ \
  --cohort cohort.parquet \
  --split test \
  --viz            # or --no-viz
```

| flag | default | purpose |
|---|---|---|
| `--task` | (required) | task name |
| `--out` | (required) | output directory for the JSON bundle |
| `--mode` | inferred | `episodic` \| `continuous` (see below) |
| `--split` | `test` | which split to analyze |
| `--cohort` | — | task cohort parquet; joins demographics + `window_number` (and the intervention timestamp, e.g. `first_abx_dttm`) when they don't flow through `preds` |
| `--viz/--no-viz` | on | render local sanity-check PNGs (needs `flair-benchmark[viz]`); never uploaded |

## The predictions parquet

Required columns (validated by the report builder):
- `hospitalization_join_id`, `prediction_id`, `split`, `<label_column>`
- **binary** → `y_prob` required; `y_pred` optional and derived at 0.5 if absent
- **multiclass** → `y_pred` + `class_probs_<class>`
- **regression** → `y_pred_value`

`hospitalization_join_id` is the required report unit: one stitched encounter / stay.
`hospitalization_id` can still be carried for audit traceability, but it is not the
evaluation key. Carry `window_number`, `prediction_dttm`, and demographics through too
(or pass `--cohort` and they're joined in). `y_prob` must be a valid probability in
`[0, 1]` with no nulls.

**Intervention tasks:** when a task's META declares an `intervention` (T4:
`first_abx_dttm`, the first CMS-qualifying antibiotic), that column is **required** —
it is generated at cohort-build time; carry it through your preds or pass `--cohort`.

## Choosing `--mode`

| mode | use for | pillars produced |
|---|---|---|
| **episodic** | one-per-stay tasks (T3, T5) | discrimination, calibration, dca, fairness |
| **continuous** | multi-row-per-stay tasks (T1, T2, T4) | ONE bundle: a **peak** section + the four pillars **per landmark** + `leadtime` with alert-vs-intervention timing |

If you omit `--mode`, it's inferred from META's `temporal_pattern`
(`episode` → `episodic`, `continuous` → `continuous`). The old `--mode peak` and
`--mode landmark` were merged into `continuous`, which emits both views at once:

- **peak section** (in `discrimination.json` / `fairness.json` under `"peak"`): each stay
  collapsed to its worst (peak) risk — deployment-like screening discrimination (+ NNE).
  Calibration/DCA have no peak section because peak-selection breaks calibration
  (the "peak-calibration trap").
- **landmark section** (`"landmarks"` list in all four pillar JSONs): FORWARD
  landmarking at **`hospitalization_time`** = day/hour 1..horizon since each stay's
  first prediction window — never anchored on the event or outcome. The risk set at
  time t is every stay still producing predictions at t (still at risk, event-free),
  so per-landmark `n` is a true Kaplan-Meier "No. at risk" count. This is the
  time-varying / dynamic AUC view (US-CRS eFigure-2 idiom). The horizon comes from
  **`META['landmark_horizon']`**, a count in the task's `window_unit`: **30 daily**
  landmarks for T1/T2, **168 hourly** (7 days) for T4. Never hardcoded.
- **kpi scoreboard** (`kpi.json`): the headline numbers — **pooled dynamic AUROC**
  (Heagerty & Zheng-style time-varying AUC, per-landmark AUCs pooled with
  cases×controls pair weights), peak AUROC/AUPRC/NNE + operating point, calibration at
  hospitalization time 1 (+ slope range = drift), pct caught + median first-alert
  hospitalization time, and the alert-vs-intervention rates.
- **leadtime** (`leadtime.json`): per-time detection sensitivity, median first-alert
  hospitalization time (how early in the stay the model first fires), mean risk
  trajectory by eventual outcome, and
  **`alert_vs_intervention`** — for tasks with an intervention config (T4 = antibiotics):
  - label-positive stays (sepsis patients): did the model fire before antibiotics, and
    the `(t_first_alert − t_antibiotics)` distribution in hours (negative = fired first);
  - label-negative stays that got antibiotics: same view;
  - label-negative stays never treated: did the model fire at all (false-alarm rate).
  Tasks without an intervention config (T1, T2) still get the generic fire/no-fire
  rates by label.

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
leaves the site. Output is one JSON per pillar under `--out` (continuous adds
`leadtime.json` + `kpi.json`); `--viz` writes matching PNGs under `report/viz/` for
local validation only — continuous mode renders `peak_*` panels and the landmark curves
on a KM-style axis: `hospitalization day` / `hospitalization hour` ascending
1 → horizon with a true "No. at risk" row beneath, reliability curves at three
hospitalization times (`landmark_reliability.png`), the sensitivity curve with a red
mean±SD intervention line, and the three-arm `alert_timing.png`.
