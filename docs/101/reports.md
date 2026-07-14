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
| **episodic** | one-per-stay tasks (T3, T5) | discrimination, calibration, dca, fairness + `operating_points` (the scorecard: every threshold-dependent metric, swept over the clinical threshold and ± 0.05 / ± 0.10) |
| **continuous** | multi-row-per-stay tasks (T1, T2, T4) | ONE bundle: a **peak** section + the four pillars **per landmark** + `leadtime` + `operating_points` (a per-threshold confusion matrix, plus optional intervention sub-analyses) |

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
  hospitalization time, and the operating-point headline (clinical-threshold
  sensitivity/specificity/FPR/PPV/NNE + intervention sub-analysis highlights).
- **leadtime** (`leadtime.json`): per-time detection sensitivity, median first-alert
  hospitalization time (how early in the stay the model first fires), and the mean
  risk trajectory by eventual outcome.
- **operating_points** (`operating_points.json`): the one decision-point section.
  It is **generic and META-driven** — the report layer knows only the task's
  `clinical_threshold`/`clinical_direction`, the `label`, and an *optional*
  `intervention` config `{"column", "display"}`; it never assumes what the task
  is about. Evaluated at THREE thresholds (clinical, clinical−0.05, clinical+0.05):
  - **confusion matrix (mandatory)** — model **alert × label** per stay
    (fired = the stay ever crosses the threshold; label = ever-positive) →
    {TP, FP, FN, TN} + sensitivity / specificity / **true FPR = FP/(FP+TN)** /
    PPV / NPV / NNE / F1 / flag-rate, each with a 95% CI. Every continuous task
    gets this (T1/T2/T4), no config needed.
  - **sub-analyses (optional)** — present only when the task declares a
    treatment-time `intervention` column (T4 = `first_abx_dttm`, `display`
    "treatment"). At each threshold, keyed on `(label, treatment, alert)`:
    - `label_positive`: how many positive stays fired + the *hours before
      treatment* the alarm first fired (positive = earlier);
    - `label_negative_treated`: of treated non-positive stays, the fire rate and
      `Δ = t_alert − t_treatment` (negative = fired before treatment);
    - `label_negative_untreated`: of never-treated non-positive stays, the misfire
      (false-alarm) count/rate — no timing.
  Tasks without an intervention column (T1, T2) still get the confusion matrix at
  all three thresholds.

See [`../eval_continuous_three_ways.md`](../eval_continuous_three_ways.md) for the full
row-pooled vs. landmark vs. stay-peak rationale.

## What each pillar reports

In **episodic** mode the split is by whether a decision threshold is involved. The
four pillars answer questions that hold wherever the threshold is put; the
scorecard owns every number that moves when it moves.

| pillar | contents | threshold-dependent? |
|---|---|---|
| **discrimination** | AUROC, AUPRC, AUPRC lift, ROC/PR curves | no |
| **calibration** | ECE, Brier, slope, intercept, reliability + smoothed curves | no |
| **fairness** | per-subgroup (sex, race, ethnicity, age-bin) AUROC/AUPRC/Brier/calibration slope — does the model *rank* equally well across groups | no |
| **dca** | decision-curve net benefit across a threshold grid, `useful_range`, **and its `verdict` at the shipped policy** | the one exception (below) |
| **operating_points** | THE scorecard, at 5 thresholds: confusion matrix, sensitivity/specificity/PPV/NPV/F1/FPR/NNE/alert-rate (`metrics`), the act taken and its net benefit (`decision`), and subgroup positive-rate/TPR/FPR + parity gaps (`fairness`) | yes — all of it |

In **continuous** mode the pillars keep their operating point inline (the peak
section and every landmark need it), and `operating_points` is the stay-level
alert×label view at clinical ± 0.05.

### Two bases, and why the scorecard keeps them apart

A `clinical_direction: "below"` task (T3: extubate when failure risk < 0.15) has two
different frames, and conflating them is how a decision curve gets read backwards:

- **Alarm basis** (`y_prob >= t`, both directions) — `confusion` + `metrics`. The
  *alarm* is always high risk, so sensitivity/PPV always mean "how well do we detect
  the event". Scoring the confusion matrix on the *action* instead would make
  TP = "extubated AND failed": a disaster booked as a true positive, with a
  sensitivity that rises as the model gets worse.
- **Decision basis** — the `decision` block and all of `dca.json`. This is the frame
  the clinician decides in. For T3 the event is *successful extubation*, the score is
  `1-p`, and the x-axis is the probability of success required before extubating: 0 =
  "unworried about failure, worried about the harms of continued intubation → extubate
  everyone", 1 = "terrified of failure → extubate nobody". The shipped policy 0.15
  therefore sits at **x = 0.85** (`decision.dca_threshold`, `metadata.clinical_threshold_on_axis`).

`decision.net_benefit` is *the same number* as `dca.json`'s curve at
`dca_threshold` — both go through `dca.to_decision_basis` / `dca.axis_threshold`, so
the two files cannot drift apart. It is deliberately **not** derivable from the
entry's alarm-basis TP/FP.

`dca.json` keeps its own `verdict` (net benefit vs act-on-all vs act-on-none at the
shipped policy) rather than surrendering it to the scorecard: for a task like T3 the
decision curve *is* the primary analysis, and a curve whose conclusion lives in
another file invites the conclusion to be missed.

Every metric carries a **500-replicate bootstrap 95% CI** (deterministic seed). Cell
suppression (`<10`) and subgroup dropping (`n < 10`) are applied so nothing identifying
leaves the site. Output is one JSON per pillar under `--out` (continuous adds
`leadtime.json` + `operating_points.json` + `kpi.json`); `--viz` writes matching PNGs
under `report/viz/` for local validation only — continuous mode renders `peak_*`
panels and the landmark curves on a KM-style axis: `hospitalization day` /
`hospitalization hour` ascending 1 → horizon with a true "No. at risk" row beneath,
reliability curves at three hospitalization times (`landmark_reliability.png`), the
sensitivity curve with a red mean±SD intervention line, the confusion-matrix grids
(`operating_points_confusion.png`), and — for intervention tasks — the per-threshold
sub-analysis grid (`operating_points_subanalyses.png`).
