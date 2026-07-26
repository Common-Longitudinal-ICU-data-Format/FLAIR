# Building a report: episodic vs continuous

[← back to FLAIR 101](../101.md)

A report turns a predictions parquet into the aggregate-only JSON bundle that a site
uploads. Every run writes **exactly two JSON files**. The key choice is **`--mode`** —
how a task's predictions are scored.

## The command

```bash
flair build-report preds.parquet \
  --task icu_daily_mortality \
  --out ./report/ \
  --cohort cohort.parquet \
  --viz            # or --no-viz
```

| flag | default | purpose |
|---|---|---|
| `--task` | (required) | task name |
| `--out` | (required) | output directory for the JSON bundle |
| `--mode` | inferred | `episodic` \| `continuous` (see below) |
| `--cohort` | — | task cohort parquet; joins demographics + `window_number` when they don't flow through `preds` |
| `--viz/--no-viz` | on | render local sanity-check PNGs (needs `flair-benchmark[viz]`); never uploaded |

Reports are **always** scored on the held-out `test` split. There is no override:
the home site trains on `train`, every site is scored on `test`, so cross-site
numbers are comparable.

## The predictions parquet

Required columns (validated by the report builder):
- `hospitalization_join_id`, `prediction_id`, `split`, `<label_column>`
- **binary** → `y_prob` required; `y_pred` optional and derived at 0.5 if absent
- **multiclass** → `y_pred` + `class_probs_<class>`
- **regression** → `y_pred_value`

`hospitalization_join_id` is the required report unit: one stitched encounter / stay.
`hospitalization_id` can still be carried for audit traceability, but it is not the
evaluation key. Carry `window_number` and demographics through too (or pass
`--cohort` and they're joined in). `y_prob` must be a valid probability in `[0, 1]`
with no nulls and no NaNs.

## The two files

| mode | file 1 — model quality | file 2 — the decision |
|---|---|---|
| **episodic** (T3, T5 — one-per-stay) | `overall.json` | `hospitalization_level.json` |
| **continuous** (T1, T2 — multi-row-per-stay) | `landmark.json` | `hospitalization_level.json` |

`hospitalization_level.json` carries the **same schema in both modes** — one reader,
one plotter, one cross-site aggregator. The only difference is where its
`(label, score)` pair came from, and that episodic adds three net-benefit keys.

If you omit `--mode`, it's inferred from META's `temporal_pattern`
(`episode` → `episodic`, `continuous` → `continuous`). The old `--mode peak` and
`--mode landmark` were merged into `continuous`.

## `landmark.json` — four numbers

FORWARD landmarking at **`hospitalization_time`** = day/hour 1..horizon since each
stay's first prediction window — never anchored on the event or outcome. The risk
set at time t is every stay still producing predictions at t (still at risk,
event-free), so per-landmark `n` is a true Kaplan-Meier "No. at risk" count. The
horizon comes from **`META['landmark_horizon']`**, a count in the task's
`window_unit`: **30 daily** landmarks for T1/T2. Never hardcoded.

Four numbers, each published **pooled** (with a CI) and **per landmark** (the curve
behind it, with a CI ribbon):

| number | pooled how | citation |
|---|---|---|
| `auroc` | pair-weighted, `w = cases × controls` — the time-varying / dynamic AUC | Heagerty & Zheng 2005 |
| `auprc` | case-weighted, since average precision is anchored on the positives | — |
| `brier` | n-weighted, which *is* the pooled mean squared error | Brier 1950 |
| `calibration_slope` | ONE logistic fit over all landmark rows | Cox 1958 |

The calibration slope is a single fit, **not** an average of per-landmark slopes:
averaging regression coefficients is a different estimator, and one degenerate
landmark would poison the mean.

**No fairness, no thresholds, no curves, no operating point** — those are
`hospitalization_level.json`'s job. A landmark with fewer than 10 stays at risk is
suppressed whole; one with a single label class is marked `undefined`.

## `overall.json` — ranking + calibration

Threshold-free, and small. AUROC and AUPRC with CIs; Brier, calibration slope and
intercept with CIs; a ten-point equal-count reliability curve.

**Curves are one array each.** ROC is stored as `tpr_at_grid` (TPR at a fixed FPR
grid, 0.00–1.00 step 0.01) and PR as `precision_at_grid` (precision at a fixed
recall grid). The x-axes are known constants, so they are never written down — which
also means two sites' curves can be averaged elementwise without interpolating one
onto the other's thresholds.

**Calibration uses equal-count (quantile) bins**, not equal-width. For a rare event
almost every prediction falls in the first equal-width bin and the curve collapses to
one point; equal-count binning puts ~n/10 stays in each bin and spreads ten readable
points across the range the model actually uses. Only the two probability arrays are
published — no counts, no edges — and a bin below 10 stays nulls its *observed* rate.

There is no `dca` block: the decision curve **is** `net_benefit` +
`net_benefit_treat_all` across the 99 rows of `hospitalization_level.json`.

## `hospitalization_level.json` — the whole decision

One hospitalization is the unit of analysis:

- **continuous** — collapse to the peak: `score = max(y_prob)` over the stay,
  `label = ever positive`. One alert per hospitalization, which is how a real-time
  alerting model is actually deployed.
- **episodic** — already one row per stay. Nothing to collapse.

It holds three things:

- **`by_threshold`** — the fixed 99-point grid (0.01 … 0.99, step 0.01), the same at
  every site and for every task, so two sites' sweeps line up row-for-row and pool
  without interpolation. Each row carries the full 2×2 plus sensitivity /
  specificity / PPV / NPV / F1 / FPR / flag-rate / NNE, each with a 95% CI band, and
  (episodic only) net benefit. **The shipped clinical policy is one flagged row**
  (`is_clinical: true`), not a separate analysis — it is already on the grid.
- **`selected_thresholds`** — Youden's J (Youden 1950) beside the clinical policy.
  Youden is stamped `in_sample: true`: choosing a cutpoint to maximise a metric on
  the rows you then score is upward biased (Leeflang 2008), so it annotates the
  sweep rather than heading it. Its `threshold_ci` is **selection-aware** — the
  cutpoint is re-chosen on every bootstrap resample — and so runs wider than the
  fixed-threshold band.
- **`fairness`** — per-subgroup (sex, race, ethnicity, age-bin) n, prevalence, AUROC
  with CI, positive-rate, TPR, FPR, plus demographic-parity and equalized-odds gaps
  with CIs. Computed **at the clinical threshold only**, not at all 99.

**Net benefit is episodic-only.** A decision curve on the peak (max-of-many) surface
is the peak-calibration trap, so continuous rows omit those keys and `metadata`
says why.

### Two bases, and why they are kept apart

A `clinical_direction: "below"` task — one whose action fires on LOW risk — has two
different frames, and conflating them is how a decision curve gets read backwards.
`extubation_failure_24h` is the live one: it predicts extubation *failure*, but the
decision it supports is two-sided — *extubate*, or *keep the patient intubated* —
so it acts on LOW risk, when P(failure) < 0.10.

- **Alarm basis** (`score >= t`, both directions) — the 2×2 and every rate. The
  *alarm* is always high risk, so sensitivity/PPV always mean "how well do we detect
  the event". Scoring the confusion matrix on the *action* instead would make
  TP = "extubated AND failed": a disaster booked as a true positive, with a
  sensitivity that rises as the model gets worse.
- **Decision basis** — `net_benefit` only. For that below task the event is
  *successful extubation* and the threshold weighs a false action against a missed
  one on the complement axis, so its 0.10 policy is weighed at **0.90**.

Only the **labels** and the **odds weight** complement. The model still emits
P(failure), and the acted arm is cut on the label scale (`p < t`) — comparing `1-p`
is not exact at a tie, which is how the curve and the sweep once disagreed about who
had been acted on. `tests/test_hospitalization_level.py` pins the vectorized sweep
against the scalar reference at every grid tie, in both directions.

## Privacy and precision

Every metric carries a **500-replicate bootstrap 95% CI** with a deterministic seed.
Confusion cells below 10 are masked **in pairs** (`tp` with `fp`, `fn` with `tn`) —
masking one alone is theatre, because the published `n` recovers it by subtraction.
Subgroup levels with fewer than 10 stays are dropped entirely, not merely suppressed:
even an AUROC on <10 patients can reveal individual outcomes.

## `--viz`

PNGs under `report/viz/`, for local validation only — never uploaded. Every plot
reads the same dicts that were written to disk, so a figure can never show a number
the JSON does not contain.

| mode | figures |
|---|---|
| **continuous** | `landmark_auroc`, `landmark_auprc`, `landmark_brier`, `landmark_slope`, `threshold_sweep` |
| **episodic** | `roc`, `pr`, `calibration`, `dca`, `fairness`, `threshold_sweep` |

`threshold_sweep.png` is the decision figure: six panels — sensitivity, specificity,
PPV, NPV, NNE, flag-rate — each against the threshold with its CI band, the shipped
policy marked in red and Youden's pick in green.
