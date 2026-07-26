# Continuous mode — every file and every number

[← back to FLAIR 101](../101.md) · [← episodic mode](episodic.md)

**Plain English:** continuous mode is used when your model scores the **same patient
over and over** — once a day, or once an hour, for as long as the stay lasts. One
stay = many rows = many probabilities.

Shipped tasks that use it:

| task | clinical threshold | direction | window | landmark horizon |
|---|---|---|---|---|
| `icu_daily_mortality` | 0.95 | `above` | one row per day (07:00) | 30 days |
| `icu_daily_ltach` | 0.50 | `above` | one row per day (07:00) | 30 days |

## The command

```bash
flair build-report preds.parquet \
  --task icu_daily_mortality \
  --out ./report/ \
  --cohort cohort.parquet \
  --mode continuous        # optional: inferred from the task
```

Your predictions must carry `window_number` (1, 2, 3 … per stay). If it is not in
your preds file, pass `--cohort` and FLAIR joins it in. As always, only the `test`
split is scored.

---

## The two honest ways to score a repeating model

This is the one idea you need before reading the files. Because a stay produces
many predictions, "the AUROC" is ambiguous. Continuous mode reports **two** views,
one per file:

| view | the question it answers | where it lives |
|---|---|---|
| **landmark** | "On hospitalization day 5, how well does the model separate the patients still at risk?" — the time-varying view | `landmark.json` |
| **hospitalization level** | "Did this stay ever alarm, vs did it ever have the event?" — the screening / deployment view | `hospitalization_level.json` |

**What a landmark is.** Landmark *t* is the slice `window_number == t` — every stay
still producing predictions on hospitalization day *t*. Because task builders only
generate windows before the event/discharge, that slice is automatically "still
alive, still event-free": a true Kaplan-Meier *No. at risk* set. Nothing is anchored
on the event, so it is exactly the group a deployed model would score that morning.
The number of landmarks comes from the task's `landmark_horizon` (30 for both
shipped tasks) — never hardcoded.

**What the hospitalization level is.** Each stay collapses to `max(y_prob)` over all
its windows, against "did the event ever happen". One alert per hospitalization,
which is how a real-time alerting model is actually deployed.

**Why there is no decision curve in continuous mode.** Taking the *maximum* risk
over 20 days is biased upward by construction, so net benefit on that surface reads
wrong — the "peak-calibration trap". The continuous sweep therefore omits the
`net_benefit` keys and says so in `metadata.net_benefit_omitted_reason`. Honest
calibration lives in `landmark.json`, where each slice has one row per stay.

---

## 1. What files come out

```
report/
  landmark.json                 four numbers, over hospitalization time
  hospitalization_level.json    everything that depends on the threshold
  viz/                          local PNGs, never uploaded
```

**Two files. That is the whole bundle** (~82 KB on a 300-stay, 30-landmark task).
Only the JSON is uploaded.

`hospitalization_level.json` is the **same schema episodic mode produces**, minus
the three net-benefit keys — so a reader written once works for every task and
every site.

---

## 2. Rules that apply to every file

**Every JSON file starts with the same `metadata` block:**

| key | meaning |
|---|---|
| `task` | task name, e.g. `icu_daily_mortality` |
| `task_script_sha256` | hash of the task file — proves every site ran the same definition |
| `flair_version` | FLAIR version that built the report |
| `generated_at` | UTC timestamp of the build |
| `report_mode` | `"continuous"` |
| `split` | always `"test"` |
| `n_stays` | number of unique stays scored |
| `prevalence` | event rate in the scored rows |
| `report_schema` | shape version of the bundle (currently `4`) |
| `evaled_at` | the task's clinical threshold |

**Confidence intervals.** Anything ending in `_ci` is a `[low, high]` pair =
95% percentile bootstrap, **500 replicates, fixed seed 20260612**. Crucially, the
resampling unit is the **hospitalization**, not the row — a stay drawn twice is
drawn twice at every landmark it appears in. That is what respects the correlation
between a patient's own repeated predictions; a row bootstrap would report intervals
that are far too narrow.

**Privacy.** Any count below **10** is written as `"<10"`, and the confusion matrix
masks **in pairs** — `tp` with `fp`, `fn` with `tn` — because `n` is published and a
lone masked cell is recoverable by subtraction. Demographic subgroups under 10 are
dropped entirely. A landmark whose risk set falls below 10 is suppressed whole.

**Rounding.** Scalars are rounded to 4 decimals.

---

## 3. `landmark.json` — four numbers over time

Each number is published **pooled** (one headline with a CI) and **per landmark**
(the curve behind it, with a CI ribbon).

### 3.1 `pooled`

| key | meaning | pooled how |
|---|---|---|
| `auroc` (+`_ci`) | the time-varying / dynamic AUC | pair-weighted, `w = cases × controls` (Heagerty & Zheng 2005) |
| `auprc` (+`_ci`) | ranking under class imbalance | case-weighted — average precision is anchored on the positives |
| `brier` (+`_ci`) | accuracy of the probabilities themselves | n-weighted, which *is* the pooled mean squared error |
| `calibration_slope` (+`_ci`) | **1.0 = the probabilities mean what they say** | ONE logistic fit over all landmark rows |

**Why the weights differ.** Each is chosen for what the metric is made of. A
landmark with no events contributes zero comparable pairs and so drops out of the
AUROC pool automatically, which is the correct behaviour rather than a special case.

**Why the slope is one fit, not an average.** Averaging regression coefficients is a
different estimator, and a single degenerate landmark would poison the mean.

### 3.2 `by_landmark`

One entry per landmark, `1 … landmark_horizon`:

| key | meaning |
|---|---|
| `hospitalization_time` | the day (or hour) — 1-based |
| `n` | stays still at risk. This is a genuine KM "No. at risk" and falls over time |
| `prevalence` | event rate in that slice |
| `auroc`, `auprc`, `brier`, `calibration_slope` (each +`_ci`) | that slice's four numbers |

Two kinds of stub replace a scored entry:

- `{"n": "<10", "suppressed": true}` — too few stays left at risk.
- `{"undefined": true, "reason": "single label class …"}` — everyone in the slice
  has the same label, so there is no contrast to score.

**What is NOT here:** no fairness, no thresholds, no ROC/PR curves, no operating
point. Storing a ROC curve per landmark was the bulk of the old bundle and nobody
plotted them. The decision belongs in the other file.

---

## 4. `hospitalization_level.json` — what happens at a cutoff?

Identical in structure to [episodic mode's](episodic.md#4-hospitalization_leveljson--what-happens-at-a-cutoff),
with two differences:

- `metadata.surface` is `"peak"` and `metadata.score_rule` is
  `"max(y_prob) per hospitalization_join_id"`.
- The rows carry **no** `net_benefit`, `net_benefit_ci` or
  `net_benefit_treat_all`; `metadata.net_benefit_omitted_reason` explains why.

Everything else is the same file: the 99-row `by_threshold` sweep with the full 2×2
and every rate + CI band, `selected_thresholds` (Youden's J beside the shipped
policy), and `fairness` at the clinical threshold.

**Note the unit.** `n` in this file is a count of **hospitalizations**, not of
prediction windows. A 30-day stay contributes exactly one row, whether it was scored
3 times or 300.

---

## 5. The PNGs (local only, never uploaded)

| file | shows |
|---|---|
| `landmark_auroc.png` | AUROC by hospitalization day, CI ribbon, pooled value as a red dashed line |
| `landmark_auprc.png` | the same for AUPRC |
| `landmark_brier.png` | the same for the Brier score |
| `landmark_slope.png` | the same for the calibration slope, with a grey reference line at 1.0 |
| `threshold_sweep.png` | six panels — sensitivity, specificity, PPV, NPV, NNE, flag-rate — each against the threshold with its CI band |

On the sweep the shipped policy is the **red dashed** line and Youden's pick is the
**green dotted** line.

Fairness is in `hospitalization_level.json` for continuous tasks too; it simply has
no auto-rendered figure in this mode.

---

## 6. Quick cheat sheet — which number do I quote?

| the question | where to look |
|---|---|
| "How well does it rank, overall?" | `landmark.json` → `pooled.auroc` (the dynamic AUC) |
| "Are the probabilities honest?" | `landmark.json` → `pooled.calibration_slope` (want 1.0), `pooled.brier` |
| "Does it get worse deeper into the stay?" | `landmark.json` → `by_landmark[].auroc` across time |
| "What happens at our cutoff?" | `hospitalization_level.json` → the `is_clinical` row |
| "How many alerts per real case?" | that row → `nne` |
| "How many stays would we alarm on?" | that row → `flag_rate` |
| "Where would the data put the cutoff?" | `selected_thresholds` → `youden_j.threshold` (and mind `in_sample`) |
| "Does it treat groups equally?" | `fairness.subgroups` (ranking) + `fairness.parity` (flag rates) |
