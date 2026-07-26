# Episodic mode — every file and every number

[← back to FLAIR 101](../101.md) · [continuous mode →](continuous.md)

**Plain English:** episodic mode is used when your model makes **one prediction per
hospital stay**. One patient stay = one row = one probability.

Shipped tasks that use it:

| task | clinical threshold | direction | meaning |
|---|---|---|---|
| `icu_readmission` | 0.20 | `above` | keep in ICU when readmission risk ≥ 0.20 |
| `extubation_failure_24h` | 0.10 | `below` | extubate when failure risk < 0.10 |

## The command

```bash
flair build-report preds.parquet \
  --task icu_readmission \
  --out ./report/ \
  --cohort cohort.parquet \
  --mode episodic          # optional: inferred from the task
```

The report is **always** scored on the `test` split. If your file has no `test`
rows, the build stops with an error.

---

## 1. What files come out

```
report/
├── discrimination.json      how well the model RANKS patients
├── calibration.json         do the probabilities mean what they say
├── dca.json                 is the model worth using at all (net benefit)
├── fairness.json            does it work equally well across groups
├── operating_points.json    THE scorecard — everything that depends on the cutoff
└── viz/                     pictures only, for your own eyes — never uploaded
    ├── roc_pr.png
    ├── calibration.png
    ├── dca.png
    ├── fairness_auroc.png
    └── operating_points.png
```

**5 JSON files, 5 PNGs.** Only the JSON is uploaded. The PNGs need
`pip install "flair-benchmark[viz]"`; without matplotlib you get a warning and the
JSON only. `--no-viz` skips them. Each rebuild wipes old PNGs in `viz/` first, so
you never look at a stale picture.

`report/` also does **not** contain `table1.json` — that one is written by
`flair load-task` next to your cohort file (`cohort.table1.json`), not by
`build-report`.

---

## 2. Rules that apply to every file

**Every JSON file starts with the same `metadata` block** (added on top of the
file's own metadata keys):

| key | meaning |
|---|---|
| `task` | task name, e.g. `icu_readmission` |
| `task_script_sha256` | hash of the task file — proves every site ran the same definition |
| `flair_version` | FLAIR version that built the report |
| `generated_at` | UTC timestamp of the build |
| `report_mode` | `"episodic"` |
| `split` | always `"test"` |
| `n_stays` | number of unique stays scored |
| `prevalence` | event rate in the scored rows |
| `report_schema` | shape version of the bundle (currently `3`) |
| `evaled_at` | the task's clinical threshold (centre of the sweep) |

**Confidence intervals.** Anything ending in `_ci` is a `[low, high]` pair =
95% percentile bootstrap, **500 replicates, fixed seed 20260612**. One row = one
stay, so it is a proper patient-level bootstrap. A CI is `null` when fewer than
250 replicates produced a usable value. Cohorts over 100,000 rows are subsampled
once before resampling.

**Privacy.** Any count below **10** is written as the string `"<10"` instead of the
number. Any demographic subgroup with fewer than 10 patients is **dropped
entirely** — it never appears in the file.

**Rounding.** Scalars are rounded to 4 decimals; curve arrays to 6.

**Two ways of reading a threshold** (this matters for `extubation_failure_24h`):

- **Alarm basis** — `y_prob >= t`, always, in both directions. This is what
  sensitivity / specificity / PPV / the confusion matrix are built on, so
  "sensitivity" always means "how well do we catch the event".
- **Decision basis** — the act actually taken. For a `below` task the act fires on
  LOW risk (`y_prob < t`, extubate). This is what `dca.json` and the `decision`
  block use.

---

## 3. `discrimination.json` — can the model tell patients apart?

Threshold-free. Nothing here changes if you move the cutoff.

| key | plain meaning |
|---|---|
| `task_type` | `binary` / `multiclass` / `regression` |
| `prevalence` | fraction of stays that had the event |
| `metrics.auroc` | AUROC: chance a random event patient scores higher than a random non-event patient. 0.5 = coin flip, 1.0 = perfect |
| `metrics.auroc_ci` | 95% CI for AUROC |
| `metrics.auprc` | area under the precision–recall curve — the honest one for rare events |
| `metrics.auprc_ci` | 95% CI for AUPRC |
| `metrics.auprc_baseline` | what AUPRC a useless model would get = the prevalence |
| `metrics.auprc_lift` | `auprc / prevalence` — how many times better than useless. Comparable across sites with different event rates |
| `curves.roc.fpr` | false-positive rate points of the ROC curve (≤500 points) |
| `curves.roc.tpr` | matching true-positive rate points |
| `curves.roc.thresholds` | the probability cutoff at each point (`null` where infinite) |
| `curves.roc.grid_fpr` | a fixed FPR ladder 0.00 → 1.00 in steps of 0.01 |
| `curves.roc.tpr_at_grid` | TPR read at that fixed ladder — lets many sites' ROC curves be averaged |
| `curves.pr.recall` | recall points of the precision–recall curve |
| `curves.pr.precision` | matching precision points |
| `curves.pr.thresholds` | the cutoff at each point |
| `metadata.n` | rows scored |
| `metadata.downsample_max` | curves are trimmed to at most 500 points (`500`) |
| `metadata.reason` | only appears when the metrics could not be computed, e.g. `"single label class"` |

**Not here:** sensitivity, PPV, confusion matrix. Those depend on a cutoff and live
in `operating_points.json`.

**Other task types.** A `multiclass` task replaces `curves` with
`curves.multiclass` = `{class_order, confusion_matrix, per_class_roc}`; a
`regression` task gets `curves.regression` =
`{quantiles, y_true_quantiles, y_pred_quantiles, residual_quantiles}`.

---

## 4. `calibration.json` — does "20%" really mean 20%?

Also threshold-free: it looks at the whole range of predicted probabilities.

| key | plain meaning |
|---|---|
| `task_type` | `binary` (calibration is only computed for binary) |
| `prevalence` | observed event rate |
| `metrics.ece` | Expected Calibration Error — average gap between predicted and observed, weighted by how many patients are in each bin. **Lower is better; 0 is perfect** |
| `metrics.ece_ci` | 95% CI |
| `metrics.brier` | Brier score = mean squared error of the probabilities. Lower is better |
| `metrics.brier_ci` | 95% CI |
| `metrics.slope` | calibration slope. **1.0 = perfect.** Below 1 = over-confident (scores too extreme); above 1 = under-confident |
| `metrics.slope_ci` | 95% CI |
| `metrics.intercept` | calibration-in-the-large. **0 = perfect.** Positive = the model under-predicts overall |
| `metrics.intercept_ci` | 95% CI |
| `metrics.o_e_ratio` | observed events ÷ expected events. 1.0 = the totals match |
| `curves.reliability.n_bins` | `10` |
| `curves.reliability.bin_edges` | the 11 edges of 10 equal-width bins (0.0, 0.1, … 1.0) |
| `curves.reliability.bin_pred_prob` | mean predicted probability inside each bin |
| `curves.reliability.bin_true_prob` | observed event rate inside each bin — **`null` if the bin holds fewer than 10 patients** |
| `curves.reliability.bin_counts` | how many patients fell in each bin |
| `curves.reliability.ece` | the ECE computed from these bins (same number as `metrics.ece`) |
| `curves.reliability_quantile.*` | the same five keys again, but with **equal-count** bins (~n/10 patients each) instead of equal-width. This is the readable curve for rare events, where everything piles up near 0. Display only — it does **not** feed ECE. Carries `binning: "quantile"` instead of `ece` |
| `curves.smoothed.method` | `"linear_fit"` |
| `curves.smoothed.fit_slope` | slope of a plain straight line fitted to observed-vs-predicted |
| `curves.smoothed.fit_intercept` | its intercept |
| `curves.smoothed.pred_grid` | 50 evenly spaced predicted probabilities (0.01 → 0.99) |
| `curves.smoothed.obs_smoothed` | the fitted line read at that grid |
| `metadata.n` | rows scored |
| `metadata.n_bins` | `10` |
| `metadata.threshold_used` | `null` — calibration never uses the clinical threshold |
| `metadata.note` | says the same thing in words |
| `metadata.reason` | only when calibration could not be computed |

---

## 5. `dca.json` — is the model actually worth using?

Decision Curve Analysis. The question it answers is not "is the model accurate"
but "**would acting on this model beat the two obvious default policies?**" — treat
everybody, or treat nobody.

Net benefit at a threshold `t` is `TP/n − (FP/n) × t/(1−t)`.

**Read the axis framing first.** For `icu_readmission` (`above`) the x-axis is
"probability of readmission required before keeping the patient in ICU", and the
0.20 policy sits at x = 0.20. For `extubation_failure_24h` (`below`) the whole
frame flips: the event becomes **successful extubation**, the x-axis is
"probability of success required before extubating", and the 0.10 failure-risk
policy sits at **x = 0.90**. That is why `prevalence` in this file can read ~0.95
for a task with a ~5% failure rate — `metadata.prevalence_of` names which event it
is counting.

| key | plain meaning |
|---|---|
| `task_type` | `binary` |
| `direction` | `above` or `below`, from the task |
| `basis` | a sentence saying whether the frame was complemented |
| `clinical_threshold_original` | the task's threshold on the label scale (e.g. 0.10) |
| `prevalence` | event rate **on this file's axis** (may be the complement) |
| `metrics.clinical_threshold` | where the shipped policy sits on the x-axis (0.90 for the extubation task) |
| `metrics.net_benefit_at_clinical_threshold.model` | the model's net benefit there |
| `…​.model_ci` | 95% CI for it |
| `…​.treat_all` | net benefit of acting on everyone |
| `…​.treat_none` | always `0.0` |
| `metrics.clinically_useful` | `true` if the model beats **both** defaults at the shipped policy |
| `metrics.verdict.clinically_useful` | same flag, repeated inside the verdict |
| `metrics.verdict.model` / `model_ci` / `treat_all` / `treat_none` | the same four numbers |
| `metrics.verdict.beats` | **one readable sentence**, e.g. "the model beats both defaults — net benefit 0.031, vs 0.004 for 'extubate' on everyone and 0 for 'keep the patient intubated' on everyone" |
| `metrics.useful_range` | `{low, high}` — the widest continuous stretch of thresholds where the model beats both defaults |
| `metrics.all_useful_intervals` | every such stretch, as `[low, high]` pairs |
| `curves.thresholds` | the x-axis grid (see `metadata.grid`) |
| `curves.net_benefit_model` | the model's curve; `null` where suppressed |
| `curves.net_benefit_all` | the treat-everyone curve |
| `curves.net_benefit_none` | `0.0` (a scalar, the flat treat-nobody line) |
| `curves.focus_range` | `[low, high]` hint for where to zoom the plot — the full grid stays in the file |
| `curves.suppressed_thresholds` | thresholds nulled because fewer than 10 patients were acted on |
| `metadata.grid` | `"0.001-0.05 step .002 + 0.01-0.99 step .01 + 0.90-0.999 step .002"` — dense at both tails where rare-event and complement decisions live |
| `metadata.min_flagged_n` | `10` |
| `metadata.n_test` | rows scored |
| `metadata.event` | what the curve counts as the event, in words |
| `metadata.action` | the act, e.g. `"extubate"` |
| `metadata.inaction` | the other real policy, e.g. `"keep the patient intubated"` |
| `metadata.x_axis` | a sentence describing the axis |
| `metadata.prevalence_of` | which event `prevalence` refers to |
| `metadata.clinical_threshold_on_axis` | policy position on this file's axis |
| `metadata.clinical_threshold_on_label_scale` | the same policy on the label scale |
| `metadata.reason` | only when DCA could not be computed |

The net benefit in this file and the one in `operating_points.json` are computed
through the same code, so **they cannot disagree**.

---

## 6. `fairness.json` — does it work equally well for everyone?

In episodic mode this file is **threshold-free**: it only asks whether the model
*ranks and calibrates* equally well across groups. The threshold-dependent equity
numbers (who gets flagged, at what error rates) are in `operating_points.json`, at
all 99 thresholds.

Four grouping columns, when the demographics are present:
`sex_category`, `race_category`, `ethnicity_category`, `age_bin`
(age is binned into `18-39`, `40-64`, `65-79`, `80+`; a missing age is dropped,
not silently counted as 80+).

Shape: `subgroups.<column>.<level>` → one dict per group level.

| key | plain meaning |
|---|---|
| `subgroups.<col>.<level>.n` | patients in that group (levels with n < 10 are **not written at all**) |
| `…​.prevalence` | event rate in that group |
| `…​.auroc` | AUROC within the group |
| `…​.auroc_ci` | 95% CI |
| `…​.auprc` | AUPRC within the group |
| `…​.brier` | Brier score within the group |
| `…​.calibration_slope` | calibration slope within the group |
| `…​.calibration_intercept` | calibration intercept within the group |
| `metadata.min_subgroup_n` | `10` |
| `metadata.threshold_free` | `true` |
| `metadata.note` | points you at `operating_points.json` for the decision-threshold view |

AUROC/AUPRC/Brier/slope are omitted for a group that has only one label class.

*(A `multiclass` task reports `accuracy` + `macro_f1` per group instead; a
`regression` task reports `mae`, `mean_y_true`, `mean_y_pred`.)*

---

## 7. `operating_points.json` — THE scorecard

This is the biggest file and the one that answers "what happens if we deploy at
cutoff X". Every threshold-dependent number in episodic mode lives here.

It sweeps a **fixed grid of 99 thresholds: 0.01, 0.02, … 0.99**. The grid is a
constant — not centred on your task — so two sites' files line up row-for-row. The
shipped clinical policy is simply the one row with `is_clinical: true`.

### 7.1 `metadata`

| key | plain meaning |
|---|---|
| `clinical_threshold` | the task's shipped cutoff |
| `clinical_direction` | `above` / `below` |
| `grid` | `{min: 0.01, max: 0.99, step: 0.01, n: 99}` |
| `surface` | `"stay"` (one row per stay) |
| `clinical_index` | which of the 99 rows is the clinical one (0-based) |
| `min_cell_count` | `10` |
| `selection_params` | the constants the cutpoint rules use: `sens_target` 0.90, `spec_target` 0.90, `ppv_target` 0.30, `alert_rate_target` 0.10, `alert_rate_targets` [0.05, 0.10, 0.20], `f_beta` 1, `cost_ratio` 1 |
| `data_driven_ci` | a paragraph warning that the fitted rules' CIs are *selection-aware* and therefore wider |
| `flag_rule` | `"y_prob >= threshold (alarm convention)"` |
| `net_benefit_basis` | explains that `decision.net_benefit` matches `dca.json`, not this row's TP/FP |
| `reason` | only when the scorecard could not be built (non-binary task, or no threshold) |

### 7.2 `by_threshold` — 99 rows, one per cutoff

Each row has:

**`threshold`** — the cutoff · **`is_clinical`** — `true` for exactly one row.

**`confusion`** — the 2×2 table at that cutoff (alarm basis, `y_prob >= t`):

| key | plain meaning |
|---|---|
| `n` | total patients |
| `tp` | flagged **and** had the event (`"<10"` if under 10) |
| `fp` | flagged but no event |
| `fn` | not flagged but had the event |
| `tn` | not flagged, no event |

**`metrics`** — the rates derived from that 2×2, each with a CI band:

| key | plain meaning |
|---|---|
| `sensitivity` (+ `_ci`) | of the patients who had the event, what share did we flag (recall) |
| `specificity` (+ `_ci`) | of the patients with no event, what share did we correctly leave alone |
| `ppv` (+ `_ci`) | of the patients we flagged, what share really had the event (precision). **`null` when fewer than 10 were flagged** |
| `npv` (+ `_ci`) | of the patients we did not flag, what share really had no event |
| `f1` (+ `_ci`) | harmonic mean of PPV and sensitivity |
| `fpr` (+ `_ci`) | true false-positive rate = FP/(FP+TN) |
| `nne` (+ `_nne_ci`) | **N**umber **N**eeded to **E**valuate = 1/PPV — how many alerts you review per real case. Lower is better |
| `flag_rate` | share of all patients flagged (your workload). No CI — it is a count, not an estimate |

**`decision`** — what the clinician actually *does* at that cutoff (direction-aware):

| key | plain meaning |
|---|---|
| `acts_on` | `"y_prob >= threshold"` or `"y_prob < threshold"` |
| `n_acted` | how many patients get the action (`"<10"` if under 10) |
| `act_rate` | that as a fraction |
| `event_rate_in_acted` | observed event rate among the acted-on patients |
| `event_rate_in_not_acted` | observed event rate among the rest |
| `dca_threshold` | where this cutoff sits on the decision-curve axis (`1−t` for a `below` task) |
| `net_benefit` | net benefit of the act here — **the same number as `dca.json`'s curve** at `dca_threshold` |
| `net_benefit_ci` | 95% CI |
| `net_benefit_treat_all` | net benefit of acting on everyone |
| `net_benefit_treat_none` | `0.0` |
| `clinically_useful` | `true` if the model beats both defaults here |
| `suppressed_reason` | present when fewer than 10 patients were acted on — the net-benefit keys are then `null` |

**`fairness`** (only when demographics are present) — the equity view **at that
cutoff**:

| key | plain meaning |
|---|---|
| `fairness.subgroups.<col>.<level>.n` | patients in the group |
| `…​.positive_rate` | share of that group flagged at this cutoff |
| `…​.tpr` | of that group's event patients, share flagged |
| `…​.fpr` | of that group's non-event patients, share flagged |
| `fairness.parity.<col>.reference_group` | the largest group, used as the comparison anchor |
| `…​.demographic_parity_gap` (+ `_ci`) | biggest minus smallest **flag rate** across groups. 0 = every group flagged at the same rate |
| `…​.equalized_odds_tpr_gap` (+ `_ci`) | biggest minus smallest **TPR** across groups |
| `…​.equalized_odds_fpr_gap` (+ `_ci`) | biggest minus smallest **FPR** across groups |

**Row-level suppression:** if fewer than 10 patients are flagged at a cutoff, the
row gets `suppressed: true` and `suppressed_reason`.

### 7.3 `data_driven_thresholds` — 12 entries

"If you had picked the cutoff by a statistical rule instead of by clinical policy,
where would you have landed?" These are **annotations, not recommendations** — a
cutoff chosen to maximise a metric on the same data it is then scored on is
optimistic, which is why every fitted rule is stamped `in_sample: true`.

The 11 rules:

| `rule` | picks the threshold that… |
|---|---|
| `youden_j` | maximises sensitivity + specificity − 1 (equal cost, ignores prevalence) |
| `closest_01` | is geometrically closest to the perfect corner (0,1) of ROC space |
| `concordance_cz` | maximises sensitivity × specificity (punishes lopsided points) |
| `index_of_union` | puts sensitivity and specificity both closest to the AUC |
| `f_beta_max` | maximises F-β (β = 1 → plain F1); ignores true negatives |
| `sensitivity_target` | gives the best specificity while sensitivity ≥ 0.90 — "a miss is worse than a false alarm" |
| `specificity_target` | gives the best sensitivity while specificity ≥ 0.90 — the reverse |
| `cost_based` | maximises sensitivity − slope × FPR with an explicit FP/FN cost ratio |
| `ppv_target` | is the lowest cutoff that still reaches PPV ≥ 0.30 |
| `alert_rate_target` | lands nearest a 10% alert workload |
| `max_accuracy` | maximises plain accuracy. **Flagged `degenerate: true`** — for rare events it collapses toward "never alert" |

Plus a 12th entry with `rule: "clinical"` — your shipped policy lined up in the
same list, with `in_sample: false` (it was not fitted).

Keys per entry:

| key | plain meaning |
|---|---|
| `rule` | rule name |
| `threshold` | where it landed (`null` if the rule is undefined on this data) |
| `threshold_ci` | 95% CI of the landing point — how unstable the rule is. `null` for the clinical row (it was never fitted) |
| `objective_value` | the value of the thing the rule maximised/minimised |
| `surface` | `"stay"` |
| `delta_from_clinical` | landing threshold minus your clinical threshold |
| `metrics_at.sensitivity` / `specificity` / `ppv` / `npv` / `f1` / `fpr` / `flag_rate` / `nne` | performance read off the sweep at that landing point |
| `metrics_at.*_ci` | CIs for all of the above except `flag_rate` |
| `in_sample` | `true` for fitted rules, `false` for the clinical row |
| `assumption` | one line saying what the rule assumes about costs |
| `citation` | where the rule comes from (e.g. `"Youden 1950"`) |
| `degenerate` | `true` only on `max_accuracy` |
| `reason` | present instead of the metrics when the rule could not be applied |

### 7.4 `prevalence`

Event rate over the scored rows.

---

## 8. The PNGs (local only, never uploaded)

| file | what it shows |
|---|---|
| `viz/roc_pr.png` | ROC curve and precision–recall curve side by side. Because episodic mode keeps no operating point inside `discrimination.json`, the ROC is annotated from the scorecard instead: the clinical point (with error bars) plus where the named data-driven rules land. The PR panel draws the prevalence baseline |
| `viz/calibration.png` | reliability curve (equal-count deciles) + the fitted line, with slope / intercept / Brier / ECE / O:E in a box, and a red warning if the slope is more than 0.3 away from 1 |
| `viz/dca.png` | the net-benefit curve against treat-all and treat-none, with the shipped policy marked |
| `viz/fairness_auroc.png` | AUROC per demographic subgroup with CIs |
| `viz/operating_points.png` | every scorecard metric as a trend across all 99 thresholds, plus a table of named thresholds |

Use these to sanity-check your run before uploading. They are rebuilt from the
same JSON dicts, so they can never show a different number than the JSON.

---

## 9. Quick cheat sheet — which number do I quote?

| the question | where to look |
|---|---|
| "Does the model rank well?" | `discrimination.json` → `metrics.auroc`, `metrics.auprc_lift` |
| "Are the probabilities honest?" | `calibration.json` → `metrics.slope`, `metrics.ece` |
| "Is it worth deploying at all?" | `dca.json` → `metrics.verdict.beats` (one sentence) |
| "What happens at our cutoff?" | `operating_points.json` → the row with `is_clinical: true` |
| "How many alerts per real case?" | that row → `metrics.nne` |
| "Does it treat groups equally?" | `fairness.json` (ranking) + that row's `fairness.parity` (flag rates) |
