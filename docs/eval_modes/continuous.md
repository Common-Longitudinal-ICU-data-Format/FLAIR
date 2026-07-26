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

Your predictions must carry `window_number` (1, 2, 3 … per stay) and — for the
timing analyses — `prediction_dttm`. If they are not in your preds file, pass
`--cohort` and FLAIR joins them in. As always, only the `test` split is scored.

---

## The three honest ways to score a repeating model

This is the one idea you need before reading the files. Because a stay produces
many predictions, "the AUROC" is ambiguous. Continuous mode reports **two** views
plus timing, all in the same bundle:

| view | the question it answers | where it lives |
|---|---|---|
| **peak** | "Did this stay ever alarm, vs did it ever have the event?" — the screening view | the `"peak"` section of `discrimination.json` / `fairness.json`, and all of `operating_points.json` |
| **landmark** | "On hospitalization day 5, how well does the model separate the patients still at risk?" — the time-varying view | the `"landmarks"` list in `discrimination.json` / `calibration.json` / `fairness.json` |
| **lead time** | "How early in the stay does it first fire?" | `leadtime.json` |

**Why calibration is missing from the peak section.** Taking the *maximum* risk
over 20 days is biased upward by construction — a reliability curve on it would
say the model is over-confident even when it is perfect. This is the
"peak-calibration trap". Honest calibration lives in the landmark section only.

**Why there is no `dca.json` in continuous mode.** Same trap: net benefit on a
max-of-many surface reads wrong, and there is no landmark decision curve either.
**A continuous task simply has no decision curve anywhere in its bundle.**

**What a landmark is.** Landmark *t* is the slice `window_number == t` — every
stay still producing predictions on hospitalization day *t*. Because task builders
only generate windows before the event/discharge, that slice is automatically
"still alive, still event-free" — a true Kaplan-Meier *No. at risk* set. Nothing is
anchored on the event, so it is exactly the group a deployed model would score
that morning. The number of landmarks comes from the task's `landmark_horizon`
(30 for both shipped tasks) — never hardcoded.

---

## 1. What files come out

```
report/
├── discrimination.json      ranking — peak section + one entry per landmark
├── calibration.json         landmarks only (no peak — see the trap above)
├── fairness.json            subgroups — peak section + one entry per landmark
├── leadtime.json            how early the model first fires
├── operating_points.json    THE decision-point file (99 thresholds, peak surface)
├── kpi.json                 the headline scoreboard
└── viz/
    ├── peak_roc_pr.png
    ├── peak_fairness_auroc.png
    ├── landmark_auc_by_hospital_time.png
    ├── landmark_calibration.png
    ├── landmark_brier.png
    ├── landmark_reliability.png
    ├── alert_burden.png
    ├── data_driven_thresholds.png
    ├── operating_points_confusion.png
    └── operating_points_subanalyses.png    (only for intervention tasks)
```

**6 JSON files, up to 10 PNGs. No `dca.json`.** Only the JSON is uploaded. PNGs
need `pip install "flair-benchmark[viz]"`; `--no-viz` skips them.

---

## 2. Rules that apply to every file

**Every JSON file carries the same audit `metadata`:** `task`,
`task_script_sha256`, `flair_version`, `generated_at`, `report_mode`
(`"continuous"`), `split` (`"test"`), `n_stays`, `prevalence`, `report_schema`
(`3`), `evaled_at` (the clinical threshold).

**Confidence intervals.** `_ci` = `[low, high]`, 95% percentile bootstrap, 500
replicates, fixed seed. Crucially, every bootstrap in continuous mode resamples
**stays, not rows** — either because the frame was first collapsed to one row per
stay (peak, lead time, operating points) or because a landmark slice already has
at most one row per stay. A row bootstrap over all windows would understate the
interval, since a patient's days are correlated.

**Privacy.** Counts under 10 become `"<10"`. Subgroups with n < 10 are dropped.
A **whole landmark** whose risk set falls under 10 stays is replaced by a stub.

**Two ways of reading a threshold.** *Alarm basis* (`y_prob >= t`, always) drives
the confusion matrix and sensitivity/specificity/PPV. *Action basis*
(direction-aware) drives first-alert timing. See
[episodic.md §2](episodic.md#2-rules-that-apply-to-every-file).

**Landmark entries come in three shapes.** Every pillar's `landmarks` list always
has exactly `landmark_horizon` entries, in order, so the arrays never go ragged:

| shape | keys | when |
|---|---|---|
| **scored** | the tag keys + the full pillar payload | normal |
| **suppressed** | `landmark_index`, `hospitalization_time`, `window_unit`, `n: "<10"`, `suppressed: true` | fewer than 10 stays still at risk |
| **undefined** | `landmark_index`, `hospitalization_time`, `window_unit`, `n`, `undefined: true`, `reason: "single label class at this hospitalization time"` | everyone at that time had the same outcome, so AUROC is meaningless |

The four tag keys on every entry:

| key | plain meaning |
|---|---|
| `landmark_index` | 0-based position in the list |
| `hospitalization_time` | 1-based day (or hour) since the stay's first prediction |
| `window_unit` | `"day"` or `"hour"`, from the task |
| `n` | stays still at risk at that time (the Kaplan-Meier *No. at risk*) |

---

## 3. `discrimination.json` — ranking

Two sections: `"peak"` and `"landmarks"`.

### 3.1 `peak` — the screening view

Each stay is collapsed to its **highest** predicted risk, and the label to "did
this stay ever have the event". Then it is scored exactly like a one-per-stay task.

| key | plain meaning |
|---|---|
| `peak.task_type` | `binary` |
| `peak.prevalence` | share of stays that ever had the event |
| `peak.metrics.auroc` (+ `_ci`) | AUROC on peak risk — "did event stays ever alarm louder than non-event stays" |
| `peak.metrics.auprc` (+ `_ci`) | AUPRC on peak risk |
| `peak.metrics.auprc_baseline` | prevalence (what a useless model scores) |
| `peak.metrics.auprc_lift` | AUPRC ÷ prevalence |
| `peak.metrics.operating_point.threshold` | the clinical threshold |
| `…​.direction` | `above` / `below` |
| `…​.sensitivity` / `specificity` / `ppv` / `npv` / `f1` (each + `_ci`) | the usual rates at that threshold |
| `…​.flag_rate` | share of stays that ever alarm |
| `…​.tp` / `fp` / `fn` / `tn` | the 2×2 counts (`"<10"` when small) |
| `…​.n_flagged` | how many stays ever alarmed |
| `…​.nne` (+ `_ci`) | **alarms per true case** = 1/PPV. This is the headline number for an alerting system — Wong 2021's "NNE ≈ 109" for the Epic sepsis model is exactly this |
| `…​.action.acts_on` | `"y_prob >= threshold"` or `"y_prob < threshold"` |
| `…​.action.n_acted` / `act_rate` | how many stays get the action |
| `…​.action.event_rate_in_acted` / `event_rate_in_not_acted` | observed event rate in each arm |
| `peak.curves.roc.*` | `fpr`, `tpr`, `thresholds`, `grid_fpr`, `tpr_at_grid` |
| `peak.curves.pr.*` | `recall`, `precision`, `thresholds` |
| `peak.metadata.n` | rows in the collapsed frame |
| `peak.metadata.downsample_max` | `500` |
| `peak.metadata.section` | `"peak"` |
| `peak.metadata.n_stays` | number of stays |
| `peak.metadata.peak_calibration_omitted` | `true` — the reminder about the trap |

### 3.2 `landmarks` — the time-varying view

One entry per hospitalization day. A scored entry carries the four tag keys plus:

| key | plain meaning |
|---|---|
| `task_type`, `prevalence` | as above, but for that day's risk set only |
| `metrics.auroc` (+ `_ci`) | **the dynamic AUC** — how well the model separates patients *on that day*. Plotting this against day is the standard time-varying-AUC figure |
| `metrics.auprc` (+ `_ci`), `auprc_baseline`, `auprc_lift` | precision-recall on that day |
| `metrics.operating_point.*` | the same operating-point block as the peak section, but for that day (landmarks keep it inline, because the point is to watch the decision drift over time) |
| `curves.roc.*`, `curves.pr.*` | that day's curves |
| `metadata.*` | `n`, `downsample_max`, plus the four tag keys repeated |

---

## 4. `calibration.json` — landmarks only

| key | plain meaning |
|---|---|
| `metadata.peak_omitted_reason` | the sentence explaining the peak-calibration trap |
| `landmarks[i]` | tag keys + one calibration payload per day |

Each scored landmark entry:

| key | plain meaning |
|---|---|
| `task_type`, `prevalence` | for that day's risk set |
| `metrics.ece` (+ `_ci`) | Expected Calibration Error — average gap between predicted and observed. Lower is better |
| `metrics.brier` (+ `_ci`) | Brier score. Lower is better |
| `metrics.slope` (+ `_ci`) | calibration slope. **1.0 = perfect**, below 1 = over-confident. Watching this across days is the drift check |
| `metrics.intercept` (+ `_ci`) | calibration-in-the-large. 0 = perfect |
| `metrics.o_e_ratio` | observed ÷ expected events |
| `curves.reliability.*` | 10 equal-width bins: `n_bins`, `bin_edges`, `bin_pred_prob`, `bin_true_prob` (`null` under 10 patients), `bin_counts`, `ece` |
| `curves.reliability_quantile.*` | the same, with equal-count bins — the readable curve for rare events. Display only, carries `binning: "quantile"` |
| `curves.smoothed.*` | `method`, `fit_slope`, `fit_intercept`, `pred_grid` (50 points), `obs_smoothed` |
| `metadata.*` | `n`, `n_bins` (10), `threshold_used` (`null`), `note`, plus the tag keys |

---

## 5. `fairness.json` — subgroups, peak + landmarks

Grouping columns: `sex_category`, `race_category`, `ethnicity_category`, `age_bin`
(`18-39` / `40-64` / `65-79` / `80+`; missing age is dropped). Groups with fewer
than 10 patients never appear.

Unlike episodic mode, continuous fairness **keeps the threshold-dependent numbers
inline** — both in the peak section and per landmark — because the whole point is
to watch the decision behave over time.

| key | plain meaning |
|---|---|
| `peak.overall.*` | the whole-cohort operating point at the clinical threshold: `threshold`, `direction`, `sensitivity`/`specificity`/`ppv`/`npv`/`f1` (+ `_ci`), `flag_rate`, `tp`/`fp`/`fn`/`tn`, `n_flagged`, `action{...}`, plus `n`, `prevalence`, `accuracy` — the baseline each group is compared against |
| `peak.subgroups.<col>.<level>.n` | patients in the group |
| `…​.prevalence` | event rate in the group |
| `…​.positive_rate` | share of the group flagged at the clinical threshold |
| `…​.tpr` | of the group's event patients, share flagged |
| `…​.fpr` | of the group's non-event patients, share flagged |
| `…​.auroc` (+ `_ci`), `auprc`, `brier` | ranking + accuracy within the group |
| `…​.calibration_slope`, `calibration_intercept` | calibration within the group |
| `peak.parity.<col>.reference_group` | the largest group (the anchor) |
| `…​.demographic_parity_gap` (+ `_ci`) | biggest minus smallest **flag rate** across groups. 0 = equal treatment |
| `…​.equalized_odds_tpr_gap` (+ `_ci`) | biggest minus smallest **TPR** |
| `…​.equalized_odds_fpr_gap` (+ `_ci`) | biggest minus smallest **FPR** |
| `peak.metadata.min_subgroup_n` | `10` |
| `peak.metadata.operating_threshold` / `threshold` | the threshold used |
| `peak.metadata.clinical_direction` | `above` / `below` |
| `peak.metadata.flag_rule` | `"y_prob >= threshold"` |
| `peak.metadata.section` / `n_stays` / `peak_calibration_omitted` | peak-section tags |
| `landmarks[i]` | the same `overall` / `subgroups` / `parity` / `metadata` shape, one per hospitalization day, plus the tag keys |

---

## 6. `leadtime.json` — how early does it fire?

One number, deliberately. (The old per-day sensitivity curve and mean-risk
trajectory were removed: the operating-point analysis is encounter-level only.)

| key | plain meaning |
|---|---|
| `task_type` | `binary` |
| `prevalence` | event rate across the prediction rows |
| `metrics.n_positive_stays` | stays that eventually had the event |
| `metrics.n_caught` | how many of those the model ever fired on |
| `metrics.pct_caught` | that as a fraction — **the "did we ever catch it" number** |
| `metrics.median_first_alert_hospitalization_time` | among caught stays, the median day (or hour) of the **first** alert. Lower = earlier warning. `null` when fewer than 10 stays were caught |
| `metrics.median_first_alert_hospitalization_time_ci` | 95% CI (stay-level bootstrap of the median) |
| `metadata.clinical_threshold` | the threshold used |
| `metadata.window_unit` | `"day"` or `"hour"` |
| `metadata.min_cell_count` | `10` |
| `metadata.first_alert_basis` | `"action (direction-aware)"` — for a `below` task "firing" means risk *dropping under* the threshold |
| `metadata.reason` | present instead of metrics if the task is not binary or has no threshold |

---

## 7. `operating_points.json` — the decision-point file

Everything that depends on where you put the cutoff. It runs on the **peak
surface**: each stay collapsed to one alarm ("did this stay ever cross the
threshold") versus one label ("did this stay ever have the event"). That is the
real-world unit — one alert per admission.

Same fixed grid as episodic mode: **99 thresholds, 0.01 … 0.99**, with the shipped
policy flagged as the single `is_clinical: true` row.

### 7.1 `metadata`

| key | plain meaning |
|---|---|
| `clinical_threshold`, `clinical_direction` | the task's policy |
| `grid` | `{min: 0.01, max: 0.99, step: 0.01, n: 99}` |
| `surface` | `"peak"` |
| `clinical_index` | which of the 99 rows is the clinical one |
| `min_cell_count` | `10` |
| `selection_params` | constants for the cutpoint rules: `sens_target` 0.90, `spec_target` 0.90, `ppv_target` 0.30, `alert_rate_target` 0.10, `alert_rate_targets` [0.05, 0.10, 0.20], `f_beta` 1, `cost_ratio` 1 |
| `data_driven_ci` | warning that the fitted rules' CIs are selection-aware (wider) |
| `intervention` | the task's treatment-column config, or `null` |
| `window_unit` | `"day"` / `"hour"` |
| `unit_of_analysis` | `"encounter (peak)"` |
| `net_benefit_omitted_reason` | the sentence explaining why there is no net benefit / no DCA here |
| `sub_analyses_reason` | present when the task declares no intervention column |
| `reason` | present when the whole section could not be built |

### 7.2 `by_threshold` — 99 rows

**`threshold`**, **`is_clinical`**, then:

**`confusion`** — stay-level alert × label 2×2: `n`, `tp`, `fp`, `fn`, `tn`
(`"<10"` when small). "Fired" = the stay ever crossed the threshold; "positive" =
the stay ever had the event.

**`metrics`** — `sensitivity`, `specificity`, `ppv`, `npv`, `f1`, `fpr` (each with
`_ci`), `nne` + `nne_ci` (alarms per true case, 1/PPV), and `flag_rate` (no CI).
`ppv`/`nne` are `null` when fewer than 10 stays were flagged.

**`alert_burden`** — the workload view, in units clinicians recognise:

| key | plain meaning |
|---|---|
| `frac_stays_alerting` | what fraction of admissions produce at least one alert |
| `ppv` | of the alerting admissions, what share were real |
| `false_alarm_fraction` | `1 − ppv` — the share of alerts that were noise |
| `nne` | admissions you must review per true case |
| `n_stays` | total stays |
| `n_stays_alerting` | how many alerted (`"<10"` when small) |

**`sub_rates`** — *only present when the task declares an `intervention`
(treatment-timestamp) column.* It answers "what is my alarm made of?" — the three
arms partition every stay, so the shares sum to 1 (unless one is suppressed):

| key | plain meaning |
|---|---|
| `positive_share` | of all firing stays, the share that really had the event |
| `treated_negative_share` | share that were treated but never labelled positive |
| `untreated_misfire_share` | share that were never treated and never positive — pure false alarms |
| `n_alarms` | total firing stays (`"<10"` when small) |

> **Note for the shipped tasks:** neither `icu_daily_mortality` nor
> `icu_daily_ltach` declares an intervention column today, so `sub_rates`,
> `intervention_timing`, `intervention_offsets` and
> `operating_points_subanalyses.png` are empty/absent for them. They activate
> automatically for any future task whose META sets `intervention`.

### 7.3 `data_driven_thresholds` — 12 entries

Identical in shape and meaning to episodic mode, but computed on the peak surface
(`surface: "peak"`). 11 statistical cutpoint rules — `youden_j`, `closest_01`,
`concordance_cz`, `index_of_union`, `f_beta_max`, `sensitivity_target`,
`specificity_target`, `cost_based`, `ppv_target`, `alert_rate_target`,
`max_accuracy` (flagged `degenerate`) — plus a `clinical` entry for your shipped
policy.

Keys per entry: `rule`, `threshold`, `threshold_ci`, `objective_value`, `surface`,
`delta_from_clinical`, `metrics_at` (`sensitivity`, `specificity`, `ppv`, `npv`,
`f1`, `fpr`, `flag_rate`, `nne`, each with a `_ci` except `flag_rate`),
`in_sample`, `assumption`, `citation`, and optionally `degenerate` / `reason`.
Full explanations of each rule are in
[episodic.md §7.3](episodic.md#73-data_driven_thresholds--12-entries).

These are **annotations, not recommendations** — a cutoff fitted on the same rows
it is then scored on is optimistic, hence `in_sample: true`.

### 7.4 `operating_budget`

"Different models put out different score scales, so comparing them 'at 0.3' is
meaningless. Compare them at **equal workload** instead." One entry per target
alert rate (5%, 10%, 20% of admissions):

| key | plain meaning |
|---|---|
| `target_flag_rate` | the workload budget (0.05 / 0.10 / 0.20) |
| `threshold` | the grid threshold that comes closest to it |
| `sensitivity` | what share of events you catch at that workload |
| `ppv` | precision at that workload |
| `nne` | reviews per true case at that workload |

### 7.5 `intervention_timing` and `intervention_offsets`

Only for tasks with an intervention column; otherwise `[]` and `null`.

| key | plain meaning |
|---|---|
| `intervention_timing[i].threshold` | one entry per named threshold (clinical + every located cutpoint) |
| `…​.is_clinical` | `true` for the shipped policy |
| `…​.median_hours_before_treatment_positive` | among event stays that fired and were treated: the median hours the alarm came **before** treatment. Positive = the model warned early. `null` under 10 stays |
| `intervention_offsets.n` | treated stays used for the reference line |
| `intervention_offsets.window_unit` | `"day"` / `"hour"` |
| `intervention_offsets.mean_hospitalization_time` | on average, at what point in the stay treatment happens |
| `intervention_offsets.sd` | its standard deviation |

### 7.6 `prevalence`

Share of stays that ever had the event.

---

## 8. `kpi.json` — the headline scoreboard

Pure aggregation of the files above (plus one recomputation), so it can never
disagree with them. This is the file to read first.

| key | plain meaning |
|---|---|
| `pooled_dynamic_auroc` | **the single headline AUROC** for a repeating model. Each day's AUC, pooled with Mann-Whitney pair weights (`cases × controls`) — the Heagerty & Zheng time-varying concordance. It pools over the **whole stay**, not just the 30 displayed landmarks |
| `pooled_dynamic_auroc_ci` | 95% CI from a **stay-level** pooled bootstrap — one stay resample per replicate, every day's AUC recomputed on it. This respects the fact that the same patient appears on many days |
| `metadata.horizon` | number of displayed landmarks |
| `metadata.window_unit` | `"day"` / `"hour"` |
| `metadata.method` | a paragraph naming the methods and citations |
| `metadata.pooled_auroc_scope` | `"full_stay"` (or `"display_horizon"` in a bare unit call) |
| `metadata.pooled_auroc_n_windows` | how many time windows were pooled |
| `peak.auroc` (+ `_ci`), `peak.auprc` (+ `_ci`) | the stay-peak screening headline |
| `peak.sensitivity`, `peak.specificity`, `peak.ppv`, `peak.flag_rate` | peak operating point at the clinical threshold |
| `peak.nne` (+ `_ci`) | alarms per true case at the peak operating point |
| `calibration.at_hospitalization_time` | which day the calibration numbers come from — the earliest *scored* landmark (day 1 may be suppressed) |
| `calibration.slope` (+ `_ci`), `intercept` (+ `_ci`), `ece`, `brier` (+ `_ci`), `o_e_ratio` | calibration on that day |
| `calibration.slope_range` | `[min, max]` slope across all scored landmarks — **the drift check**. A wide range means calibration changes as the stay goes on |
| `leadtime.pct_caught` | share of event stays the model ever fired on |
| `leadtime.median_first_alert_hospitalization_time` (+ `_ci`) | median day of first alert |
| `operating_points.clinical_threshold` | the shipped policy |
| `operating_points.sensitivity` / `specificity` / `fpr` / `ppv` | at that policy |
| `operating_points.nne` (+ `_ci`) | alarms per true case at that policy |
| `operating_points.fpr_by_threshold` | 99 `{threshold, fpr}` pairs — the whole false-alarm sweep in one array |
| `operating_points.alert_burden` | the clinical row's burden block (see §7.2) |
| `operating_points.selected_thresholds.youden_j` | `{threshold, sensitivity, ppv}` — where Youden would have put the cutoff |
| `operating_points.selected_thresholds.f_beta_max` | the same for the F1 rule |
| `operating_points.sub_analyses` | `null` unless the task declares an intervention; otherwise `positive_share`, `treated_negative_share`, `untreated_misfire_share`, `n_alarms`, `median_hours_before_treatment_positive` |

---

## 9. The PNGs (local only, never uploaded)

| file | what it shows |
|---|---|
| `viz/peak_roc_pr.png` | ROC + precision-recall on the stay-peak collapse, with the operating point marked |
| `viz/peak_fairness_auroc.png` | AUROC per demographic subgroup on the peak surface |
| `viz/landmark_auc_by_hospital_time.png` | **the time-varying AUC curve** — AUROC ± 95% CI by hospitalization day, with a Kaplan-Meier style *No. at risk* row under the axis |
| `viz/landmark_calibration.png` | calibration slope ± CI by day, with a dashed line at 1.0 — the drift picture |
| `viz/landmark_brier.png` | Brier score ± CI by day (lower is better) |
| `viz/landmark_reliability.png` | reliability curves at three days: earliest, middle, latest scored |
| `viz/alert_burden.png` | twin axis: detection (sensitivity / PPV) against the fraction of admissions alerting, with the alarm-fatigue budget shaded |
| `viz/data_driven_thresholds.png` | where each statistical rule lands versus the shipped policy |
| `viz/operating_points_confusion.png` | the full alert × label 2×2 at the named thresholds (clinical / Youden / F1) |
| `viz/operating_points_subanalyses.png` | treatment fire-rate curves + median lead time — **only for intervention tasks** |

Suppressed or undefined landmarks leave a gap in the curves but still show their
`"<10"` in the at-risk row, so the risk table stays complete.

---

## 10. Quick cheat sheet — which number do I quote?

| the question | where to look |
|---|---|
| "One number for how good it is" | `kpi.json` → `pooled_dynamic_auroc` (+ CI) |
| "Does it get worse later in the stay?" | `viz/landmark_auc_by_hospital_time.png`, or `discrimination.json` → `landmarks[*].metrics.auroc` |
| "Are the probabilities drifting?" | `kpi.json` → `calibration.slope_range` |
| "How many alerts will my nurses get?" | `operating_points.json` → clinical row → `alert_burden.frac_stays_alerting` |
| "How many alerts per real case?" | same row → `alert_burden.nne` |
| "Do we catch the events, and how early?" | `leadtime.json` → `pct_caught`, `median_first_alert_hospitalization_time` |
| "Is it fair?" | `fairness.json` → `peak.parity` (and per landmark) |
| "What if we could only handle 5% alerts?" | `operating_points.json` → `operating_budget` |
