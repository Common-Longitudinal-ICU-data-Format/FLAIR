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
  --mode episodic          # optional: inferred from temporal_pattern (warns if omitted)
```

The report is **always** scored on the `test` split. If your file has no `test`
rows, the build stops with an error.

---

## 1. What files come out

```
report/
  overall.json                  ranking + calibration, threshold-free
  hospitalization_level.json    everything that depends on the threshold
  viz/                          local PNGs, never uploaded
```

**Two files. That is the whole bundle** (~95 KB on a 700-stay cohort). Only the
JSON is uploaded.

`hospitalization_level.json` is the **same schema continuous mode produces**, so a
reader written once works for every task and every site.

---

## 2. Rules that apply to every file

**Every JSON file starts with the same `metadata` block** (added on top of the
file's own metadata keys):

| key | meaning |
|---|---|
| `task` | task name, e.g. `icu_readmission` |
| `task_script_sha256` | hash of the task file, its META policy and `constants.py` (benchmark window, stitch gap) — proves every site ran the same definition |
| `flair_version` | FLAIR version that built the report |
| `generated_at` | UTC timestamp of the build |
| `report_mode` | `"episodic"` |
| `split` | always `"test"` |
| `n_stays` | number of unique stays scored |
| `prevalence` | event rate in the scored rows |
| `report_schema` | shape version of the bundle (currently `4`) |
| `evaled_at` | the task's clinical threshold |

**Confidence intervals.** Anything ending in `_ci` is a `[low, high]` pair
(`nne_ci` is the inverted PPV CI, so its `high` is `null` when PPV's lower bound is 0) =
95% percentile bootstrap, **500 replicates, fixed seed 20260612**. One row = one
stay, so it is a proper patient-level bootstrap. A CI is `null` when fewer than
250 replicates produced a usable value. Cohorts over 100,000 rows are subsampled
once before resampling (the fairness parity bootstrap uses all rows).

**Privacy.** Any count below **10** is written as the string `"<10"` instead of the
number, and the confusion matrix masks **in pairs** — `tp` with `fp`, `fn` with
`tn` — because `n` is published and a lone masked cell is recoverable by
subtraction. Any demographic subgroup with fewer than 10 patients is **dropped
entirely**.

**Rounding.** Scalars are rounded to 4 decimals and curve arrays to 6, except:
`prevalence`, `net_benefit` and `net_benefit_treat_all` (6), `nne` (1), and the
calibration `quantile_bins` arrays (4).

**Two ways of reading a threshold** (this matters for `extubation_failure_24h`):

- **Alarm basis** — `y_prob >= t`, always, in both directions. This is what
  sensitivity / specificity / PPV / the confusion matrix are built on, so
  "sensitivity" always means "how well do we catch the event".
- **Decision basis** — the act actually taken. For a `below` task the act fires on
  LOW risk (`y_prob < t`, extubate). Only `net_benefit` uses it.

---

## 3. `overall.json` — how good is the model, ignoring the cutoff?

### 3.1 `discrimination` — can it tell patients apart?

| key | meaning |
|---|---|
| `auroc`, `auroc_ci` | probability a random event outranks a random non-event. 0.5 = coin flip |
| `auprc`, `auprc_ci` | area under the precision-recall curve — the one that matters for rare events |
| `auprc_baseline` | the prevalence; a no-skill model scores exactly this |
| `tpr_at_grid` | 101 numbers: the ROC curve, sampled at FPR 0.00, 0.01 … 1.00 |
| `precision_at_grid` | 101 numbers: the PR curve, sampled at recall 0.00, 0.01 … 1.00 |

**Why the curves are single arrays.** The x-axes are fixed constants, so they are
never written down — and two sites' curves can be averaged point-by-point without
interpolating one onto the other's thresholds. Both curves are stored as their
**upper envelope**: at FPR 0 several TPRs are valid (every event that outranks
every non-event adds one), and at recall 1.0 several precisions are valid, so the
best value at each grid point is the one published.

### 3.2 `calibration` — does "20%" really mean 20%?

| key | meaning |
|---|---|
| `brier`, `brier_ci` | mean squared error of the probabilities. Lower is better |
| `slope`, `slope_ci` | **1.0 = perfect.** Below 1 = overconfident (the classic overfit signature); above 1 = underconfident |
| `intercept`, `intercept_ci` | **0.0 = perfect.** Negative = the model over-predicts risk overall |
| `quantile_bins.bin_pred_prob` | 10 numbers: mean predicted probability per bin |
| `quantile_bins.bin_true_prob` | 10 numbers: observed event rate per bin. `null` where the bin holds <10 stays |

**Bins hold equal COUNTS, not equal widths.** For a rare event nearly every
prediction falls into the first equal-width bin and the curve collapses to a single
point; equal-count binning puts ~n/10 stays in each bin and spreads ten readable
points across the range the model actually uses. Counts and edges are not published
— the two arrays are all a reliability plot needs.

There is **no ECE** (it is defined on the equal-width bins that were removed) and
**no separate DCA file** — see §4.

---

## 4. `hospitalization_level.json` — what happens at a cutoff?

### 4.1 `metadata`

| key | meaning |
|---|---|
| `surface` | `"stay"` — one prediction per hospitalization, nothing collapsed |
| `score_rule` | `"y_prob (one prediction per hospitalization)"` |
| `clinical_threshold` / `clinical_direction` | the shipped policy |
| `grid` | `{min: 0.01, max: 0.99, step: 0.01, n: 99}` — identical at every site |
| `clinical_index` | which of the 99 rows is the shipped policy |
| `min_cell_count` | `10` |
| `boot_reps` | `500` — bootstrap replicates behind every `_ci` in this file |
| `flag_rule` | `"score >= threshold (alarm convention)"` — the basis the 2×2 is built on |
| `selection_note` | why `youden_j`'s `threshold_ci` is selection-aware and the clinical policy's is null |

### 4.2 `by_threshold` — 99 rows

One row per cutoff from 0.01 to 0.99. The same grid at every site and for every
task, so two sites' sweeps line up row-for-row and pool without interpolation.

| key | meaning |
|---|---|
| `threshold` | the cutoff |
| `is_clinical` | `true` on exactly one row — the shipped policy |
| `n` | stays scored |
| `tp`, `fp`, `fn`, `tn` | the 2×2, masked in pairs when small |
| `sensitivity` (+`_ci`) | of the events, how many did we catch |
| `specificity` (+`_ci`) | of the non-events, how many did we leave alone |
| `ppv` (+`_ci`) | of the alarms, how many were real. `null` when <10 stays are flagged |
| `npv` (+`_ci`) | of the non-alarms, how many were truly fine |
| `f1` (+`_ci`) | harmonic mean of PPV and sensitivity |
| `fpr` (+`_ci`) | false alarm rate = 1 − specificity |
| `flag_rate` | fraction of all stays flagged — the workload |
| `nne` (+`_ci`) | **alarms per true case** = 1/PPV. The number a screener lives or dies by |
| `net_benefit` (+`_ci`) | the decision curve, on the decision basis |
| `net_benefit_treat_all` | the "act on everyone" comparator. "Act on no one" is 0 by definition |

**The decision curve lives here.** `net_benefit` and `net_benefit_treat_all` across
these 99 rows *are* the DCA — there is no `dca.json`, because storing the same array
twice is how two files end up disagreeing.

**Reading it:** the model earns its keep at threshold t when `net_benefit` exceeds
both `net_benefit_treat_all` and 0 at that row.

### 4.3 `selected_thresholds` — a list of one or two entries

A list of dicts, one per `rule`. The `youden_j` entry is omitted when Youden's J
cannot be computed; the `clinical` entry is always present.

| key | meaning |
|---|---|
| `rule` | `"youden_j"` or `"clinical"` |
| `threshold` | where it lands |
| `threshold_ci` | for Youden: **selection-aware** — the cutpoint is re-picked on every bootstrap resample, so this is wider than a fixed-threshold band. `null` for the clinical policy, which is not fitted |
| `in_sample` | `true` for Youden, `false` for the clinical policy |
| `citation` | `"Youden 1950"` for Youden, `null` for the clinical policy |
| `delta_from_clinical` | how far the rule sits from the shipped policy |
| `sensitivity`, `specificity`, `ppv`, `npv`, `f1`, `fpr`, `flag_rate`, `nne` | that row's metrics |

**Why only Youden.** Ten other cutpoint rules were dropped. Youden's J
(sensitivity + specificity − 1) is the most-cited and the easiest to explain, and
the honest object is the full sweep anyway — every rule is just an annotation on it.
`in_sample: true` is the warning label: a cutpoint chosen to maximise a metric on
the rows you then report is upward biased (Leeflang 2008).

### 4.4 `fairness` — at the clinical threshold only

| key | meaning |
|---|---|
| `at_threshold` | the clinical threshold — this block is ONE point, not 99 |
| `subgroups.<column>.<level>.n` | stays in that group. Groups under 10 never appear |
| `… .prevalence` | event rate in the group |
| `… .auroc` (+`_ci`) | does the model **rank** this group as well as the others |
| `… .positive_rate` | how often the group gets flagged |
| `… .tpr`, `… .fpr` | catch rate and false-alarm rate within the group |
| `parity.<column>.reference_group` | the largest level |
| `… .demographic_parity_gap` (+`_ci`) | max − min flag rate across levels |
| `… .equalized_odds_tpr_gap` (+`_ci`) | max − min TPR across levels |
| `… .equalized_odds_fpr_gap` (+`_ci`) | max − min FPR across levels |

Columns: `sex_category`, `race_category`, `ethnicity_category`, `age_bin`.
The whole block is `null` when the predictions frame carries no demographics —
pass `--cohort` and they are joined in.

---

## 5. The PNGs (local only, never uploaded)

| file | shows |
|---|---|
| `roc.png` | the ROC curve with AUROC + CI |
| `pr.png` | the PR curve with AUPRC + CI and the prevalence baseline |
| `calibration.png` | the ten quantile reliability points against the diagonal |
| `dca.png` | net benefit vs treat-all vs treat-none |
| `fairness.png` | subgroup AUROC with CIs |
| `threshold_sweep.png` | six panels — sensitivity, specificity, PPV, NPV, NNE, flag-rate — each against the threshold, all but flag-rate with a CI band |

On every threshold plot the shipped policy is the **red dashed** line and Youden's
pick is the **green dotted** line.

---

## 6. Quick cheat sheet — which number do I quote?

| the question | where to look |
|---|---|
| "Does the model rank well?" | `overall.json` → `discrimination.auroc`, `discrimination.auprc` |
| "Are the probabilities honest?" | `overall.json` → `calibration.slope` (want 1.0), `calibration.brier` |
| "Is it worth deploying at all?" | `hospitalization_level.json` → the `is_clinical` row's `net_benefit` vs `net_benefit_treat_all` |
| "What happens at our cutoff?" | that same `is_clinical` row |
| "How many alerts per real case?" | that row → `nne` |
| "Where would the data put the cutoff?" | `selected_thresholds` → the entry with `rule == "youden_j"` → `threshold` (and mind `in_sample`) |
| "Does it treat groups equally?" | `fairness.subgroups` (ranking) + `fairness.parity` (flag rates) |
