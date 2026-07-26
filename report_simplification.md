# Report simplification — design

**Status:** IMPLEMENTED. Sections 1–10 describe the shipped design; §11 carries measured results.
**Date:** 2026-07-26
**Scope:** `flair_benchmark/report/` (both `--mode continuous` and `--mode episodic`)

---

## 1. The goal, in one sentence

Every site should upload **two small JSON files per task**, holding the fewest
numbers that still let anyone evaluate the model — and nothing else.

---

## 2. What we ship today

### Continuous mode — 6 files

| file | what it holds |
|---|---|
| `discrimination.json` | peak block + one full block **per landmark** (each with a 500-point ROC curve and a 500-point PR curve) |
| `calibration.json` | one reliability curve **per landmark** (equal-width bins + quantile bins + a smoothed curve) |
| `fairness.json` | peak subgroups + subgroups **per landmark** |
| `leadtime.json` | median first-alert time |
| `operating_points.json` | 99-threshold sweep + alert burden + operating budget + 11 threshold rules + treatment sub-rates + intervention timing |
| `kpi.json` | a scoreboard that re-states numbers already in the five files above |

### Episodic mode — 5 files, **407 KB measured**

Measured on the checked-in sample (`tests/artifacts/example_report/`):

| file | bytes | share |
|---:|---:|---:|
| `operating_points.json` | 358,863 | 86% |
| `discrimination.json` | 40,994 | 10% |
| `dca.json` | 9,908 | 2% |
| `calibration.json` | 4,202 | 1% |
| `fairness.json` | 3,097 | 1% |
| **total** | **417,064** | |

Inside `operating_points.json`, each of the 99 threshold rows is 1,879 bytes and
**1,080 of those bytes (57%) are the per-threshold fairness block** — four
demographic columns re-computed at every one of the 99 thresholds.

### Three problems

1. **Curves are stored per landmark.** A 30-day task writes ~30 ROC curves and
   ~30 PR curves. A 168-hour task writes ~168 of each. Nobody plots them.
   *(Estimated at ~1.2 MB / ~6.9 MB for `discrimination.json` alone, scaled from
   the measured 41 KB episodic file — no continuous sample is checked in.)*
2. **Fairness is swept 99 times.** Four demographic columns × 99 thresholds ×
   500 bootstrap replicates, to produce a table no reader reads at 99 points.
3. **The same number appears in three files.** `kpi.json` restates
   `discrimination.json`, which restates `operating_points.json`.

---

## 3. What we ship after

**Two files per mode. Always.**

| mode | file 1 — model quality | file 2 — the decision |
|---|---|---|
| **continuous** | `landmark.json` | `hospitalization_level.json` |
| **episodic** | `overall.json` | `hospitalization_level.json` |

`hospitalization_level.json` has an **identical schema in both modes.** One
reader, one plotter, one aggregator on the FLAIR website. The only difference is
where the `(label, score)` pair came from:

- **continuous** — `score = max(y_prob)` over the stay, `label = ever positive`.
  One row per `hospitalization_join_id`.
- **episodic** — already one row per stay. Nothing to collapse.

---

## 4. `landmark.json` — 4 numbers

Landmark `t` = the slice `window_number == t`, for `t = 1 … META['landmark_horizon']`.
One row per stay per landmark, so a stay-level bootstrap is honest.

**No fairness. No thresholds. No curves. No operating point.** Four pooled
numbers with CIs, and the per-landmark values behind them so they can be plotted.

### The four pooled numbers

| number | how it is pooled | citation |
|---|---|---|
| `auroc` | `Σ AUC(t)·w(t) / Σ w(t)`, `w(t) = cases(t) × controls(t)` | Heagerty & Zheng 2005 |
| `auprc` | `Σ AP(t)·cases(t) / Σ cases(t)` — case-weighted, since AP is anchored on the positives | — |
| `brier` | `Σ n(t)·Brier(t) / Σ n(t)` — identical to the plain Brier over all landmark rows | Brier 1950 |
| `calibration_slope` | **one** logistic fit of `y` on `logit(p)` across all landmark rows | Cox 1958 |

Calibration slope is a single fit, **not** an average of per-landmark slopes:
averaging regression coefficients is a different estimator, and one degenerate
landmark would poison the mean.

### The CIs

**One** stay-level bootstrap for all four. Per replicate: resample
hospitalizations with replacement → recompute every landmark's four values →
recompute the four pooled values. Percentile (2.5, 97.5) over 500 replicates,
seed `20260612`.

One resample stream means the pooled number and the per-landmark ribbons are
mutually coherent, and it replaces today's `3 pillars × 500 reps × N landmarks`
process pool with `1 × 500 reps`.

### Schema

```json
{
  "metadata": { "...audit...", "window_unit": "day", "horizon": 30,
                "n_landmarks_scored": 26, "min_cell_count": 10,
                "boot_reps": 500, "method": "..." },
  "prevalence": 0.0312,

  "pooled": {
    "auroc": 0.812, "auroc_ci": [0.79, 0.83],
    "auprc": 0.191, "auprc_ci": [0.16, 0.23],
    "brier": 0.0284, "brier_ci": [0.027, 0.030],
    "calibration_slope": 0.94, "calibration_slope_ci": [0.86, 1.03]
  },

  "by_landmark": [
    { "hospitalization_time": 1, "n": 4210, "prevalence": 0.031,
      "auroc": 0.81, "auroc_ci": [0.78, 0.84],
      "auprc": 0.19, "auprc_ci": [0.15, 0.24],
      "brier": 0.028, "brier_ci": [0.026, 0.031],
      "calibration_slope": 0.94, "calibration_slope_ci": [0.81, 1.08] },

    { "hospitalization_time": 27, "n": "<10", "suppressed": true },
    { "hospitalization_time": 28, "undefined": true,
      "reason": "single label class at this hospitalization time" }
  ]
}
```

A landmark is scored when `n ≥ 10` **and** both label classes are present.
Otherwise it emits a stub, exactly as today.

---

## 5. `hospitalization_level.json` — the whole decision, in one file

This is today's `peak` + `operating_points` + `kpi`, merged. Two pieces of
information in one place: **where the threshold is**, and **what every metric
does there**.

The threshold grid is the existing fixed 99-point grid — `0.01 … 0.99`, step
`0.01`, site- and task-invariant, so two sites' sweeps line up row-for-row and
pool without interpolation. The shipped clinical policy is **one flagged row**
inside it (`is_clinical: true`), never a separate test — it is already on the
grid.

### Schema

```json
{
  "metadata": {
    "...audit...",
    "surface": "peak",
    "score_rule": "max(y_prob) per hospitalization_join_id",
    "clinical_threshold": 0.10,
    "clinical_direction": "above",
    "grid": { "min": 0.01, "max": 0.99, "step": 0.01, "n": 99 },
    "min_cell_count": 10,
    "boot_reps": 500
  },
  "prevalence": 0.0512,
  "n_hospitalizations": 4210,

  "by_threshold": [
    { "threshold": 0.35, "is_clinical": false,
      "tp": 120, "fp": 400, "fn": 60, "tn": 3630,
      "sensitivity": 0.667, "sensitivity_ci": [0.61, 0.72],
      "specificity": 0.901, "specificity_ci": [0.89, 0.91],
      "ppv": 0.231, "ppv_ci": [0.20, 0.26],
      "npv": 0.984, "npv_ci": [0.98, 0.99],
      "f1": 0.343, "f1_ci": [0.31, 0.38],
      "fpr": 0.099, "fpr_ci": [0.09, 0.11],
      "flag_rate": 0.1235,
      "nne": 4.3, "nne_ci": [3.8, 5.0],
      "net_benefit": 0.0123, "net_benefit_ci": [0.008, 0.017],
      "net_benefit_treat_all": 0.0041 }
  ],

  "selected_thresholds": [
    { "rule": "youden_j", "threshold": 0.28, "threshold_ci": [0.22, 0.34],
      "in_sample": true, "citation": "Youden 1950",
      "sensitivity": 0.74, "specificity": 0.84, "ppv": 0.19,
      "npv": 0.99, "f1": 0.30, "fpr": 0.16, "flag_rate": 0.20, "nne": 5.3 },
    { "rule": "clinical", "threshold": 0.10, "threshold_ci": null,
      "in_sample": false, "citation": null,
      "sensitivity": 0.91, "specificity": 0.52, "ppv": 0.09,
      "npv": 0.99, "f1": 0.16, "fpr": 0.48, "flag_rate": 0.52, "nne": 11.1 }
  ],

  "fairness": {
    "at_threshold": 0.10,
    "subgroups": {
      "sex_category": {
        "Female": { "n": 1904, "prevalence": 0.048, "auroc": 0.80,
                    "auroc_ci": [0.76, 0.84], "positive_rate": 0.51,
                    "tpr": 0.89, "fpr": 0.49 }
      }
    },
    "parity": {
      "sex_category": { "reference_group": "Male",
        "demographic_parity_gap": 0.03, "demographic_parity_gap_ci": [0.01, 0.06],
        "equalized_odds_tpr_gap": 0.04, "equalized_odds_tpr_gap_ci": [0.00, 0.09],
        "equalized_odds_fpr_gap": 0.03, "equalized_odds_fpr_gap_ci": [0.00, 0.07] }
    }
  }
}
```

### Three deliberate rules

- **`net_benefit*` appear only in episodic mode.** Net benefit on a
  max-of-many surface is the peak-calibration trap, so continuous rows omit
  those three keys and `metadata` carries the reason. Everything else is
  identical.
- **Fairness is computed at the clinical threshold only** — not at all 99.
  That single change removes 57% of the bytes and ~99× the bootstrap work.
- **One threshold rule survives: Youden's J** (Youden 1950). It appears
  alongside the shipped `clinical` row so a reader can see, at a glance, where a
  data-driven rule would have landed. Both carry `in_sample` so nobody mistakes
  a fitted cutpoint for a validated one.

---

## 6. `overall.json` — episodic model quality

Threshold-free. Two blocks, ~4 KB.

```json
{
  "metadata": { "...audit...", "roc_grid": "fpr 0.00..1.00 step 0.01",
                "pr_grid": "recall 0.00..1.00 step 0.01" },
  "prevalence": 0.0512,
  "n": 4210,

  "discrimination": {
    "auroc": 0.812, "auroc_ci": [0.79, 0.83],
    "auprc": 0.191, "auprc_ci": [0.16, 0.23],
    "auprc_baseline": 0.0512,
    "tpr_at_grid": [101 floats],
    "precision_at_grid": [101 floats]
  },

  "calibration": {
    "brier": 0.0284, "brier_ci": [0.027, 0.030],
    "slope": 0.94, "slope_ci": [0.86, 1.03],
    "intercept": -0.08, "intercept_ci": [-0.19, 0.02],
    "quantile_bins": {
      "bin_pred_prob": [10 floats],
      "bin_true_prob": [10 floats]
    }
  }
}
```

**Curves become one array each.** `tpr_at_grid` is TPR at a fixed FPR grid
(`0.00 … 1.00` step `0.01`); `precision_at_grid` is precision at a fixed recall
grid. The x-axes are known constants, so they are never stored. This drops the
ROC + PR payload from 22 KB to ~2 KB **and** makes cross-site vertical averaging
exact — today's variable-length curves cannot be averaged without interpolation.

**Calibration keeps only the quantile (equal-count) curve** — 10 predicted
probabilities and 10 observed rates, no counts, no edges. Bins with fewer than
10 stays still null their observed rate.

**There is no `dca` block.** The decision curve *is* `net_benefit` +
`net_benefit_treat_all` across the 99 rows of `hospitalization_level.json`
(`net_benefit_treat_none` is 0 by definition). Storing it twice is what got us
here. The DCA plot reads that file.

---

## 7. Complete key table

Everything a site uploads, after this change.

### `landmark.json` — continuous only

| key | type | notes |
|---|---|---|
| `metadata.window_unit` | str | `day` or `hour` |
| `metadata.horizon` | int | `META['landmark_horizon']` |
| `metadata.n_landmarks_scored` | int | landmarks that cleared n≥10 + 2 classes |
| `metadata.min_cell_count` | int | 10 |
| `metadata.boot_reps` | int | 500 |
| `prevalence` | float | |
| `pooled.auroc` / `_ci` | float / [lo,hi] | Heagerty pair-weighted |
| `pooled.auprc` / `_ci` | float / [lo,hi] | case-weighted |
| `pooled.brier` / `_ci` | float / [lo,hi] | n-weighted |
| `pooled.calibration_slope` / `_ci` | float / [lo,hi] | single pooled logistic fit |
| `by_landmark[].hospitalization_time` | int | 1-based |
| `by_landmark[].n` | int or `"<10"` | risk set |
| `by_landmark[].prevalence` | float | |
| `by_landmark[].auroc` / `_ci` | float / [lo,hi] | |
| `by_landmark[].auprc` / `_ci` | float / [lo,hi] | |
| `by_landmark[].brier` / `_ci` | float / [lo,hi] | |
| `by_landmark[].calibration_slope` / `_ci` | float / [lo,hi] | |
| `by_landmark[].suppressed` | bool | present only when n < 10 |
| `by_landmark[].undefined` | bool | present only when single-class |

**24 keys.**

### `overall.json` — episodic only

| key | type | notes |
|---|---|---|
| `prevalence`, `n` | float, int | |
| `discrimination.auroc` / `_ci` | float / [lo,hi] | |
| `discrimination.auprc` / `_ci` | float / [lo,hi] | |
| `discrimination.auprc_baseline` | float | = prevalence |
| `discrimination.tpr_at_grid` | 101 floats | fixed FPR grid |
| `discrimination.precision_at_grid` | 101 floats | fixed recall grid |
| `calibration.brier` / `_ci` | float / [lo,hi] | |
| `calibration.slope` / `_ci` | float / [lo,hi] | |
| `calibration.intercept` / `_ci` | float / [lo,hi] | |
| `calibration.quantile_bins.bin_pred_prob` | 10 floats | |
| `calibration.quantile_bins.bin_true_prob` | 10 floats | null where bin n < 10 |

**17 keys.**

### `hospitalization_level.json` — both modes

| key | type | notes |
|---|---|---|
| `metadata.surface` | str | `peak` (continuous) or `stay` (episodic) |
| `metadata.score_rule` | str | continuous only |
| `metadata.clinical_threshold` | float | |
| `metadata.clinical_direction` | str | `above` / `below` |
| `metadata.grid` | obj | `{min, max, step, n}` |
| `metadata.min_cell_count`, `.boot_reps` | int, int | 10, 500 |
| `prevalence`, `n_hospitalizations` | float, int | |
| `by_threshold[].threshold` | float | 0.01 … 0.99 |
| `by_threshold[].is_clinical` | bool | exactly one row is true |
| `by_threshold[].tp/fp/fn/tn` | int or `"<10"` | masked in pairs |
| `by_threshold[].sensitivity` / `_ci` | float / [lo,hi] | |
| `by_threshold[].specificity` / `_ci` | float / [lo,hi] | |
| `by_threshold[].ppv` / `_ci` | float / [lo,hi] | null when < 10 flagged |
| `by_threshold[].npv` / `_ci` | float / [lo,hi] | |
| `by_threshold[].f1` / `_ci` | float / [lo,hi] | |
| `by_threshold[].fpr` / `_ci` | float / [lo,hi] | |
| `by_threshold[].flag_rate` | float | point only |
| `by_threshold[].nne` / `_ci` | float / [lo,hi] | 1/PPV |
| `by_threshold[].net_benefit` / `_ci` | float / [lo,hi] | **episodic only** |
| `by_threshold[].net_benefit_treat_all` | float | **episodic only** |
| `selected_thresholds[].rule` | str | `youden_j` or `clinical` |
| `selected_thresholds[].threshold` / `_ci` | float / [lo,hi] | CI is selection-aware |
| `selected_thresholds[].in_sample` | bool | true for `youden_j` |
| `selected_thresholds[].citation` | str | |
| `selected_thresholds[].{sens,spec,ppv,npv,f1,fpr,flag_rate,nne}` | float | |
| `fairness.at_threshold` | float | = clinical threshold |
| `fairness.subgroups.<col>.<level>.n` | int | levels with n < 10 are dropped |
| `… .prevalence` | float | |
| `… .auroc` / `_ci` | float / [lo,hi] | |
| `… .positive_rate`, `.tpr`, `.fpr` | float | |
| `fairness.parity.<col>.reference_group` | str | largest level |
| `… .demographic_parity_gap` / `_ci` | float / [lo,hi] | |
| `… .equalized_odds_tpr_gap` / `_ci` | float / [lo,hi] | |
| `… .equalized_odds_fpr_gap` / `_ci` | float / [lo,hi] | |

**~45 keys.** Subgroup columns: `sex_category`, `race_category`,
`ethnicity_category`, `age_bin`.

---

## 8. Plots

Rendered locally with `--viz`, never exported.

### Continuous — 5 PNGs

| png | source | content |
|---|---|---|
| `landmark_auroc.png` | `landmark.json` | AUROC vs hospitalization time, CI ribbon, pooled value as a horizontal line |
| `landmark_auprc.png` | `landmark.json` | same, AUPRC |
| `landmark_brier.png` | `landmark.json` | same, Brier |
| `landmark_slope.png` | `landmark.json` | same, calibration slope, reference line at 1.0 |
| `threshold_sweep.png` | `hospitalization_level.json` | **5 panels** — Sensitivity, Specificity, PPV, NPV, NNE — each vs threshold with a CI band, in the JAMA layout |

### Episodic — 6 PNGs

| png | source |
|---|---|
| `roc.png` | `overall.json` — ROC curve, AUROC + CI in the corner |
| `pr.png` | `overall.json` — PR curve, AUPRC + CI, prevalence baseline |
| `calibration.png` | `overall.json` — quantile reliability curve + diagonal |
| `dca.png` | `hospitalization_level.json` — net benefit, treat-all, treat-none |
| `threshold_sweep.png` | `hospitalization_level.json` — the same 5 panels |
| `fairness.png` | `hospitalization_level.json` — subgroup AUROC with CIs |

---

## 9. What is deleted

| removed | why |
|---|---|
| `kpi.json` | every number restates another file |
| `leadtime.json` | timing analysis removed; the collapse is `max(score)` |
| `intervention_timing`, `intervention_offsets`, `sub_rates` | timing analysis removed |
| `alert_burden` | `frac_stays_alerting` = `flag_rate`, `false_alarm_fraction` = `1 − ppv`, `nne` already in the row |
| `operating_budget` | derivable — find the row whose `flag_rate` is nearest the target |
| 10 of 11 threshold rules | `closest_01`, `concordance_cz`, `index_of_union`, `f_beta_max`, `sensitivity_target`, `specificity_target`, `cost_based`, `ppv_target`, `alert_rate_target`, `max_accuracy` |
| per-threshold fairness (99 →  1) | 57% of `operating_points.json`; nobody reads a parity gap at 99 points |
| per-landmark ROC / PR curves | ~30–168 curves per task that were never plotted |
| per-landmark reliability bins, ECE, o/e ratio, fairness | landmark is 4 numbers now |
| `ece`, `ece_ci`, equal-width bins, `bin_counts`, `bin_edges`, `smoothed` | quantile curve only |
| `dca.json` and its dense 0.002-step grid | net benefit lives in the 99 sweep rows |
| `useful_range`, `all_useful_intervals`, `verdict`, `beats`, `focus_range` | prose conclusions; the reader has the curve |
| `decision.{acts_on, n_acted, act_rate, event_rate_in_acted, event_rate_in_not_acted, clinically_useful}` | derivable from the 2×2 in the same row |
| per-subgroup `auprc`, `brier`, `calibration_slope`, `calibration_intercept` | subgroup AUROC + the three gaps are the equity question |

---

## 10. Remove the treatment / intervention machinery entirely

It was built for the sepsis task ("did the model fire before the antibiotics").
Sepsis is gone, so **no shipped task declares `META['intervention']`** — every
branch below is unreachable on all four tasks. It is dead code, not a feature
being cut.

**`flair_benchmark/cohort/` is already clean** — grep finds no treatment column,
no intervention plumbing, nothing to remove there.

| file | what goes |
|---|---|
| `constants.py` | `"intervention"` from the allowed-META-keys tuple (line 120) |
| `tasks/__init__.py` | the `META['intervention']` validation block (lines 99–104) |
| `report/__init__.py` | `_validate_intervention_column()`, and the intervention column appended to `_CONTEXT_COLS` |
| `report/operating_points.py` | `_arm_masks`, `_sub_rates_row`, `_timing_at`, `_intervention_offsets`, `_named_thresholds`, the `intervention_timing` / `intervention_offsets` / `sub_rates` outputs, and `_sweep.arm_fire_rate` (its only caller) |
| `report/kpi.py` | the whole `sub_analyses` block — file is deleted anyway |
| `report/_viz.py` | `_render_operating_points_sub()` → `operating_points_subanalyses.png` |
| `report/_basis.py` | the "did the model fire before treatment" line in the docstring |
| `cli.py` | "optional intervention sub-analyses" in the `--mode` help text |
| docs | `docs/101/reports.md`, `docs/101/tasks.md`, `docs/eval_modes/continuous.md` |
| tests | the intervention cases in `test_audit_fixes.py`, `test_cli_build_report.py`, `test_report_continuous.py` |

### Two things that look related but are not — keep both

- **`task_class: "intervention"`** on `extubation_failure_24h` is a taxonomy
  label (`outcome` / `deterioration` / `intervention` / `operational`), not a
  treatment column. `TASK_CLASS_VALUES` keeps all four values.
- **`net_benefit_treat_all`** is a decision-curve reference strategy (treat
  everyone vs treat no one), standard DCA. Nothing to do with the treatment
  timing analysis.

---

## 11. Expected impact

### Bytes

All figures below are **measured** on the same synthetic cohorts, before and after.

| | before | after | change |
|---|---:|---:|---|
| **episodic bundle** (700 stays) | **417,064 B** | **95,275 B** | **−77.2%** |
| — `overall.json` | — | 4,265 B | |
| — `hospitalization_level.json` | — | 91,010 B | |
| **continuous bundle** (300 stays, 30 landmarks) | — | **81,595 B** | |
| — `landmark.json` | — | 6,220 B | |
| — `hospitalization_level.json` | — | 75,375 B | |

The episodic "before" is the checked-in baseline that
`tests/artifacts/example_report/` used to hold; that directory now holds the new
two-file bundle. No continuous "before" sample was ever checked in, so it has no
measured baseline — the estimate in §2 stood at ~1.6 MB for a 30-landmark task,
driven almost entirely by one ROC and one PR curve per landmark.

`hospitalization_level.json` is now ~95% of both bundles, which is the intended
shape: it is the only file holding a 99-row array, and every row carries six
metrics with CI bands because the threshold-sweep figure needs the ribbons.

### Compute

Three structural reductions (not yet timed):

1. **Landmark:** `3 pillars × 500 reps × N landmarks` → `1 × 500 reps` over all
   landmarks at once. The `ProcessPoolExecutor` and its BLAS-pinning machinery
   are no longer needed.
2. **Fairness:** `500 reps × 99 thresholds × 4 columns` → `500 reps × 1
   threshold × 4 columns`.
3. **Locators:** 11 rules re-selected on every replicate → 1.

### Code

| | before | after |
|---|---:|---:|
| files in `report/` | 15 | 8 |
| lines in `report/` | 5,110 | 2,225 |
| report tests | 4 files | 5 files, 134 tests |

Proposed layout:

```
report/
  __init__.py                 orchestrator, validation, audit block   (keep)
  _stats.py                   bootstrap + suppression primitives      (trim)
  _sweep.py                   vectorized 99-point sweep engine        (keep)
  _basis.py                   direction / decision basis              (trim)
  landmark.py                 -> landmark.json                        (rewrite)
  overall.py                  -> overall.json                         (new)
  hospitalization_level.py    -> hospitalization_level.json           (new)
  _viz.py                     5-6 plots                               (rewrite)
```

Deleted: `continuous.py`, `episodic.py`, `kpi.py`, `dca.py`, `calibration.py`,
`discrimination.py`, `fairness.py`, `operating_points.py`, `_locators.py`.

---

## 12. Numbers that will change

Sites must re-run and re-upload. These are not refactors.

| change | effect |
|---|---|
| Pooled AUROC now pools over `META['landmark_horizon']` only | today it pools over **every** window of the full stay while displaying a capped curve; the published number becomes reproducible from the published file, but its **value moves** |
| Calibration slope is one pooled fit | replaces "slope at the earliest scored landmark" plus a min/max range |
| ROC / PR stored on fixed grids | curve points shift to grid points; AUROC / AUPRC scalars are unchanged (still computed on raw scores) |
| Fairness parity gaps at one threshold | the 99-threshold table is gone |
| Threshold rules 11 → 1 | any site comparing against `f_beta_max` etc. loses that column |
| `ece` removed | no replacement; use `calibration_slope` + `brier` |

`REPORT_SCHEMA` bumps **3 → 4**.

---

## 13. Order of work

1. **Continuous first**, as requested — `landmark.py` + `hospitalization_level.py`,
   plus the 5 plots.
2. **Episodic second** — `overall.py`, reusing `hospitalization_level.py`
   unchanged, plus its 6 plots.
3. Delete the 9 dead modules, bump `REPORT_SCHEMA`, update `docs/101/reports.md`
   and `docs/eval_modes/`.
4. Rewrite `tests/test_report_continuous.py` and `tests/test_report_extensions.py`
   against the new schema; regenerate `tests/artifacts/example_report/`.

---

## 14. Open questions

1. **Should continuous also render `fairness.png`?** The data is in
   `hospitalization_level.json` either way. The spec called for 5 continuous
   plots, so it is currently excluded — adding it is one function call.
2. **Is `age_bin` still wanted as a fairness column?** It is the one derived
   column (from `age_at_admission`); the other three come straight from CLIF.
3. **`npv` at low thresholds is ~1.0 everywhere** (visible in panel D of the
   reference figure). Keep it in the sweep table, or drop it from the plot and
   promote `flag_rate` — the workload number — to that panel instead?
