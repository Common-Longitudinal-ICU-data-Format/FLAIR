# How `flair build-report` Works

A plain-language guide to the FLAIR reporting step: what it does, what it computes,
how the output changes for different tasks and prediction frequencies, and whether
it's ready for the baseline proof-of-concept (POC).

---

## 1. What this does (in one paragraph)

Every FLAIR site trains its model **locally** on its own ICU data, then adds a single
`y_prob` column to the cohort file `flair load-task` produced (the label, split, and
demographics are already in that file — see §2). The command `flair build-report`
turns that file into **seven JSON files**. Those JSON files are what the central FLAIR
website reads to draw interactive charts comparing all 12+ sites. **No patient-level
data ever leaves the site** — every number in the JSON is an aggregate, and any group
smaller than **10 patients is suppressed** (shown as `"<10"` or `null`), so nothing in
the output can be traced back to an individual.

```
your model  ──►  cohort.parquet + y_prob  ──►  flair build-report  ──►  7 JSON files  ──►  FLAIR website
   (local)            (one file, local)             (local)             (shareable)        (compares sites)
```

---

## 2. How to run it

```bash
flair build-report PREDS.parquet --task TASK_NAME --out OUT_DIR [--cohort COHORT.parquet] [--viz]
```

| Argument | Meaning |
|---|---|
| `PREDS.parquet` | Your predictions file (see columns below). |
| `--task` | Which task these predictions are for, e.g. `task1_icu_daily_mortality`. |
| `--out` | Folder where the 7 JSON files are written. |
| `--cohort` | *(optional)* The task cohort parquet from `flair load-task`. Only needed to fill in over-time charts when your predictions don't already carry `window_number` / `prediction_dttm`. |
| `--viz` | *(optional)* Also write PNG charts into `OUT_DIR/viz/`. These are **local sanity checks only**, rendered *from the JSON* — the website does its own plotting. Needs `pip install 'flair-benchmark[viz]'`. |

### What the predictions file must contain — just add `y_prob`

There is **no separate predictions file to author**. Start from the cohort file
`flair load-task` gave you and add **one column, `y_prob`**:

```python
import polars as pl
cohort = pl.read_parquet("cohort.parquet")
preds = cohort.with_columns(pl.Series("y_prob", y_prob))   # binary tasks
preds.write_parquet("preds.parquet")
```

Because you're writing to the same file, every prediction stays glued to its row —
there's no second file to join and no way for windows to drift out of alignment.

**Already in the cohort file (you don't add these):**

- `hospitalization_id` — the stay ID
- `split` — `"train"` or `"test"` (all metrics are computed on **test** only)
- the task's label column, e.g. `label_mortality` (no need to re-supply a `y_true`)
- `age_at_admission`, `sex_category`, `race_category`, `ethnicity_category` (fairness)
- `prediction_id`, `window_number`, `prediction_dttm` (the over-time charts use these)

**What you add:**

- `y_prob` — the model's predicted probability (the one column to add; without it
  you lose ROC, calibration, the over-time charts, DCA, and threshold analysis).
  It's validated to be non-null and within `[0, 1]`.
- `y_pred` — *optional.* If you don't supply it, FLAIR derives it as `y_prob ≥ 0.5`.
  Add your own `pl.Series("y_pred", …)` only if you threshold somewhere other than 0.5.

If you wrote predictions to a *different* file that lacks the cohort context columns,
pass `--cohort cohort.parquet` and FLAIR joins them back on `prediction_id`.

---

## 3. The seven files, in plain English

| File | The question it answers |
|---|---|
| `table1.json` | *Who is in the cohort?* (the classic "Table 1") |
| `tripod_ai.json` | *How good is the model overall?* (headline accuracy + calibration) |
| `curves.json` | *The raw points to draw ROC, PR, and calibration curves.* |
| `fairness.json` | *Does the model work equally well across subgroups?* |
| `temporal.json` | *Does performance change over the stay and over the years?* |
| `dca.json` | *Is acting on the model actually worth it, across thresholds?* |
| `threshold_analysis.json` | *How does it behave at the one threshold a clinician would use?* |

Every file also carries a `metadata` block with the task name, a SHA-256 hash of the
exact task script used, the FLAIR version, and a UTC timestamp — so a result is
always traceable to the code that produced it.

### table1.json — who's in the cohort

Counts and summary stats (age mean/median/quartiles; sex/race/ethnicity counts),
broken down by split and by outcome. Any cell with fewer than 10 patients is shown
as `"<10"`.

### tripod_ai.json — the headline numbers

A TRIPOD+AI-style summary. For a binary task the `results.performance` block has
**discrimination + classification** metrics (AUROC, AUPRC, accuracy, precision,
recall, F1, specificity, NPV, and the confusion-matrix counts), and
`results.calibration` has **calibration** metrics (slope, intercept, observed:expected
ratio, Brier score, ECE). *In the example report:* AUROC `0.8214`, AUPRC `0.497`,
calibration slope `0.969`, intercept `-0.080`, O:E `0.973`, Brier `0.104`.

> **The math, briefly.** *Calibration slope* comes from re-fitting a logistic
> regression of the true outcome on `logit(y_prob)`. Slope ≈ 1 and intercept ≈ 0
> means the predicted probabilities are trustworthy as stated. *O:E* is the average
> predicted risk vs. the actual event rate — close to 1 is good.

### curves.json — points to draw the curves

The actual `(fpr, tpr)` and `(recall, precision)` arrays for ROC and PR, plus the
calibration bins **and** a smooth calibration curve (50-point logistic
recalibration — smooth, no extra dependency, and PHI-safe because only two fitted
numbers leave the site). Curves are down-sampled to ≤500 points to keep files small.

### fairness.json — equal performance across groups

Per-subgroup metrics (AUROC, AUPRC, Brier, TPR, FPR, positive rate) for sex, race,
ethnicity, and age band, plus **parity gaps**: demographic-parity gap and
equalized-odds TPR/FPR gaps (the max−min spread across groups). A second copy,
`parity_at_clinical_threshold`, recomputes the gaps at the task's clinical threshold
instead of the default 0.5.

> **Privacy note:** fairness is stricter than Table 1 — a subgroup with **fewer
> than 10 patients is dropped entirely**, not just suppressed. A metric on a handful
> of patients could leak an individual outcome, so small minorities won't appear here.

### temporal.json — performance over time

Three sections (binary tasks only):

- **`by_window`** — AUROC/AUPRC for each prediction window in the stay (windows
  `1`–`6`, then a `7+` bucket), with a bootstrap confidence interval when a window
  has ≥100 rows. *Only populated for one-per-timestep tasks.*
- **`by_year`** — AUROC trend by year of admission (drift detection). The year comes
  from `admission_dttm` if present, otherwise from `min(prediction_dttm)` per stay.
- **`trajectory_by_outcome`** — the "two-line" chart: mean predicted risk per window,
  split into patients who **did** vs **did not** have the event — so you can see risk
  diverge over the stay. *One-per-timestep tasks only.*

### dca.json — is the model worth acting on?

**Decision Curve Analysis.** It sweeps every threshold from 0.01 to 0.99 and computes
the model's **net benefit** versus the two trivial strategies "treat everyone" and
"treat no one," then reports the threshold **range** where the model beats both. It
also marks the task's specific clinical threshold. *In the example:* prevalence
`0.159`, model is net-beneficial over `0.02–0.66`.

> **The math, briefly.** Net benefit = `TP/n − FP/n · t/(1−t)`, where `t` is the
> threshold. It's the true-positive rate penalised by false positives, weighted by
> how bad a false positive is at that threshold.

### threshold_analysis.json — behavior at the clinical threshold

Each task has **one** threshold a clinician actually acts on (DCA shows the whole
sweep; this zooms to the operating point). It reports, at that threshold:
**sensitivity, specificity, PPV, NPV, F1** and the confusion counts (`label_metrics`);
**who gets acted on and their observed event rate** (`action_metrics`); a
**partial AUROC** measuring discrimination only in a ±0.05 band around the threshold;
and a **robustness sweep** at threshold ±0.02 and ±0.05. If too few patients fall on
either side of the threshold, the whole block is `null` rather than fabricated — see
the per-task notes below.

---

## 4. The evaluation flow — from predictions to a verdict

The report answers five questions, in order. Each builds on the previous one: a model
has to *discriminate* before it's worth *calibrating*, and it has to be calibrated
before a *decision threshold* means anything.

```
 cohort + y_prob
       │
       ▼
 [0] Filter to the TEST split          ← every metric below uses test rows only
       │                                  (the model never saw these)
       ▼
 [1] CAN it tell apart sick vs not?    → AUROC, AUPRC        (tripod_ai, curves)
       │   discrimination               "rank a true case above a non-case"
       ▼
 [2] Are the PROBABILITIES honest?     → slope, intercept,   (tripod_ai, curves)
       │   calibration                   O:E, Brier, ECE
       ▼                                 "when it says 30%, does 30% happen?"
 [3] Does it HOLD UP over time &        → AUROC by window,    (temporal, fairness)
       │   across people?                AUROC by year,
       │   stability + equity            parity gaps
       ▼                                 "same skill on day 7? on every subgroup?"
 [4] Is ACTING on it worth it?          → net benefit,        (dca,
       │   clinical usefulness            sens/spec/PPV/NPV    threshold_analysis)
       ▼                                  at the threshold
   a verdict the site can compare       "treat at t=0.10 — net positive vs doing nothing?"
   against every other site
```

**Why this order tells a story.** A model can be a strong *ranker* (high AUROC) yet
output dishonest probabilities (slope ≠ 1) — so step 2 checks the numbers are usable
as stated. It can look good *pooled* yet decay on later ICU days or underperform for
one subgroup — step 3 catches that. And even a good, honest, stable model is only
useful if **acting at the clinical threshold beats the do-nothing / treat-everyone
defaults** — that's step 4, and it's the one a clinician actually cares about.

Two ideas run through all of it:

- **Test split only.** Discrimination, calibration, everything — computed on rows the
  model was not trained on (`split == "test"`), assigned at `load-task` time by a
  temporal cut (train on earlier admissions, test on later ones). This is the honest
  estimate of how the model behaves on the next patient.
- **Suppress anything under 10 patients.** Every bin, subgroup, and confusion cell with
  n < 10 is dropped or shown as `"<10"`/`null`. A metric on a handful of people can leak
  an individual outcome, so the report refuses to compute it.

---

## 5. Every metric: the formula and what it tells you

Notation: `TP/FP/TN/FN` = true/false positives/negatives at a threshold; `n` = number of
test rows; `p` = predicted probability `y_prob`; `y` = true label (0/1);
`prev` = prevalence = mean(`y`).

### Discrimination — *can it rank a case above a non-case?*

| Metric | Formula / definition | What it tells you | Good value |
|---|---|---|---|
| **AUROC** | area under TPR-vs-FPR as the threshold sweeps; = P(score of a random positive > score of a random negative) | how well the *ranking* separates events from non-events, at **every** threshold at once | 0.5 = coin flip, 1.0 = perfect; >0.8 is strong |
| **AUPRC** | area under precision-vs-recall | same idea but focused on the **positive** class — the honest yardstick when events are **rare** (most ICU tasks) | compare to `prev`; AUPRC ≫ prevalence is good |

> AUROC can look flattering when positives are rare (a model gets the easy negatives
> right). AUPRC doesn't let it — that's why both are reported.

### Calibration — *are the probabilities honest?*

Refit a logistic regression of the outcome on the model's log-odds:
`logit(P(y=1)) = intercept + slope · logit(p)`.

| Metric | Formula / definition | What it tells you | Good value |
|---|---|---|---|
| **Calibration slope** | the fitted `slope` above | are risks too extreme or too timid? <1 = over-confident (stretch toward 0/1), >1 = under-confident | **1.0** |
| **Calibration intercept** | the fitted `intercept` (calibration-in-the-large) | does the model systematically over- or under-predict risk overall? | **0.0** |
| **O:E ratio** | mean(`y`) / mean(`p`) = observed events ÷ expected events | a one-number gut check on overall risk level | **1.0** (>1 under-predicts, <1 over-predicts) |
| **Brier score** | mean((`p` − `y`)²) | overall accuracy of the probabilities (discrimination **and** calibration together); lower is better | 0 = perfect; compare to `prev·(1−prev)` |
| **ECE** (expected calibration error) | bin p into 10 bins; ECE = Σ (n_bin/n)·\|mean_p − mean_y\| per bin | average gap between predicted and observed risk across the probability range | near 0 |
| **smoothed curve** | logistic-recalibration fit evaluated on a 50-point grid | the calibration plot the website draws (predicted vs observed), without leaking raw points | hugs the diagonal |

### Stability & equity — *does the skill hold up?*

| Metric | Formula / definition | What it tells you |
|---|---|---|
| **AUROC by window** | AUROC computed within each day/hour-in-stay bin (1…6, 7+) | whether the model is as sharp on ICU day 7 as on day 1 |
| **AUROC by year** | AUROC within each admission-year bin | dataset/temporal **drift** — is performance sliding over calendar time? |
| **trajectory by outcome** | per-window mean `p`, split by eventual outcome | whether predicted risk **diverges** between patients who do vs don't have the event |
| **demographic-parity gap** | max−min of the positive (flag) rate across subgroups | are some groups flagged far more often than others? |
| **equalized-odds TPR gap** | max−min of TPR across subgroups | does the model **miss** events more in some groups (unequal sensitivity)? |
| **equalized-odds FPR gap** | max−min of FPR across subgroups | does it **false-alarm** more in some groups? |

> Parity gaps are reported both at the generic 0.5 threshold and at the task's clinical
> threshold (`parity_at_clinical_threshold`), because fairness can change once you move
> the operating point to where clinicians actually act.

### Decision value — *is acting on it worth it?*

| Metric | Formula / definition | What it tells you | Good value |
|---|---|---|---|
| **Net benefit (model)** | `TP/n − (FP/n)·(t/(1−t))` at threshold `t` | true positives gained, **minus** false positives weighted by how costly they are at that threshold — all on one "true-positives-per-patient" scale | higher than both default lines |
| **Net benefit (treat all)** | `prev − (1−prev)·(t/(1−t))` | the "treat everyone" baseline | model should beat it |
| **Net benefit (treat none)** | `0` | the "treat no one" baseline | model should beat it |
| **Useful range** | the contiguous `t` interval where model NB > max(treat-all, 0) | the band of risk-thresholds where using the model genuinely helps | should contain the clinical threshold |
| **Sensitivity (recall/TPR)** | `TP/(TP+FN)` | of the patients who **will** have the event, what fraction does it flag? | task-dependent |
| **Specificity (TNR)** | `TN/(TN+FP)` | of those who **won't**, what fraction does it correctly clear? | task-dependent |
| **PPV (precision)** | `TP/(TP+FP)` | when it flags a patient, how often is it right? — drives alarm trust | depends on `prev` |
| **NPV** | `TN/(TN+FN)` | when it clears a patient, how often is that safe? | high for rule-out use |
| **F1** | `2·TP/(2·TP+FP+FN)` | single balance of precision and recall | — |
| **Partial AUROC** | trapezoidal AUROC restricted to the ROC region within ±0.05 of the clinical threshold, McClish-standardized | discrimination **right where the decision is made**, not averaged over thresholds nobody uses | 0.5 = chance, 1.0 = perfect |

> **Why net benefit, not just accuracy?** Accuracy treats a missed sepsis case and a
> false antibiotic course as equally bad. Net benefit encodes the real trade-off: the
> threshold `t` itself says how many false alarms you'd tolerate to catch one true case
> (at `t = 0.10`, you accept ~9 false positives per true positive). A model is only
> "clinically useful" if it beats *both* doing-nothing and treating-everyone **at the
> threshold the task actually uses** — that single boolean is `clinically_useful` in
> `dca.json`.

---

## 6. How the report changes by **task type**

The report adapts to `task_type`. **All five baseline tasks are binary**, so the
binary column is the one that matters for the POC — the others exist but aren't
exercised yet.

| Report | binary *(the POC path)* | multiclass | regression |
|---|---|---|---|
| table1 | full | full | by-quartile split |
| tripod_ai | perf **+ calibration** | perf only | perf only |
| curves | ROC + PR + calibration | per-class ROC + confusion | quantile plots |
| fairness | metrics + parity gaps | accuracy / macro-F1 | MAE per group |
| temporal | **yes** | *binary-only* → `reason` | *binary-only* → `reason` |
| dca | **yes** | *binary-only* → `reason` | *binary-only* → `reason` |
| threshold_analysis | **yes** | *binary-only* → `reason` | *binary-only* → `reason` |

When a section doesn't apply, it isn't omitted silently — it returns a `reason`
string explaining why, so the website can show "not applicable" instead of a blank.

---

## 7. How the report changes by **prediction frequency**

"Frequency" = how often the model predicts per stay. This decides whether the
over-time charts can exist at all.

- **One-per-stay** tasks (task3 extubation, task5 readmission): there's a single
  prediction per stay, so there's nothing to plot *across* a stay. In
  `temporal.json`, `by_window` and `trajectory_by_outcome` are `null` with
  `reason: "one_per_stay_task"`; only `by_year` is produced. Everything else
  (overall metrics, calibration, fairness, DCA, threshold analysis) is identical.
- **One-per-timestep** tasks (task1, task2, task4): multiple predictions per stay,
  so the window charts populate fully.

> ⚠️ **Important — what a "window" means depends on the task's grid.** The window
> bins are labelled `1, 2, 3, … 7+`, but the unit isn't always a day:
>
> | Task | grid | "window 3" means |
> |---|---|---|
> | task1, task2 | daily 07:00 | **day 3** of the stay |
> | task4 sepsis | **hourly** | **hour 3** of the stay |
>
> Same JSON key, different physical unit. Read `by_window` against the task's grid,
> not as "days" by default.

---

## 8. What each of the five tasks produces

All five are **binary**, so all seven JSONs are produced for each. They differ in
frequency (which sets the over-time charts) and in the clinical threshold the
`threshold_analysis.json` zooms to.

| Task | frequency | clinical threshold | direction | action |
|---|---|---|---|---|
| task1 ICU daily mortality | daily (one-per-timestep) | 0.95 | above | withdraw life support |
| task2 ICU daily LTACH | daily (one-per-timestep) | 0.50 | above | early tracheostomy |
| task3 extubation failure 24h | once (one-per-stay) | 0.15 | **below** | extubate |
| task4 sepsis (CDC ASE) 6h | hourly (one-per-timestep) | 0.10 | above | treat with antibiotics |
| task5 ICU readmission | once (one-per-stay) | 0.20 | above | keep in ICU |

A few task-specific things to know:

- **task3 is the "below" case.** You extubate when the *failure* risk is **below**
  15%. So "acting" means `y_prob < 0.15`, the opposite of the others. The JSON keeps
  this unambiguous by reporting both `label_metrics` (always sensitivity/specificity
  for the failure label) and `action_metrics` (who gets extubated and their actual
  failure rate — which should be **low** if the model is good).
- **task1's threshold is 0.95** (withdraw life support only at very high confidence).
  On the synthetic example data almost no prediction exceeds 0.95, so
  `threshold_analysis.json` correctly reports `label_metrics: null` — it refuses to
  compute sensitivity from a handful of patients rather than fabricate a number.
  That suppression path is a feature, and it's covered by the test suite.

---

## 9. What the output actually looks like

A complete, ready-to-browse example bundle lives in the repo:

```
tests/artifacts/example_report/
├── table1.json   tripod_ai.json   fairness.json   curves.json
├── temporal.json   dca.json   threshold_analysis.json
└── viz/
    ├── roc_pr.png   calibration.png   dca.png
    ├── auroc_by_window.png   auroc_by_year.png
    ├── trajectory_by_outcome.png   fairness_auroc.png
```

It's generated (and kept in sync) by the CLI regression test. Regenerate it any time
with:

```bash
pytest tests/test_cli_build_report.py
open tests/artifacts/example_report/viz/*.png
```

That example is `task1` on seeded synthetic data, so its windows are exactly
`1..6, 7+` and its years span `2017–2023`.

---

## 10. Is it ready for the baseline POC?

**Yes.** Every report path used by the five baseline tasks is implemented, tested
(37 tests passing), and suppression-safe. The items below are known limitations —
**none block the POC** — listed honestly so nobody is surprised later.

| # | Limitation | Blocks POC? | Notes |
|---|---|---|---|
| 1 | `by_window` bin unit is grid-dependent (day for task1/2, **hour** for task4) | No | Documentation matter (§7), not a bug. |
| 2 | Multiclass / regression calibration, temporal, and DCA aren't implemented | No | All 5 baseline tasks are binary; these paths are unused. |
| 3 | Fairness drops any subgroup with n<10 entirely | No | Intentional PHI choice; stricter than Table 1. Small minorities won't appear. |
| 4 | Predictions are one file (cohort + `y_prob`), so no window mismatch | — | `y_prob` is validated for null / `[0,1]` range; `y_pred` is derived at 0.5 if absent. This is a safeguard, not a gap. |
| 5 | `by_year` uses `min(prediction_dttm)`'s year | No | A stay straddling Dec 31 could be binned to the earlier year. Already noted in code. |

**Separate, unrelated:** the project `README.md` task table still describes task5 as
"AKI KDIGO-2" while the code's task5 is **ICU readmission** — pre-existing drift
worth fixing in a separate edit (not part of the report system).
