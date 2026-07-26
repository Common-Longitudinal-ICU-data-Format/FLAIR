# Threshold-Sweep + Data-Driven Threshold Analysis — Design Notes

> **Status:** design / ideas doc. No code yet. This is the plan we review before
> touching `flair_benchmark/report/`.
>
> **Record note (2026-07-25):** this document describes a `TASK_POLICY` dict in
> `_constants.py` and a per-task `cost_ratio` key. Neither exists. Clinical policy
> moved inline into each task's `META` (validated in `tasks/__init__.py`, hashed in
> `_hash.py`), and `_constants.py` retains only `SELECTION_DEFAULTS`. Those mentions
> are left in place because they are part of the design record; the policy *tables*
> below have been kept factually current. `flair list-tasks` is authoritative.
>
> **Companion:** `docs/eval_continuous_three_ways.md` §3 already argues *why* FLAIR
> should report data-driven thresholds next to the shipped clinical policy. This
> doc is the concrete **how** — one universal sweep, a menu of threshold-selection
> rules as *locators*, and an alert-burden axis for continuous tasks.

---

## 0. One-paragraph summary

Replace the current "clinical threshold ± 0.05 / ± 0.10" operating-point grid with a
single, task-agnostic, site-invariant sweep over **every threshold from 0.01 to 0.99
in steps of 0.01** (99 points). Every threshold-dependent number (confusion matrix,
sensitivity / specificity / PPV / NPV / F1 / FPR / NNE / flag-rate, net benefit, the
direction-aware action) is reported at all 99 thresholds. The clinical policy
threshold is one flagged row inside that grid (`is_clinical`) — the "clinical
intrusion" marker. On top of the sweep, FLAIR reports, per model / per site / per
test-set, **where a menu of statistical selection rules (Youden, closest-to-(0,1),
concordance-probability, Index of Union, F1/F-β, sensitivity/specificity targets,
cost-based, PPV-target, alert-rate target) would put the cutpoint** — as *locators*,
not as a graded headline. For continuous tasks, each threshold additionally carries an
**encounter-level alert-burden** block (fraction of stays ever alerting, PPV,
false-alarm fraction, NNE) on the stay-peak surface — the hospitalization-level,
one-alert-per-stay evaluation of Kamran et al. (NEJM AI 2024). The full sweep is the honest
object; the selected thresholds are annotations on it, explicitly flagged as in-sample.

---

## 1. Why change what we have

### 1.1 What FLAIR does today

`flair_benchmark/report/operating_points.py` builds the one threshold-dependent
section. It sweeps a **small grid centred on the clinical threshold**:

| Mode | Grid | Offsets | Points |
|---|---|---|---|
| episodic | `t0 ± {0.05, 0.10}` | `_EPISODIC_OFFSETS = (-0.10,-0.05,0,0.05,0.10)` | 5 |
| continuous | `t0 ± 0.05` | `_DELTAS = (-0.05,0,0.05)` | 3 |

`threshold_grid(t0, offsets)` rounds each `t0 + offset`, drops any that fall outside
the open interval `(0,1)` or collide after rounding, and records the drops in
`dropped_offsets`. This offset arithmetic is the source of most of the module's
complexity (the `icu_daily_mortality @ 0.95 → +0.05 = 1.0` edge case, duplicate-after-rounding, etc).

Two gaps:

1. **The grid is tiny and clinical-anchored.** A reader sees sensitivity at 0.95 and
   its immediate neighbours, but never the *shape* of the trade-off across the whole
   score range. There is no way to see that the model's useful operating region is,
   say, 0.30–0.55, nowhere near where the policy stands.
2. **No data-driven thresholds exist in code.** `docs/eval_continuous_three_ways.md`
   §3 designs them (Youden, closest-to-top-left, F1-max, sensitivity-target) but
   nothing computes them. There is no "on *this* site's test set, Youden lands here."
3. **No explicit alert-burden axis across the sweep.** Continuous tasks carry NNE
   (`1/PPV`) and first-alert timing at the single clinical threshold, but never the
   encounter-level burden *as a function of `t`* — "what fraction of admissions alert,
   and how many alerts are false, at threshold t" — the alarm-fatigue trade-off a
   clinician actually weighs (§4.2, Kamran et al.).

### 1.2 What we want (this design)

- **Simplify** the operating-point machinery: one fixed universal grid, no per-task
  offset arithmetic, no dropped-offset bookkeeping. *"Make the operating points simple
  — we check for all the thresholds."*
- **Full sweep 0.01→0.99** for both modes, clinical threshold flagged inside it.
- **Data-driven threshold locators**: the full menu of selection rules, reported as
  "where would this rule stand on this test set", with the honesty caveat handled.
- **Alert burden** per threshold for continuous tasks, on the peak surface, with a
  clearly disclosed unit of analysis and denominator.

---

## 2. The universal sweep

### 2.1 The grid

```
SWEEP_GRID = [0.01, 0.02, 0.03, ... , 0.98, 0.99]        # 99 points, fixed
```

Properties that make this the right primitive:

- **Task-agnostic and site-invariant.** Same 99 thresholds for every task, every
  site, every model — exactly the property that already makes `GRID_FPR`
  (`discrimination.py`) poolable across sites. Two sites' sweeps line up row-for-row,
  so cross-site averaging of any curve (sensitivity vs threshold, PPV vs threshold,
  encounter alert-rate vs threshold) is a clean vertical average with no interpolation.
- **The clinical threshold is always on the grid.** Every shipped policy threshold is
  a multiple of 0.01 (`0.95, 0.50, 0.10, 0.20`), so exactly one grid row gets
  `is_clinical = true`. Add a one-line validation to `_validate_taxonomy`:
  `clinical_threshold` must be a multiple of the grid step (already constrained to
  `0 < t < 1`). This is the **"clinical intrusion"** highlight — the policy shown as a
  single flagged row intruding into the neutral statistical grid.
- **No offsets, no drops.** `threshold_grid`, `_DELTAS`, `_EPISODIC_OFFSETS`, and the
  entire dropped-offset story disappear. Replaced by one constant array. This is the
  requested simplification.

Trade-off (accepted): 99 confusion matrices instead of 3–5. Mitigation in §2.3.

```
        clinical intrusion (icu_daily_mortality, t0 = 0.95)
                                                                 v
  t:  0.01  0.02  ...  0.10  ...  0.20  ...  0.50  ...  0.90  0.95  0.96 ... 0.99
      ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬════╪════┬────┬────┐
      │    │    │    │    │    │    │    │    │    │    │    │    ║ ██ ║    │    │
      └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴════╪════┴────┴────┘
       every column = one full confusion matrix + metrics (+ burden, continuous)
       ██ = is_clinical row (also carries the DCA verdict, per docs §3)
```

### 2.2 What each grid row carries

Per threshold `t` (both modes):

```
{
  "threshold": t,
  "is_clinical": (t == clinical_threshold),
  "confusion":  { n, tp, fp, fn, tn },        # FULL alert×label 2×2 (all 4 cells), suppressed
  "metrics":    { sensitivity, specificity, ppv, npv, f1,   # ALARM basis (p >= t)
                  fpr, nne, flag_rate, ... , *_ci },         # in risk-classifier
  "decision":   { acts_on, n_acted, act_rate,               # ACTION basis
                  event_rate_in_acted, event_rate_in_not_acted,
                  dca_threshold, net_benefit, net_benefit_ci,
                  net_benefit_treat_all, clinically_useful }
}
```

- **The confusion matrix is the FULL alert×label 2×2 — all four cells (TP, FP, FN, TN),
  never the TP cell alone.** Every rate (sensitivity, specificity, PPV, NPV, F1, FPR) is
  derived from all four counts, so the whole 2×2 is single-sourced at each threshold
  (`_confusion.confusion_counts` already returns the full tuple — this is unchanged, just
  stated). It is **one flat encounter-level table — no per-subgroup / per-arm drill-down
  here.** The *only* place counts are split into arms (treated / untreated / by label) is
  the treatment sub-analysis (§4.4), and only when an `intervention` column exists.
  **Kamran et al. do exactly this**: they report PPV *and* sensitivity at the alert
  threshold, which need TP, FP and FN — the full 2×2 on the hospitalization-level (peak)
  collapse, with no drill-down.
- **ALARM basis** (`p >= t`, both directions) drives the confusion matrix and
  se/sp/PPV — unchanged from today: "how well do we DETECT the event", comparable
  across tasks regardless of `clinical_direction`. (Scoring the *action* on a `below`
  task would book "extubated AND failed" as a true positive — see
  `report/_basis.py`.)
- **ACTION / DECISION basis** (direction-aware) drives the decision block and net
  benefit, identical to `dca.json`'s curve at `dca_threshold` — reused verbatim from
  the existing `_decision_at` / `to_decision_basis` primitives. Net benefit is now a
  full curve across 99 thresholds, of which `dca.json` already publishes the shape;
  the sweep and `dca.json` agree by construction because both route through
  `_basis.net_benefit`.

Confusion counts continue through `suppress()`; rates that derive from `< MIN_CELL_COUNT`
flagged patients are nulled with a reason, exactly as today.

### 2.3 Computing 99 points without 99× the work

The point estimates are **vectorized over the threshold axis** — the same trick
`sklearn.roc_curve` uses. Sort the scores once; sensitivity, specificity, PPV, NPV,
F1, FPR, flag-rate at all 99 thresholds are cumulative functions of that one sort.
No per-threshold Python loop over the data.

Bootstrap CIs use **one shared resample stream** (same `BOOT_SEED`, `BOOT_REPS` as the
rest of the report). Each replicate recomputes the whole 99-long metric vector
(vectorized), giving an honest band across the entire sweep from a single loop — this
is how you'd draw a sensitivity confidence *band*, not 99 independent CIs. Cost is
`BOOT_REPS × (one vectorized sweep)`, comparable to a handful of today's single-point
bootstraps.

```
scores ──sort──> cumulative TP/FP/FN/TN(t) for all t  ──> metrics(t) vector  (point est.)
   │
   └── resample stays (BOOT_REPS, seeded) ──> metrics(t) vector per replicate ──> 2.5/97.5 band
```

### 2.4 Metric dictionary — every number, how it is computed, on what label

**Surface & label conventions** (continuous → peak surface; episodic → the stay itself):

- **stay** = one `hospitalization_join_id` (encounter). After the peak collapse every
  number is **one row per stay**.
- **label = 1** (positive) ⇔ the stay is **ever-positive**: `max(label_column) > 0` over
  the stay. Binary, one per stay.
- **fired / alert** (ALARM basis) ⇔ the stay's **peak** risk crosses the threshold:
  `max(y_prob) ≥ t`, in BOTH directions (detection convention — `report/_basis.py`).
- **acted** (ACTION basis) ⇔ direction-aware crossing (`p < t` for a `below` task) —
  used ONLY for timing / the treatment sub-analysis, never for the confusion matrix.
- Confusion counts are always `(label, fired)`. `n` = stays; `p` = prevalence = `(TP+FN)/n`.

| JSON key | Name | Formula | Basis | What it means / label |
|---|---|---|---|---|
| `tp` | true positives | `fired ∧ label` | alarm | alerted stays that were positive |
| `fp` | false positives | `fired ∧ ¬label` | alarm | alerted stays that were negative |
| `fn` | false negatives | `¬fired ∧ label` | alarm | missed positive stays |
| `tn` | true negatives | `¬fired ∧ ¬label` | alarm | correctly-silent negatives |
| `n` | stays | count | — | rows after peak collapse |
| `sensitivity` | recall / TPR | `TP/(TP+FN)` | alarm | of positive stays, frac alerted |
| `specificity` | TNR | `TN/(TN+FP)` | alarm | of negative stays, frac silent |
| `fpr` | false-positive rate | `FP/(FP+TN)` = `1−spec` | alarm | of negatives, frac alerted |
| `ppv` | precision | `TP/(TP+FP)` | alarm | of alerts, frac truly positive |
| `npv` | | `TN/(TN+FN)` | alarm | of silents, frac truly negative |
| `f1` | | `2·TP/(2·TP+FP+FN)` | alarm | harmonic mean(PPV, sens) |
| `flag_rate` | | `(TP+FP)/n` | alarm | of all stays, frac alerted |
| `nne` | number-needed-to-evaluate | `1/PPV` (= WDR) | alarm | alerts per true case caught |
| `frac_stays_alerting` | encounter alert-rate | `(TP+FP)/n` (= `flag_rate` on peak) | alarm | admissions that ever alert (§4.2) |
| `false_alarm_fraction` | | `1 − PPV` = `FP/(TP+FP)` | alarm | Kamran's "95% of alerts false" |
| `net_benefit` | | `TP/n − (FP/n)·t/(1−t)` on decision basis | decision | == `dca.json` at `dca_threshold` |
| `net_benefit_treat_all` | | `p − (1−p)·t/(1−t)` | decision | treat-everyone baseline |
| `clinically_useful` | | `net_benefit > max(nb_all, 0)` | decision | beats treat-all & treat-none |
| `pct_fired` (arm) | | `n_acted / n_arm` | action | §4.4, per treatment arm |
| `misfire_rate` | | `n_misfired / n_untreated_neg` | action | §4.4, false alarm on never-treated |
| `median_hours_before_treatment` | | `median(t_treat − t_first_alert)/3600` | action | §4.4, single scalar, named thr only |

The selection-rule objective values (`Youden J = se+sp−1`, `IU = |se−AUC|+|sp−AUC|`, …)
live in the §3.2 rule menu and are not repeated here.

---

## 3. Data-driven threshold locators

### 3.1 The honesty problem, resolved by the clean sweep

> *"if we do it on the test set it will be optimistic, right?"* — yes.

Choosing a cutpoint to maximise a metric on the same rows you then report is an
in-sample argmax → an **upward-biased** sensitivity/specificity (Leeflang 2008). The
textbook fix (select on train, report on test) is unavailable to inference-only
sites, which never produce train-split predictions.

**How this design stays honest without heavy machinery:**

1. **The full sweep is not a selection.** Publishing every threshold's performance is
   not optimistic — there is no argmax, the reader sees the entire curve. The sweep is
   the honest, un-graded object.
2. **Data-driven thresholds are LOCATORS, not headlines.** For each rule we report the
   *threshold value it picks* — "on this model / this site / this test set, Youden
   lands at 0.42." The performance at that threshold is simply **read off the already-
   published sweep** and stamped `in_sample: true`. We do not manufacture a separate
   optimism-corrected headline number in v1.
3. **Never on the leaderboard.** Ranking stays on pooled AUROC, which needs no
   threshold and cannot be gamed by overfitting a cutpoint (docs §3). The locators
   describe a model's operating behaviour; they do not score it.
4. **The real signal is cross-site stability, not within-site CI.** The step-function
   argmax has no honest confidence interval (Thiele & Hirschfeld 2021), so we do **not**
   put a CI on the cutpoint *location*. Instead the long game is aggregation across
   sites: if four sites' Youden thresholds cluster at ~0.40 while FLAIR ships 0.95,
   that gap is the finding (docs §3, "the policy stops being an assumption and starts
   being a claim").

Optional **v2** (documented, not built now): bootstrap optimism-correction — re-select
the cutpoint inside each resample, report the out-of-fold performance and an
optimism-corrected headline, plus a threshold *stability range* (a range, explicitly
not a CI). Deferred by choice; the clean sweep + locator framing is v1.

### 3.2 The rule menu

All rules operate in the **risk-classifier convention** (predicted-positive = `p >= t`,
label = event happened), independent of `clinical_direction` — so Youden/F1/etc.
separate events from non-events the same way for `above` and `below` tasks. Direction
only changes the *action* view reported alongside (who gets acted on), never the
selection. Notation: `se` sensitivity, `sp` specificity, `p` prevalence, `PPV`, `AUC`.

| # | Rule | Picks the `t` that … | Assumption it bakes in | Prevalence-aware? | Citation |
|---|---|---|---|---|---|
| 1 | **Youden J** | maximises `J = se + sp − 1` | FP and FN cost the same | no | Youden 1950 |
| 2 | **Closest-to-(0,1)** (ER) | minimises `√((1−se)² + (1−sp)²)` | equal weight, geometric | no | Perkins & Schisterman 2006 |
| 3 | **Concordance prob (CZ)** | maximises `se · sp` | equal value, penalises lopsided points | no | Liu 2012 |
| 4 | **Index of Union (IU)** | minimises `\|se−AUC\| + \|sp−AUC\|` | se and sp both near the model's AUC | no | Ünal 2017 |
| 5 | **F1 / F-β max** | maximises `F_β = (1+β²)·PPV·se / (β²·PPV + se)` | ignores TN; β = recall-vs-precision cost | **yes** | van Rijsbergen 1979 |
| 6 | **Sensitivity target** | max `sp` s.t. `se ≥ s₀` (default 0.90) | a miss is worse than a false alarm | via constraint | standard / cutpointr |
| 6b | **Specificity target** | max `se` s.t. `sp ≥ sp₀` | a false alarm is worse than a miss | via constraint | standard / cutpointr |
| 7 | **Cost-based** | maximises `se − m·(1−sp)`, `m = ((1−p)/p)·(C_FP/C_FN)` | explicit cost ratio `C_FP/C_FN` + `p` | **yes** | Metz 1978 |
| 8 | **PPV target** | smallest `t` with `PPV(t) ≥ target` (e.g. 0.30) | a minimum precision to be worth acting | **yes** | deployment practice |
| 9 | **Alert-rate target** | `t` s.t. flag-rate `≈ a₀` (e.g. flag 10%) | a fixed workload budget | **yes** | deployment practice |
| 10 | **Max accuracy** | maximises `se·p + sp·(1−p)` | equal cost, prevalence-weighted | **yes** | Metz 1978 |

Notes:

- **#10 (max accuracy) is reported but flagged degenerate for rare events** — at
  a low-prevalence task (~2%) it collapses toward "never alert." Kept so the reader sees
  *why* it is a bad ICU rule, not hidden.
- **#7 cost-based** needs a cost ratio. Default `C_FP/C_FN = 1` (reduces toward the
  prevalence-weighted point) with an optional per-task `cost_ratio` in `TASK_POLICY`
  when clinicians can state one. When `cost_ratio = 1` and `p = 0.5`, #7 collapses to
  Youden — a nice internal consistency check.
- **F-β**: default β = 1 (F1); optional per-task β.
- **s₀, sp₀, PPV target, a₀** are constants with sensible defaults, overridable per
  task in `TASK_POLICY`. Defaults are recorded in the JSON so a reader knows what was
  asked for.
- **NNE / WDR are the same number.** `NNE = 1/PPV` and the work-up-to-detection ratio
  are identical (or off by the `(1−PPV)/PPV` convention). Report **once**, name it
  `nne`, note `wdr` as a synonym — do not double-count them as two metrics.

### 3.3 What each locator emits

```
"data_driven_thresholds": [
  { "rule": "youden_j",
    "threshold": 0.42,               # where it lands on THIS test set
    "objective_value": 0.61,         # J at that t
    "surface": "stay"|"peak",        # episodic → stay; continuous → peak (§4)
    "delta_from_clinical": -0.53,    # 0.42 − 0.95
    "metrics_at": { sensitivity, specificity, ppv, npv, f1, fpr, nne, flag_rate },
    "in_sample": true,               # honesty flag: read off the sweep, optimistic
    "assumption": "equal FP/FN cost; prevalence-blind",
    "citation": "Youden 1950" },
  ...
]
```

`metrics_at` are pointers into the sweep at the nearest grid threshold (or computed at
the exact selected score) — never a separately-graded headline. `in_sample: true` is on
every entry. The clinical threshold appears here too, as a rule named `clinical` with
`in_sample: false` (it was never fitted) — so the reader can line the shipped policy up
against all ten data-driven picks in one list.

```
   ROC space — where the rules land (schematic, one model, one site)
   TPR
   1.0┤            ● closest-(0,1)    ▲ = clinical policy (0.95): far down here
      │        ● Youden                  → high spec, low sens
      │      ● CZ
      │    ● F1                       every ● is a locator on the SAME curve;
      │   ●IU                         the sweep publishes the whole curve, the
      │  ●                            locators just say "a rule would stand here"
      │ ●                        ▲ clinical
      └───────────────────────────────── FPR
      0                                 1
```

---

## 4. Continuous tasks: surface + alert burden

### 4.1 Which surface the rules and sweep use

A continuous task has many rows per stay. We do **not** pool rows (long stays would
dominate — the trap the peak/landmark split exists to avoid). Decision:

- **Data-driven thresholds and the operating-point sweep are computed on the PEAK
  surface** — each stay collapsed to its maximum risk, one row per stay. **This is the
  established evaluation for real-time alert models, not just a convenience:** Kamran et
  al. (NEJM AI 2024) compute the hospitalization-level operating point on the max score,
  firing "the first time the [sepsis] threshold was exceeded" — precisely this peak /
  any-alert collapse. It is the alarm/screening workflow: "if the model ever crosses `t`,
  the stay is flagged." On the peak surface the frame looks episodic again, so §2–§3
  apply unchanged, and — see §4.2 — the burden is horizon-invariant here.
- Calibration stays **out** of the peak surface (the peak-calibration trap, docs §2)
  and remains in the landmark section. DCA / net benefit is removed from continuous
  mode **entirely** (2026-07-15) — not merely off the peak: the same peak-calibration
  trap makes net benefit on max-of-many risk misleading, and net benefit is now an
  **episodic-only** view. Continuous tasks emit no `dca.json`; landmark reports carry
  three pillars (discrimination / calibration / fairness).

```
   continuous preds (many rows / stay)          episodic preds (1 row / stay)
   stay A: .08 .11 .40 .22 .09  ──max──> 0.40        stay A: 0.40
   stay B: .30 .55 .48         ──max──> 0.55         stay B: 0.55     ← already
   stay C: .05 .04 .06 .03     ──max──> 0.06         stay C: 0.06       one row
              PEAK SURFACE (1 row/stay)                  same math as peak
                     │                                          │
                     └──────────► §2 sweep + §3 locators ◄──────┘
```

### 4.2 Alert burden — encounter-level, on the peak surface (Kamran et al.)

Selection rules read the ROC axis (se, sp). Alarm fatigue lives on a different axis:
**how many stays alert, and how many alerts are false**. FLAIR measures it exactly as
Kamran et al. (NEJM AI 2024) evaluate the Epic Sepsis Model: at the **hospitalization
level, on the maximum score**, where the model "fires an alert the first time the
threshold was exceeded" — **one alert per hospitalization**. Their headline alarm-fatigue
statement, *"95% of alerts were for individuals who did not acquire sepsis,"* is exactly
**1 − PPV on the peak/encounter collapse** (PPV ≈ 5% at their alert threshold). So the
operating-point burden rides the same peak surface the sweep already uses (§4.1) — no
separate per-timepoint machinery:

```
"alert_burden": {                    # encounter-level, peak surface — one alert / stay
  "frac_stays_alerting":  0.31,  # peak flag-rate: hospitalizations that EVER fire (peak >= t)
  "ppv":                  0.05,  # of alerted hospitalizations, frac truly positive
  "false_alarm_fraction": 0.95,  # 1 - PPV — Kamran's "95% of alerts were false"
  "nne":                  20.0,  # 1/PPV — hospitalizations alerted per true case (== WDR)
  "n_stays":              8140,
  "n_stays_alerting":     2523   # suppressed if < MIN_CELL_COUNT
}
```

- **`frac_stays_alerting`** = peak flag-rate = "one alert per hospitalization, fired the
  first time the score crosses `t`" (Kamran's alert model). The workload axis: what
  fraction of admissions generate an alert as `t` sweeps.
- **`nne = 1/PPV`** = alarms per true case — the honest alarm-fatigue number. At a
  low-prevalence hourly task's clinical `t` (matching Kamran et al.'s Epic Sepsis Model
  operating point), NNE ≈ 20 (PPV ≈ 5%) = 20 alerted admissions per true case caught;
  `false_alarm_fraction = 1 − PPV = 0.95`. NNE is *already* computed on the peak surface
  in `discrimination`/`kpi` today — the sweep just reports it at all 99 thresholds.
- **Horizon- and unit-invariant — this is why peak-only also fixes the daily-vs-hourly
  mismatch.** Because burden is one-alert-per-hospitalization on the peak collapse, it
  carries **no patient-day denominator**. The daily ICU tasks (30-day, day-windowed) and a
  hypothetical hourly task (7-day, hour-windowed) therefore produce directly comparable encounter-level burden
  with **no window_unit conversion**. The peak collapse *dissolves* the horizon mismatch
  raised above rather than correcting for it — a second reason (beyond matching Kamran)
  to keep the operating point on the peak surface.

**No per-timepoint / per-day burden (RESOLVED, §10 Q9).** The operating-point analysis is
**encounter / stay (`hospitalization_join_id`) level only — never day or window level.**
The per-timepoint view (window crossings over patient-days, raw + first-alert, window_unit
denominator, refractory bounds, day-granularity cap) is a *different unit of analysis*
(JMIR 2026) and is **not produced** in v1. The single day/hour-level number the analysis
is allowed to emit is the **treatment lead-time scalar** (§4.4) — and only when an
`intervention` column is attached: "a number, not a distribution." Everything else on the
operating point is one row per stay.

```
   Two coupled axes as t sweeps 0.01 → 0.99  (continuous, ALL on the peak surface)
   perf                                    burden (encounter-level)
   1.0┤●●●                          frac    ┤●●●
      │   ●● sensitivity (peak)     alerting │   ●●●   frac_stays_alerting ↓ as t ↑
      │     ●●●                              │      ●●●
      │        ●●●●                   NNE    │ ●         NNE = 1/PPV (alarms per case)
      │            ●●●●●●●                   │  ●●●●●●●●● ↑ as t ↓ (more false alarms)
      └────────────────────── t             └────────────────────── t
      a threshold is "good" only if the encounter alert-rate AND NNE are acceptable,
      not just se/sp — the same PPV-60%-vs-alert-fatigue trade-off Kamran frames.
```

### 4.3 Operating-budget view (optional, derived)

Because score scales differ between models, comparing two models "at threshold 0.3" is
meaningless. The honest comparison is **at equal workload** — and on the peak surface,
workload = the encounter flag-rate (`frac_stays_alerting`). Invert the sweep: given a
target flag-rate (e.g. alert {5%, 10%, 20%} of admissions), report the threshold that
hits it and the se / PPV / NNE there — "sensitivity at a fixed encounter alert budget."
This is the alert-rate-target rule (#9) surfaced as a model-comparison view, cheap to
derive from the sweep; include as `operating_budget` in the continuous JSON.

### 4.4 Intervention / treatment sub-analysis under the sweep

**Scope: intervention tasks only.** Only tasks that declare an `intervention` config get this block.
The daily ICU tasks (daily mortality, daily LTACH) declare **no** intervention → the entire §4.4
block is simply **absent** for them: the confusion sweep (§2) and alert burden (§4.2) are
still produced in full, while `sub_rates`, `intervention_timing` and `intervention_offsets`
are all null with a recorded `sub_analyses_reason` ("no intervention/treatment column
configured") — exactly the existing `if not iv_col:` behaviour, unchanged. Nothing below
fires unless the task names a treatment-time column. The rest of §4.4 concerns that case.

Continuous tasks that declare an `intervention` config (e.g. a hypothetical hourly task, onset → treatment)
today attach, **at every threshold**, a three-arm sub-analysis (`_positive_arm`,
`_treated_negative_arm`, `_untreated_negative_arm`) where the two treated arms carry a
**full** `hours_before_treatment` distribution — median + median-CI, p10/p25/p75/p90,
and a binned histogram (`_delta_summary`). Multiplying that by the 99-point sweep is the
one place the sweep gets genuinely expensive *and* genuinely disclosive: 99 lead-time
histograms leaving each site, most of them near-identical to their neighbours, each a
bootstrap for its median CI. So under the sweep the sub-analysis changes shape.

Decision (per the "keep the number, not the distribution" steer):

1. **Rates stay swept — cheap, and they vary with `t`.** At every threshold each arm
   keeps only its count + rate: `pct_fired` (positive arm, treated-negative arm) and
   `misfire_rate` (+CI) (untreated-negative arm). These fall straight out of the same
   one-row-per-stay agg the confusion matrix already builds (no extra passes), and they
   are the informative part — they **decompose the false-alarm axis** into "fired on a
   stay that got treated anyway" vs "pure misfire on a never-treated stay," which the
   plain FPR cannot show. Keep them on all 99 rows.
2. **Drop the hours-before-treatment DISTRIBUTION from the sweep.** No quantiles, no
   histogram, no median-CI at any swept threshold. `_delta_summary`'s distribution output
   no longer rides the 99-point axis. This removes the only real privacy/cost blow-up the
   sweep would introduce.
3. **Keep ONE lead-time number, at the named thresholds only.** A single median "hours
   before treatment" scalar per treated arm — "the model fired ~6 h before treatment" —
   computed only at the **named** thresholds (clinical + the drawn locators), NOT across
   the 99-point sweep. Lead-time is a *timing descriptor of an operating point*, not a
   *threshold-selection signal*; it does not need to be a curve, a CI, or a distribution
   to answer the clinical question — one number where the decision actually sits.
4. **`_intervention_offsets` unchanged.** The mean ± SD treatment time on the forward
   landmark axis (the red reference line on the lead-time sensitivity curve) is
   threshold-independent, computed once — untouched.

Net effect: the treatment sub-analysis stops being 99 distributions and becomes 99 cheap
rate rows + a few single lead-time scalars at the thresholds a reader actually reads.

```
   before (per threshold, ×99)              after
   ┌─ positive arm ───────────┐             ┌─ swept row (×99) ─────────────┐
   │ pct_fired                │             │ pos.pct_fired                 │
   │ hours_before_treatment:  │    ──►       │ treated_neg.pct_fired         │  cheap,
   │   median + median_ci     │             │ untreated.misfire_rate (+ci)  │  varies
   │   p10 p25 p75 p90         │             └───────────────────────────────┘  with t
   │   histogram[bins]        │             ┌─ named thresholds only ───────┐
   └──────────────────────────┘             │ median_hours_before_treatment │  one
    (× 2 treated arms × 99)                 │   @ clinical, @ youden, @ f1…  │  scalar,
                                             └───────────────────────────────┘  no dist
```

Residual toggle in §10 Q8: named-only scalar (default) vs one cheap swept median curve.

---

## 5. Output schema

Extend the existing `operating_points.json` (both modes) — it is already "the one
self-contained decision-point view", so the sweep, the locators and the burden belong
there, single-sourced. No new file.

```
operating_points.json
├── metadata
│   ├── mode                 "episodic" | "continuous"
│   ├── surface              "stay" | "peak"
│   ├── grid                 {min:0.01, max:0.99, step:0.01, n:99}
│   ├── clinical_threshold   0.95
│   ├── clinical_index       94                # row where is_clinical=true
│   ├── clinical_direction   "above" | "below"
│   ├── unit_of_analysis     "encounter (peak)"    # continuous op-point (Kamran); NEVER day/timepoint (§9)
│   ├── selection_params     {sens_target:0.90, spec_target:.., ppv_target:0.30,
│   │                         alert_rate_target:0.10, f_beta:1, cost_ratio:1}
│   └── min_cell_count       10
├── prevalence               0.021
├── by_threshold[99]
│   └── { threshold, is_clinical, confusion{...}, metrics{...,_ci},
│         decision{...},                          # direction-aware, == dca.json
│         alert_burden{...},                      # continuous, §4.2 — ENCOUNTER-level (peak):
│                                                 #   frac_stays_alerting, ppv, false_alarm_fraction,
│                                                 #   nne. No patient-day denominator (Kamran).
│         sub_rates{...} }                        # continuous + intervention, §4.4:
│                                                 #   pos.pct_fired, treated_neg.pct_fired,
│                                                 #   untreated.misfire_rate(+ci) — RATES ONLY
├── data_driven_thresholds[]                      # §3.3 — one per rule + clinical
│   └── { rule, threshold, objective_value, surface, delta_from_clinical,
│         metrics_at{...}, in_sample, assumption, citation }
├── operating_budget[]                            # continuous only, §4.3 (optional)
│   └── { target_flag_rate, threshold, sensitivity, ppv, nne }   # equal ENCOUNTER workload
├── intervention_timing[]                         # continuous + intervention, §4.4
│   └── { threshold, is_clinical,                 #   NAMED thresholds only (clinical+locators)
│         median_hours_before_treatment_positive,       # single scalar, no dist / no CI
│         median_hours_before_treatment_treated_negative }
└── intervention_offsets{...}                     # UNCHANGED — mean±SD treat time (red line)
```

- `REPORT_SCHEMA` bumps (2 → 3). The ± offset grid, `_DELTAS`, `_EPISODIC_OFFSETS`,
  `threshold_grid` and `dropped_offsets` are removed. `_delta_summary`'s per-threshold
  hours-before-treatment **distribution** (quantiles + histogram + median-CI) leaves the
  swept rows (§4.4): swept rows keep `sub_rates` only; the median lead-time survives as a
  single scalar in `intervention_timing` at the named thresholds.
- **Removed: the lead-time `sensitivity` curve by hospitalization time.** Per the steer
  "remove `sensitivity_by_hospital_time`", drop the per-hospitalization-time sensitivity
  curve — `leadtime.py::_sensitivity_curve` → `leadtime.json["curve"]` and its rendering
  in `_viz.py::_lead_png`. It is a day/hour-level breakdown, which the encounter-only
  decision (§10 Q9) rules out. **Kept:** `median_first_alert_hospitalization_time` — a
  single first-alert timing *number*, not a per-time curve. Open: whether the
  per-hospitalization-time mean-risk `trajectory` (also day-level) goes too — §10 Q10.
- `kpi.py::_operating_points_block` keeps reading `by_threshold` + `is_clinical`
  (still valid); extend its headline to add a `selected_thresholds` digest (Youden &
  F1 landing points) and, for continuous, `alert_burden` at the clinical row.
- `discrimination.json`, `calibration.json`, `dca.json`, `fairness.json` stay
  **threshold-free**. The sweep does not leak back into them.
- Privacy: every confusion count and burden count passes `suppress()`; any metric
  from `< MIN_CELL_COUNT` flagged/acted patients is nulled with a reason. 99 rows do
  not change the suppression rule, only its frequency at the sparse tails
  (t → 0.99 flags almost nobody; t → 0.01 flags almost everyone).

---

## 6. Visualization (`--viz`): trends across all thresholds

`--viz` renders local site-validation PNGs **from the report JSON dicts** (never the
parquet, never exported). Good news: `report/_viz.py` already has the right skeleton —
`_render_scorecard` draws **one trend panel per KPI against the threshold grid** (point
line + 95% CI band + clinical `vline` + a value table), and `_mark_grid_on_roc` marks
the grid points on the ROC. Today those run on the 3–5-point grid. On the 99-point
sweep the 5-dot scatter becomes a real **trend curve**, which is exactly what we want —
but three things break at 99 points and must change, and the new metrics + locators
need to appear.

### 6.1 Scorecard → `threshold_trends.png` (upgrade of `operating_points.png`)

- **Panels become continuous trend lines** over 99 thresholds (drop the per-point `o`
  markers, keep the `#5b86a6` line + CI band). sens ↓, spec ↑, PPV ↑, flag-rate ↓,
  NNE ↓ as `t` rises — clean monotone-ish curves that read far better at 99 points
  than at 5.
- **x-ticks go sparse.** Line 365's `ax.set_xticks(thr)` (all 99) → ~11 ticks
  (`0.0, 0.1, … 1.0`). The clinical red-dashed `vline` stays.
- **Data-driven locators drawn on every panel.** A thin vertical marker per rule
  (Youden, F1, CZ, IU, closest-(0,1), sens-target, …) so you can read "at Youden's `t`,
  sensitivity = 0.71" straight off each trend. One legend maps line-style → rule.
  Default drawn-as-vline set is small to avoid clutter (see §9); the rest live only in
  the table.
- **The value table is rebuilt.** A 99-column table is unreadable, so drop the
  all-thresholds table. Replace it with a **compact table whose columns are only the
  named thresholds** — `clinical` + one column per data-driven locator — rows =
  sens / spec / PPV / NPV / F1 / FPR / NNE / alert-rate (+ burden, continuous). This
  IS the episodic ask made visual: *"for this model on this site's test set, here is
  the threshold each rule picks and what it buys."* The clinical column keeps its blue
  highlight.

```
   threshold_trends.png  (schematic)
   ┌── Sensitivity ──────┐ ┌── Specificity ─────┐ ┌── PPV ─────────────┐ ┌── NNE ──┐
   │‾‾‾●╲                │ │            ╱●‾‾‾    │ │          ╱‾‾‾      │ │╲        │
   │     ╲___  CI band   │ │      ___╱          │ │    ___╱            │ │ ╲___    │
   │ ┆  ┆ ╲__╲           │ │ ┆  ┆ ╱             │ │ ┆  ┆╱              │ │ ┆ ┆╲__  │
   └─┴──┴────┴── t ──────┘ └─┴──┴─── t ─────────┘ └─┴──┴─── t ────────┘ └─┴─┴──── ┘
     Y  F1  clin(red)        Y  F1 clin            Y  F1  clin            (locator vlines
                                                                          on every panel)
   ┌─ compact table: columns = named thresholds only ──────────────────────────────┐
   │            clinical   youden  closest01   f1max   sens_tgt   ppv_tgt  ...       │
   │ threshold    0.95      0.41     0.38       0.33     0.22       0.55             │
   │ sens         0.20      0.71     0.74       0.80     0.90       0.55             │
   │ spec         0.99      0.88     0.85       0.79     0.66       0.93             │
   │ ppv / nne …                                                                     │
   └────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 ROC / PR overlay → mark locators, not 99 dots

`_mark_grid_on_roc` currently plots every grid point on the ROC. At 99 points that just
**retraces the ROC line** — redundant. Change it to mark only the **named** points: the
clinical operating point (with its sens/FPR CI error bars, as now) plus each
data-driven locator as a labelled dot on the curve. The ROC line already shows the full
sweep; the dots show where each rule *stands* on it. Same treatment on the PR panel
(clinical + F1 / PPV-target markers). Legend text `"threshold grid (±0.05, ±0.10)"` →
`"data-driven thresholds"`.

### 6.3 Continuous: `alert_burden.png` (new) — the two coupled axes

Renders §4.2. Twin-y trend across the full sweep:

- **left y:** detection performance on the peak surface (sensitivity, PPV).
- **right y:** encounter-level burden — `frac_stays_alerting` (fraction of admissions
  alerting) and `nne` (alarms per true case). **No patient-day axis** (§4.2).
- **x:** threshold, clinical + locator `vline`s.
- Shade the region where the encounter alert-rate exceeds a configurable **budget line**
  (alarm-fatigue zone, Sendelbach 2013), so a threshold that wins on ROC but alerts 60%
  of admissions — or needs 20 alerts per true case — is visibly disqualified.
- If `operating_budget` is present (§4.3), add markers: "sensitivity at a 10% flag-rate."

```
   alert_burden.png (continuous, twin axis — all encounter-level, peak surface)
   perf                                                  frac alerting / NNE
   1.0┤‾‾●╲  sensitivity (peak)                                  ┤ hi
      │    ╲___                              ░░░ alarm-fatigue   │
      │  ╱‾‾   ╲__  PPV            ‾‾╲__      ░░░ zone (>budget)  │
      │ ╱          ╲___   frac_stays_alerting ╲___              │
      │╱   ┆    ┆      ╲__╱‾NNE=1/PPV↗            ╲____          │ lo
      └────┴────┴──────────────────────────────── t ────────────┘
           Y   clin(red)      budget line ──────────
```

### 6.4 Confusion panels → named thresholds only

`_render_operating_points` draws one 2×2 confusion panel per grid threshold. 99 panels
is absurd; render the 2×2 only at a **small named set** — default `clinical`, `youden`,
`f1` (configurable). Remove the `dropped_offsets` footnote entirely (no offsets exist
anymore). The title states which thresholds are shown and why. Sub-analysis rendering
(`_render_operating_points_sub`, intervention timing) changes shape per §4.4: the
per-threshold hours-before-treatment **histogram panel is gone**. In its place, (a) the
swept **fire-rate curves** — positive-arm `pct_fired`, treated-negative `pct_fired`,
untreated `misfire_rate` vs `t`, which decompose the false-alarm axis — and (b) a single
**median lead-time annotation** at each named threshold ("~6 h before treatment @ clinical").
No distribution panel, no per-threshold histogram.

### 6.5 `data_driven_thresholds.png` (new) — where every rule lands

A lollipop / strip figure: one row per rule, a marker at its threshold on a 0–1 axis,
the clinical policy drawn as a reference line, each row annotated with sens / PPV at
that `t`. One glance answers "where does every statistical rule put the cutpoint versus
the shipped policy." This is the per-site tile that, aggregated across sites, becomes
the docs §3 "long game" figure (do four sites' Youden thresholds cluster far from the
policy?).

```
   data_driven_thresholds.png            clinical policy (0.95) ──┐
   youden        ●─────0.41                                       ┊
   closest01    ●──────0.38                                       ┊
   cz           ●─────0.43                                        ┊
   iu           ●────0.36                                         ┊
   f1max       ●───0.33                                           ┊
   sens_tgt   ●──0.22                                             ┊
   ppv_tgt          ●──────0.55                                   ┊
   cost              ●───────0.60                                 ┊
   alert_rate         ●────────0.68                               ┊
   maxacc     ●0.05 (flagged degenerate)                          ┊
   └──────────────────────────────────────────────────── t ──────●──┘
   0                                                             0.95
```

### 6.6 Cleanups

- Drop `dropped_offsets` handling (lines ~650–655, ~246) — the concept is gone.
- Update legend / title strings that name "±0.05, ±0.10" to "full sweep + locators."
- No JSON schema change is needed for viz — it consumes §5's dicts. The locators and
  burden it draws are already in the JSON.

---

## 7. Pipeline (end to end)

```
                         preds (test split, one task)
                                    │
                 ┌──────────────────┴───────────────────┐
                 │ mode = episodic                       │ mode = continuous
                 ▼                                        ▼
          one row / stay                         collapse each stay → PEAK
          (as-is)                                (max risk); ALL op-point
                 │                                work encounter-level
                 └──────────────────┬───────────────────┘
                                    ▼
                        SWEEP_GRID = 0.01 … 0.99  (99)
                                    │
        ┌───────────────┬───────────┴───────────┬────────────────────┐
        ▼               ▼                       ▼                     ▼
  per-threshold    per-threshold          data-driven          alert_burden
  confusion +      decision / net         locators (§3):        (§4.2, continuous):
  metrics (§2)     benefit (dir-aware)    Youden, CZ, IU,       encounter-level —
  vectorized,      == dca.json            F1, sens/spec tgt,    frac_stays_alerting,
  1 bootstrap                             cost, PPV/rate tgt    PPV, NNE (peak)
        │               │                       │                     │
        └───────────────┴───────────┬───────────┴─────────────────────┘
                                     ▼
                        operating_points.json  (schema 3)
                                     │
                        is_clinical row flagged  ("clinical intrusion")
                                     │
                    leaderboard UNCHANGED (pooled AUROC only)
```

---

## 8. Per-task implications (sanity check)

| Task | Clinical `t` | Dir | Mode | What the sweep newly shows |
|---|---|---|---|---|
| icu_daily_mortality | 0.95 | above | continuous (30-day, day) | Is 0.95 near any data-driven pick, or is the useful region far lower? Encounter alert-rate + NNE at 0.95 vs at Youden. No intervention → §4.4 absent. |
| icu_daily_ltach | 0.50 | above | continuous (30-day, day) | Youden vs the more-likely-than-not policy; encounter burden at 0.50. No intervention → §4.4 absent. |
| extubation_failure_24h | 0.10 | **below** | episodic | Locators computed on risk (alarm) basis; action view = who gets extubated. Does 0.10 sit where the data separates fail/no-fail? |
| _hypothetical hourly deterioration task_ (illustrative — no such task ships today) | 0.10 | above | continuous (7-day, hour) | The headline case for an intervention task: Youden (prevalence-blind) vs F1/PPV-target (sees the ~2% prevalence). Encounter alert-rate + NNE across the sweep (frac admissions alerting; 1−PPV false-alarm) — the Kamran alarm-fatigue curve. **The one task shape with an intervention** → §4.4 sub_rates + median lead-time apply. |
| icu_readmission | 0.20 | above | episodic | Where four rules land vs the top-20%-risk policy. |

Note the shipped clinical thresholds `{0.95, 0.50, 0.10, 0.20}` (and the illustrative
hourly task's 0.10) are all multiples of
0.01 → all land exactly on the grid. The validation in §2.1 guards future edits.

Two continuous-mode facts the table makes explicit: (a) **only an intervention task (the
illustrative hourly one) has an intervention config**, so §4.4's treatment sub-analysis
exists there alone — the daily ICU tasks skip it entirely
(§4.4). (b) The continuous tasks run on **different horizons/window units** (the daily ICU tasks
= 30-day, day-windowed; a hypothetical hourly task = 7-day = 168-h, hour-windowed), but because the operating
point and its burden are **encounter-level on the peak surface** (§4.2, Kamran), that
difference **does not touch the operating point** — there is no patient-day denominator to
convert. It would matter only for the deferred per-timepoint texture view (§9).

---

## 9. What is explicitly NOT in v1

- **Bootstrap optimism-correction** of selected-threshold performance, and the
  threshold **stability range** — designed in docs §3, deferred here by choice. The
  clean sweep + `in_sample` locator framing is the v1 honesty story.
- **Per-timepoint / per-day alert burden — not produced at all (§10 Q9 resolved).** The
  operating-point analysis is encounter/stay-level only (§4.2, Kamran). The timepoint view
  (alerts/patient-day raw & first-alert, window_unit denominator, refractory window,
  day-granularity cap) is a *different unit of analysis* (JMIR 2026) and is excluded from
  v1 entirely. If it ever returns (v2), that is the one place the 30-day/day vs 7-day/hour
  reconciliation would live.
- **The lead-time `sensitivity`-by-hospitalization-time curve — removed (per your steer,
  §5).** A day/hour-level breakdown, incompatible with the encounter-only decision.
- **Per-landmark optimal-threshold curve** (a data-driven threshold at each landmark,
  showing the fixed cutpoint is wrong at most lead times) — attractive (docs §3 calls
  it "the view that did not exist anywhere before") but heavier; v2.
- **Census / caseload-based utility** (`clinical_utility.json`) — already scoped as a
  future add-on in docs §2; not part of this change.
- **Full hours-before-treatment distribution** (quantiles + histogram + median-CI) on the
  swept rows — deliberately collapsed to a single median at the named thresholds (§4.4).
  If a site needs the full lead-time *shape*, it can return in v2 at the clinical
  threshold only (one distribution, not 99), never across the sweep.
- Any change to the **leaderboard** — it stays on pooled AUROC.

---

## 10. Open questions for review

1. **Grid step 0.01 vs 0.02.** 99 points is chosen. Confirm the bootstrap-band cost is
   acceptable at every site's cohort size, or fall back to 0.02 (49 points) if a
   large-cohort site is slow. (Point estimates are cheap either way; only the shared
   bootstrap scales.)
2. **Rule menu scope.** All ten families are in. Any to drop from the *default* set to
   keep the JSON lean (e.g. max-accuracy is kept only as a cautionary "bad rule")?
3. **Cost ratio source.** Do we want a per-task `cost_ratio` in `TASK_POLICY` now, or
   ship cost-based with ratio = 1 and add real ratios when clinicians state them?
4. **Alert-budget targets.** Default encounter flag-rate grid for §4.3 — alert {5%, 10%,
   20%} of admissions? Per-task override?
5. **kpi.py headline.** Which one or two locators go in the scoreboard digest (Youden +
   F1?), so `kpi.json` stays compact.
6. **Viz locator set (§6).** Which locators get drawn as `vline`s on every trend panel
   and rendered as confusion 2×2s — all ten clutters. Default proposal: draw
   `clinical + youden + f1 + closest-(0,1)` as vlines and 2×2s; list the rest only in
   the compact table and the `data_driven_thresholds.png` strip. Confirm.
7. **Alert-budget line (§6.3).** Default encounter alert-rate (or NNE) threshold for the
   alarm-fatigue shading on `alert_burden.png` — per-task, or one global default?
8. **~~Lead-time scalar placement (§4.4).~~ RESOLVED.** One median hours-before-treatment
   per treated arm — a single **number** (no distribution, no CI), only when a treatment
   column is attached. The sole day/hour-level number the operating-point analysis emits.
9. **~~Per-timepoint burden (§4.2).~~ RESOLVED.** Operating points are **encounter /
   `hospitalization_join_id` level only, never day.** No per-day / per-timepoint burden in
   v1. The only day-level number allowed is the treatment lead-time scalar (§4.4), and
   only when treatment is attached.
10. **~~Lead-time `trajectory` (§5).~~ RESOLVED.** Both day-level lead-time curves are
    dropped — the sensitivity-by-hospitalization-time `curve` *and* the mean-risk
    `trajectory` (`leadtime.py::_sensitivity_curve` + `_risk_trajectory`, and their two
    PNGs). `leadtime.json` keeps only the first-alert **number**
    (`median_first_alert_hospitalization_time`). Consistent with the encounter-only
    decision (Q9).

**Implementation-time decisions (recorded for the plan):** (a) **fairness is swept at
all 99 thresholds** — the episodic per-threshold subgroup fairness runs on the full grid
(its one-shared-resample-index design scales); (b) net benefit is **omitted from
continuous mode entirely** (peak-DCA trap) and is an episodic-only view. It was
initially kept on the landmark surface (dca.json's landmarks), but that was removed
(2026-07-15) — DCA is now episodic-only, and continuous tasks emit no `dca.json`.

---

## 11. References

- Youden WJ. *Index for rating diagnostic tests.* Cancer 1950;3(1):32–35.
- Metz CE. *Basic principles of ROC analysis.* Semin Nucl Med 1978;8(4):283–298.
  https://pubmed.ncbi.nlm.nih.gov/112681/
- Perkins NJ, Schisterman EF. *The inconsistency of "optimal" cutpoints obtained using
  two criteria based on the ROC curve.* Am J Epidemiol 2006;163(7):670–675.
  https://academic.oup.com/aje/article/163/7/670/77813
- Liu X. *Classification accuracy and cut point selection.* Stat Med
  2012;31(23):2676–2686. https://onlinelibrary.wiley.com/doi/10.1002/sim.4509
- Ünal İ. *Defining an optimal cut-point value in ROC analysis: an alternative
  approach.* Comput Math Methods Med 2017:3762651.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5470053/
- van Rijsbergen CJ. *Information Retrieval*, 2nd ed., 1979 (F-measure).
- Leeflang MMG, Moons KGM, Reitsma JB, Zwinderman AH. *Bias in sensitivity and
  specificity caused by data-driven selection of optimal cutoff values.* Clin Chem
  2008;54(4):729–737.
- Thiele C, Hirschfeld G. *cutpointr: improved estimation and validation of optimal
  cutpoints in R.* J Stat Softw 2021;98(11). https://www.jstatsoft.org/article/view/v098i11
- Sendelbach S, Funk M. *Alarm fatigue: a patient safety concern.* AACN Adv Crit Care
  2013;24(4):378–386. https://pubmed.ncbi.nlm.nih.gov/24153215/
- Reyna MA et al. *Early prediction of sepsis from clinical data: the PhysioNet/CinC
  Challenge 2019.* Crit Care Med 2020;48(2):210–217.
  https://physionet.org/content/challenge-2019/1.0.0/
- **Kamran F, Tjandra D, Heiler A, Virzi J, Singh K, King JE, Valley TS, Wiens J.
  *Evaluation of Sepsis Prediction Models before Onset of Treatment.* NEJM AI
  2024;1(3):AIoa2300032. https://doi.org/10.1056/AIoa2300032** — the peak/encounter
  operating-point basis (§4.1–§4.2): hospitalization-level max score, "fires the first
  time the threshold is exceeded," PPV/sensitivity + the "95% of alerts were false"
  (1−PPV) alarm-fatigue framing. Local copy: `docs/paper/lit/AIoa2300032.pdf`.
- *The impact of evaluation strategy on sepsis prediction model performance metrics in
  intensive care data.* JMIR 2026:e72083. https://www.jmir.org/2026/1/e72083
- Workload-capacity / census-based EWS evaluation.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12819968/
- Companion design rationale: `docs/eval_continuous_three_ways.md` §3.

> Two attribution caveats flagged during research: (a) the census/patient-day burden
> framing is well supported (workload-capacity + Churpek/Mayampurath deterioration
> work), but no single Mayampurath paper is titled around "concurrently-hospitalized
> patients" — verify before quoting that exact phrase. (b) NNE and WDR are the same
> `1/PPV` quantity under different names — reported once (§3.2).
