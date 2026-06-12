# flair-baseline — end-to-end flow

XGBoost **count-feature** baseline for all 5 FLAIR ICU tasks, trained on MIMIC-IV with a leak-free **75/25 split on `hospitalization_join_id`**. Features are counts of every ELF event code that occurs **strictly before** each prediction's `prediction_dttm` (point-in-time, no leakage). FLAIR is all-in-one: the CLIF→MEDS ETL is vendored into `flair_benchmark`, so this subproject only depends on `flair-benchmark` (editable path dep) + `xgboost`.

```         
                         ┌─────────────────────────── flair-baseline run ───────────────────────────┐
 task.build()            │                                                                            │
 (flair_benchmark) ──► cohort ──► FE-meds ──► MEDS.parquet ──► count featurizer ──► XGBoost ──► preds ──► report (+viz)
                       (split)   (CLIF→MEDS)  + codes.parquet   (sparse, leak-free)  (model)   .parquet   9 JSON / PNG
```

| stage | code | output |
|------------------------|------------------------|------------------------|
| ① cohort | `flair_benchmark/tasks/taskN_*.py` + `_split.py` | `cohort.parquet` |
| ② FE-meds | `flair_benchmark/features/fe_meds.py` + `flair_benchmark/meds_etl/` | `data/MEDS.parquet`, `metadata/codes.parquet` |
| ③ featurize | `flair_baseline/featurize.py` | in-memory sparse CSR |
| ④ train/score | `flair_baseline/train.py` | `model.json`, `vocab.json`, `preds.parquet` |
| ⑤ report | `flair_benchmark.report.build_report` | `report/*.json` (+ `report/viz/*.png`) |

------------------------------------------------------------------------

## 0. Install (uv, local)

``` bash
cd flair_baseline
uv sync          # installs flair-benchmark from ../ (editable) + xgboost + matplotlib
```

`pyproject.toml` declares `flair-benchmark` as a path source (`{ path = "..", editable = true }`), so edits to the vendored ETL / tasks are picked up directly. Python 3.12.

------------------------------------------------------------------------

## 1. Commands

### `flair-baseline run`

```         
flair-baseline run --clif-config CFG.json [--elf-config flair_elf_config.yaml]
                   [--out out] [--task NAME] [--train-end YYYY-MM-DD] [--test-start YYYY-MM-DD]
                   [--report/--no-report] [--viz/--no-viz] [--reuse/--no-reuse]
```

| flag | default | meaning |
|------------------------|------------------------|------------------------|
| `--clif-config` | (required) | `clif_config.json` (path, filetype, timezone, `site`). `site=mimic` ⇒ 75/25 split |
| `--elf-config` | `flair_elf_config.yaml` | event concepts to extract + `feature_exclude_prefixes` |
| `--out` | `out` | output root; each task writes to `out/<task_name>/` |
| `--task` | all 5 | task name or prefix (`task1` → `task1_icu_daily_mortality`) |
| `--train-end` / `--test-start` | none | date cutoff for non-mimic sites (omit on mimic) |
| `--report` / `--no-report` | report | build the 9-JSON FLAIR report bundle |
| `--viz` / `--no-viz` | no-viz | also render sanity-check PNGs into `report/viz/` |
| `--reuse` / `--no-reuse` | no-reuse | reuse existing `cohort.parquet` + `data/MEDS.parquet` (skip ETL) |

Examples:

``` bash
# all 5 tasks on MIMIC, with report + PNGs
uv run flair-baseline run --clif-config ../clif_config_mimic.json --out out --viz

# one task
uv run flair-baseline run --task task4 --clif-config ../clif_config_mimic.json --out out

# re-featurize/train/report without re-running the ETL (MEDS already on disk)
uv run flair-baseline run --clif-config ../clif_config_mimic.json --out out --reuse --viz
```

### `flair-baseline score` (external site)

```         
flair-baseline score --task NAME --clif-config OTHER_SITE.json
                     --model out/<task>/model.json --vocab out/<task>/vocab.json
                     [--elf-config …] [--out out_external]
```

Builds the task cohort on a non-mimic site (→ all rows `test`), featurizes against the **saved vocabulary**, and scores with the saved model. No training.

------------------------------------------------------------------------

## 2. Stage detail

### ① Cohort — `task.build()` + `_split.py`

`taskN_*.build(clif_config, train_end, test_start) -> pl.DataFrame`, one row per `prediction_id`:

```         
hospitalization_id, hospitalization_join_id, prediction_dttm,
<label_column>, split, age_at_admission, sex_category, race_category,
ethnicity_category, window_number, prediction_id
```

- `hospitalization_join_id` = stitched-encounter key (clifpy `stitch_encounters`, 6 h gap; one block can span several hospitalizations).
- **Split** (`flair_benchmark/_split.py`, `assign_split`): on MIMIC the split is decided at the **block grain** — 75 % of distinct `hospitalization_join_id` → `train`, 25 % → `test`, broadcast to all rows. A block never spans both splits (no split leakage). Non-mimic sites with no cutoff dates → **all `test`** (external validation on the full cohort).
- `prediction_id = {join_id}-{hospitalization_id}-{patient_id}-{window_number}`.

Written to `out/<task>/cohort.parquet`.

### ② FE-meds — `fe_meds.build_meds_tables()` + `meds_etl/`

Runs the **vendored CLIF→MEDS ETL** over the cohort's `hospitalization_id`s for the domains listed in `flair_elf_config.yaml`, attaches `hospitalization_join_id`, and writes two MEDS-standard tables.

`data/MEDS.parquet` — one row per clinical event:

```         
subject_id (=patient_id), time, code, numeric_value, text_value,
hospitalization_id, hospitalization_join_id
```

ELF code format (`//` separates hierarchy levels):

```         
VITAL//heart_rate                         LAB//lactate//mmol/l//bmp
MED_CON//norepinephrine//mcg/kg/min//start    MED_INT//vancomycin//mg//given
RESP//device_category//imv   PA//gcs_total   PA//rass   CRRT//crrt_mode_category//cvvhdf
HOSP_DX//ICD10CM//A41.9
```

`metadata/codes.parquet` — code registry: `code, description, parent_codes, concept_version, event_count, is_numeric_value, is_text_value` (counts \< 10 floored to 10 for PHI).

Notes: - **Timestamps**: clifpy loads each table converting UTC → site-local **once**, then `strip_tz` drops the tz (`replace_time_zone(None)`) — wall-clock preserved, no second conversion. - The ETL emit is **vectorized polars** (one frame per concept), not row-by-row. - The data table is **not** stamped with `prediction_id`/`prediction_dttm` — the prediction join and the leak filter live in the featurizer, keeping this table compact and reusable.

### ③ Count featurizer — `flair_baseline/featurize.py`

`count_features(events, task_df, label_col, vocab=None, n_chunks=8, exclude_prefixes=…)` → `(X_csr, prediction_ids, vocab)`.

```         
count[prediction_id, code] = #events of code in the same hospitalization_join_id
                             with  time < prediction_dttm        ← strict point-in-time
```

- **No leak**: events join predictions on `hospitalization_join_id`, then `time < prediction_dttm` (strict `<`). One feature row per `prediction_id`.
- **Sparse only**: result is a scipy CSR (`n_predictions × |vocab|`). A dense matrix would be hundreds of GB on the big tasks.
- **Scale**: codes + encounter blocks are factorized to int32 *before* the join (no strings in the \~100 M-row heavy path), and the join runs in `n_chunks` block-hash partitions so peak memory is \~1/`n_chunks` of the aggregation.
- **Vocabulary**: fit on the **train split only**; test predictions align to it (unseen codes dropped). `score` passes a saved vocab instead.
- **HOSP_DX excluded**: `feature_exclude_prefixes: [HOSP_DX]` — discharge diagnoses are post-hoc (assigned for the whole stay at coding time) and leak. They stay in `MEDS.parquet` for audit but never become features.

### ④ XGBoost — `flair_baseline/train.py`

`train_and_score(X, prediction_ids, vocab, task_df, label_col, …)`:

- Appends `age_at_admission` as a dense column after the count columns.
- `xgboost.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", scale_pos_weight=neg/pos)`.
- Fits on `split=="train"`, scores **every** row (so the report can bin by split).
- Writes `model.json` (booster), `vocab.json` (`{vocab, extras}`), and `preds.parquet`:

```         
prediction_id, hospitalization_id, hospitalization_join_id, split, <label_column>, y_prob, y_pred
```

### ⑤ Report — `flair_benchmark.report.build_report`

`build_report(preds, task_module, out, cohort_path=cohort, viz=…, site=…)` → 9 JSONs (+ `viz/` PNGs). Demographics (`sex/race/ethnicity/age`) flow in from `cohort.parquet` for fairness + table1.

**Two reporting levels — row-level vs encounter-level.** Most JSONs (`curves`, `temporal`, `threshold_analysis`, …) score at **prediction-row** granularity. For a continuous / per-window model (task1 daily, task2 daily, task4 hourly) each stay contributes many correlated rows, so row-level AUROC measures **discrimination only** — it overstates deployed utility and says nothing about alarm burden or how early the alert fires. `encounter.json` (`flair_benchmark/report/encounter.py`) adds the clinically meaningful **encounter (stay) level**:

- `counts` — stays / event stays / control stays.
- `at_clinical_threshold` — one-alarm-per-stay **sensitivity** (caught = an alarm at/before the first positive-label row) & **specificity**, **control-alarm fraction**, **alarms per patient-day** (raw + silenced), **number-needed-to-alert**, and the **lead-time** distribution (median/IQR).
- `lead_time_hist` — histogram of caught lead times (the TREWS Fig 2 analog).
- `operating_curve` — stay-sensitivity vs alarms-per-patient-day across thresholds: the deployment trade-off that replaces row-ROC, with the clinical threshold marked.

Gated to continuous binary tasks; one-per-stay tasks (3, 5) emit `{"reason":"one_per_stay_task"}` (row == stay there). HIPAA suppression (n \< 10) applies to event-stay counts and histogram bins.

> Lead-time semantics follow each task's **label horizon**: task4's `label_sepsis_6h` flags the 6 h before ASE onset, so lead time is the genuine pre-onset warning; task1/2's *daily* mortality/LTACH label fires on the event day itself, so most lead times collapse to ≈0 — there the encounter view is useful mainly for alarm burden + sensitivity, not lead time.
>
> Basis: JMIR 2026 (evaluation strategy / alarm fatigue), Henry 2022 *Nat Med* (TREWS lead-time figure), SepsisAI 2024 (false-alarms + silencing window).

------------------------------------------------------------------------

## 3. Output tree (per task)

```         
out/<task_name>/
  cohort.parquet              # one row per prediction_id (+ split, label, demographics)
  data/MEDS.parquet           # MEDS events: subject_id,time,code,numeric_value,text_value,
                              #   hospitalization_id, hospitalization_join_id
  metadata/codes.parquet      # code registry (description, version, counts, value flags)
  vocab.json                  # {vocab:[…codes…], extras:["age_at_admission"]}
  model.json                  # trained XGBoost booster
  preds.parquet               # prediction_id, …, split, label, y_prob, y_pred
  report/
    table1.json               # cohort characteristics by split & label
    tripod_ai.json            # TRIPOD-AI reporting skeleton
    fairness.json             # per-subgroup performance + parity gaps
    curves.json               # ROC (raw + fixed-grid) / PR (+prevalence) / calibration
    temporal.json             # discrimination binned over time (by_window / by_year)
    dca.json                  # decision-curve analysis (+ prevalence focus_range)
    threshold_analysis.json   # metrics at the clinical threshold
    encounter.json            # stay-level: sensitivity, alarms/patient-day, lead time,
                              #   operating curve (continuous tasks; stub for one-per-stay)
    scorecard.json            # FLAT cross-site headline row (poolable; see §9)
    viz/                       # only with --viz; stale PNGs cleared each run
      roc_pr.png  calibration.png  dca.png  threshold_analysis.png  fairness_auroc.png
      [auroc_by_year.png]                                      ← only on real-date sites
      [auroc_by_window.png  trajectory_by_outcome.png          ← one-per-window tasks only
       lead_time.png  encounter_operating_curve.png]           ← continuous tasks only
```

------------------------------------------------------------------------

## 4. What's used

### Feature domains — `flair_elf_config.yaml`

| domain | what | as feature? |
|------------------------|------------------------|------------------------|
| VITAL | all vital signs | ✅ |
| LAB | all labs | ✅ |
| RESP | respiratory-support settings | ✅ |
| PA | patient assessments — **GCS + RASS only** | ✅ |
| CRRT | CRRT presence (count \> 0 ⇒ on CRRT) | ✅ |
| MED_CON | continuous meds (vasopressors, sedatives, …) | ✅ |
| MED_INT | intermittent meds (antibiotics, …) | ✅ |
| HOSP_DX | hospital discharge diagnoses | ❌ extracted, **excluded** (post-hoc/leaky) |

`PATIENT` / `HOSP` load implicitly for subject + hospitalization mapping (not featurized).

### Rules

- **Split**: 75/25 on `hospitalization_join_id` (mimic); a block never spans splits.
- **No leak**: features use only events with `time < prediction_dttm` (strict).
- **Vocabulary**: fit on train split only.
- **HOSP_DX excluded**: post-hoc; see §2③.
- **Timestamps**: UTC → site-local once (clifpy), then tz stripped; no second conversion.

------------------------------------------------------------------------

## 5. Per-task breakdown (MIMIC-IV)

### task1 — ICU Daily In-Hospital Mortality (7 AM)

- `label_mortality` · outcome · one-per-window (daily 07:00 grid) · threshold 0.95.
- Cohort **198,125** rows (148,763 train / 49,362 test; 9,838 test positives).
- MEDS **94,137,739** events / 17,404 codes. Per-domain: VITAL 44.999M, LAB 17.584M, RESP 11.726M, PA 8.282M, MED_CON 6.224M, MED_INT 2.681M, CRRT 1.468M, HOSP_DX 1.174M.
- Features **198,125 × 433**, 14.66M nnz. **Test AUROC 0.790**. 9 PNGs (+ encounter lead_time / operating_curve). Encounter: 16,102 stays (2,135 event); at t=0.95 stay-sens 0.017, ctrl-alarm 0.001, NNA 3.58 — lead ≈0 (daily mortality label fires on the event day).

### task2 — ICU Daily LTACH Discharge (7 AM)

- `label_ltach` · outcome · one-per-window (daily 07:00) · threshold 0.5.
- Cohort **198,125** (148,763 / 49,362; 7,152 test pos). Same base ICU cohort + events as task1.
- Features **198,125 × 433**, 14.66M nnz. **Test AUROC 0.757**. 9 PNGs (+ encounter).

### task3 — Extubation failure within 24 h

- `label_extubation_failure_24h` · intervention · one-per-stay (episode) · threshold 0.15.
- Cohort **26,771** (20,077 / 6,694; 364 test pos).
- MEDS **57,832,570** events / 13,543 codes.
- Features **26,771 × 412**, 2.21M nnz. **Test AUROC 0.701**. 5 PNGs (one-per-stay → no window plots; `encounter.json` = `one_per_stay_task` stub, no lead-time/operating-curve PNGs).

### task4 — Sepsis (CDC ASE onset) within 6 h

- `label_sepsis_6h` · intervention · one-per-window (hourly grid, **hospital-wide**) · threshold 0.1.
- Cohort **14,343,501** (10,735,052 train / 3,608,449 test; 5,541 test pos — 0.15 % prevalence).
- MEDS **95,512,509** events / 18,635 codes.
- Features **14,343,501 × 425**, **931.6M nnz**. **Test AUROC 0.812**. 9 PNGs (+ encounter — the canonical TREWS-style case: `label_sepsis_6h` gives genuine pre-onset lead time).
- Scale: featurize \~1,283 s, train \~288 s, peak RSS \~12 GB (16 GB box) thanks to the chunked integerized featurizer.

### task5 — Unplanned ICU Readmission

- `label_icu_readmission` · outcome · one-per-stay (episode) · threshold 0.2.
- Cohort **67,250** (50,436 / 16,814; 1,562 test pos).
- MEDS **78,185,174** events / 17,985 codes.
- Features **67,250 × 429**. **Test AUROC 0.645**. 5 PNGs (one-per-stay → `encounter.json` stub).

### Results summary

| task                         | test AUROC | features | n_test (pos)      |
|------------------------------|-----------:|---------:|-------------------|
| task1 ICU daily mortality    |  **0.790** |      433 | 49,362 (9,838)    |
| task2 ICU daily LTACH        |  **0.757** |      433 | 49,362 (7,152)    |
| task3 extubation failure 24h |  **0.701** |      412 | 6,694 (364)       |
| task4 sepsis ABX 6h          |  **0.812** |      425 | 3,608,449 (5,541) |
| task5 ICU readmission        |  **0.645** |      429 | 16,814 (1,562)    |

Excluding HOSP_DX moved AUROC by ≤ 0.02 vs the leaky version — the post-hoc diagnoses were not real predictive signal, just leakage (and they had ballooned the vocab, e.g. task1 1,840 → 433).

------------------------------------------------------------------------

## 6. Report catalog

| file | shows |
|------------------------------------|------------------------------------|
| `table1.json` | cohort counts/characteristics by split & label |
| `tripod_ai.json` | TRIPOD-AI reporting skeleton |
| `fairness.json` | per-subgroup AUROC + parity gaps (subgroups n \< 10 dropped) |
| `curves.json` | ROC (raw + **fixed FPR grid** `grid_fpr`/`tpr_at_grid`, `auroc`+CI) / PR (`auprc`, `baseline`=prevalence, `auprc_lift`) / calibration (10 fixed bins + counts) |
| `temporal.json` | discrimination binned over time (`by_window`; `by_year` skipped on date-shifted sites) |
| `dca.json` | decision-curve analysis (prevalence-anchored grid + `focus_range`) |
| `threshold_analysis.json` | confusion-matrix metrics at the clinical threshold |
| `encounter.json` | **stay-level**: sensitivity/specificity, control-alarm fraction, alarms/patient-day (raw+silenced), number-needed-to-alert, lead-time dist (**fixed clinical bins**), operating curve (continuous tasks; `one_per_stay_task` stub for 3/5) |
| `scorecard.json` | **flat cross-site headline row** — discrimination/calibration/row-operating/encounter scalars (see §9) |

### Viz catalog — how to read each PNG

PNGs are **local sanity only** (the dashboard renders the JSON). Each is rendered *from the JSON dict*, proving the JSON alone suffices.

| png | what it plots | healthy ✅ / watch ⚠️ |
|---|---|---|
| `roc_pr.png` | ROC (titled AUROC) + PR with the **prevalence baseline** dashed line (titled AUPRC + lift) | ✅ ROC bowed to top-left. ⚠️ PR collapses toward the baseline on rare events — judge it by **lift** (AUPRC/prevalence), not raw precision |
| `calibration.png` | reliability points (10 bins, n≥10) + logistic recalibration; title = slope/intercept/O:E | ✅ points on the diagonal, slope≈1. ⚠️ points hugging y=0 + slope≪1 ⇒ **probabilities uncalibrated** (`scale_pos_weight` inflates risk) — a red note is drawn |
| `dca.png` | net benefit vs threshold, **zoomed to `focus_range`** (rare-event decision region, includes the clinical threshold) | ✅ model curve above treat-all & treat-none near the clinical threshold. ⚠️ below 0 ⇒ acting on the score loses to "treat none" |
| `threshold_analysis.png` | ROC with the clinical operating point + pAUC band | ✅ operating point high-left. ⚠️ point far right ⇒ high FPR (alarm fatigue) at that threshold |
| `auroc_by_window.png` | AUROC per in-stay window (x-label = `hour`/`day` from `window_unit`) | ✅ flat/stable across the stay. ⚠️ decay = model weaker later in stay |
| `trajectory_by_outcome.png` | mean predicted risk per window, positives vs negatives | ✅ positive line clearly above negative. ⚠️ lines overlapping ⇒ no separation |
| `auroc_by_year.png` | AUROC per admission year — **only on real-date sites** | skipped (no PNG) when dates are de-id-shifted (MIMIC → 2105-2214) or > 25 distinct years |
| `fairness_auroc.png` | AUROC per demographic subgroup (sex/race/ethnicity/age) | ✅ bars tightly clustered. ⚠️ wide spread = disparate performance (tiny subgroups n≥10 are noisy) |
| `lead_time.png` | lead-time distribution (first alarm → event) over **fixed clinical bins** `0-6h…>7d`; title = median | ✅ mass at useful pre-event lead. ⚠️ mass only at 0-6h ⇒ alarm fires too late to act |
| `encounter_operating_curve.png` | stay-sensitivity vs **alarms/patient-day** across thresholds (clinical point marked) | ✅ high sensitivity at low alarm burden (knee far left). ⚠️ sensitivity only reachable at many alarms/day ⇒ unworkable |

**Calibration caveat (rare-event tasks, esp. task4):** `scale_pos_weight` inflates predicted probabilities for the rare positive class — the model **ranks** well (AUROC) but is not calibrated (task4 ECE ≈ 0.17, slope ≪ 1, predicted ≫ observed). This corrupts the calibration plot, the PR precision, *and* DCA net benefit simultaneously. Highest-impact follow-up: Platt/isotonic-recalibrate on a held-out split (or drop `scale_pos_weight`) → all three become meaningful and cross-site comparable.

------------------------------------------------------------------------

## 7. Scale & memory

- Featurizer factorizes codes + blocks to int32 and runs the event→prediction join in `n_chunks` (default 8) block-hash partitions → peak ≈ 1/`n_chunks` of the aggregation. Even task4 (14.3M predictions, \~95M events, \~932M nnz) fits a 16 GB box (\~12 GB peak).
- Everything downstream of the ELF events is sparse CSR.
- `--reuse` skips the ETL (reads existing `cohort.parquet` + `data/MEDS.parquet`) — re-featurize / retrain / re-report in minutes (seconds for the small tasks; \~26 min for task4).
- Full cold sweep ≈ ETL (a few min per task; loads full CLIF tables once) + task4 featurize \~21 min.

------------------------------------------------------------------------

## 8. Correctness guarantees (and how to re-check)

| guarantee | check |
|------------------------------------|------------------------------------|
| no leak | every counted event has `time < prediction_dttm` (strict, in `featurize.py`); sum of CSR counts == #events before each `prediction_dttm` |
| split disjoint | `train` vs `test` `hospitalization_join_id` sets disjoint, block ratio = 0.7500 |
| vocab train-only | vocab built from `split=='train'` predictions only |
| HOSP_DX excluded | `HOSP_DX=0` in every `vocab.json`, but present in `data/MEDS.parquet` |
| timestamps | tz stripped after one UTC→local convert; all datetimes tz-naive |

------------------------------------------------------------------------

## 9. Standardized output & cross-site aggregation

FLAIR is federated: each site runs locally and ships **JSON** (PNGs never leave the site; the central
dashboard renders the JSON). For results to be *compared and pooled* across sites, the output must be
standardized — which drives three rules baked into the report:

**(1) The JSON is the canonical, poolable artifact; PNGs are local sanity.** Every PNG is rendered
*from* its JSON dict, so the JSON alone is sufficient.

**(2) Every curve sits on a FIXED, site-invariant grid with per-cell sufficient statistics**, so a
coordinator can meta-analyze without raw patient data:

| output | fixed grid | how a coordinator pools across sites |
|---|---|---|
| ROC | `grid_fpr` 0:1 @0.01 → `tpr_at_grid` | n-weighted mean of `tpr_at_grid` per FPR (+ CI) = pooled ROC |
| calibration | 10 equal-width prob bins + `bin_counts` | Σ observed, Σ expected per bin → pooled reliability + ECE |
| DCA | `0.001–0.05` + `0.01–0.99` thresholds | average net benefit per threshold (n-weighted) |
| `by_window` | windows `1..7+` | pooled AUROC per in-stay window |
| lead-time | **fixed clinical bins** `0-6h…>7d` + counts | Σ counts per bin → pooled lead-time distribution |
| operating curve | thresholds `0.02..0.98` | pooled stay-sensitivity vs alarms/patient-day |

PHI: any cell with n < 10 is suppressed (`"<10"`) and excluded from pools. Site-specific things that
**cannot** pool are guarded off: `by_year` is skipped on date-shifted sites (see §6).

**(3) `scorecard.json` — one flat headline row per (site, task)**, identical schema everywhere, built
by reading the other report dicts (never recomputed). A coordinator concatenates scorecards → a table
and forest plots. Schema:

``` json
{ "task","task_type","prediction_unit","site","n_test","prevalence",
  "discrimination": {"auroc","auroc_ci","auprc","auprc_lift"},
  "calibration":    {"slope","intercept","ece","o_e_ratio","brier"},
  "row_operating_at_clinical_threshold": {"threshold","sensitivity","specificity","ppv","npv"},
  "encounter": {"stay_sensitivity","stay_specificity","control_alarm_fraction",
                "alarms_per_patient_day_silenced","number_needed_to_alert",
                "median_lead_hours","n_event_stays"}   // null for one-per-stay tasks
}
```

Prevalence-awareness is built in for rare-event sites: AUPRC is reported as **lift** over prevalence,
PR carries the prevalence `baseline`, and DCA zooms to a prevalence-anchored `focus_range` — so a
0.15 %-prevalence site and a 20 %-prevalence site stay comparable.

------------------------------------------------------------------------

## 10. External-site validation

``` bash
uv run flair-baseline score --task task4 \
  --clif-config ../other_site.json \
  --model out/task4_sepsis_abx_6h/model.json \
  --vocab out/task4_sepsis_abx_6h/vocab.json --out out_external
```

Non-mimic site → all rows `test`; cohort featurized against the saved vocabulary and scored with the saved model; writes `out_external/<task>/preds.parquet`. Because ELF standardizes the code vocabulary across CLIF sites, the MIMIC-trained model applies without code remapping.