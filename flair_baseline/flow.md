# flair-baseline — end-to-end flow

XGBoost **count-feature** baseline for all 5 FLAIR ICU tasks, trained on MIMIC-IV with a leak-free **75/25 split on `hospitalization_join_id`**. Features are counts of every ELF event code that occurs **strictly before** each prediction's `prediction_dttm` (point-in-time, no leakage). This is a **self-contained sub-project**: the FLAIR library is cloned in beside it (`./flair`) and pinned as an editable path dep; the baseline carries its own config (`config/`). Train once at a source site, ship the **models folder** to other sites, re-run there — the code travels to the data.

Every run partitions its artifacts into three **site-prefixed** folders (see §3); only the non-PHI folder leaves the site.

```         
                         ┌──────────────────── flair-baseline train | infer ────────────────────┐
 task.build()            │                                                                        │
 (flair_benchmark) ──► cohort ──► FE-meds ──► MEDS.parquet ──► count featurizer ──► XGBoost ──► preds ──► report (+viz)
                       (split)   (CLIF→MEDS)  + codes.parquet   (sparse, leak-free)  (model)   .parquet   JSON / PNG
```

| stage | code | output (folder) |
|------------------------|------------------------|------------------------|
| ① cohort | `flair_benchmark/tasks/taskN_*.py` + `_split.py` | `cohort.parquet` (PHI) |
| ② FE-meds | `flair_benchmark/features/fe_meds.py` + `flair_benchmark/meds_etl/` | `MEDS.parquet` (PHI), `codes.parquet` (non-PHI) |
| ③ featurize | `flair_baseline/featurize.py` | in-memory sparse CSR |
| ④ train/score | `flair_baseline/train.py` | `model.json`, `vocab.json` (models), `preds.parquet` (PHI) |
| ⑤ report | `flair_benchmark.report.build_report` | `report/*.json` (+ `viz/*.png`) (non-PHI) |

Folder routing is owned by `flair_baseline/layout.py` (`TaskPaths`, `report_mode`).

------------------------------------------------------------------------

## 0. Setup (uv, self-contained)

``` bash
git clone <baseline-repo> flair_baseline && cd flair_baseline
git clone <flair-repo> flair          # FLAIR lib cloned in beside the baseline
uv sync                               # resolves flair-benchmark from ./flair (editable) + xgboost + matplotlib

cp config/clif_config.template.json config/clif_config.json   # set "site" + "data_directory"
```

`pyproject.toml` declares `flair-benchmark` as a path source (`{ path = "flair", editable = true }`). Developing inside the FLAIR repo? `ln -s .. flair` instead of cloning. Python 3.12.

------------------------------------------------------------------------

## 1. Commands

Two subcommands. `train` fits models at a source site; `infer` scores a shipped model at a new site. Both emit the three site-prefixed folders (§3); the `site` comes from the clif config and prefixes every folder.

### `flair-baseline train`

```         
flair-baseline train [--clif-config config/clif_config.json] [--elf-config flair_elf_config.yaml]
                     [--out .] [--task NAME] [--train-end YYYY-MM-DD] [--test-start YYYY-MM-DD]
                     [--report/--no-report] [--viz/--no-viz] [--reuse/--no-reuse]
```

| flag | default | meaning |
|------------------------|------------------------|------------------------|
| `--clif-config` | `config/clif_config.template.json` | `clif_config.json` (path, filetype, timezone, `site`). `site=mimic` ⇒ 75/25 split |
| `--elf-config` | `flair_elf_config.yaml` | event concepts to extract (= the feature set) |
| `--out` | `.` | root for the three `<site>_baseline_*` folders |
| `--task` | all 5 | task name or prefix (`task1` → `task1_icu_daily_mortality`) |
| `--train-end` / `--test-start` | none | date cutoff for non-mimic sites (omit on mimic) |
| `--report` / `--no-report` | report | build the FLAIR report bundle (mode per task; see §2⑤) |
| `--viz` / `--no-viz` | no-viz | also render sanity-check PNGs into `report/viz/` |
| `--reuse` / `--no-reuse` | no-reuse | reuse existing `cohort.parquet` + `MEDS.parquet` (skip ETL) |

``` bash
# all 5 tasks on MIMIC, with report + PNGs
uv run flair-baseline train --clif-config config/clif_config.json --out . --viz

# one task
uv run flair-baseline train --task task4 --clif-config config/clif_config.json --out .
```

### `flair-baseline infer` (new site — only the models folder travels)

```         
flair-baseline infer --models-dir <SITE>_baseline_models [--clif-config config/clif_config.json]
                     [--elf-config …] [--out .] [--task NAME] [--report/--no-report] [--viz/--no-viz]
```

Copy a training site's `*_baseline_models/` to the new site. `infer` rebuilds the cohort + features from the **local** data, re-applies the deterministic **75/25 block split**, and scores the shipped model on the resulting **25% test** set — then writes this site's three folders. Send back only `<site>_baseline_non_phi_for_upload/`. Because ELF standardizes the code vocabulary across CLIF sites, the model applies without code remapping.

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
- **Split** (`flair_benchmark/_split.py`, `assign_split`): on MIMIC the split is decided at the **block grain** — 75 % of distinct `hospitalization_join_id` → `train`, 25 % → `test`, broadcast to all rows. A block never spans both splits (no split leakage). `infer` re-applies this same deterministic 75/25 block split at any site (ordered by `prediction_dttm`) so external evaluation runs on a **25% holdout**.
- `prediction_id = {join_id}-{hospitalization_id}-{patient_id}-{window_number}`.

Written to `<site>_baseline_phi/<task>/cohort.parquet`.

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

`build_report(preds, task_module, out, cohort_path=cohort, viz=…, site=…, mode=…)`. Demographics (`sex/race/ethnicity/age`) flow in from `cohort.parquet` for fairness. Each pillar is one JSON of metrics + raw curve arrays + **95% bootstrap CIs**, evaluated at the task's fixed clinical threshold. `--viz` renders PNGs from those same dicts (local sanity only).

The **mode** (chosen per task by `flair_baseline/layout.py:report_mode`) sets how a stay's many rows are scored:

| mode | tasks | files |
|---|---|---|
| **episodic** | task3, task5 | `discrimination.json`, `calibration.json`, `dca.json`, `fairness.json` |
| **landmark** | task1, task2 | the four pillars **per lead-time landmark** + `leadtime.json` |
| **peak** | task4 | `discrimination.json` + `fairness.json` (+ NNE); calibration/DCA omitted (peak-calibration trap) |

- **episodic** — one prediction per stay; the four pillars straight up.
- **landmark** — scores at 12 risk-set landmarks counting back from each stay's last window, so discrimination/calibration/DCA/fairness are reported *per lead-time*. `leadtime.json` adds per-window detection sensitivity, median lead-time (+CI), and the **predicted-risk trajectory by outcome** (mean risk for ever-positive vs negative stays at each landmark).
- **peak** — collapses each stay to its peak risk (a screening view); calibration/DCA are dropped because the max-of-many-windows is inflated and not honestly calibratable.

Why split this way: for a continuous / per-window model each stay contributes many correlated rows, so plain row-level metrics overstate deployed utility. Landmark (per lead-time) and peak (screening) are the two honest reductions; episode tasks have one row per stay already.

------------------------------------------------------------------------

## 3. Output tree (three site-prefixed folders)

`<site>` is the `site` field from the clif config (slugified). Only the non-PHI folder
is meant to leave the site; models are what `train` ships to other sites.

```         
<out>/
  <site>_baseline_phi/<task_name>/            ← stays local (PHI)
    cohort.parquet            # one row per prediction_id (+ split, label, demographics)
    data/MEDS.parquet         # MEDS events: subject_id,time,code,numeric/text_value, hosp ids
    preds.parquet             # prediction_id, …, split, label, y_prob, y_pred

  <site>_baseline_non_phi_for_upload/<task_name>/   ← upload this
    codes.parquet             # code registry (description, version, counts<10 floored, value flags)
    table1.json               # cohort characteristics by split & label
    report/
      discrimination.json     # AUROC/AUPRC + ROC/PR curves + operating point (+CIs)
      calibration.json        # reliability bins, ECE, Brier, slope/intercept  (episodic/landmark)
      dca.json                # decision-curve net benefit                     (episodic/landmark)
      fairness.json           # per-subgroup metrics + parity gaps
      leadtime.json           # per-window sensitivity, median lead-time, risk trajectory (landmark)
      viz/                    # only with --viz; PNGs rendered from the JSONs (local sanity)

  <site>_baseline_models/<task_name>/         ← ship to other sites
    model.json                # trained XGBoost booster
    vocab.json                # {vocab:[…codes…], extras:["age_at_admission"]}
```

Landmark tasks (1, 2) wrap each pillar as `{metadata, landmarks:[…]}` (one entry per
lead-time) and add `leadtime.json`. Peak task (4) omits `calibration.json`/`dca.json`.

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

| file | shows | modes |
|------------------------------------|------------------------------------|---|
| `discrimination.json` | AUROC/AUPRC (+CIs), raw ROC/PR curves, PR `baseline`=prevalence + `auprc_lift`, operating point at the clinical threshold (direction-aware action block; NNE in peak) | all |
| `calibration.json` | reliability bins, ECE, Brier, logistic slope/intercept, O:E ratio | episodic, landmark |
| `dca.json` | decision-curve net benefit vs threshold (+ clinical operating point; complement transform for `clinical_direction: below`) | episodic, landmark |
| `fairness.json` | per-subgroup metrics + parity gaps at the operating threshold (pinned subgroup set; subgroups n \< 10 dropped) | all |
| `leadtime.json` | per-window detection sensitivity (+CI), median lead-time (+CI), `trajectory` = predicted risk by eventual outcome | landmark |

Landmark files wrap each pillar as `{metadata, landmarks:[…]}` (one entry per lead-time).

### Viz catalog (only with `--viz`; local sanity, rendered from the JSON)

| png | mode | what it plots |
|---|---|---|
| `roc_pr.png` | all | ROC (AUROC) + PR with the prevalence baseline (AUPRC + lift) |
| `calibration.png` | episodic | reliability points + logistic recalibration; title = slope/intercept/O:E |
| `dca.png` | episodic | net benefit vs threshold, model vs treat-all/treat-none |
| `fairness_auroc.png` | all | AUROC per demographic subgroup (sex/race/ethnicity/age) |
| `landmark_skill.png`, `landmark_sensitivity.png`, `landmark_calibration.png` | landmark | metric vs windows-before-end, per-point `n` annotated |
| `landmark_risk_trajectory.png` | landmark | mean predicted risk vs windows-before-end, positive (label=1) vs negative (label=0) |

**Calibration caveat (rare-event tasks, esp. task4):** `scale_pos_weight` inflates predicted probabilities for the rare positive class — the model **ranks** well (AUROC) but is not calibrated (slope ≪ 1, predicted ≫ observed). This is exactly why task4 runs in **peak** mode (calibration/DCA omitted). For tasks that do report calibration, recalibrate (Platt/isotonic) on a held-out split → slope/ECE/DCA become meaningful and cross-site comparable.

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

FLAIR is federated: each site runs locally and shares only `<site>_baseline_non_phi_for_upload/`
(the report JSONs + Table 1 + codes registry). PHI (cohort, MEDS, preds) and PNGs never leave the
site. The report is built to be *pooled* without raw patient data:

**(1) JSON is the canonical, poolable artifact; PNGs are local sanity.** Every PNG is rendered
*from* its JSON dict, so the JSON alone suffices.

**(2) Each pillar carries raw curve arrays + per-cell counts + 95% bootstrap CIs**, so a coordinator
can meta-analyze: n-weighted ROC/PR curves, Σ observed/expected per calibration bin → pooled
reliability + ECE, n-weighted DCA net benefit, and (landmark) per-lead-time sensitivity + risk
trajectory. Prevalence-awareness is built in — AUPRC reported as **lift** over prevalence and PR
carries the prevalence `baseline` — so a 0.15%-prevalence site and a 20%-prevalence site stay
comparable.

**(3) PHI suppression:** any cell with n < 10 is suppressed (`"<10"`) and excluded from pools;
codes.parquet counts are floored at the same threshold.

Each JSON's `metadata` block stamps `task`, `task_script_sha256` (same hash ⇒ identical label logic),
`flair_version`, `report_mode`, `split`, `n_stays`, `prevalence`, and `evaled_at` (the operating
threshold) — enough for a coordinator to align and pool sites.

------------------------------------------------------------------------

## 10. External-site validation (`infer`)

``` bash
# copy a training site's models folder to this site, then:
uv run flair-baseline infer --task task4 \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json --out . --viz
```

`infer` rebuilds the cohort + features from the **local** data, re-applies the deterministic 75/25
block split, featurizes against the **saved vocabulary**, and scores the saved model on the resulting
**25% test** set — then writes this site's three folders. Because ELF standardizes the code vocabulary
across CLIF sites, the MIMIC-trained model applies without code remapping. Share back only
`<site>_baseline_non_phi_for_upload/`.