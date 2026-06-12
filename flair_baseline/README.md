# flair-baseline

XGBoost count-feature baseline for the FLAIR ICU benchmark (all 5 tasks).

## What it does

For each task:
1. `task.build()` → cohort (one row per `prediction_id`, with `split`).
   On MIMIC the split is **75/25 on `hospitalization_join_id`** (an encounter never
   spans train and test). Other sites get no split (full-data test).
2. **FE-meds** (`flair_benchmark.features.fe_meds`) runs the vendored CLIF→MEDS ETL
   over the cohort and writes `data/MEDS.parquet` (events) + `metadata/codes.parquet`.
3. **Count featurizer** — counts each ELF code with `time < prediction_dttm`
   (strict point-in-time, no leakage). Vocabulary is fit on the train split only.
4. **XGBoost** fits on train, scores all rows → `preds.parquet`.
5. **FLAIR report** bundle (9 JSONs) is generated per task. Row-level JSONs (ROC/PR, calibration,
   temporal, threshold) measure **discrimination only**; for continuous tasks `encounter.json` adds the
   **stay-level** operating view — one-alarm-per-stay sensitivity, control-alarm fraction,
   alarms/patient-day, number-needed-to-alert, lead-time distribution + operating curve (TREWS-style).
   One-per-stay tasks (3, 5) get a `one_per_stay_task` stub. Outputs are **standardized for cross-site
   pooling**: every curve on a fixed grid with per-cell counts, plus a flat `scorecard.json` headline
   row per (site, task) for tables/forest plots. See `flow.md` §6 (viz catalog) and §9 (aggregation).

Feature domains (`flair_elf_config.yaml`): all vitals, all labs, respiratory
support, patient assessments (GCS + RASS), CRRT (presence), continuous +
intermittent medications.

> The ELF config is the **single source of truth**: the domains listed there are exactly
> what gets extracted into MEDS *and* exactly what the model can use as features (no separate
> exclude list). `HOSP_DX` (discharge diagnoses, timestamped at discharge → leaky) is intentionally
> omitted, so it is neither extracted nor featured. To include/exclude a domain, add/remove it there.

## Install (uv, local)

```bash
cd flair_baseline
uv sync          # installs flair-benchmark from ../ (editable) + xgboost
```

## Run

```bash
# all 5 tasks on MIMIC
uv run flair-baseline run \
  --clif-config ../clif_config_mimic.json \
  --elf-config flair_elf_config.yaml \
  --out out

# a single task
uv run flair-baseline run --task task1 --clif-config ../clif_config_mimic.json --out out
```

Outputs per task under `out/<task_name>/`:
`cohort.parquet`, `data/MEDS.parquet`, `metadata/codes.parquet`, `model.json`,
`vocab.json`, `preds.parquet`, `report/`.

## Results (MIMIC-IV, 75/25 join-id split)

| task | n_test | test AUROC |
|---|---:|---:|
| task1 ICU daily mortality | 49,362 | 0.788 |
| task2 ICU daily LTACH | 49,362 | 0.757 |
| task3 extubation failure 24h | 6,694 | 0.720 |
| task4 sepsis ABX 6h | 3,608,449 | 0.812 |
| task5 ICU readmission | 16,814 | 0.649 |

## Scale notes

- The featurizer factorizes codes + encounter blocks to int32 and runs the
  event→prediction join in `n_chunks` block-hash partitions (default 8), so peak
  memory is ~1/n_chunks of the aggregation. This keeps even task4 — **14.3M hourly
  predictions, ~95M events, ~930M feature non-zeros** — within a 16 GB machine
  (peak ≈ 12 GB, ~21 min featurize + ~5 min train). Raise `n_chunks` if memory is
  tighter. Features are sparse CSR throughout (a dense 14.3M×1k matrix would be ~60 GB).
- ETL loads full CLIF tables per task (~few min each); FE-meds writes
  `data/MEDS.parquet` once per task and the featurizer scans it lazily.

## External-site validation

```bash
uv run flair-baseline score --task task1 \
  --clif-config ../other_site_config.json \
  --model out/task1_icu_daily_mortality/model.json \
  --vocab out/task1_icu_daily_mortality/vocab.json
```
