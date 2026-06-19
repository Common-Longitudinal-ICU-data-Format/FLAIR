# flair-baseline

XGBoost count-feature baseline for the FLAIR ICU benchmark (all 5 tasks).

A **self-contained sub-project**: it carries its own config and pins the FLAIR
library, which is cloned in beside it. Train once at a source site (MIMIC), ship
the **models folder** to other sites, and re-run there — the code travels to the
data, not the other way around.

## What it does

For each task:
1. `task.build()` → cohort (one row per `prediction_id`, with `split`).
   On MIMIC the split is **75/25 on `hospitalization_join_id`** (an encounter never
   spans train and test). At an external site `infer` re-applies the same deterministic
   75/25 block split, so the model is evaluated on a **25% holdout** of that site's data.
2. **FE-meds** (`flair_benchmark.features.fe_meds`) runs the vendored CLIF→MEDS ETL
   over the cohort and writes `MEDS.parquet` (events) + `codes.parquet` (code registry,
   counts floored to the suppression threshold).
3. **Count featurizer** — counts each ELF code with `time < prediction_dttm`
   (strict point-in-time, no leakage). Vocabulary is fit on the train split only.
4. **XGBoost** fits on train (or loads a shipped model), scores all rows → `preds.parquet`.
5. **FLAIR report** bundle, scored by the per-task mode below.

### Report bundle (per task)

`build_report` is keyed on the unit of prediction. Each pillar is one JSON of
metrics + raw curve arrays + 95% bootstrap CIs, evaluated at the task's fixed
clinical threshold; `--viz` also renders sanity-check PNGs (never uploaded).

| mode | tasks | pillars |
|---|---|---|
| **episodic** | task3, task5 | `discrimination`, `calibration`, `dca`, `fairness` |
| **landmark** | task1, task2 | the four pillars **per lead-time landmark** + `leadtime` (per-window sensitivity, median lead-time, risk trajectory by outcome) |
| **peak** | task4 | `discrimination` + `fairness` (+ NNE); calibration/DCA omitted (peak-calibration trap) |

Feature domains (`flair_elf_config.yaml`): all vitals, all labs, respiratory
support, patient assessments (GCS + RASS), CRRT (presence), continuous +
intermittent medications.

> The ELF config is the **single source of truth**: the domains listed there are exactly
> what gets extracted into MEDS *and* exactly what the model can use as features (no separate
> exclude list). `HOSP_DX` (discharge diagnoses, timestamped at discharge → leaky) is intentionally
> omitted. To include/exclude a domain, add/remove it there.

## Output: three site-prefixed folders

Every run writes under one `--out` root (default `.`), into three folders prefixed
with the `site` from your config. Only the **non-PHI** folder is meant to leave the site.

```
<site>_baseline_phi/<task>/                  cohort.parquet, data/MEDS.parquet, preds.parquet   ← stays local (PHI)
<site>_baseline_non_phi_for_upload/<task>/   codes.parquet, table1.json, report/*.json, report/viz/*.png   ← upload this
<site>_baseline_models/<task>/               model.json, vocab.json                            ← ship to other sites
```

Model folders keep their **training-site** prefix; PHI + non-PHI folders carry the
**running-site** prefix.

## Setup (at any site)

```bash
git clone <baseline-repo> flair_baseline && cd flair_baseline
git clone <flair-repo> flair          # FLAIR library, cloned in beside the baseline
uv sync                               # resolves flair-benchmark from ./flair (editable) + xgboost

cp config/clif_config.template.json config/clif_config.json
# edit config/clif_config.json: set "site" (prefixes every output folder) and "data_directory"
```

> Developing inside the FLAIR repo? `ln -s .. flair` instead of cloning, then `uv sync`.

## Train (source site, e.g. MIMIC)

```bash
# all 5 tasks
uv run flair-baseline train --clif-config config/clif_config.json --out . --viz

# a single task
uv run flair-baseline train --task task1 --clif-config config/clif_config.json --out .
```

Produces `mimic_baseline_models/` (the artifact to ship), plus `mimic_baseline_phi/`
and `mimic_baseline_non_phi_for_upload/`.

## Infer (new site — only the models folder travels)

Copy a training site's `*_baseline_models/` to the new site, then:

```bash
uv run flair-baseline infer \
  --models-dir mimic_baseline_models \
  --clif-config config/clif_config.json \
  --out . --viz
# single task: add --task task1
```

This rebuilds the cohort + features from the **local** data, scores the shipped
model on a deterministic 25% test split, and writes `<site>_baseline_*` folders.
Send back only `<site>_baseline_non_phi_for_upload/`.

## Results (MIMIC-IV, 75/25 join-id split)

| task | n_test | test AUROC | mode |
|---|---:|---:|---|
| task1 ICU daily mortality | 49,362 | 0.788 | landmark |
| task2 ICU daily LTACH | 49,362 | 0.757 | landmark |
| task3 extubation failure 24h | 6,694 | 0.720 | episodic |
| task4 sepsis ABX 6h | 3,608,449 | 0.812 | peak |
| task5 ICU readmission | 16,814 | 0.649 | episodic |

## Scale notes

- The featurizer factorizes codes + encounter blocks to int32 and runs the
  event→prediction join in `n_chunks` block-hash partitions (default 8), so peak
  memory is ~1/n_chunks of the aggregation. This keeps even task4 — **14.3M hourly
  predictions, ~95M events, ~930M feature non-zeros** — within a 16 GB machine
  (peak ≈ 12 GB, ~21 min featurize + ~5 min train). Features are sparse CSR throughout.
- ETL loads full CLIF tables per task (~few min each); FE-meds writes
  `MEDS.parquet` once per task and the featurizer scans it lazily.
