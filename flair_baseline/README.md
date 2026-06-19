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

## Two roles

| role | who | does | publishes |
|---|---|---|---|
| **Model owner** | the FLAIR baseline team (trains on MIMIC) | `train` → fits the 5 models | uploads `mimic_baseline_models/` to the FLAIR website |
| **CLIF site** | each participating ICU site | downloads the models, `infer` on local data | uploads its `*_non_phi_for_upload/` results back |

Only two things ever move: the **models folder** (owner → site) and the **non-PHI
results folder** (site → FLAIR). Patient data never leaves a site.

### Shared setup (both roles, once)

```bash
git clone <baseline-repo> flair_baseline && cd flair_baseline
git clone <flair-repo> flair          # FLAIR library, cloned in beside the baseline
uv sync                               # resolves flair-benchmark from ./flair (editable) + xgboost

cp config/clif_config.template.json config/clif_config.json
# edit config/clif_config.json: set "site" (prefixes every output folder) and "data_directory"
```

> Developing inside the FLAIR repo? `ln -s .. flair` instead of cloning, then `uv sync`.

---

### Role A — Model owner (FLAIR baseline, on MIMIC)

1. Set `"site": "mimic"` and the MIMIC `data_directory` in `config/clif_config.json`.
2. Train all 5 tasks:
   ```bash
   uv run flair-baseline train --clif-config config/clif_config.json --out . --viz
   ```
3. Three folders appear: `mimic_baseline_phi/` (local), `mimic_baseline_non_phi_for_upload/`
   (MIMIC's own results), and **`mimic_baseline_models/`** — the per-task `model.json` + `vocab.json`.
4. **Publish `mimic_baseline_models/`** to the FLAIR website as the downloadable model bundle.
   That folder is all a CLIF site needs.

---

### Role B — CLIF site (participating ICU)

1. Finish the shared setup above; set your own `"site"` (e.g. `"rush"`) and `data_directory`.
2. **Download the models folder** from the FLAIR website and drop it into the baseline folder,
   e.g. `flair_baseline/mimic_baseline_models/` (unzip here; keep the per-task subfolders).
3. Run inference — scores the downloaded models on a deterministic 25% holdout of your data:
   ```bash
   uv run flair-baseline infer \
     --models-dir mimic_baseline_models \
     --clif-config config/clif_config.json \
     --out . --viz
   # one task only: add --task task1
   ```
4. Three folders appear, prefixed with **your** site name:
   - `rush_baseline_phi/` — cohort, MEDS, preds → **stays on your machine** (PHI, never upload).
   - `rush_baseline_models/` — the models you ran (already public).
   - **`rush_baseline_non_phi_for_upload/`** — codes registry, Table 1, report JSONs + PNGs.
5. **Upload only `rush_baseline_non_phi_for_upload/`** to the FLAIR website (or the provided
   secure Box link). Nothing else leaves your site.

> Quick check before upload: the folder holds only `codes.parquet`, `table1.json`, and
> `report/` — no `cohort.parquet`, `MEDS.parquet`, or `preds.parquet`. All cell counts < 10
> are already suppressed (`"<10"`).

## Results (MIMIC-IV, 75/25 join-id split)

**Report AUROC** is the headline metric in `discrimination.json` — the one that gets
uploaded. It is mode-matched (episodic = one row/stay; peak = stay-peak risk; landmark =
the lead-0 landmark, i.e. each stay's last window). The **row-level** column is the raw
per-prediction-row AUROC for context — for peak/landmark tasks it differs because a stay
contributes many correlated rows, so it is not the deployed metric.

| task | mode | report AUROC (test) | n_stays | row-level |
|---|---|---:|---:|---:|
| task1 ICU daily mortality | landmark (lead-0) | 0.864 [0.856, 0.872] | 18,195 | 0.790 |
| task2 ICU daily LTACH | landmark (lead-0) | 0.789 [0.775, 0.803] | 18,195 | 0.752 |
| task3 extubation failure 24h | episodic | 0.696 [0.665, 0.719] | 6,693 | 0.696 |
| task4 sepsis ABX 6h | peak | 0.693 [0.678, 0.709] | 19,943 | 0.815 |
| task5 ICU readmission | episodic | 0.641 [0.626, 0.658] | 16,806 | 0.641 |

> Landmark tasks report 12 per-lead-time AUROCs (see `leadtime.json` / the `landmarks[]`
> array); the table shows lead-0 as a single summary. task4 peak (0.69) vs its row-level
> (0.81) is the peak-trap in action — per-window pooling overstates deployed skill.

## Scale notes

- The featurizer factorizes codes + encounter blocks to int32 and runs the
  event→prediction join in `n_chunks` block-hash partitions (default 8), so peak
  memory is ~1/n_chunks of the aggregation. This keeps even task4 — **14.3M hourly
  predictions, ~95M events, ~930M feature non-zeros** — within a 16 GB machine
  (peak ≈ 12 GB, ~21 min featurize + ~5 min train). Features are sparse CSR throughout.
- ETL loads full CLIF tables per task (~few min each); FE-meds writes
  `MEDS.parquet` once per task and the featurizer scans it lazily.
