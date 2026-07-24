# The reference baseline (flair-baseline)

[← back to FLAIR 101](../101.md)

`flair-baseline` is an **optional** reference model: an XGBoost head over count features.
It implements the full model-to-data loop end-to-end, so it's the easiest way to see a
report bundle without writing your own model — and it's what ships frozen to sites for
external validation.

For each task it runs: `task.build()` → Table 1 → CLIF→MEDS ETL → count featurizer →
XGBoost (fit or score) → report bundle.

## Two subcommands

**Train** at a source site (MIMIC), producing models you can ship:

```bash
flair-baseline train \
  --clif-config clif_config.json \
  --elf-config flair_elf_config.yaml \
  --out . \
  [--task icu_daily_mortality]   # default: all 4 tasks
```

On MIMIC the split is **75/25 on `hospitalization_join_id`** (an encounter never spans
train and test).

**Infer** at another site, scoring shipped models on local data:

```bash
flair-baseline infer \
  --models-dir mimic_baseline_models \
  --clif-config site_clif_config.json \
  --elf-config flair_elf_config.yaml \
  --out .
```

`infer` re-applies the same deterministic 75/25 block split, so the model is evaluated on a
**25% holdout** of *that* site's own data. The report mode per task is chosen automatically
(continuous for the daily ICU tasks, episodic for the one-per-stay tasks).

## Output: three site-prefixed folders

Every run writes under `--out` into three folders prefixed with `site` from the config.
**Only the non-PHI folder is meant to leave the site.**

```
<site>_baseline_phi/<task>/                cohort.parquet, data/MEDS.parquet, preds.parquet   ← stays local (PHI)
<site>_baseline_non_phi_for_upload/<task>/ codes.parquet, table1.json, report/*.json (+viz)   ← upload this
<site>_baseline_models/<task>/             model.json, vocab.json                             ← ship to other sites
```

Model folders keep their **training-site** prefix; the PHI and non-PHI folders carry the
**running-site** prefix.

## flair_elf_config.yaml — the feature source of truth

This YAML lists the feature **domains** extracted into MEDS, and is the single source of
truth for what the model may use (no separate exclude list):

```
VITAL, LAB, RESP, PA (GCS + RASS), CRRT (presence), MED_CON, MED_INT
```

`HOSP_DX` (discharge diagnoses) is **intentionally omitted** — it is timestamped at
discharge and would leak into real-time predictions. To include or exclude a domain, edit
this file.
