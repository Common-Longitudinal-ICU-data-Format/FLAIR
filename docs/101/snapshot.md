# Snapshots — the same cohort for every model you test

A benchmark only compares models fairly if every model is scored on the **same cohort**:
the same stitched encounters, rows, labels and train/test split. A **snapshot** freezes that
at a site and lets every later run prove it rebuilt the same data.

## What a snapshot is

One folder per site per FLAIR version, created by `flair load-task`:

```
<site>_flair_v<version>/          e.g. mimic_flair_v0.0.1/
  encounter_index.parquet         frozen encounter IDs: hospitalization_id → hospitalization_join_id, patient_id
  cohort_scores.parquet           SOFA / Charlson / Elixhauser per hospitalization (reused by Table 1)
  hashes.json                     reference hashes — the audit record
```

`hashes.json` holds, for every task built on the snapshot, four reference hashes:

| hash | what it covers |
|---|---|
| `task_script_sha256` | the task's code, its META policy, and all of `constants.py` (benchmark window, stitch gap, …) |
| `cohort_sha256` | every cohort row: `hospitalization_join_id, prediction_dttm, feature_cutoff_dttm, label, split` |
| `train_cohort_sha256` | the same, train arm only |
| `test_cohort_sha256` | the same, test arm only |

plus `encounter_index_sha256`, the hash of the frozen IDs themselves.

**The task cohorts are not stored in the snapshot.** Each run rebuilds them from your CLIF
data on the frozen IDs, writes them to `<out>/<site>_flair_phi/<task>/cohort.parquet` as
usual, and the hashes prove they're the same as before.

## Content, not machine

Every hash is computed from data values: rows sorted, values cast to strings, code with line
endings normalised. Nothing depends on file modification times, paths, operating system or
the machine. A snapshot copied to a new server, or restored from a backup, audits exactly
as it did where it was made.

## The process

**First run — no `--snapshot`.** Everything is regenerated from CLIF:

```bash
flair load-task icu_readmission --clif-config clif_config.json --out output
# ...
# Recorded reference hashes → output/<site>_flair_v0.0.1/hashes.json
# Snapshot → output/<site>_flair_v0.0.1  (keep it; reuse with --snapshot ...)
```

This stitches encounters, writes `encounter_index.parquet`, builds the cohort and records
its four hashes as the reference. Run the other tasks the same way, but **with `--snapshot`**
pointing at this folder so they share the same IDs (see below). A run without `--snapshot`
**replaces** an existing `<out>/<site>_flair_v<version>/`, since its IDs would no longer match.

**Every later run — with `--snapshot`.**

```bash
flair load-task icu_readmission --clif-config clif_config.json --out output \
  --snapshot output/<site>_flair_v0.0.1
# Using snapshot output/<site>_flair_v0.0.1 (frozen encounter IDs)
#   match     task_script_sha256
#   match     cohort_sha256
#   match     train_cohort_sha256
#   match     test_cohort_sha256
```

- The stored IDs are used as they are. A `--snapshot` run never re-stitches; a folder
  without `encounter_index.parquet` is refused.
- The IDs file is re-hashed and checked against `encounter_index_sha256`. A warning is
  printed if it was altered.
- The cohort is rebuilt and its four hashes are compared to the reference. A task that has
  no reference yet (its first build on this snapshot) records one.
- The reference is **never overwritten**. It is the record of the original cohort.
- `flair fe-meds --snapshot <same folder>` uses the same IDs, so a feature pull expands each
  encounter to the same member hospitalizations.

The audit result is stamped into the uploaded `table1.json` under `metadata.snapshot`:

```json
"snapshot": {
  "name": "mimic_flair_v0.0.1",
  "encounter_index_sha256": "...",
  "reference": {"task_script_sha256": "...", "cohort_sha256": "...", ...},
  "match": {"task_script_sha256": true, "cohort_sha256": true, ...}
}
```

`match` is `null` on the run that recorded the reference. `build-report --cohort` also
stamps the scored cohort's `cohort_sha256` / `test_cohort_sha256` into the report audit, so
each report can be tied back to the snapshot.

## Reading a mismatch

Nothing is blocked locally. A mismatch is printed and recorded, and **flagged at the
leaderboard**, which compares hashes across a site's submissions.

| mismatch | usual cause |
|---|---|
| `task_script_sha256` | different FLAIR code or constants than when the snapshot was made (edited task, other checkout) |
| `cohort_sha256` / train / test, task hash matches | the CLIF data changed (re-ETL, new rows, corrected values) |
| encounter IDs warning | `encounter_index.parquet` was edited or replaced |

## Storing the snapshot

> [!IMPORTANT]
> **Data stewards must keep the snapshot folder** (`<site>_flair_v<version>/`) for as long
> as models are being tested against that FLAIR version, and pass it with `--snapshot` on
> every later run. Losing it means the next run creates new IDs and new reference hashes,
> and results stop being comparable with earlier models.

- The snapshot contains patient-level identifiers (`encounter_index.parquet`). Store and
  back it up **as PHI**, next to `<site>_flair_phi/`, and never upload it. `hashes.json`
  alone contains no PHI.
- A new FLAIR version gets a new snapshot folder (`..._flair_v<new version>/`), because the
  version is in its name.

## Who uses it

- **Data steward** creates the snapshot (the one run without `--snapshot`), stores it, and
  passes it on every later run.
- **Model developer, collaborative project.** When the developer is authorized to train on
  the site's CLIF data, they use the **same** snapshot inside the site's environment:

  ```bash
  flair load-task icu_readmission --clif-config clif_config.json --out output \
    --snapshot output/<site>_flair_v<version>
  flair fe-meds --feature-config feature_config.yaml \
    --task output/<site>_flair_phi/icu_readmission/cohort.parquet \
    --out output/meds.parquet --snapshot output/<site>_flair_v<version>
  ```

  The model is then trained on exactly the cohort and IDs the benchmark scores, and the
  four-hash audit confirms it on every run. The developer never runs without `--snapshot`:
  that would replace the steward's snapshot with new IDs. The snapshot never leaves the site.
- **Model developer, federated project** (no site data access) doesn't need the snapshot.
  They develop on MIMIC-CLIF and hand reviewed code to the steward, who runs it with `--snapshot`.
