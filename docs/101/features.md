# Feature engineering: direct, fe-meds, fe-wide

[← back to FLAIR 101](../101.md)

FLAIR gives you **three ways** to turn CLIF data into model inputs. All three honor the
per-row `feature_cutoff_dttm` (no data after the prediction point). Pick by how much
control vs. convenience you want.

| mode | how | output | when to use |
|---|---|---|---|
| **direct** | your own code over `cohort.parquet` + `flair_benchmark._clif` loaders / clifpy | whatever you build | full control; custom or novel features |
| **fe-meds** | `flair fe-meds` | ELF-long **medication** events — raw dose + raw unit, **no conversion** | lightweight; only selected meds; you convert doses downstream |
| **fe-wide** | `flair fe-wide` | ELF-long events for **all configured** vitals/labs/meds/assessments/resp | config-driven, leak-guarded, everything at once |

## fe-wide / fe-meds commands

```bash
flair fe-wide \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out features.parquet

flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out meds.parquet
```

Both read `cohort.parquet` (for `hospitalization_id`s and per-stay `feature_cutoff_dttm`),
load CLIF via clifpy, truncate at the prediction point, and write **ELF-long** parquet.

## The ELF long schema

Both extractors emit the same Event List Format (`flair_benchmark/features/_elf.py`) — one
row per clinical event — so downstream code works with either:

| column | type | notes |
|---|---|---|
| `subject_id` | str | = `patient_id` |
| `time` | datetime (tz-naive) | always `≤ feature_cutoff_dttm` |
| `code` | str | `<DOMAIN>//<category>//<unit>` |
| `numeric_value` | float | nullable (numeric events) |
| `unit` | str | required where `numeric_value` is set |
| `text_value` | str | nullable (categorical events) |
| `hospitalization_id` | str | required |
| `hospitalization_join_id` | str | required (stitched-encounter key) |

Example rows:

```
subject_id | time             | code                    | numeric_value | unit  | text_value
P101       | 2150-03-01 06:10 | VITAL//heart_rate//bpm  | 88            | bpm   |
P101       | 2150-03-01 06:30 | LAB//creatinine//mg/dl  | 1.8           | mg/dl |
P101       | 2150-03-01 07:45 | RESP//device_category   |               |       | imv
P101       | 2150-03-01 04:00 | MED_INT//vancomycin//mg | 1000          | mg    |
```

## feature_config.yaml

`flair init-config` scaffolds it. It drives both extractors:

```yaml
clif_config: clif_config.json
output_directory: ./output/features

# fe-wide: time-series -> clifpy create_wide_dataset -> melted to ELF long
wide:
  vitals:
    categories: [heart_rate, sbp, dbp, map, spo2, respiratory_rate, temp_c]
  labs:
    categories: [creatinine, lactate, wbc, hemoglobin, platelet_count, sodium, potassium]
  medication_admin_continuous:
    categories: [norepinephrine, epinephrine, vasopressin, propofol]
  patient_assessments:
    categories: [gcs_total, RASS]
  respiratory_support:
    columns: [device_category, mode_category, fio2_set, peep_set]

# fe-meds: lightweight med stream, raw dose + raw unit, NO conversion
meds_stream:
  medication_admin_continuous:
    categories: [norepinephrine, propofol]
  medication_admin_intermittent:
    categories: [vancomycin, cefepime, piperacillin_tazobactam]

# optional per-category unit overrides
# units:
#   labs:
#     creatinine: mg/dl
```

Categories are CLIF 2.1 mCIDE values; unknown ones warn but don't fail.

## Units

Units are **deterministic per mCIDE category** and are **not converted at extraction**
(`flair_benchmark/features/_units.py`): labs read their reference unit from clifpy's schema,
vitals/respiratory use fixed mCIDE units. The `fe-meds` path keeps the **raw** per-row
`med_dose_unit` (charted as `"UNK"` when a dose has no unit) so you can convert later. Use
the optional `units:` block to override any default.

## From ELF to model features (count truncation)

The reference baseline counts ELF codes into a sparse matrix. Before counting, codes are
**truncated to fixed levels** so units and order-types don't leak in as separate features
(`flair_baseline/.../featurize.py`):

- **numeric** codes → 2 levels: `LAB//lactate//mmol/l//bmp` → `LAB//lactate`
- **categorical** codes → 3 levels: `RESP//device_category//imv` (unchanged)

Then events with `time ≤ feature_cutoff_dttm` are counted per `prediction_id`, and the
**vocabulary is fit on the train split only**. Every lactate measurement counts as one
feature regardless of unit; every vancomycin dose counts as one regardless of action.
