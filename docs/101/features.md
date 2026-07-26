# Feature engineering: direct, fe-meds

[← back to FLAIR 101](../101.md)

FLAIR gives you **two ways** to turn CLIF data into model inputs. Both must honor the
per-row `feature_cutoff_dttm` (no data after the prediction point) — `fe-meds` enforces it
for you, `direct` is on you. Pick by how much control vs. convenience you want.

| mode | how | output | when to use |
|---|---|---|---|
| **direct** | your own code over `cohort.parquet` + `flair_benchmark._clif` loaders / clifpy | whatever you build | full control; custom or novel features; **all wide time-series** |
| **fe-meds** | `flair fe-meds` | ELF-long **medication** events — raw dose + raw unit, **no conversion** | lightweight; only selected meds; you convert doses downstream |

## Wide time-series data (vitals, labs, assessments, respiratory support)

Use clifpy directly — FLAIR no longer wraps it:

```python
from clifpy import ClifOrchestrator

co = ClifOrchestrator(data_directory=..., filetype="parquet", timezone="US/Eastern")
co.create_wide_dataset(
    tables_to_load=["vitals", "labs", "respiratory_support"],
    category_filters={"vitals": ["heart_rate", "spo2"], "labs": ["creatinine", "lactate"]},
    hospitalization_ids=cohort["hospitalization_id"].unique().to_list(),
    output_format="dataframe",
)
wide = co.wide_df
```

> ⚠️ **clifpy applies no leakage guard.** You must enforce the rule yourself: join
> `cohort.parquet` and keep only rows with `event_time <= feature_cutoff_dttm` for that
> stay. Pull `hospitalization_id`s from the cohort's *stitched* encounter
> (`hospitalization_join_id`) so sibling hospitalizations in the same encounter aren't
> dropped — see `flair_benchmark/features/_scope.py:feature_scope` for how `fe-meds` does
> both.

## fe-meds command

```bash
flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out meds.parquet
```

It reads `cohort.parquet` (for `hospitalization_id`s and per-stay `feature_cutoff_dttm`),
loads CLIF via clifpy, truncates at the prediction point, and writes **ELF-long** parquet.

## The ELF long schema

`fe-meds` emits Event List Format (`flair_benchmark/features/_elf.py`) — one row per
clinical event:

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
P101       | 2150-03-01 04:00 | MED_INT//vancomycin//mg | 1000          | mg    |
P101       | 2150-03-01 06:15 | MED_CON//norepinephrine//mcg/kg/min | 0.08 | mcg/kg/min |
```

If you build wide features with clifpy, nothing forces you into this schema — but matching
it keeps downstream baselines reusable.

## feature_config.yaml

`flair init-config` scaffolds it. It drives `fe-meds`:

```yaml
clif_config: clif_config.json
output_directory: ./output/features

# fe-meds: lightweight med stream, raw dose + raw unit, NO conversion
meds_stream:
  medication_admin_continuous:
    categories: [norepinephrine, propofol]
  medication_admin_intermittent:
    categories: [vancomycin, cefepime, piperacillin_tazobactam]
```

Categories are CLIF 2.1 mCIDE values; unknown ones warn but don't fail.

## Units

`fe-meds` does **no conversion at extraction**: it keeps the **raw** per-row
`med_dose_unit` (charted as `"UNK"` when a dose has no unit) and writes it into both the
`unit` column and the code, so a downstream step can convert. For dose normalization
against a preferred-unit table, see `flair_benchmark/meds_etl/` (which wraps clifpy's
`convert_dose_units_by_med_category`).

## From ELF to model features (count truncation)

The reference baseline counts ELF codes into a sparse matrix, whatever produced them
(`fe-meds`, or your own wide extraction). Before counting, codes are **truncated to fixed
levels** so units and order-types don't leak in as separate features
(`flair_baseline/.../featurize.py`):

- **numeric** codes → 2 levels: `LAB//lactate//mmol/l//bmp` → `LAB//lactate`
- **categorical** codes → 3 levels: `RESP//device_category//imv` (unchanged)

Then events with `time ≤ feature_cutoff_dttm` are counted per `prediction_id`, and the
**vocabulary is fit on the train split only**. Every lactate measurement counts as one
feature regardless of unit; every vancomycin dose counts as one regardless of action.
