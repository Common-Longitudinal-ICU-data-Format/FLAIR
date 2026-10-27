# Feature engineering: direct, fe-meds

[← back to FLAIR 101](../101.md)

FLAIR gives you **two ways** to turn CLIF data into model inputs. Both must honor the
per-row `feature_cutoff_dttm` (no data after the prediction point). `fe-meds` only cuts
each encounter at its **latest** `feature_cutoff_dttm`; the per-row cut is on you in both
modes. Pick by how much control vs. convenience you want.

| mode | how | output | when to use |
|---|---|---|---|
| **direct** | your own code over `cohort.parquet` + `flair_benchmark.cohort.clif` loaders / clifpy | whatever you build | full control; custom or novel features; **all wide time-series** |
| **fe-meds** | `flair fe-meds` | ELF-long **medication** events — raw dose + raw unit, **no conversion** | lightweight; only selected meds; you convert doses downstream |

## Wide time-series data (vitals, labs, assessments, respiratory support)

Use clifpy directly — FLAIR does not wrap it:

```python
from clifpy import ClifOrchestrator
from flair_benchmark.cohort.clif import read_clif_config
from flair_benchmark.cohort.stitch import load_or_build_encounter_index, members_of_joins

cfg = read_clif_config("clif_config.json")
idx = load_or_build_encounter_index(cfg)   # fixed 6 h gap
hosp_ids = members_of_joins(idx, cohort["hospitalization_join_id"].unique().to_list())

co = ClifOrchestrator(data_directory=..., filetype="parquet", timezone="US/Eastern")
co.create_wide_dataset(
    tables_to_load=["vitals", "labs", "respiratory_support"],
    category_filters={"vitals": ["heart_rate", "spo2"], "labs": ["creatinine", "lactate"]},
    hospitalization_ids=hosp_ids,  # every member of each stitched encounter
    output_format="dataframe",
)
wide = co.wide_df
```

> ⚠️ **clifpy applies no leakage guard.** You must enforce the rule yourself: join
> `cohort.parquet` and keep only rows with `event_time <= feature_cutoff_dttm` for that
> stay. Pull `hospitalization_id`s from the cohort's *stitched* encounter
> (`hospitalization_join_id`) so sibling hospitalizations in the same encounter aren't
> dropped (that is what `members_of_joins` above does) — see
> `flair_benchmark/features/_scope.py:feature_scope`, which expands sibling ids and sets
> the per-encounter loading bound for `fe-meds`.

## fe-meds command

```bash
flair fe-meds \
  --feature-config feature_config.yaml \
  --task cohort.parquet \
  --out meds.parquet
```

It reads `cohort.parquet` (which must carry `prediction_dttm` and `feature_cutoff_dttm`;
a cohort without `feature_cutoff_dttm` raises `ValueError` — regenerate it with
`flair load-task`), expands each `hospitalization_join_id` to all its member
hospitalizations, loads CLIF via clifpy, cuts every event at the **encounter's latest**
`feature_cutoff_dttm`, and writes **ELF-long** parquet.

> ⚠️ For a daily task an encounter's events run up to its *last* day's cutoff, so a row
> predicting on day 2 can see day 5's events. Before training, join on
> `hospitalization_join_id` and keep `time <= feature_cutoff_dttm` for each row.

## The ELF long schema

`fe-meds` emits Event List Format (`flair_benchmark/features/_config.py`, `ELF_COLUMNS` /
`ELF_SCHEMA`) — one row per
clinical event:

| column | type | notes |
|---|---|---|
| `subject_id` | str | = `patient_id` |
| `time` | datetime (tz-naive) | always `≤` the encounter's max `feature_cutoff_dttm` |
| `code` | str | `<DOMAIN>//<category>//<unit>`; just `<DOMAIN>//<category>` when a row has no dose and no unit |
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
it keeps downstream feature code reusable across tasks and sites.

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

## Beyond `fe-meds`: the vendored CLIF→MEDS ETL

`fe-meds` is the supported CLI surface, and it covers medications only. The
package also ships a full vendored CLIF→MEDS ETL under
`flair_benchmark/meds_etl/` — `build_meds_events(clif_cfg, elf_cfg, hosp_ids, *,
max_workers=None)`, returning `(events, versions)`, over ten per-domain concept YAMLs
(`VITAL`, `LAB`, `RESP`, `PA`, `CRRT`, `HOSP`, `HOSP_DX`, `PATIENT`, `MED_CON`,
`MED_INT`) selected by the `domains:` mapping in `elf_cfg` (`hosp_ids=None` means every
hospitalization), which
`flair_benchmark/features/extractors.py` uses to build a shared MEDS store
across task cohorts. It is a **library-level API, not exposed by the `flair`
CLI**, and its interface is still moving; treat `fe-meds` plus clifpy as the
stable path unless you are working inside the repo. Its output is MEDS-shaped, not the
ELF schema above: there is no `unit` column, and `numeric_value` is Float32. Two
environment variables tune it (`fe-meds` reads neither):

- `FLAIR_MEDS_WORKERS` — domains run in parallel. Precedence: the `max_workers`
  argument, then `elf_cfg["max_workers"]`, then this variable, then `min(4, n_domains)`.
- `FLAIR_PUSHDOWN_MAX_IDS` (default `50000`) — above this many ids, a table is loaded in
  full and filtered in memory instead of pushing the id filter down to DuckDB.

## From ELF to model features (count truncation)

*Guidance, not shipped code — FLAIR does not include a count featurizer.*

A common way to turn ELF events into model features is to count codes into a sparse
matrix, whatever produced them (`fe-meds`, or your own wide extraction). Before counting,
truncate codes to **fixed levels** so units and order-types don't leak in as separate
features:

- **numeric** codes → 2 levels: `LAB//lactate//mmol/l//bmp` → `LAB//lactate`
- **categorical** codes → 3 levels: `RESP//device_category//imv` (unchanged)

Then events with `time ≤ feature_cutoff_dttm` are counted per `prediction_id`, and the
**vocabulary is fit on the train split only**. Every lactate measurement counts as one
feature regardless of unit; every vancomycin dose counts as one regardless of action.
