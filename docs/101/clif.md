# CLIF data layer

[← back to FLAIR 101](../101.md)

FLAIR is anchored on the **Common Longitudinal ICU Format (CLIF)** — a critical-care
minimum dataset that harmonizes ICU EHR content into a small set of relational tables
with standardized vocabularies (mCIDE) for vital-sign, lab, medication, and respiratory
categories. Because CLIF standardizes the data *elements* themselves, a model meets the
same coded vocabulary at every site, so FLAIR measures genuine cross-site generalizability.

## The tables FLAIR uses

| domain | CLIF table(s) |
|---|---|
| vitals | `vitals` |
| labs | `labs` |
| medications | `medication_admin_continuous`, `medication_admin_intermittent` |
| respiratory support | `respiratory_support` |
| assessments | `patient_assessments` (e.g. GCS, RASS) |
| identity / timing | `hospitalization`, `patient`, `adt` |
| infection | `microbiology_culture` |

## Two equivalent ways to consume CLIF

1. **Native clifpy tables** — the canonical CLIF Python interface, one class per table.
2. **MEDS event stream** — via the maintained CLIF→MEDS bridge.

FLAIR supports both; nothing locks you into one representation.

## How FLAIR loads CLIF

The loaders in `flair_benchmark/_clif.py` wrap clifpy and standardize the load:

```
clifpy table.from_file(data_directory, filetype, timezone, filters)
        → pandas → polars (pd_to_pl) → strip timezone to local wall-clock
```

Public loaders (each filters by `hospitalization_id` and, where relevant, category):

- `load_hospitalization(cfg, hospitalization_ids, columns)`
- `load_patient(cfg, patient_ids, columns)`
- `load_vitals(cfg, hospitalization_ids, vital_categories)`
- `load_labs(cfg, hospitalization_ids, lab_categories)`
- `load_medication_admin_continuous(...)` / `load_medication_admin_intermittent(...)`
- `load_respiratory_support(cfg, hospitalization_ids, device_categories)`

**Timezone note.** clifpy returns tz-aware datetimes typed to `cfg["timezone"]`; FLAIR's
`pd_to_pl` strips the tz so all datetimes are naive wall-clock already local to the site.
A literal anchor like `"07:00"` in a task therefore means local hospital time.

## Stitched encounters

Adjacent hospitalizations less than `stitch_time_interval_hours` apart (default 6) are
merged into a single **stitched encounter**, keyed by `hospitalization_join_id`
(`flair_benchmark/_stitch.py`). This is the **stay-level grain**: a stitched encounter is
one entity even when it spans several raw hospitalizations, and an encounter is never
split across train and test.

## Going deeper: clifpy directly

For custom work — pulling raw tables, or computing severity/comorbidity scores — use
clifpy directly. FLAIR already uses clifpy's `compute_sofa_polars` (SOFA), `calculate_cci`
(Charlson), and `calculate_elix` (Elixhauser) when building each task's Table 1. The
`clif-icu` skill in this repo is a ready guide for clifpy table loading and clinical scoring.
