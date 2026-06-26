# FLAIR Task Schematics

Self-contained task-definition documents (HTML) that specify cohort, anchors,
prediction window, label rule, and censoring logic for the FLAIR benchmark
tasks. These pages **track the Python implementation** in
`flair_benchmark/tasks/`; where the code and the earlier LaTeX spec drifted, the
HTML follows the **code** and flags the change in an audit callout.

## Read

Open [`index.html`](index.html) in any browser. Equations render via MathJax and
flowcharts via Mermaid, both loaded from CDN — an internet connection is needed
on first view. No build step.

## Contents

| Task | Page | Grain | Label |
|------|------|-------|-------|
| T1 ICU Daily In-Hospital Mortality (7AM) | [`icu_daily_mortality.html`](icu_daily_mortality.html) | one-per-window (daily) | `label_mortality` |
| T2 ICU Daily LTACH Discharge (7AM) | [`icu_daily_ltach.html`](icu_daily_ltach.html) | one-per-window (daily) | `label_ltach` |
| T3 Extubation Failure within 24h | [`extubation_failure_24h.html`](extubation_failure_24h.html) | one-per-stay | `label_extubation_failure_24h` |
| T4 Sepsis (CDC ASE onset) within 6h | [`sepsis_ase_6h.html`](sepsis_ase_6h.html) | one-per-window (hourly) | `label_sepsis_6h` |
| T5 Unplanned ICU Readmission | [`icu_readmission.html`](icu_readmission.html) | one-per-stay | `label_icu_readmission` |

`references.html` renders the shared bibliography; `references.bib` remains the
BibTeX source of record. `flair.css` is the shared stylesheet.

T1 and T2 are paired tasks (complementary labels on the same cohort and pipeline)
and share a flowchart. T4 (sepsis) carries an additional literature survey and a
timeline diagram because its time-zero anchor is contested in the sepsis-AI
literature.

## Parity with the Python implementation

These schematics mirror the constants and predicates in
`flair_benchmark/tasks/task<N>_*.py`. When updating either side, update both.

- **Stitched-encounter grain (all tasks).** Every task operates on
  `hospitalization_join_id`: member hospitalizations within
  `stitch_time_interval_hours` (default 6h) are folded to one encounter, with
  anchors taken as min/max over members. Read each schematic's per-hospitalization
  `H_i` as the stitched encounter.
- **Constants** such as `HORIZON_HOURS`, `GRID_STEP_HOURS`, `LABEL_HORIZON_HOURS`,
  `INTUB_WINDOW_MIN_DEFAULT`, `EXTUB_WINDOW_MIN_DEFAULT`,
  `TRACH_CLOSURE_TOLERANCE_MIN`, `MIN_ICU_STAY_HOURS`, and the `LOCATION_MAPPING`
  buckets appear with matching values in both the schematic and the module.
- **`META.inclusion_criteria` / `META.exclusion_criteria`** read as English
  equivalents of each schematic's cohort bullets.
- **CLIF tables and categories** — each schematic enumerates every CLIF v2.1 table
  and category-column filter, mirroring the loader calls in
  `flair_benchmark/_clif.py` and the per-task `build()`. Task T4 is the only task
  whose `adt` filter is exclusion-style (drops `psychiatric`/`rehabilitation`); the
  other four restrict to `location_category == "icu"`.

### Task T4 — anchor changed to CDC ASE

Task T4's time-zero was re-anchored from the **first qualifying IV antibiotic**
(the original spec) to **CDC Adult Sepsis Event (ASE) onset** via
`clifpy.utils.ase.compute_ase`. The HTML documents the ASE implementation; the
antibiotic-anchored label is retained only as a documented fallback for sites
lacking the `microbiology_culture` table. See the audit callout in
`sepsis_ase_6h.html`.

## History

These pages were converted from the original LaTeX schematics (`*.tex` +
`tectonic`-compiled PDFs). The LaTeX sources and build artifacts have been removed
in favor of the HTML; `references.bib` is kept as the citation source of record.
