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
| T5 Unplanned ICU Readmission | [`icu_readmission.html`](icu_readmission.html) | one-per-stay | `label_icu_readmission` |

`references.html` renders the shared bibliography; `references.bib` remains the
BibTeX source of record. `flair.css` is the shared stylesheet.

T1 and T2 are paired tasks (complementary labels on the same cohort and pipeline)
and share a flowchart.

## Parity with the Python implementation

These schematics mirror the constants and predicates in
`flair_benchmark/tasks/<name>.py`. When updating either side, update both.

- **Stitched-encounter grain (all tasks).** Every task operates on
  `hospitalization_join_id`: member hospitalizations within
  `stitch_time_interval_hours` (default 6h) are folded to one encounter, with
  anchors taken as min/max over members. Read each schematic's per-hospitalization
  `H_i` as the stitched encounter.
- **Constants** such as `HORIZON_HOURS`,
  `INTUB_WINDOW_MIN_DEFAULT`, `EXTUB_WINDOW_MIN_DEFAULT`,
  `TRACH_CLOSURE_TOLERANCE_MIN`, `MIN_IMV_HOURS`, `MIN_ICU_STAY_HOURS`, and the `LOCATION_MAPPING`
  buckets appear with matching values in both the schematic and the module.
- **`META.inclusion_criteria` / `META.exclusion_criteria`** read as English
  equivalents of each schematic's cohort bullets.
- **CLIF tables and categories** — each schematic enumerates every CLIF v2.1 table
  and category-column filter, mirroring the loader calls in
  `flair_benchmark/_clif.py` and the per-task `build()`. Every task restricts to
  `location_category == "icu"`.

## History

These pages were converted from the original LaTeX schematics (`*.tex` +
`tectonic`-compiled PDFs). The LaTeX sources and build artifacts have been removed
in favor of the HTML; `references.bib` is kept as the citation source of record.
