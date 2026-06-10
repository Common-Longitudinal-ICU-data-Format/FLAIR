# FLAIR Task Schematics

Self-contained task-definition documents (LaTeX + compiled PDF) that
specify cohort, anchors, prediction window, label rule, and censoring
logic for the FLAIR benchmark tasks. Each schematic predates the
Python implementation in `flair_benchmark/tasks/` and is the
authoritative spec --- the module docstrings defer to these documents.

## Contents

| Task | Schematic | Flowchart |
|------|-----------|-----------|
| T1 ICU Daily In-Hospital Mortality (7AM) | `icu_daily_mortality.tex` | `figures/icu_daily_disposition_flowchart.tex` |
| T2 ICU Daily LTACH Discharge (7AM) | `icu_daily_ltach.tex` | shares T1's flowchart |
| T3 Extubation Failure within 24h | `extubation_failure_24h.tex` | `figures/extubation_failure_24h_flowchart.tex` |
| T4 Sepsis (antibiotic-anchored) within 6h | `sepsis_antibiotic_timezero.tex` | `figures/flowchart.tex` + `figures/timeline.tex` |
| T5 Unplanned ICU Readmission | `icu_readmission.tex` | `figures/icu_readmission_flowchart.tex` |

T1 and T2 are paired tasks (complementary labels on the same cohort
and pipeline); T2 cites T1 as the pipeline authority and shares the
flowchart. T4 (sepsis) carries an additional literature survey and a
timeline diagram because its time-zero anchor is contested in the
sepsis-AI literature; the other tasks have less contested anchors and
keep to a focused algorithm + flowchart + rationale structure.

`references.bib` carries the BibTeX for all schematics.

## Compile

Compile any single schematic:

```
cd docs/task_schematics
tectonic <schematic>.tex
```

Or compile all five in place:

```
cd docs/task_schematics
tectonic *.tex
```

Tectonic auto-fetches missing TeX packages on first run. Produces
`<schematic>.pdf` for each `.tex` source.

## Authority and parity with Python

Each schematic is the authoritative spec; the sibling Python module
in `flair_benchmark/tasks/task<N>_*.py` mirrors the constants and
predicates verbatim. When updating either side, update both:

- Constants such as `INPUT_WINDOW_HOURS`, `GRID_STEP_HOURS`,
  `HORIZON_HOURS`, `INTUB_WINDOW_MIN_DEFAULT`,
  `EXTUB_WINDOW_MIN_DEFAULT`, `TRACH_CLOSURE_TOLERANCE_MIN`,
  `MIN_ICU_STAY_HOURS`, and the `LOCATION_MAPPING` buckets appear with
  matching values in both the schematic and the corresponding module.
- `META.inclusion_criteria` / `META.exclusion_criteria` strings should
  read as English equivalents of the schematic's cohort bullets.
- Each schematic's "CLIF tables and categories used" subsection
  enumerates every CLIF v2.1 table touched and every category-column
  filter (literal values, case-sensitivity). This mirrors the loader
  calls in `flair_benchmark/_clif.py` and the per-task `build()`
  filters; update both sides together when adding or removing a table
  or category filter. Task~T4 is the only task whose `adt` filter is
  exclusion-style (drops `psychiatric`/`rehabilitation`); the other
  four restrict to `location_category == "icu"`.
