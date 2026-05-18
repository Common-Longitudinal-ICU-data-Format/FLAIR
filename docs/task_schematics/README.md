# FLAIR Task Schematics

Self-contained task-definition documents (LaTeX + compiled PDF) that
specify cohort, time-zero anchor, prediction window, label rule, and
censoring logic for FLAIR benchmark tasks. Each schematic predates the
Python implementation in `flair_benchmark/tasks/` and is the
authoritative spec.

## Contents

- `sepsis_antibiotic_timezero.tex` / `.pdf` — sepsis prediction with
  first-qualifying-IV-antibiotic as time-zero (Kamran et al. NEJM AI
  2024 framework, hospital-wide cohort, 6h horizon).
- `figures/timeline.tex` — TikZ hospitalization timeline.
- `figures/flowchart.tex` — TikZ build-pipeline flowchart.
- `references.bib` — BibTeX for the 10 prior sepsis-prediction papers.

## Compile

```
cd docs/task_schematics
tectonic sepsis_antibiotic_timezero.tex
```

Tectonic auto-fetches missing TeX packages on first run. Produces
`sepsis_antibiotic_timezero.pdf` in place.
