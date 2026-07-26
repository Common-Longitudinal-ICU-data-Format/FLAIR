# plan.md — `flair_benchmark/` restructure

**Status:** EXECUTED. See §11 for what actually happened, including two
deviations from this plan. The 15 code-review findings this restructure deferred
were fixed in a later commit — see the Appendix.
**Goal:** fewer files, clearer grouping, plainer comments — with the CLI surface unchanged.
**Branch:** `add_new_task`
**Verification gate:** `pytest tests/` must report **169 passed** before and after.
Baseline measured on this branch: `169 passed, 10 warnings in 450.61s`. The suite takes
about **7m30s**, so budget for that on each verification run.

---

## 1. Why

Three concrete problems in the current layout, in priority order.

### 1.1 Table 1 generation is split across four files in two packages

| file | lines | job |
|---|---:|---|
| `flair_benchmark/_table1.py` | 131 | orchestration + per-site score cache |
| `flair_benchmark/report/table1.py` | 205 | the actual table builder |
| `flair_benchmark/report/_severity.py` | 49 | SOFA max |
| `flair_benchmark/report/_comorbidity.py` | 67 | Charlson / Elixhauser |

None of it is reachable from `build_report`. It all runs at `load-task` time
(`cli.py:95` → `_table1.generate_table1`). It sits under `report/` for historical
reasons only, so anyone reading `report/` has to grep to discover that three of its
modules are not report code.

### 1.2 The package root is ungrouped

Eleven private modules sit flat at the root doing three unrelated jobs, with nothing
in the layout to distinguish them:

- **packaging / identity** — `_constants` (5 importers), `_templates` (1), `_hash` (2)
- **CLIF I/O** — `_clif` (14), `_stitch` (11), `_split` (6)
- **clinical derivation** — `_encounter` (5), `_readmission` (3), `_ventilation_episodes` (2), `_table1` (1)

### 1.3 "Fewer files" and "cleaner files" pull in opposite directions

`report/_suppress.py` is 15 lines — too small to earn a file. `report/_viz.py` is 924
lines and `report/operating_points.py` is 688 — already past what fits in one reading.
Naive file-count reduction makes the second problem worse.

**This plan merges only where the merged file stays under ~460 lines, and never merges
into `_viz.py` or `operating_points.py`.**

### 1.4 Secondary: the leading underscore means three different things

- at the root — "private"
- in `tasks/` — "**not a task**", load-bearing, enforced by `pkgutil.iter_modules`
  filtering `not name.startswith("_")` at `tasks/__init__.py:37`
- in `report/` — "helper"

`tasks/` is therefore **left completely untouched** by this plan. Renaming
`tasks/_finalize.py` to `finalize.py` would register a phantom task named `finalize`.

---

## 2. Decisions taken

| decision | choice | rationale |
|---|---|---|
| Aggressiveness | **Grouped** | Fixes all three problems; no new module over ~460 lines. |
| Concept YAMLs | **Keep all 10 separate** | Per-domain clinical review matters — see Appendix findings 4 and 5, both content bugs in a single domain YAML. |
| Bug fixes | **Excluded from this commit** | Structural change only. The 15 review findings were recorded in the Appendix and fixed in a **follow-up commit** — see the Appendix for the resolution of each. |
| Back-compat | **Clean break, no shims** | Package is at `0.0.1`. Shims would add ~8 files back, defeating the goal. Old paths raise `ModuleNotFoundError` immediately and greppably. |
| Commits | **One single commit** | Chosen deliberately; note the cost in §7. |
| Stale `.pyc` | **Delete** | 14 stale `.pyc` files for task modules that no longer exist. |

---

## 3. Target layout

```
flair_benchmark/
  __init__.py             package surface                        (unchanged)
  cli.py                  the CLI                                (unchanged)
  constants.py            version, shared constants, templates, audit hash

  cohort/                 everything that BUILDS a cohort
    __init__.py
    clif.py               CLIF table loaders
    stitch.py             encounter key + train/test split
    encounter.py          stitched-encounter ICU cohort + daily grid
    events.py             clinical event detection from CLIF tables
    table1.py             cohort characteristics (all four sources reunited)

  tasks/                  UNTOUCHED — discovery contract
    __init__.py  _finalize.py
    extubation_failure_24h.py  icu_daily_ltach.py
    icu_daily_mortality.py     icu_readmission.py

  features/
    __init__.py
    _config.py            feature-config loading + ELF schema
    _scope.py             cohort scope + no-future cutoff
    extractors.py         fe_meds + meds_stream

  meds_etl/
    __init__.py
    pipeline.py           config load, domain runners, codes metadata
    _emit.py              emission core + code/time resolution + transforms
    concepts/             10 YAMLs, unchanged

  report/                 report-time code ONLY
    __init__.py           build_report, mode dispatch, preds validation
    episodic.py           episodic mode
    continuous.py         continuous mode + peak section + leadtime
    landmark.py  kpi.py
    discrimination.py  calibration.py  dca.py  fairness.py
    operating_points.py
    _stats.py             CIs, suppression, confusion arithmetic, calibration fit
    _basis.py  _sweep.py  _locators.py  _viz.py
```

### File counts

|  | now | after | delta |
|---|---:|---:|---:|
| Python | 54 | 37 | **−17** |
| YAML | 10 | 10 | 0 |
| **Total** | **64** | **47** | **−17** |

| package | now | after |
|---|---:|---:|
| root | 12 | 3 |
| `cohort/` | 0 | 6 |
| `tasks/` | 6 | 6 (untouched) |
| `features/` | 6 | 4 |
| `meds_etl/` | 7 py | 3 py |
| `report/` | 23 | 15 |
| **py total** | **54** | **37** |

---

## 4. File mapping

### 4.1 Root → `constants.py` and `cohort/`

| from | to | new size |
|---|---|---:|
| `_constants.py` + `_templates.py` + `_hash.py` | `constants.py` | ~131 |
| `_clif.py` | `cohort/clif.py` | 260 |
| `_stitch.py` + `_split.py` | `cohort/stitch.py` | ~345 |
| `_encounter.py` | `cohort/encounter.py` | 262 |
| `_readmission.py` + `_ventilation_episodes.py` | `cohort/events.py` | ~341 |
| `_table1.py` + `report/table1.py` + `report/_severity.py` + `report/_comorbidity.py` | `cohort/table1.py` | ~452 |

`_stitch` + `_split` merge because the split is assigned *at* the stitched-encounter
grain — `_split.py:46` keys on `hospitalization_join_id`, which `_stitch.py` owns.
They are one concept.

`_readmission` + `_ventilation_episodes` merge because both answer the same shape of
question: "when did a clinical event happen, per hospitalization, read from one CLIF
table". Neither imports the other today; the merged file has two clearly separated
sections.

### 4.2 `report/` 23 → 15

| from | to | new size |
|---|---|---:|
| `_ci.py` + `_suppress.py` + `_confusion.py` + `_calib.py` | `_stats.py` | ~272 |
| `continuous.py` + `peak.py` + `leadtime.py` | `continuous.py` | ~250 |
| `table1.py`, `_severity.py`, `_comorbidity.py` | → `cohort/table1.py` (§4.1) | — |

The four merged into `_stats.py` are all genuinely shared primitives:
`_ci` 10 consumers, `_suppress` 10, `_confusion` 5, `_calib` 3.

`peak.py` and `leadtime.py` fold into `continuous.py` because `continuous.py` is
already the sole composer of both, making `episodic.py` / `continuous.py` symmetric
mode entry points.

**Not merged, deliberately:** `_viz.py` (924), `operating_points.py` (688),
`_locators.py` (310), `_sweep.py` (203), `_basis.py` (157) — each is already
substantial and single-purpose.

### 4.3 `features/` 6 → 4

| from | to | new size |
|---|---|---:|
| `_config.py` + `_elf.py` | `_config.py` | ~207 |
| `fe_meds.py` + `meds_stream.py` | `extractors.py` | ~259 |
| `_scope.py` | unchanged | 89 |

`_elf.py` is a schema definition; it belongs with the config that selects what to emit.
`_scope.py` stays separate — the no-future cutoff is a distinct correctness concern.

### 4.4 `meds_etl/` 7 py → 3 py

| from | to | new size |
|---|---|---:|
| `_emit.py` + `resolve.py` + `transforms.py` | `_emit.py` | ~367 |
| `pipeline.py` + `domains.py` + `metadata.py` | `pipeline.py` | ~405 |

`resolve.py` (45) and `transforms.py` (33) are small pure-function helpers used only by
the emit path. `domains.py` and `metadata.py` are only ever driven by `pipeline.py`.

---

## 5. Import path mapping

Every old path below stops existing. There are no shims.

### 5.1 Root modules

| old | new |
|---|---|
| `flair_benchmark._constants` | `flair_benchmark.constants` |
| `flair_benchmark._templates` | `flair_benchmark.constants` |
| `flair_benchmark._hash` | `flair_benchmark.constants` |
| `flair_benchmark._clif` | `flair_benchmark.cohort.clif` |
| `flair_benchmark._stitch` | `flair_benchmark.cohort.stitch` |
| `flair_benchmark._split` | `flair_benchmark.cohort.stitch` |
| `flair_benchmark._encounter` | `flair_benchmark.cohort.encounter` |
| `flair_benchmark._readmission` | `flair_benchmark.cohort.events` |
| `flair_benchmark._ventilation_episodes` | `flair_benchmark.cohort.events` |
| `flair_benchmark._table1` | `flair_benchmark.cohort.table1` |

### 5.2 `report/`

| old | new |
|---|---|
| `flair_benchmark.report._ci` | `flair_benchmark.report._stats` |
| `flair_benchmark.report._suppress` | `flair_benchmark.report._stats` |
| `flair_benchmark.report._confusion` | `flair_benchmark.report._stats` |
| `flair_benchmark.report._calib` | `flair_benchmark.report._stats` |
| `flair_benchmark.report.peak` | `flair_benchmark.report.continuous` |
| `flair_benchmark.report.leadtime` | `flair_benchmark.report.continuous` |
| `flair_benchmark.report.table1` | `flair_benchmark.cohort.table1` |
| `flair_benchmark.report._severity` | `flair_benchmark.cohort.table1` |
| `flair_benchmark.report._comorbidity` | `flair_benchmark.cohort.table1` |

### 5.3 `features/` and `meds_etl/`

| old | new |
|---|---|
| `flair_benchmark.features._elf` | `flair_benchmark.features._config` |
| `flair_benchmark.features.fe_meds` | `flair_benchmark.features.extractors` |
| `flair_benchmark.features.meds_stream` | `flair_benchmark.features.extractors` |
| `flair_benchmark.meds_etl.resolve` | `flair_benchmark.meds_etl._emit` |
| `flair_benchmark.meds_etl.transforms` | `flair_benchmark.meds_etl._emit` |
| `flair_benchmark.meds_etl.domains` | `flair_benchmark.meds_etl.pipeline` |
| `flair_benchmark.meds_etl.metadata` | `flair_benchmark.meds_etl.pipeline` |

**All public function and class names are preserved.** Only the module they live in
changes, so every rewrite is a one-line edit at the import site.

Roughly **110 import statements** across the 54 source files and 12 test files.

---

## 6. Non-source files to update

| file | why |
|---|---|
| `pyproject.toml` | `[tool.hatch.version] path = "flair_benchmark/_constants.py"` → `constants.py`. **The build breaks if this is missed.** |
| `docs/101/features.md:11` | Documents `flair_benchmark._clif` loaders as a supported user path. |
| `docs/task_schematics/index.html:113` | Cites `flair_benchmark._split.assign_split`. |
| `docs/task_schematics/icu_daily_mortality.html:179,204` | Cites `_split.assign_split` and `_clif.pd_to_pl`. |
| `flair_benchmark/tasks/__pycache__/` | Delete 14 stale `.pyc` for removed task modules (`task1_discharged_home`, `task10_aki_kdigo2_24h`, `task11_sepsis_abx_6h`, …). Gitignored, so this is a working-tree cleanup only. |

`docs/task_schematics/*.log` are LaTeX build output and are **not** hand-edited; they
regenerate.

### Out-of-repo consumers

None known. The `flair-baseline` reference model no longer exists and all mention
of it has been removed from this repo. If another project turns out to import
these paths, §5 is the mapping it needs.

---

## 7. Execution

One commit, as chosen. Order of operations within it:

1. Create `flair_benchmark/cohort/` with `__init__.py`.
2. `git mv` each root module to its destination (single-file moves first, so git can
   still detect the renames it is able to).
3. Concatenate the merge groups from §4, resolving duplicate imports and keeping each
   merged file's sections in the order listed.
4. Rewrite all ~110 import statements per §5.
5. Rewrite module docstrings per §8.
6. Update `pyproject.toml` and the four doc references from §6.
7. Delete the stale `__pycache__` entries.
8. Run the verification in §9.

**Known cost of the single-commit choice:** because renames and content edits land
together, `git` will show most of these files as delete + add rather than as renames,
so `git log --follow` will not trace history through the move. This is accepted.

---

## 8. Comment and docstring convention

Every module docstring follows the same three-part shape, in this order:

1. **What it does** — one plain sentence, no jargon, no forward references.
2. **Why it exists / the non-obvious constraint** — the thing a reader would
   otherwise get wrong.
3. **What it depends on** — upstream inputs and downstream consumers.

The current docstrings front-load jargon. `report/_basis.py` opens with
*"Shared decision-basis primitives: alarm vs action, and the DCA frame"* — three
undefined terms before the first full stop. The rewrite states the plain fact first,
then the nuance.

Inline comments keep the existing high standard — the codebase already explains *why*
rather than *what* (for example the note at `tasks/__init__.py:74` explaining why the
clinical threshold must be a multiple of 0.01). Nothing in that class is removed.

**Scope limit:** docstrings and comments only. No logic, no signatures, no behaviour.

---

## 9. Verification

```bash
pytest tests/                       # must be 169 passed  (~7m30s)
ruff check flair_benchmark tests    # E, F, W, I per pyproject
python -c "import flair_benchmark; print(flair_benchmark.__version__)"   # 0.0.1
flair list-tasks                    # must list exactly 4 tasks
python -m build                     # confirms the hatch version path still resolves
grep -rn "flair_benchmark\._" flair_benchmark tests   # must return nothing
```

The last check is the real guard: after the move, no `flair_benchmark._<name>`
top-level private path should remain anywhere.

`flair list-tasks` must show exactly `extubation_failure_24h`, `icu_daily_ltach`,
`icu_daily_mortality`, `icu_readmission` — it catches an accidental rename inside
`tasks/`, which is the one directory this plan must not disturb.

---

## 10. Risks

| risk | likelihood | mitigation |
|---|---|---|
| `pyproject.toml` version path missed → build breaks | medium | `python -m build` in §9 |
| A `tasks/` file gains or loses an underscore → phantom or missing task | low | `flair list-tasks` in §9 |
| Circular import between `cohort/` modules | low | `constants.py` imports nothing from the package (already true today); `cohort/table1.py` imports `clif` and `stitch`, never the reverse |
| A merged file exceeds what fits in one reading | low | Largest new file is `cohort/table1.py` at ~452 lines; nothing merges into `_viz.py` or `operating_points.py` |
| An unknown out-of-repo importer breaks | low | No known consumers; §5 mapping table, and `ModuleNotFoundError` is immediate and names the module |

---

## 11. Execution record

Final counts, measured from the built wheel rather than estimated:

|  | before | after | delta |
|---|---:|---:|---:|
| Python | 54 | **37** | **−17** |
| YAML | 10 | 10 | 0 |
| **Total** | **64** | **47** | **−17** |

Verified after the change:

- `pytest tests/` — see §9 gate
- `uvx --from build pyproject-build --wheel` — succeeds; the wheel contains
  exactly 37 `.py` + 10 `.yaml`, confirming the `[tool.hatch.version]` path move
- `flair list-tasks` — lists exactly the 4 expected tasks, so nothing in
  `tasks/` gained or lost an underscore
- `flair init-config` — writes both templates from the merged `constants.py`
- `grep -rn "flair_benchmark\._" flair_benchmark tests` — no hits
- `ruff check` — 9 violations, all pre-existing (13 before; 4 import-sort
  violations were fixed, and the remaining diff is only shifted line numbers on
  untouched E702/E731/F401 findings). No new violations introduced.
- Top-level definition inventory: 271 -> 269. The only changes are the two
  private renames forced by a name collision in a merged file (`_empty` ->
  `_empty_readmission_frame`, `_empty_frame` -> `_empty_comorbidity_frame`) and
  the two dead functions in the deleted `resolve.py`. No public function moved
  or disappeared — only the module it lives in changed.

### Deviations from the plan

**1. `meds_etl/resolve.py` was DELETED, not merged.** It turned out to be dead
code: nothing imports `resolve_code` or `resolve_time`. `_emit._code_expr` is the
vectorized replacement, and its docstring said it was "mirroring
resolve.resolve_code" — the row-wise original had been superseded and left
behind. Deleting it serves the goal better than merging it would have.

**2. `meds_etl/pipeline.write_codes_parquet` is also unused** but was KEPT, since
removing a function is a behaviour-surface change rather than a structural one
and the plan scoped this work to structure. Its docstring now says so, making it
a one-line follow-up if you want it gone.

### Two failures caught by the suite, and what they teach

The first full run came back **2 failed, 167 passed**. Both were caused by this
restructure, and both are worth recording because neither would have been caught
by reading the diff.

**1. A mechanical rewrite missed an import form.** The path rewrite matched
`flair_benchmark.report._comorbidity` but not
`from flair_benchmark.report import _comorbidity as comorb`, where the module
name is not part of a dotted path. Fixed in `tests/test_audit_fixes.py`, which
now imports `compute_cci_elix` from `cohort/table1.py` and asserts on that
function's source rather than the whole module's.

*Lesson:* a path-based find-and-replace does not see `from <pkg> import
<module>`. The follow-up grep for that specific form found no other instances.

**2. Turning a module-level import into a function-local one is NOT a
structural change.** While merging `fe_meds.py` into `extractors.py` I moved
`from flair_benchmark.meds_etl import build_meds_events` inside the function
bodies. That removed `build_meds_events` from the module's attribute surface, so
`monkeypatch.setattr(fe_meds_mod, "build_meds_events", ...)` in
`tests/test_feature_stitch_scope.py` had nothing to patch — the test then
exercised the real ETL instead of the fake. Reverted to a module-level import,
which is also what the original file had.

*Lesson:* lazy imports are part of a module's public surface when tests patch
them. This plan scoped the work to "structure only", and moving that import was
a deviation that bought nothing — `meds_etl` imports nothing from `features`, so
there was no cycle to avoid.

### Duplicate imports from the merges

Collapsing four modules into `report/_stats.py` left several files importing from
the same module 2-5 times. All were consolidated by hand into single statements
(`_locators`, `fairness`, `discrimination`, `dca`, `calibration`, `kpi`,
`operating_points`, `extubation_failure_24h`, and two test files). One lazy
import in `fairness.py` was hoisted to module level: it existed to defer loading
`_calib`, which is now the same module already imported at the top, so the
laziness bought nothing.

---

## Appendix — code-review findings (fixed in a follow-up commit)

A `/code-review` pass over `git diff main...HEAD` plus the working tree returned 15
findings. They are **content** bugs, unrelated to layout, and were deliberately excluded
from the restructure so that diff stayed purely mechanical.

**All 15 are now fixed**, plus the five unranked notes, in a separate commit. File paths
below are the PRE-restructure names the review used; the resolution line gives the
current one. Regression tests live in `tests/test_review_fixes.py` — 5 of the 6
behavioural bugs there were confirmed to reproduce against the pre-fix code before the
fix landed (the sixth, the `.replace()` note, has no reachable failing input today).

### Changes to published output — re-run affected artifacts

Four fixes change numbers a site has already produced:

| change | effect |
|---|---|
| Split grain is now the **patient**, not the encounter | Every cohort's train/test assignment is re-dealt. A model trained on an older cohort must be re-scored, not just re-reported. |
| Outliers are **nulled**, not clipped | `height_cm` (whole column was 96), `pao2` and every other limited value now go missing instead of sitting at the boundary. |
| `allow_other_meds: true` | 31 of 75 MED_CON and 116 of 165 MED_INT categories re-enter the extract. MEDS/ELF tables get materially bigger. |
| MEDS ids are **Utf8** | `MEDS.parquet` `subject_id` / `hospitalization_id` change dtype (they already matched the documented ELF schema — the MEDS writer was the outlier). |

The encounter-index and cohort-score caches are now fingerprinted on the source files,
so the first run after this commit rebuilds them rather than reusing a stale one.

### Critical

1. `meds_etl/_emit.py:243` — `subject_id` is cast with
   `pl.col(...).cast(pl.Int64, strict=False)` and then null-filtered. CLIF
   `patient_id` / `hospitalization_id` are **strings** in mCIDE, so at any site with
   alphanumeric IDs (`"P101"`, `"HOSP_0012"`) the cast yields null and **every event
   row is silently dropped** — `MEDS.parquet` is written with 0 rows and no error.
   Leading-zero IDs (`"007"` → `7`) are corrupted the same way. Tests pass only because
   the fixtures use numeric IDs.

### High

2. `_clif.py:92` (repeats at `:100`, `:107`, `:188`, `_encounter.py:214`) —
   `filters = {...} if hospitalization_ids else None`. An **empty** list is falsy, so
   clifpy loads the entire table instead of nothing. In `base_icu_cohort`, a site
   charting `micu` instead of `icu` yields `icu_hosp_ids == []` and the "adult ICU base
   cohort" silently becomes the whole adult inpatient population.
3. `_readmission.py:77` — the planned `ICU → Procedural → ICU` removal runs on **raw**
   ADT rows, before consecutive same-location rows are merged (~line 92), so it only
   recognises a *single-row* procedural interval. A routine OR trip charted as two rows
   (OR then PACU) is labelled an unplanned readmission.
4. `meds_etl/concepts/VITAL.yaml:8` — `height_cm: [30, 96]` is an **inches** range on a
   centimetre category, and `_emit._numeric_expr` **clips** rather than nulls, so every
   adult height (150–200 cm) is rewritten to exactly 96. clifpy's own `vital_ranges`
   has `[70, 255]`; every other limit in the file matches clifpy exactly.
5. `meds_etl/concepts/MED_CON.yaml:22` (same shape at `MED_INT.yaml:21`) — with
   `allow_other_meds: false`, `domains.py:103` uses the `preferred_units` **key set as
   a row filter**. 9 MED_CON keys match no mCIDE category (`angiotension` →
   `angiotensin`, `sodium chloride` → `sodium_chloride`, …) and **31 of 75 valid
   categories are dropped**, including every crystalloid. MED_INT: 18 phantom keys,
   116 of 165 dropped. `override: true` downgrades clifpy's error to a log line.
6. `report/operating_points.py:102` — `_confusion_row` publishes the un-suppressed `n`
   alongside all four suppressed cells. Since `tp+fp+fn+tn == n`, any single suppressed
   cell is recoverable by subtraction. Same family at `:265`, `:441`,
   `discrimination.py:135`, `fairness.py:161`.

### Medium-high

7. `report/_comorbidity.py:44` — `_slim()`'s failure path returns a **three**-column
   `_empty_frame()` where the caller joins two-column frames, producing a stray
   `charlson_score_right` that `_table1._compute_scores` then caches permanently into
   `cohort_scores.parquet`.
8. `_clif.py:252` / `tasks/extubation_failure_24h.py:164` — ADT is truncated to the
   cohort's `hospitalization_id`s before the stitch key is attached, so
   encounter-grain detectors run over incomplete data. `members_of_joins` exists for
   exactly this and is unused on both paths.

### Medium

9. `report/__init__.py:302` — the `y_prob` contract check misses NaN.
   `Series([0.1, nan, 0.9]).null_count() == 0` and `.min()`/`.max()` skip NaN, so NaN
   scores pass both guards and later die inside sklearn. Add `is_nan().any()`.
10. `_ventilation_episodes.py:88` — `pre_existing_trach` requires
    `first_trach_idx < first_imv_idx` (strict), but the loop `break`s on the trach
    sentinel at line 97 first. When both land on the same row (a trached patient
    ventilated through the trach), the function returns nothing at all — no episodes
    *and* no sentinel. Use `<=`.
11. `_split.py:46` — splits are assigned at `hospitalization_join_id` grain only;
    `patient_id` is never consulted. A patient with two admissions further apart than
    `stitch_time_interval_hours` gets two join IDs and can land in train **and** test.
12. `meds_etl/_emit.py:216` — `outlier_shaping` uses `.clip(lo, hi)`, i.e. it
    winsorises out-of-range values into plausible-looking measurements rather than
    nulling them (clifpy sets out-of-range to NaN). With `LAB.yaml:46`
    `pao2: [0, 200]`, every hyperoxic ABG is emitted as exactly 200.
13. `report/table1.py:166` — `_stratum` suppresses `"n"` but then publishes exact
    `min`/`max`/`median`/`q25`/`q75` of `age_at_admission`. For a stratum of 4, `n`
    prints `"<10"` while `min`/`max` are two individual patients' exact ages.
    `_categorical_stats` *is* suppressed in the same block.
14. `features/_scope.py:83` — the horizon is `max(feature_cutoff_dttm)` over the whole
    join-ID block, so for one-per-window daily tasks the loading bound is the **last**
    day's prediction time.
15. `_stitch.py:109` — `load_or_build_encounter_index` returns the cached parquet on
    `path.exists()` alone, with no fingerprint of the underlying data. Same shape at
    `_table1.load_or_build_cohort_scores:84`.

### Unranked notes

- `_readmission.py:73` uses polars `.replace()`, which leaves unmatched values
  **unchanged** despite the docstring's "anything not explicitly relevant becomes
  'Other'". Harmless today because `LOCATION_MAPPING` covers the mCIDE set exactly;
  `replace_strict(..., default="Other")` is the fix.
- `meds_etl/domains.py:114` stamps `med_dose_unit_converted = "UNK"` on the
  no-conversion fallback while keeping the raw `med_dose`, collapsing mcg/kg/min and
  mg/hr into one indistinguishable code.
- `cli.py:73` help text says "MIMIC 70/30 split"; the code is 75/25.
- `_split.py:130` — `int(1 * 0.75) == 0`, so a single-block cohort produces zero train
  rows despite the module's "always carries BOTH" contract.
- `report/operating_points.py:595` — `_fairness_by_threshold` recomputes `gap_triplet`
  500 × 99 × n_columns times; measured at 207 s for 20k stays.

### Resolutions

| # | where it lives now | fix |
|---|---|---|
| 1 | `meds_etl/_emit.py` | `MEDS_SCHEMA` ids are `Utf8`; `emit_events` casts to `Utf8`. Matches the ELF schema `features/_config.py` already documented. |
| 2 | `cohort/clif.py` | New `_id_filter()` keeps `None` ("load all") and `[]` ("load nothing") distinct, via a sentinel id that matches no row. Applied at all 9 loader call sites, not just the 4 flagged. |
| 3 | `cohort/events.py` | Steps reordered: merge same-location runs FIRST, then drop sandwiched procedural GROUPS, then re-merge. New `_merge_runs()` helper, called twice. |
| 4 | `concepts/VITAL.yaml` | `height_cm: [70, 255]`, matching clifpy's `vitals_schema`. |
| 5 | `concepts/MED_CON.yaml`, `MED_INT.yaml` | `allow_other_meds: true` — the conversion table stops acting as a vocabulary filter. clifpy already falls back to base units for an unlisted category. Seven MED_CON keys corrected to real mCIDE spellings (`angiotension`→`angiotensin`, `sodium chloride`→`sodium_chloride`, …); the remaining inert keys are documented in place. |
| 6 | `report/_stats.py` | New `suppress_confusion()` applies **secondary suppression**: masking a cell also masks its partner in the same published total, so no cell is recoverable by subtraction. A true zero is exempt and published as 0. Used by `operating_points._confusion_row` and `discrimination._operating_point`; `n_flagged` / `act_rate` / the arm event rates now follow their counts. |
| 7 | `cohort/table1.py` | `_slim()`'s failure path returns the TWO columns its caller joins on. The score cache is additionally pinned to `_SCORES_SCHEMA` on write. |
| 8 | `cohort/clif.py`, `tasks/extubation_failure_24h.py` | Both expand through `members_of_joins()` before loading, so encounter-grain detectors see the whole encounter. |
| 9 | `report/__init__.py` | Explicit `is_nan()` guard with its own error message. |
| 10 | `cohort/events.py` | `first_trach_idx <= first_imv_idx`. A trached patient ventilated through the trach now yields the sentinel row instead of nothing. |
| 11 | `cohort/stitch.py` | `assign_split` decides at `patient_id` grain when the column is present, falling back to the join id. **Re-deals every existing split.** |
| 12 | `meds_etl/_emit.py` | `_numeric_expr` nulls out-of-range values (new `_outside()`); it no longer winsorises them to the limit. |
| 13 | `cohort/table1.py` | `_continuous_stats` returns all-None below `MIN_CELL_COUNT` non-null values, so a stratum can no longer publish two patients' exact ages next to an `n` of `"<10"`. |
| 14 | `features/extractors.py`, `features/_scope.py` | Docs corrected: the bound is per-ENCOUNTER, and neither output is safe to model without the downstream per-row cut. The behaviour was already intentional; the docstring claimed otherwise. |
| 15 | `cohort/stitch.py`, `cohort/table1.py` | New `source_fingerprint()` (path + size + mtime of the source files) is folded into both cache filenames, so re-ETL'd data rebuilds instead of silently reusing yesterday's. |

Unranked notes: `.replace()` → `replace_strict(default="Other")`; the no-conversion
fallback keeps the raw `med_dose_unit` instead of stamping `"UNK"`; the 70/30 help text
and two doc pages corrected to 75/25; `_split_cutoff()` clamps to `[1, n-1]` so a small
cohort cannot get an empty arm; `_fairness_by_threshold` vectorized over the grid via
`fairness.gap_triplet_by_threshold` — **207 s → 2.0 s** at 20k stays, verified
bit-identical to the scalar `gap_triplet` over 300 random cohorts × 99 thresholds.

Suite after the fixes: **189 passed** (169 before, +20 new regression tests). Lint
unchanged at 9 pre-existing violations, none in a touched file.
