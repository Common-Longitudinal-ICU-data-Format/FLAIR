# Task 7: ICU Readmission Prediction - Complete Flow

## What is Being Predicted?

**Unplanned ICU Readmission** = Will the patient return to ICU during the **same hospitalization** after being discharged from their first ICU stay?

-   **Label = 1**: Patient returns to ICU (unplanned)
-   **Label = 0**: Patient does NOT return to ICU

------------------------------------------------------------------------

## The Flow (Visual Diagram)

```         
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAW CLIF DATA                                     │
│  (ADT table, Hospitalization table, Patient table)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STEP 1: BUILD BASE COHORT (FLAIRCohortBuilder)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Load ADT → Filter to ICU locations → Get all hospitalizations with ICU   │
│  • Load Hospitalization table (with patient demographics)                   │
│  • Apply base filters:                                                      │
│    - age >= 18                                                              │
│    - valid admission/discharge dates                                        │
│    - hospital stay > 0 days                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            STEP 2: COMPUTE UNPLANNED ICU TIMING (The Magic!)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw location sequence per patient:                                        │
│   ┌─────┐ ┌──────────┐ ┌─────┐ ┌──────┐ ┌─────┐                            │
│   │ ICU │→│Procedural│→│ ICU │→│ Ward │→│ ICU │  (example patient)         │
│   └─────┘ └──────────┘ └─────┘ └──────┘ └─────┘                            │
│                                                                             │
│   2a. Map locations → ICU, Ward, ER, Procedural, Other                     │
│   2b. Keep ALL locations (including Other) for readmission detection       │
│                                                                             │
│   2c. REMOVE "PROCEDURAL SANDWICHES":                                       │
│       ICU → Procedural → ICU  becomes  ICU → ICU (procedural removed!)     │
│       ┌─────┐ ┌──────────┐ ┌─────┐           ┌─────┐ ┌─────┐              │
│       │ ICU │→│Procedural│→│ ICU │    →→→    │ ICU │→│ ICU │              │
│       └─────┘ └──────────┘ └─────┘           └─────┘ └─────┘              │
│                                                                             │
│   2d. MERGE CONSECUTIVE SAME-LOCATION:                                      │
│       ICU → ICU  becomes  ICU (single stay!)                               │
│       ┌─────┐ ┌─────┐           ┌───────────────┐                          │
│       │ ICU │→│ ICU │    →→→    │  Single ICU   │                          │
│       └─────┘ └─────┘           └───────────────┘                          │
│                                                                             │
│   After processing the example patient:                                     │
│   ┌───────────────┐ ┌──────┐ ┌─────┐                                       │
│   │ First ICU     │→│ Ward │→│ ICU │  ← This is UNPLANNED readmission!    │
│   │ (merged stays)│ └──────┘ └─────┘    (Label = 1)                        │
│   └───────────────┘                                                         │
│                                                                             │
│   OUTPUT: first_icu_start, first_icu_end, second_icu_start (if exists)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 3: EXCLUSION FILTERS (5 filters)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ Filter 1: No ICU data → EXCLUDE                                        │
│                                                                             │
│   ❌ Filter 2: Died during first ICU stay → EXCLUDE                         │
│      (dead patients can't be readmitted)                                    │
│                                                                             │
│   ❌ Filter 3: Direct ICU discharge → EXCLUDE                               │
│      (patient went home from ICU, no ward time, no opportunity)             │
│                                                                             │
│   ❌ Filter 4: First ICU stay < 1 hour → EXCLUDE                            │
│      (too short, likely transfer/error)                                     │
│                                                                             │
│   ❌ Filter 5: Has intermediate ER visit → EXCLUDE                          │
│      (ICU → ER → ICU complicates analysis)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: BUILD LABELS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each remaining patient:                                               │
│                                                                             │
│   if second_icu_start_time EXISTS:                                          │
│       label_icu_readmission = 1  (✓ READMITTED)                            │
│   else:                                                                     │
│       label_icu_readmission = 0  (✗ NOT readmitted)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: BUILD TIME WINDOWS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║  Unlike Tasks 5 & 6 (fixed 24hr window), Task 7 uses the          ║    │
│   ║  ENTIRE FIRST ICU STAY as the observation window!                 ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│                                                                             │
│   window_start = first_icu_start_time                                       │
│   window_end   = first_icu_end_time                                         │
│                                                                             │
│   Timeline:                                                                 │
│   ─────────────────────────────────────────────────────────────────────    │
│   │◄──────── OBSERVATION WINDOW ────────►│                                 │
│   │    (collect all features here)       │                                 │
│   │                                      │                                 │
│   ICU                                   ICU         Ward        ICU         │
│   ADMISSION                          DISCHARGE    (maybe)   READMISSION?   │
│   (window_start)                    (window_end)           (label=1 if yes)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 6: TEMPORAL TRAIN/TEST SPLIT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Based on admission_dttm:                                                  │
│   • Train: 2020-01-01 to 2022-12-31 → split = "train"                      │
│   • Test:  2023-01-01 to 2023-12-31 → split = "test"                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT DATASET                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   13 columns:                                                               │
│   ┌────────────────────┬─────────────────────────────────────────────────┐ │
│   │ hospitalization_id │ Unique ID                                       │ │
│   │ admission_dttm     │ Hospital admission time                         │ │
│   │ discharge_dttm     │ Hospital discharge time                         │ │
│   │ window_start       │ First ICU start (features start here)          │ │
│   │ window_end         │ First ICU end (features end here)              │ │
│   │ label_icu_readmission │ 0 or 1 (THE TARGET!)                        │ │
│   │ split              │ "train" or "test"                               │ │
│   │ age_at_admission   │ Patient age                                     │ │
│   │ sex_category       │ M/F                                             │ │
│   │ race_category      │ Race                                            │ │
│   │ ethnicity_category │ Ethnicity                                       │ │
│   │ hospital_id        │ Which hospital                                  │ │
│   │ hospital_type      │ Hospital type                                   │ │
│   └────────────────────┴─────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

## Key Insight: Planned vs Unplanned

The **magic** of Task 7 is distinguishing PLANNED from UNPLANNED readmissions:

```         
PLANNED (Label = 0):
ICU → Procedural → ICU
      (surgery)
These get MERGED into ONE ICU stay. Not counted as readmission.

UNPLANNED (Label = 1):
ICU → Ward → ICU
      (patient got worse)
This IS a readmission. The patient was discharged to ward and deteriorated.

ICU → Other → ICU
      (stepdown, hospice, psych, rehab, radiology, dialysis)
This IS ALSO a readmission. Any non-procedural intermediate location counts.
```

------------------------------------------------------------------------

## Comparison with Other Tasks

|   | Task 5 (LOS) | Task 6 (Mortality) | **Task 7 (Readmission)** |
|------------------|------------------|------------------|------------------|
| **Window** | First 24 hours | First 24 hours | **Entire 1st ICU stay** |
| **Min stay** | \>= 24 hours | \>= 24 hours | **\>= 1 hour** |
| **Target** | Remaining hours | Death (0/1) | **Readmission (0/1)** |
| **Special** | None | Uses discharge_category | **Merges procedural visits** |

------------------------------------------------------------------------

## Location Mapping

The task maps CLIF location categories to simplified groups:

| CLIF Location | Mapped To  | Included?              |
|---------------|------------|------------------------|
| `icu`         | ICU        | Yes                    |
| `ward`        | Ward       | Yes                    |
| `ed`          | ER         | Yes                    |
| `procedural`  | Procedural | Yes (special handling) |
| `stepdown`    | Other      | No                     |
| `hospice`     | Other      | No                     |
| `psych`       | Other      | No                     |
| `rehab`       | Other      | No                     |
| `radiology`   | Other      | No                     |
| `dialysis`    | Other      | No                     |

------------------------------------------------------------------------

## Code References

-   **Task implementation**: `flair/tasks/task7_icu_readmission.py`
-   **Base task class**: `flair/tasks/base.py`
-   **Cohort builder**: `flair/cohort/builder.py`

### Key Methods

| Method | Location | Purpose |
|----------------------|---------------------------|------------------------|
| `_compute_unplanned_icu_timing()` | task7:90-279 | Core logic for merging procedural visits |
| `filter_cohort()` | task7:303-395 | Apply 5 exclusion filters |
| `build_labels()` | task7:468-507 | Create binary readmission labels |
| `build_time_windows()` | task7:433-466 | Define observation window (entire ICU stay) |

------------------------------------------------------------------------

## Usage Example

``` python
from flair_benchmark import generate_task_dataset

df = generate_task_dataset(
    config_path="clif_config.json",
    task_name="task7_icu_readmission",
    train_start="2020-01-01",
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2023-12-31",
)
```