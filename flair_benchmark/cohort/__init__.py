"""Everything that turns raw CLIF tables into a task cohort.

This is the first half of FLAIR. A task module composes these pieces to produce
its cohort parquet; `report/` then scores predictions made against it. Nothing in
here reads predictions, and nothing in `report/` reads raw CLIF data.

    clif.py       load CLIF tables through clifpy
    stitch.py     build hospitalization_join_id, and split train/test on it
    encounter.py  the stitched-encounter ICU cohort and its daily grid
    events.py     detect clinical events (ICU readmission, IMV episodes)
    table1.py     cohort characteristics JSON, written beside the parquet

Import the submodule you need directly — there is no re-export here, so the
import site names the file that owns the function:

    from flair_benchmark.cohort.clif import read_clif_config, base_icu_cohort
    from flair_benchmark.cohort.stitch import assign_split, attach_join_id
"""
