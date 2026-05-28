# FLAIR Examples

## End-to-end run

```bash
python examples/end_to_end.py \
    --clif-config clif_config.json \
    --task task1_icu_daily_mortality \
    --out ./flair_run/
```

This:

1. calls `flair load-task` to produce `./flair_run/cohort.parquet`
2. trains a trivial sklearn baseline on demographics only (you'd swap in real features built with `clifpy`),
3. writes `./flair_run/preds.parquet` with `y_true` / `y_pred` / `y_prob`,
4. calls `flair build-report` to write the four JSONs into `./flair_run/report/`.

All five tasks are binary. The script still switches on `META['task_type']` to support multiclass/regression tasks added later.

## What you actually do

For a real model:

```python
from clifpy import ClifOrchestrator

co = ClifOrchestrator(
    data_directory=cfg["data_directory"],
    filetype=cfg["filetype"],
    timezone=cfg["timezone"],
)
wide = co.create_wide_dataset(
    hospitalization_ids=cohort["hospitalization_id"].to_list(),
    # vitals / labs / respiratory_support categories you want as features
    ...,
)
```

or query individual tables (`Vitals.from_file`, `Labs.from_file`, …). FLAIR does not prescribe feature engineering — that's the user's job.
