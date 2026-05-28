"""End-to-end FLAIR workflow on a single task.

Runs:
  1. flair load-task task1_icu_daily_mortality  -> cohort.parquet
  2. (toy) feature engineering directly from clifpy
  3. sklearn LogisticRegression baseline
  4. append y_pred / y_prob columns to a preds.parquet
  5. flair build-report                         -> report/{table1,tripod_ai,fairness,curves}.json

Run:
  python examples/end_to_end.py --clif-config clif_config.json \
                                --task task1_icu_daily_mortality \
                                --out ./flair_run/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clif-config", required=True)
    ap.add_argument("--task", default="task1_icu_daily_mortality")
    ap.add_argument("--out", default="./flair_run")
    ap.add_argument("--train-end", default=None,
                    help="YYYY-MM-DD (omit for MIMIC 70/30 split)")
    ap.add_argument("--test-start", default=None,
                    help="YYYY-MM-DD (omit for MIMIC 70/30 split)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort_pq = out / "cohort.parquet"
    preds_pq = out / "preds.parquet"
    report_dir = out / "report"

    # 1. load-task via CLI
    cmd = ["flair", "load-task", args.task,
           "--clif-config", args.clif_config, "--out", str(cohort_pq)]
    if args.train_end:
        cmd += ["--train-end", args.train_end]
    if args.test_start:
        cmd += ["--test-start", args.test_start]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2. Load cohort + look up task META
    from flair_benchmark.tasks import get_task
    task_module = get_task(args.task)
    meta = task_module.META
    label_col = meta["label_column"]
    cohort = pl.read_parquet(cohort_pq)
    print(f"Loaded cohort: {cohort.height} rows, label_col={label_col}")

    # 3. Toy features: just age + one-hot demographics (no clifpy joins for this demo).
    #    In real use, you'd call clifpy's ClifOrchestrator.create_wide_dataset()
    #    or load Vitals/Labs/RespiratorySupport directly and aggregate to features.
    feat_df = cohort.to_pandas()
    feat_df = feat_df.join(
        feat_df[["sex_category", "race_category", "ethnicity_category"]]
        .pipe(lambda d: __import__("pandas").get_dummies(d, dummy_na=True))
    )
    X = feat_df.drop(columns=["hospitalization_id", "window_start", "window_end",
                              label_col, "split", "sex_category", "race_category",
                              "ethnicity_category"]).to_numpy().astype(float)
    y = feat_df[label_col].to_numpy()
    is_train = (feat_df["split"] == "train").to_numpy()

    if meta["task_type"] == "binary":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000).fit(X[is_train], y[is_train])
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        preds = cohort.with_columns([
            pl.Series("y_true", y),
            pl.Series("y_pred", y_pred),
            pl.Series("y_prob", y_prob),
        ])
    elif meta["task_type"] == "regression":
        from sklearn.linear_model import Ridge
        model = Ridge().fit(X[is_train], y[is_train])
        y_pred_value = model.predict(X)
        preds = cohort.with_columns([
            pl.Series("y_true", y),
            pl.Series("y_pred_value", y_pred_value),
        ])
    else:  # multiclass
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000, multi_class="multinomial").fit(
            X[is_train], y[is_train])
        proba = model.predict_proba(X)
        cols = [pl.Series("y_true", y), pl.Series("y_pred", model.predict(X))]
        for i, c in enumerate(model.classes_):
            cols.append(pl.Series(f"class_probs_{int(c)}", proba[:, i]))
        preds = cohort.with_columns(cols)

    preds.write_parquet(preds_pq)
    print(f"Wrote preds: {preds_pq}")

    # 4. build-report via CLI
    cmd = ["flair", "build-report", str(preds_pq),
           "--task", args.task, "--out", str(report_dir)]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\nDone. Inspect:")
    for f in sorted(report_dir.glob("*.json")):
        print(f"  {f}")


if __name__ == "__main__":
    sys.exit(main())
