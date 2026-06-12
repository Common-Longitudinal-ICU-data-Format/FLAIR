"""XGBoost training + scoring for one task (sparse features).

Fits on the train split, scores every row (train+test) so the FLAIR report can bin
by split. Demographics flow into the report via the cohort parquet, so the preds
table carries only the report-required keys + predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import xgboost as xgb

JOIN_ID = "hospitalization_join_id"

# Extra dense numeric features pulled from the cohort (appended after the count columns).
_EXTRA_FEATURES = ["age_at_admission"]


def train_and_score(X: sp.csr_matrix, prediction_ids: list[str], vocab: list[str],
                    task_df: pl.DataFrame, label_col: str, *,
                    model_out: str | None = None, vocab_out: str | None = None,
                    model_in: str | None = None) -> pl.DataFrame:
    """Train (unless model_in given) and score every prediction. Returns the preds frame."""
    extras = [c for c in _EXTRA_FEATURES if c in task_df.columns]
    meta = task_df.select("prediction_id", "hospitalization_id", JOIN_ID,
                          "split", label_col, *extras)
    # Align cohort rows to X's row order (prediction_ids).
    order = pl.DataFrame({"prediction_id": prediction_ids}).with_row_index("_r")
    d = order.join(meta, on="prediction_id", how="left").sort("_r")

    if extras:
        age = d.select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in extras]).to_numpy()
        X_full = sp.hstack([X, sp.csr_matrix(age.astype("float32"))], format="csr")
    else:
        X_full = X.tocsr()

    y = d[label_col].cast(pl.Int32).fill_null(0).to_numpy()
    split = d["split"].to_numpy()
    train_mask = split == "train"

    if model_in:
        booster = xgb.Booster()
        booster.load_model(model_in)
        prob = booster.inplace_predict(X_full)
    elif train_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2 or X_full.shape[1] == 0:
        base = float(y[train_mask].mean()) if train_mask.sum() else 0.0
        prob = np.full(X_full.shape[0], base, dtype="float32")
    else:
        pos = float((y[train_mask] == 1).sum())
        neg = float((y[train_mask] == 0).sum())
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
            reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
            scale_pos_weight=(neg / pos if pos > 0 else 1.0),
            n_jobs=-1, random_state=0,
        )
        clf.fit(X_full[train_mask], y[train_mask])
        prob = clf.predict_proba(X_full)[:, 1]
        if model_out:
            clf.get_booster().save_model(model_out)
        if vocab_out:
            Path(vocab_out).write_text(json.dumps({"vocab": vocab, "extras": extras}))

    return d.select(
        "prediction_id", "hospitalization_id", JOIN_ID, "split", label_col,
    ).with_columns(
        pl.Series("y_prob", prob).cast(pl.Float64),
        (pl.Series("y_prob", prob) >= 0.5).cast(pl.Int32).alias("y_pred"),
    )
