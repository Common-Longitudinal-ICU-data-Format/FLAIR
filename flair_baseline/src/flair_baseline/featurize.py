"""Leak-free count featurizer (sparse, integerized, block-chunked).

count[prediction_id, code] = #events of that (level-truncated) code in the same
encounter (``hospitalization_join_id``) with ``time < prediction_dttm`` — strict
point-in-time, no leakage.

Feature granularity — ELF codes are ``//``-delimited hierarchies whose raw depth
varies (``LAB//lactate//mmol/l//bmp`` is 4 levels, ``RESP//device_category//imv`` is
3). Before counting, each code is truncated to a fixed depth by value type so that the
**unit is never a feature**:

* **numeric** codes (any event carries a ``numeric_value``) → **2 levels**
  ``DOMAIN//concept``  (``LAB//lactate``, ``MED_INT//vancomycin``, ``VITAL//heart_rate``)
* **categorical** codes (text-only) → **3 levels**
  ``DOMAIN//category//value``  (``RESP//device_category//imv``, ``CRRT//crrt_mode_category//cvvhdf``)

So every lactate draw counts as one feature regardless of unit (mmol/l vs mg/dl) or
order type, and every vancomycin dose regardless of unit/action — the count answers
"how many lactates / vanc doses before now", not "…in these specific units". The
numeric/categorical split mirrors ``is_numeric_value`` in ``metadata/codes.parquet``.

Scale: the event→prediction join is integerized (truncated codes and encounter blocks
are factorized to int32 first, so the ~100M-row join never carries strings) and then
run in ``n_chunks`` passes partitioned by a block-hash, so peak memory is ~1/n_chunks
of the full aggregation — this keeps the 14M-row sepsis grid under a 16 GB box. The
result is a scipy CSR matrix (dense would be hundreds of GB).

Vocabulary is fit on the train split only, or supplied via ``vocab=`` (scoring a new
site against a saved model). A saved vocab is in the same truncated space.
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl
import scipy.sparse as sp

JOIN_ID = "hospitalization_join_id"

# Truncation depth by value type: numeric ⇒ DOMAIN//concept, categorical ⇒
# DOMAIN//category//value. Dropping deeper levels removes the unit (and order/action)
# from the feature space.
NUMERIC_LEVELS = 2
CATEGORICAL_LEVELS = 3


def truncated_code_map(ev: pl.LazyFrame) -> pl.DataFrame:
    """Map every raw code → (truncated code, ``cint`` feature id).

    A code is *numeric* if any of its events has a non-null ``numeric_value`` (the same
    rule ``codes.parquet`` uses for ``is_numeric_value``); numeric codes keep the first
    ``NUMERIC_LEVELS`` ``//`` levels, the rest keep ``CATEGORICAL_LEVELS``. The truncated
    codes are factorized to a contiguous ``cint`` so the heavy join stays integerized and
    raw codes that collapse to the same feature share a ``cint`` (their counts merge).

    Returns a frame with columns ``code`` (raw), ``trunc``, ``cint`` — one row per raw code.
    """
    code_meta = (
        ev.select("code", "numeric_value")
        .group_by("code")
        .agg(pl.col("numeric_value").is_not_null().any().alias("is_numeric"))
        .collect(engine="streaming")
    )
    parts = pl.col("code").str.split("//")
    trunc = (
        pl.when(pl.col("is_numeric"))
        .then(parts.list.slice(0, NUMERIC_LEVELS).list.join("//"))
        .otherwise(parts.list.slice(0, CATEGORICAL_LEVELS).list.join("//"))
        .alias("trunc")
    )
    code_meta = code_meta.with_columns(trunc)
    trunc_ids = (
        code_meta.select("trunc").unique().sort("trunc").with_row_index("cint")
    )
    return code_meta.join(trunc_ids, on="trunc").select("code", "trunc", "cint")


def count_features(events, task_df: pl.DataFrame, label_col: str,
                   vocab: list[str] | None = None, n_chunks: int = 8
                   ) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Return (X_csr, prediction_ids, vocab). Row order = prediction_ids.

    The vocabulary is every code in the MEDS table (fit on the train split, or
    supplied via ``vocab=``). Extraction is the single source of truth: whatever the
    ELF config extracted into MEDS is exactly what can be featured — there is no
    feature-time exclusion. To drop a domain, remove it from flair_elf_config.yaml.
    """
    ev = events if isinstance(events, pl.LazyFrame) else events.lazy()

    preds = (
        task_df.select("prediction_id", JOIN_ID,
                       pl.col("prediction_dttm").cast(pl.Datetime("us")), "split")
        .with_row_index("r")
    )
    prediction_ids = preds["prediction_id"].to_list()
    n_rows = preds.height

    # Factorize blocks (+ a hash chunk) and codes to int32 so the join stays lean.
    blocks = (
        preds.select(JOIN_ID).unique()
        .with_row_index("jint")
        .with_columns((pl.col("jint") % n_chunks).alias("chunk"))
    )
    # Truncate codes by value type (numeric ⇒ 2 levels, categorical ⇒ 3) and factorize
    # the truncated codes → cint. Raw codes that share a truncated code share a cint, so
    # the per-(r, cint) aggregation below merges their counts (e.g. lactate in any unit).
    code_map = truncated_code_map(ev)                       # code → trunc → cint
    trunc_by_cint = (
        code_map.select("cint", "trunc").unique().sort("cint")["trunc"].to_list()
    )
    n_codes = len(trunc_by_cint)
    raw_to_cint = code_map.select("code", "cint")           # raw code → feature id

    ev_int = (
        ev.select(JOIN_ID, pl.col("time").cast(pl.Datetime("us")), "code")
        .join(blocks.lazy(), on=JOIN_ID, how="inner")       # drops events outside cohort
        .join(raw_to_cint.lazy(), on="code", how="inner")
        .select("jint", "chunk", "time", "cint")
    )
    preds_int = (
        preds.select("r", JOIN_ID, "prediction_dttm")
        .join(blocks, on=JOIN_ID)
        .select("r", "jint", "chunk", "prediction_dttm")
        .lazy()
    )

    # Aggregate counts one block-hash partition at a time → bounded peak memory.
    r_parts, c_parts, n_parts = [], [], []
    for k in range(n_chunks):
        longk = (
            ev_int.filter(pl.col("chunk") == k)
            .join(preds_int.filter(pl.col("chunk") == k), on="jint", how="inner")
            .filter(pl.col("time") < pl.col("prediction_dttm"))   # strict point-in-time
            .group_by("r", "cint")
            .agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        r_parts.append(longk["r"].to_numpy())
        c_parts.append(longk["cint"].to_numpy())
        n_parts.append(longk["n"].to_numpy().astype("float32"))
        del longk
        gc.collect()

    rr = np.concatenate(r_parts) if r_parts else np.empty(0, "uint32")
    cc = np.concatenate(c_parts) if c_parts else np.empty(0, "uint32")
    nn = np.concatenate(n_parts) if n_parts else np.empty(0, "float32")
    del r_parts, c_parts, n_parts
    gc.collect()

    # Resolve vocabulary → a cint → column-index lookup array. Vocab entries are
    # truncated codes, aligned to the cint factorization above.
    if vocab is None:
        train_r = preds.filter(pl.col("split") == "train")["r"].to_numpy()
        is_train = np.zeros(n_rows, dtype=bool)
        is_train[train_r] = True
        train_cints = np.unique(cc[is_train[rr]]) if rr.size else np.empty(0, "int64")
        order = sorted(train_cints.tolist(), key=lambda ci: trunc_by_cint[ci])
        vocab = [trunc_by_cint[ci] for ci in order]
        col_of = np.full(n_codes, -1, dtype="int64")
        for col, ci in enumerate(order):
            col_of[ci] = col
    else:
        name_to_cint = {trunc: ci for ci, trunc in enumerate(trunc_by_cint)}
        col_of = np.full(n_codes, -1, dtype="int64")
        for col, code in enumerate(vocab):
            ci = name_to_cint.get(code)
            if ci is not None:
                col_of[ci] = col

    if not vocab:
        return sp.csr_matrix((n_rows, 0), dtype="float32"), prediction_ids, []

    cols = col_of[cc]
    keep = cols >= 0
    X = sp.csr_matrix(
        (nn[keep], (rr[keep], cols[keep])),
        shape=(n_rows, len(vocab)), dtype="float32",
    )
    return X, prediction_ids, vocab
