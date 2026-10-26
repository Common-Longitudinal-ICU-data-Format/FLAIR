"""Table 1 — cohort characteristics, overall and stratified by label and split.

Written next to the cohort parquet at `flair load-task` time and consumed by the
FLAIR website (`r.table1` in flair_website/static/js/report.js). This is NOT
report-time code: it never sees predictions, only the generated cohort.

The whole job in three parts, which is why they share a file:

  Part 1  generate_table1()  — the public entry point; joins scores, calls the
                               builder, stamps audit metadata
  Part 2  scores             — SOFA max + Charlson/Elixhauser, and their cache
  Part 3  build_table1()     — pure cohort frame -> dict, no I/O

The non-obvious constraint is GRAIN. A cohort row is one *prediction*, but the
headline numbers must read as patients, so the flat top-level keys are at
ENCOUNTER grain (one row per hospitalization_join_id) and `by_grain` carries both
the encounter and per-prediction-row strata explicitly. Reporting the flat keys at
row grain would multiply-count any patient with several prediction windows.

The second constraint is DISCLOSURE. Every published count passes through
`suppress()`, and `_prevalence` additionally returns None whenever prevalence x n
would recover a suppressed cell.

Score failures are non-fatal by design: a site missing the tables clifpy needs
still gets a Table 1 built from the demographics it has, with the severity and
comorbidity columns simply absent.

Output JSON shape:
  {
    "n_total": int,            # encounters (flat keys are encounter-grain)
    "n_rows":  int,            # prediction rows
    "prevalence": float|None,  # binary: positive rate (6 dp). None when either the
                               # positive or negative count is a nonzero value < 10
                               # (prevalence x n would recover a suppressed cell);
                               # true extremes 0.0 / 1.0 are shown. Multiclass:
                               # {level: proportion_or_None}; regression: None.
    "continuous":  { "age_at_admission": {mean, std, median, q25, q75, min, max, n_missing},
                     "sofa_max": {...}, "charlson_score": {...}, "elix_score": {...} },
                   # every stat is None where the stratum has < 10 non-null
                   # values; n_missing is 0 or a suppressed count.
    "categorical": { "sex_category": {level: count_or_"<10"}, ... },
    "by_split":  { "total": {n, date_range, prevalence, continuous, categorical},
                   "train": {...}, "test": {...} },   # ALWAYS 3 keys; empty -> n=0
                   # date_range = {"start": "Jan 2020", "end": "Mar 2021"}
    "temporal_overlap": { test_start, train_end, overlap_days,
                          train_rows_after_test_start, train_frac_after_test_start },
                   # how far train reaches into the test period — the mass, which
                   # date_range's min/max cannot show. See _temporal_overlap.
    "by_label":  { "1": {...positive...}, "0": {...negative...} },
    "by_grain":  { "encounter": {n_total, continuous, categorical, by_split, by_label},
                   "row":       {... same shape at prediction-row grain ...} },
    "metadata":  { ... audit fields stamped by generate_table1, incl.
                   cohort_sha256 / train_cohort_sha256 / test_cohort_sha256 ... }
    "attrition": [ {step, n_rows, n_encounters, ...}, ... ]   # STROBE flow, added
                   # by cli.py load-task from cohort/strobe.py; suppressed
  }

Depends on: cohort/clif.py, cohort/stitch.py, constants.py, report/_stats.py
            (suppression policy only), clifpy (SOFA + comorbidity backends).
Used by: cli.py `load-task`.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from flair_benchmark.cohort.clif import load_hospital_diagnosis, load_hospitalization
from flair_benchmark.cohort.stitch import JOIN_ID, _cache_dir, source_fingerprint
from flair_benchmark.constants import task_script_sha256
from flair_benchmark.report._stats import MIN_CELL_COUNT, suppress, suppress_dict

CONTINUOUS_VARS = ["age_at_admission", "sofa_max", "charlson_score", "elix_score"]
CATEGORICAL_VARS = ["sex_category", "race_category", "ethnicity_category"]
# Date column used for the per-split calendar range (month-year only). First present wins.
DATE_COLS = ["admission_dttm", "prediction_dttm"]


def _warn(msg: str) -> None:
    print(f"[flair table1] WARN: {msg}", file=sys.stderr)


# ===========================================================================
# Part 1 — Public entry point
# ===========================================================================


def generate_table1(cfg: dict, cohort: pl.DataFrame, task_module) -> dict:
    """Build the Table 1 dict for a generated task cohort.

    Covers overall + by_label + by_split, at both encounter and row grain, with
    severity/comorbidity joined on and an audit `metadata` stamp.

    A cohort carries demographics but not severity or comorbidity, so those are
    computed here (and cached per site) before the table is built. Any failure in
    that step is swallowed — Table 1 must never break cohort generation.
    """
    from flair_benchmark import __version__

    meta = task_module.META
    cohort_x = cohort
    if "hospitalization_id" in cohort.columns and cohort.height:
        hosp_ids = cohort["hospitalization_id"].cast(pl.Utf8).unique().to_list()
        try:
            scores = load_or_build_cohort_scores(cfg, hosp_ids)
            cohort_x = cohort.with_columns(
                pl.col("hospitalization_id").cast(pl.Utf8)
            ).join(scores, on="hospitalization_id", how="left")
        except Exception as exc:  # noqa: BLE001 — table1 must not break generation
            _warn(f"cohort scores skipped: {exc}")

    t1 = build_table1(cohort_x, meta)
    t1["metadata"] = {
        "task": meta["name"],
        "task_script_sha256": task_script_sha256(task_module),
        "flair_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_cell_count": MIN_CELL_COUNT,
        "cohort_sha256": cohort_sha256(cohort, meta["label_column"]),
        "train_cohort_sha256": cohort_sha256(
            cohort.filter(pl.col("split") == "train"), meta["label_column"]),
        "test_cohort_sha256": cohort_sha256(
            cohort.filter(pl.col("split") == "test"), meta["label_column"]),
    }
    return t1


HASH_COLS = ["hospitalization_join_id", "prediction_dttm", "feature_cutoff_dttm",
             "label", "split"]


def cohort_sha256(cohort: pl.DataFrame, label_col: str) -> str:
    """SHA-256 fingerprint of a cohort: same rows/labels/split -> same hash, at any site.

    Rows sorted by (hospitalization_join_id, prediction_dttm); the task's label
    column is renamed to "label" and every value cast to its string form (ISO
    datetimes), so the digest is independent of row order, dtype and time unit.
    """
    df = (cohort.rename({label_col: "label"}).select(HASH_COLS)
          .sort(["hospitalization_join_id", "prediction_dttm"])
          .with_columns(pl.all().cast(pl.Utf8)))
    return hashlib.sha256(df.write_csv().encode()).hexdigest()


# ===========================================================================
# Part 2 — Severity and comorbidity scores (with a per-site cache)
#
# These are the heavy part of Table 1 (clifpy SOFA over raw files, plus
# HospitalDiagnosis), so results are cached per site to
# <cache_dir>/cohort_scores.parquet — the same build-once-reuse pattern as the
# encounter index — and reused across all tasks.
# ===========================================================================

_SCORES_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "sofa_max": pl.Float64,
    "charlson_score": pl.Float64,
    "elix_score": pl.Float64,
}

_COMORBIDITY_SCHEMA = {
    "hospitalization_id": pl.Utf8,
    "charlson_score": pl.Float64,
    "elix_score": pl.Float64,
}


def _scores_cache_path(cfg: dict) -> Path:
    # Fingerprinted like the encounter index: SOFA and the comorbidity scores are
    # derived from the raw CLIF files, so a re-ETL must not be answered out of a
    # cache built from the previous extract.
    return _cache_dir(cfg) / f"cohort_scores_{source_fingerprint(cfg)}.parquet"


def load_or_build_cohort_scores(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """Per-site cached severity/comorbidity scores.

    Computes only the hospitalization_ids not already cached, appends them, and
    returns the requested subset.
    """
    hosp_ids = [str(h) for h in hosp_ids]
    path = _scores_cache_path(cfg)
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=_SCORES_SCHEMA)

    cached = set(have["hospitalization_id"].to_list()) if have.height else set()
    missing = [h for h in hosp_ids if h not in cached]
    if missing:
        new = _compute_scores(cfg, missing)
        # Pin the cache to the declared schema. Without the select, any extra
        # column a backend happens to emit is carried by diagonal_relaxed into
        # the parquet and never leaves again.
        have = (
            pl.concat([have, new], how="diagonal_relaxed")
            .select(list(_SCORES_SCHEMA))
            .unique(subset=["hospitalization_id"], keep="first")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        have.write_parquet(path)

    return have.filter(pl.col("hospitalization_id").is_in(hosp_ids))


def _compute_scores(cfg: dict, hosp_ids: list[str]) -> pl.DataFrame:
    """[hospitalization_id, sofa_max, charlson_score, elix_score] for hosp_ids.

    Each score is best-effort: a failure leaves that column null rather than
    raising, so one unavailable clifpy backend cannot block the whole table.
    """
    base = (
        load_hospitalization(
            cfg, hospitalization_ids=hosp_ids,
            columns=["hospitalization_id", "admission_dttm", "discharge_dttm"],
        )
        .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
    )

    try:
        sofa = compute_hospitalization_sofa_max(cfg, base)
    except Exception as exc:  # noqa: BLE001
        _warn(f"SOFA unavailable: {exc}")
        sofa = pl.DataFrame(schema={"hospitalization_id": pl.Utf8, "sofa_max": pl.Float64})

    try:
        comorb = compute_cci_elix(cfg, hosp_ids)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Charlson/Elixhauser unavailable: {exc}")
        comorb = pl.DataFrame(schema=_COMORBIDITY_SCHEMA)

    return (
        base.select("hospitalization_id")
        .join(sofa.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
              on="hospitalization_id", how="left")
        .join(comorb.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
              on="hospitalization_id", how="left")
    )


def compute_hospitalization_sofa_max(cfg: dict,
                                     cohort: pl.DataFrame) -> pl.DataFrame:
    """Return [hospitalization_id, sofa_max] across the full hospitalization.

    Scored over the whole `[admission_dttm, discharge_dttm]` window — worst value
    in that window — so it is independent of which task is being summarized.
    Uses clifpy's polars-native SOFA backend, which reads the raw files directly
    and handles vasopressor unit conversion and the respiratory waterfall itself.

    cohort: pl.DataFrame with [hospitalization_id, admission_dttm, discharge_dttm].
    """
    if cohort.height == 0:
        return pl.DataFrame(
            schema={"hospitalization_id": pl.Utf8, "sofa_max": pl.Float64}
        )

    from clifpy.utils.sofa_polars import compute_sofa_polars

    cohort_in = cohort.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("admission_dttm").alias("start_dttm"),
        pl.col("discharge_dttm").alias("end_dttm"),
    ])

    sofa = compute_sofa_polars(
        data_directory=cfg["data_directory"],
        cohort_df=cohort_in,
        filetype=cfg["filetype"],
        id_name="hospitalization_id",
        extremal_type="worst",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        timezone=cfg["timezone"],
    )

    return sofa.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("sofa_total").alias("sofa_max"),
    ])


def compute_cci_elix(cfg: dict, hospitalization_ids: list[str]) -> pl.DataFrame:
    """Return one row per hospitalization with charlson_score + elix_score.

    Both scores derive from the ICD codes on the hospitalization record, so the
    result is fixed at admission — there is no temporal leakage. A hospitalization
    with no ICD codes comes through as null and is handled by the summarizer.
    """
    from clifpy.utils.comorbidity import calculate_cci, calculate_elix

    if not hospitalization_ids:
        return _empty_comorbidity_frame()

    dx = load_hospital_diagnosis(cfg, hospitalization_ids=hospitalization_ids)
    if dx.height == 0:
        return _empty_comorbidity_frame()

    cci = calculate_cci(dx, hierarchy=True)
    elix = calculate_elix(dx, hierarchy=True)

    cci_slim = _slim(cci, "cci_score", "charlson_score")
    elix_slim = _slim(elix, "elix_score", "elix_score")

    # how="full" + coalesce: polars' full join does NOT coalesce keys by default,
    # so elix-only rows would get a null hospitalization_id (their score dropped
    # by downstream joins) plus a stray hospitalization_id_right column leaking
    # into the cohort_scores cache.
    return cci_slim.join(elix_slim, on="hospitalization_id", how="full",
                         coalesce=True)


def _slim(df, score_col: str, out_name: str) -> pl.DataFrame:
    """Reduce a clifpy comorbidity output to [hospitalization_id, <out_name>].

    Accepts both polars and pandas DataFrames (clifpy returns pandas). The
    failure path must return those TWO columns and no more: handing back the
    full three-column comorbidity schema makes the caller's join see
    charlson_score (or elix_score) on both sides, so polars suffixes one to
    `..._right` — and that stray column is then written permanently into the
    cohort_scores.parquet cache.
    """
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    if "hospitalization_id" not in df.columns or score_col not in df.columns:
        return pl.DataFrame(schema={"hospitalization_id": pl.Utf8,
                                    out_name: pl.Float64})
    return df.select([
        "hospitalization_id",
        pl.col(score_col).cast(pl.Float64).alias(out_name),
    ])


def _empty_comorbidity_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_COMORBIDITY_SCHEMA)


# ===========================================================================
# Part 3 — The table builder (pure: frame in, dict out, no I/O)
# ===========================================================================


def build_table1(cohort: pl.DataFrame, meta: dict) -> dict:
    """Compute the table1 dict from a cohort frame.

    Flat keys are encounter-grain; `by_grain` exposes both the encounter and
    prediction-row strata. The severity/comorbidity continuous vars appear only
    when those columns were joined on upstream — this function works with or
    without them.
    """
    label_col = meta["label_column"]
    task_type = meta["task_type"]

    enc = _to_encounter(cohort, label_col, task_type)

    out = _block(enc, label_col, task_type)      # flat headline = encounter grain
    out["n_rows"] = cohort.height
    out["by_grain"] = {
        "encounter": _block(enc, label_col, task_type),
        "row": _block(cohort, label_col, task_type),
    }
    return out


def _to_encounter(cohort: pl.DataFrame, label_col: str, task_type: str) -> pl.DataFrame:
    """One row per stitched encounter.

    The encounter's label is ever-positive (max over its rows) for binary tasks;
    covariates are constant per encounter, so `first()` is safe for them.
    """
    if JOIN_ID not in cohort.columns:
        raise ValueError(
            "Table 1 requires hospitalization_join_id so headline counts are at "
            "stitched-encounter grain. Add hospitalization_join_id to the cohort "
            "instead of falling back to hospitalization_id or prediction rows."
        )
    date_col = _first_date_col(cohort)
    keep = [c for c in (CONTINUOUS_VARS + CATEGORICAL_VARS + ["split"]
                        + ([date_col] if date_col else []))
            if c in cohort.columns]
    aggs = [pl.col(c).first() for c in keep]
    if label_col in cohort.columns:
        label_agg = (pl.col(label_col).max() if task_type == "binary"
                     else pl.col(label_col).first())
        aggs = [label_agg.alias(label_col), *aggs]
    return cohort.group_by(JOIN_ID).agg(aggs)


def _first_date_col(df: pl.DataFrame) -> str | None:
    return next((c for c in DATE_COLS if c in df.columns), None)


def _date_range(df: pl.DataFrame) -> dict:
    """Calendar span of a stratum as month-year strings (e.g. 'Jan 2020'), no day."""
    col = _first_date_col(df)
    if col is None or df.height == 0:
        return {"start": None, "end": None}
    s = df[col].drop_nulls()
    if s.len() == 0:
        return {"start": None, "end": None}
    return {"start": s.min().strftime("%b %Y"), "end": s.max().strftime("%b %Y")}


def _temporal_overlap(df: pl.DataFrame) -> dict:
    """How far the train arm reaches past the start of the test arm.

    `by_split[*].date_range` publishes only min and max, so a reader can see THAT
    the arms overlap but not how much of train sits inside the test period — one
    late-returning encounter moves the extreme just as far as half the cohort
    would. This block carries the mass, which is the number that says whether a
    temporal split actually held.

    Expect a small nonzero overlap even on a correct split: a stay open across the
    boundary belongs wholly to one arm (the encounter is the split grain — see
    cohort/stitch.py), so its later rows land past `test_start` legitimately. Days
    to a few weeks is normal; months means the deal was made at the wrong grain.

    Meaningless for `site="mimic"`, whose dates are de-identified/shifted and whose
    deal is random rather than temporal — it is reported anyway rather than
    special-cased, since table1 does not know the site.

    Suppression: `train_rows_after_test_start` is a raw count, so it is masked
    below MIN_CELL_COUNT, and the fraction goes with it (frac x n_train recovers
    the count). A true zero is not PHI and is reported as 0.
    """
    empty = {"test_start": None, "train_end": None, "overlap_days": None,
             "train_rows_after_test_start": None,
             "train_frac_after_test_start": None}
    col = _first_date_col(df)
    if col is None or "split" not in df.columns or df.height == 0:
        return empty

    train = df.filter(pl.col("split") == "train")[col].drop_nulls()
    test = df.filter(pl.col("split") == "test")[col].drop_nulls()
    if train.len() == 0 or test.len() == 0:
        return empty

    test_start, train_end = test.min(), train.max()
    n_after = int((train >= test_start).sum())
    disclosable = n_after == 0 or n_after >= MIN_CELL_COUNT
    return {
        "test_start": test_start.strftime("%b %Y"),
        "train_end": train_end.strftime("%b %Y"),
        "overlap_days": max(0, (train_end - test_start).days),
        "train_rows_after_test_start": n_after if disclosable else suppress(n_after),
        "train_frac_after_test_start": (round(n_after / train.len(), 6)
                                        if disclosable else None),
    }


def _prevalence(df: pl.DataFrame, label_col: str | None, task_type: str | None):
    """Outcome prevalence of a stratum, suppression-aware.

    Disclosable only if the underlying label counts would survive `suppress()`:
    a published prevalence times the stratum's n recovers the positive count (and
    1-p the negative), so any nonzero count below MIN_CELL_COUNT forces None.
    True extremes (0.0 / 1.0) are shown — a true zero is not PHI.
    """
    if label_col is None or label_col not in df.columns:
        return None
    s = df[label_col].drop_nulls()
    n = s.len()
    if n == 0:
        return None
    if task_type == "binary":
        n_pos = int((s > 0).sum())
        n_neg = n - n_pos
        if 0 < n_pos < MIN_CELL_COUNT or 0 < n_neg < MIN_CELL_COUNT:
            return None
        return round(n_pos / n, 6)
    if task_type == "multiclass":
        counts = s.value_counts()
        return {
            str(level): (round(cnt / n, 6) if cnt >= MIN_CELL_COUNT else None)
            for level, cnt in counts.iter_rows()
        }
    return None


def _block(df: pl.DataFrame, label_col: str, task_type: str) -> dict:
    out: dict = {
        "n_total": df.height,
        "prevalence": _prevalence(df, label_col, task_type),
        "continuous": {v: _continuous_stats(df, v) for v in CONTINUOUS_VARS
                       if v in df.columns},
        "categorical": {v: _categorical_stats(df, v) for v in CATEGORICAL_VARS
                        if v in df.columns},
        "by_split": {},
        "by_label": {},
    }
    # Always emit the same 3 columns — total / train / test — so every site
    # (including inference-only, where only 'test' exists) produces an identical
    # schema for cross-site aggregation. Missing splits are zero-filled by
    # calling _stratum on an empty frame.
    has_split = "split" in df.columns
    out["by_split"] = {
        "total": _stratum(df, label_col, task_type),
        "train": _stratum(df.filter(pl.col("split") == "train"), label_col, task_type)
                 if has_split else _stratum(df.head(0), label_col, task_type),
        "test":  _stratum(df.filter(pl.col("split") == "test"), label_col, task_type)
                 if has_split else _stratum(df.head(0), label_col, task_type),
    }
    # Sibling of by_split, not a member of it: by_split's three strata must keep an
    # identical schema for cross-site aggregation, and this describes the deal
    # between them rather than any one arm.
    out["temporal_overlap"] = _temporal_overlap(df)

    if label_col not in df.columns:
        return out

    if task_type in ("binary", "multiclass"):
        for level in df[label_col].unique().drop_nulls().to_list():
            out["by_label"][str(level)] = _stratum(df.filter(pl.col(label_col) == level))
    elif task_type == "regression":
        q1, q3 = df[label_col].quantile(0.25), df[label_col].quantile(0.75)
        out["by_label"]["low_q1"] = _stratum(df.filter(pl.col(label_col) <= q1))
        out["by_label"]["high_q4"] = _stratum(df.filter(pl.col(label_col) >= q3))
    return out


def _stratum(df: pl.DataFrame, label_col: str | None = None,
             task_type: str | None = None) -> dict:
    # True zero (e.g. no training at an inference-only site) is not PHI, so report
    # it as 0 rather than the '<10' suppression string used for real small counts.
    # by_label strata omit the label args: prevalence conditioned on the label is
    # degenerate, so the key stays present but None for a uniform stratum schema.
    return {
        "n": 0 if df.height == 0 else suppress(df.height),
        "date_range": _date_range(df),
        "prevalence": _prevalence(df, label_col, task_type),
        "continuous": {v: _continuous_stats(df, v) for v in CONTINUOUS_VARS
                       if v in df.columns},
        "categorical": {v: _categorical_stats(df, v) for v in CATEGORICAL_VARS
                        if v in df.columns},
    }


_EMPTY_CONTINUOUS = ("mean", "std", "median", "q25", "q75", "min", "max")


def _continuous_stats(df: pl.DataFrame, var: str) -> dict:
    """Summary stats for one continuous variable, suppression-aware.

    Below MIN_CELL_COUNT non-null values every statistic is withheld. `min` and
    `max` are literally one patient's value each, and on a stratum of 4 the
    quartiles are barely more aggregated — publishing them next to an `n` that
    reads "<10" suppresses the count while disclosing the individuals it was
    hiding. `n_missing` survives because it is itself a count, so it goes
    through the same `suppress()` as every other published count.
    """
    series = df[var]
    n_null = int(series.null_count())
    n_missing = 0 if n_null == 0 else suppress(n_null)   # a true zero is not PHI
    nn = series.drop_nulls()
    if nn.len() < MIN_CELL_COUNT:
        return {**{k: None for k in _EMPTY_CONTINUOUS}, "n_missing": n_missing}
    return {
        "mean": round(float(nn.mean()), 2),
        "std": round(float(nn.std()), 2) if nn.len() > 1 else 0.0,
        "median": round(float(nn.median()), 2),
        "q25": round(float(nn.quantile(0.25)), 2),
        "q75": round(float(nn.quantile(0.75)), 2),
        "min": round(float(nn.min()), 2),
        "max": round(float(nn.max()), 2),
        "n_missing": n_missing,
    }


def _categorical_stats(df: pl.DataFrame, var: str) -> dict:
    counts = (df.group_by(var).agg(pl.len().alias("n"))
              .filter(pl.col(var).is_not_null()))
    return suppress_dict({str(row[var]): row["n"] for row in counts.iter_rows(named=True)})
