"""The encounter key every task is built on, and the train/test split over it.

Two halves of one idea, which is why they share a file:

  1. STITCHING builds `hospitalization_join_id` — the canonical encounter key.
  2. SPLITTING assigns train/test *at that key's grain*.

A single CLIF `hospitalization_id` is one admission-to-discharge record. When a
patient is discharged and re-admitted within a few hours (an ED-to-inpatient
transfer, a facility switch), that one continuous encounter is split across *two*
hospitalization_ids. clifpy's `stitch_encounters` links such records into one
`encounter_block`; we rename it `hospitalization_join_id` and thread it through
every FLAIR stage.

The split is then decided once per stitched ENCOUNTER and broadcast to every row
of it, so no encounter's prediction windows can span train and test. That is what
prevents the within-stay leakage count-feature and time-series models are exposed
to, and it keeps the two arms separated in calendar time — the property an
external-validity read of the test set depends on. See `assign_split` for why the
grain is the encounter and not the patient.

The index is built ONCE per (config, time_interval) and cached to parquet, so the
same stitched mapping is reused by every task, feature and metadata run.

Depends on: cohort/clif.py (loaders), clifpy.
Used by: every task, features/_scope.py, cohort/table1.py, cohort/encounter.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import polars as pl

from flair_benchmark.cohort.clif import load_adt, load_hospitalization

JOIN_ID = "hospitalization_join_id"


# ===========================================================================
# Part 1 — Encounter stitching
#
# Public API:
#   build_encounter_index(cfg, time_interval)         -> always rebuilds
#   load_or_build_encounter_index(cfg, time_interval) -> cached
#   attach_join_id(df, idx)                           -> left-join + self-fallback
#   members_of_joins(idx, join_ids)                   -> join_id -> all member hosp_ids
#
# The mapping frame has columns:
#   hospitalization_id, hospitalization_join_id, patient_id
# ===========================================================================


def _cache_dir(cfg: dict) -> Path:
    """Where the encounter index parquet lives.

    Precedence:
      1. cfg["cache_directory"] — set this to the folder where you write task
         parquet so the index sits alongside your outputs.
      2. ./output/_flair_cache (default, relative to the working directory).
         FLAIR never writes into the CLIF data directory, which may be shared
         or read-only.
    """
    explicit = cfg.get("cache_directory")
    base = Path(explicit) if explicit else Path("output") / "_flair_cache"
    return base


def source_fingerprint(cfg: dict) -> str:
    """Short digest of the CLIF source files an index/score cache was built from.

    Covers path, size and mtime of every file directly under `data_directory`.
    A cache keyed on `path.exists()` alone is stale the moment a site re-ETLs its
    data or repoints `data_directory`, and nothing about the stale result looks
    wrong — the split just silently belongs to yesterday's cohort. Folding this
    into the filename means a changed source builds a new cache instead of
    quietly reusing the old one.

    Directory listing only (never file contents), so it stays O(number of files)
    on a multi-GB data directory.
    """
    root = Path(cfg["data_directory"])
    parts = [str(root)]
    try:
        for p in sorted(root.iterdir()):
            if p.is_file():
                st = p.stat()
                parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    except OSError:
        # Unreadable directory: fall back to the path alone rather than failing.
        pass
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _cache_path(cfg: dict, time_interval: int) -> Path:
    return (_cache_dir(cfg)
            / f"encounter_index_{time_interval}h_{source_fingerprint(cfg)}.parquet")


def build_encounter_index(cfg: dict, time_interval: int = 6) -> pl.DataFrame:
    """Run clifpy `stitch_encounters` on the FULL hospitalization + ADT set.

    Stitching shifts within `patient_id`, so it must see every hospitalization for
    the relevant patients — not just the ICU subset a task ends up using. Passing
    a pre-filtered frame here would break encounters apart.

    Returns one row per hospitalization_id:
        hospitalization_id, hospitalization_join_id (Utf8), patient_id
    """
    from clifpy.utils.stitching_encounters import stitch_encounters

    # stitch_encounters requires admission_type_category (absent from base_icu_cohort).
    hosp = load_hospitalization(
        cfg,
        columns=["hospitalization_id", "patient_id", "admission_dttm",
                 "discharge_dttm", "age_at_admission",
                 "admission_type_category", "discharge_category"],
    )
    adt = load_adt(cfg)

    # stitch_encounters is pandas-based. clifpy hands us pandas upstream but our
    # loaders convert to polars, so convert back just for this call.
    _, _, mapping = stitch_encounters(
        hospitalization=hosp.to_pandas(),
        adt=adt.to_pandas(),
        time_interval=time_interval,
    )

    idx = (
        pl.from_pandas(mapping)
        .with_columns([
            pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col("encounter_block").cast(pl.Utf8).alias(JOIN_ID),
        ])
        .select(["hospitalization_id", JOIN_ID])
        .unique(subset=["hospitalization_id"])
    )

    # Carry patient_id for downstream block-level analysis / traceability.
    pat = hosp.select([
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col("patient_id").cast(pl.Utf8),
    ]).unique(subset=["hospitalization_id"])

    return idx.join(pat, on="hospitalization_id", how="left")


# Process-level memo for the cached index. One task build calls
# load_or_build_encounter_index several times (base cohort, encounter attributes,
# per-table attach) and the parquet can be large, so we read it once per
# (path, mtime) rather than once per call.
_INDEX_MEMO: dict[tuple[str, int], pl.DataFrame] = {}


def load_or_build_encounter_index(cfg: dict, time_interval: int = 6,
                                  cache_dir: Optional[str] = None) -> pl.DataFrame:
    """Read the cached encounter index, building (and saving) it on first use.

    This is the "run before all tasks, saved" step: the first cohort build writes
    the parquet, every later run reads it. The filename carries a fingerprint of
    the source files (`source_fingerprint`), so re-ETL'd data rebuilds rather
    than silently reusing the previous stitching.
    """
    if cache_dir is not None:
        cfg = {**cfg, "cache_directory": cache_dir}
    path = _cache_path(cfg, time_interval)
    if path.exists():
        key = (str(path), path.stat().st_mtime_ns)
        cached = _INDEX_MEMO.get(key)
        if cached is None:
            cached = pl.read_parquet(path)
            _INDEX_MEMO.clear()          # one live index per process is plenty
            _INDEX_MEMO[key] = cached
        return cached

    idx = build_encounter_index(cfg, time_interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx.write_parquet(path)
    return idx


def attach_join_id(df: pl.DataFrame, idx: pl.DataFrame) -> pl.DataFrame:
    """Left-join hospitalization_join_id onto any frame keyed on hospitalization_id.

    A hospitalization_id absent from the index (never stitched) maps to *itself*,
    so the join key is always non-null and a non-stitched stay is its own block.
    """
    mapping = idx.select(["hospitalization_id", JOIN_ID]).unique(subset=["hospitalization_id"])
    out = df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).join(
        mapping.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)),
        on="hospitalization_id", how="left",
    )
    return out.with_columns(
        pl.col(JOIN_ID).fill_null(pl.col("hospitalization_id"))
    )


def members_of_joins(idx: pl.DataFrame, join_ids: list[str]) -> list[str]:
    """All member hospitalization_ids of the given stitched encounters.

    A cohort knows its own hospitalization_ids, but one `hospitalization_join_id`
    can stitch several admissions together — and a sibling member may be absent
    from the cohort (e.g. a non-ICU transfer). Pulling data by cohort ids alone
    would silently drop that sibling's events. This resolves each join_id back to
    its FULL member set via the stitch index, so extraction sees the whole
    encounter.

    join_ids absent from the index (never stitched) map to themselves, so the
    returned list always covers the input even for self-blocks. Matched as Utf8.
    """
    wanted = {str(j) for j in join_ids}
    sidx = idx.with_columns(
        pl.col("hospitalization_id").cast(pl.Utf8),
        pl.col(JOIN_ID).cast(pl.Utf8),
    )
    matched = sidx.filter(pl.col(JOIN_ID).is_in(list(wanted)))
    members = set(matched["hospitalization_id"].to_list())
    # Self-fallback: a never-stitched join_id (== its own hospitalization_id) may be
    # absent from the index's join column; add those raw ids so none are lost. Only
    # the unmatched ones, so synthetic block ids never leak into the hosp_id list.
    present = set(matched[JOIN_ID].to_list())
    members |= (wanted - present)
    return sorted(members)


# ===========================================================================
# Part 2 — Train/test split assignment
#
# Every cohort always carries BOTH 'train' and 'test', whether or not the site
# will fit a model, so the cohort and Table 1 schemas are identical across sites.
#
# One rule at every site: 75% of stitched encounters train, 25% test.
#   site == 'mimic'  -> encounters ordered by sha256 of the join id (dates are
#                       shifted); the same split wherever it runs.
#   other site       -> encounters ordered by earliest admission_col: oldest 75%
#                       train, newest 25% test. The benchmark window has already
#                       dropped whole encounters upstream (cohort/clif.py).
# ===========================================================================


def _split_cutoff(n: int, train_ratio: float) -> int:
    """Index of the first test group: how many of `n` ordered groups go to train.

    Clamped to [1, n-1] so both arms are non-empty whenever that is possible at
    all. A bare `int(n * train_ratio)` is 0 for n == 1 and for any n below
    1/train_ratio, which hands a small cohort ZERO training rows — the exact
    opposite of this module's "always carries BOTH" contract, and it fails
    silently. n == 1 cannot honour the contract; that lone group goes to train.
    """
    if n <= 1:
        return n
    return min(max(1, int(n * train_ratio)), n - 1)


def _stable_digest(group_id: str) -> str:
    """Version-stable shuffle key: sha256 of the group id string.

    Not `pl.Expr.hash`: polars documents its hash as unstable across releases.
    sha256 of the id string is identical on every machine and every library
    version, so the MIMIC split is the same wherever it runs.
    """
    return hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()


def assign_split(cohort: pl.DataFrame, site: str, *,
                 admission_col: str = "admission_dttm",
                 train_ratio: float = 0.75) -> pl.DataFrame:
    """Add a 'split' column: 75% of encounters 'train', 25% 'test', at every site.

    The unit is always the stitched encounter (`hospitalization_join_id`); every
    row of an encounter lands in the same arm, and a cohort without the join id
    raises. Only the order the encounters are cut in depends on the site:

      site == "mimic"  sha256 of the join id (dates are de-identified/shifted, so
                       calendar order means nothing). Identical wherever it runs.
      any other site   earliest `admission_col` per encounter: oldest 75% train,
                       newest 25% test. No date filter here: base_icu_cohort
                       already kept only encounters inside the benchmark window.

    An encounter spans days to weeks, so the temporal arms are separated in
    calendar time with at most one stay open across the boundary. A patient with
    several encounters can appear in both arms; that is the accepted cost of a
    test set that is genuinely later than training.
    """
    if JOIN_ID not in cohort.columns:
        raise ValueError("assign_split needs hospitalization_join_id: the split is always "
                         "dealt per stitched encounter (attach_join_id first).")
    if bool(site) and site.lower() == "mimic":
        return _random_split_by_group(cohort, train_ratio)
    return _ratio_split_by_group(cohort, admission_col, train_ratio)


def _random_split_by_group(cohort: pl.DataFrame, train_ratio: float) -> pl.DataFrame:
    """Assign 'train'/'test' once per encounter at RANDOM (block grain).

    Used for MIMIC, whose admission dates are de-identified/shifted so temporal
    order carries no meaning. Order is the sha256 digest of the group id (see
    `_stable_digest`), so the split is random-looking yet reproducible across
    runs, machines AND library versions; the first `train_ratio` of the
    digest-ordered groups are 'train'. Never `pl.Expr.hash`, which is not stable
    across polars versions.
    """
    groups = (
        cohort.select(pl.col(JOIN_ID).unique())
        .with_columns(
            pl.col(JOIN_ID).cast(pl.Utf8)
            .map_elements(_stable_digest, return_dtype=pl.Utf8).alias("_h"))
        .sort(["_h", JOIN_ID])
        .with_row_index("_gi")
    )
    cutoff = _split_cutoff(groups.height, train_ratio)
    groups = groups.with_columns(
        pl.when(pl.col("_gi") < cutoff).then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).select([JOIN_ID, "split"])
    return cohort.join(groups, on=JOIN_ID, how="left")


def _ratio_split_by_group(cohort: pl.DataFrame, order_col: str,
                          train_ratio: float) -> pl.DataFrame:
    """Assign 'train'/'test' once per encounter; broadcast to all its rows.

    Groups are ordered by their earliest `order_col` (then group id) so the split
    is deterministic; the first `train_ratio` of groups are 'train'.
    """

    groups = (
        cohort.group_by(JOIN_ID)
        .agg(pl.col(order_col).min().alias("_g_order"))
        .sort(["_g_order", JOIN_ID])
        .with_row_index("_gi")
    )
    cutoff = _split_cutoff(groups.height, train_ratio)
    groups = groups.with_columns(
        pl.when(pl.col("_gi") < cutoff).then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    ).select([JOIN_ID, "split"])
    return cohort.join(groups, on=JOIN_ID, how="left")
