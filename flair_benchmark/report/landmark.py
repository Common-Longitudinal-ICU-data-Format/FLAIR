"""Landmark report: Way 3 of docs/eval_continuous_three_ways.md.

FORWARD landmarking (van Houwelingen; the time-dependent-AUC idiom of the
US-CRS eFigure-2): landmark ``t`` is the slice ``window_number == t`` for
``t = 1..n_landmarks`` — **hospitalization day t** for the daily tasks,
**hospitalization hour t** for hourly task 4 (unit from META's
``window_unit``). ``n_landmarks`` comes from ``META['landmark_horizon']``
(30 daily landmarks for tasks 1/2, 168 hourly for task 4).

The risk set at landmark t is every stay still producing predictions at t.
Task builders only generate windows inside ``[pred_start, pred_stop)`` (stop =
event onset / discharge / death), so each slice is automatically restricted to
stays still at risk and event-free — the per-landmark ``n`` is a true
Kaplan-Meier "No. at risk" count that falls as events occur and stays end.
Nothing anchors on the event or outcome: the slice at day t is exactly the
cohort a deployed model would score on day t.

Each slice has (at most) one row per stay — the day-t prediction — so it is an
"episodic" cohort and runs through the three pillars (`_landmark_pillars`), with
the threshold-dependent blocks kept inline. Stacking the slices gives skill,
calibration and fairness AS A FUNCTION OF HOSPITALIZATION TIME (the time-varying
/ dynamic AUC view). Net benefit / DCA is deliberately not among them — it is an
episodic-only view (the same peak trap that keeps calibration off the peak).

Why this is honest: one row per stay per landmark means report/_ci.py's row
bootstrap is again a patient-level bootstrap (unlike row-pooling all windows).

Notes:
  - Early landmarks of windowed tasks (e.g. task4's 6h label) can be
    single-class when no event falls within the horizon yet; those slices are
    undefined (no contrast) and emit a compact marker.
  - Deep landmarks shrink as stays end; a risk set below MIN_LANDMARK_N has its
    ENTIRE entry suppressed (not just per-cell), matching the house n<10 rule.
  - Scored landmarks are built concurrently on a ProcessPoolExecutor (the
    bootstrap loops are GIL-bound, so processes — not threads — give the
    multicore speedup); output is deterministic and order-identical to a serial
    run. Set FLAIR_REPORT_WORKERS to cap workers, or pass max_workers=1 for serial.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from flair_benchmark.report._suppress import MIN_CELL_COUNT
from flair_benchmark.report.calibration import build_calibration
from flair_benchmark.report.discrimination import build_discrimination
from flair_benchmark.report.fairness import build_fairness

# Pin linear-algebra (BLAS/LAPACK) to ONE thread while building a landmark. A
# degenerate logistic calibration fit on a tiny subgroup is sensitive to BLAS
# thread count, so without this the numbers would drift with the machine's core
# count / pool size. Pinning makes every landmark bit-reproducible regardless of
# how many worker processes run — the pool only affects speed, never results.
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:                                # pragma: no cover - always present with sklearn
    from contextlib import nullcontext as _nullcontext

    def _threadpool_limits(*_a, **_k):
        return _nullcontext()


# THE pillar registry: one source for what gets built per slice AND which keys
# the suppressed/undefined stubs and the assembly loop must cover. A pillar
# added here automatically stays aligned across scored and stub landmarks —
# hand-syncing a separate tuple was one forgotten edit away from ragged
# per-landmark arrays.
PILLAR_BUILDERS = {
    "discrimination": build_discrimination,
    "calibration": build_calibration,
    "fairness": build_fairness,
}


def _landmark_pillars(sl: pl.DataFrame, meta: dict) -> dict[str, dict]:
    """The four pillars for ONE landmark slice, threshold blocks INCLUDED.

    Landmarks keep the operating point and the fairness parity gaps inline,
    because the point of the landmark report is to show how the decision behaves
    as a function of hospitalization time — and kpi.py reads those per-landmark
    numbers.

    The EPISODIC report deliberately does NOT compose the pillars this way: it
    hoists every threshold-dependent number into operating_points.json (swept over
    five thresholds). Landmarks therefore build the pillars themselves rather than
    borrowing build_episodic_report, so the two reports can evolve independently.
    """
    return {name: build(sl, meta) for name, build in PILLAR_BUILDERS.items()}


def _build_deterministic(sl, meta):
    # limit ALL native thread pools (BLAS *and* OpenMP): a non-converged logistic
    # calibration fit on a tiny subgroup is sensitive to both, so pinning only
    # BLAS still lets OpenMP thread count (which varies with machine load) perturb
    # the result. limits=1 with no user_api caps every pool.
    with _threadpool_limits(limits=1):
        return _landmark_pillars(sl, meta)

MIN_LANDMARK_N = MIN_CELL_COUNT
_PILLARS = tuple(PILLAR_BUILDERS)   # derived — cannot drift from the registry
# only spin up a process pool when there are enough scored landmarks to amortize
# the spawn + pickle overhead; smaller runs (tests, short stays) stay serial.
_MIN_PARALLEL = 16


def _worker_count(n_scored: int) -> int:
    """Worker PROCESSES for the per-landmark builds: min(#scored, env/cpu), >=1.

    FLAIR_REPORT_WORKERS overrides; otherwise os.cpu_count(). Each landmark's
    4-pillar build is CPU-bound and mostly pure-Python bootstrap loops (GIL-bound),
    so processes — not threads — give the real multicore speedup. Builds are
    independent and bootstrap with fixed seeds, so the pooled result is
    deterministic and order-identical to a serial run.
    """
    env = os.environ.get("FLAIR_REPORT_WORKERS")
    cap = int(env) if (env and env.isdigit() and int(env) > 0) else (os.cpu_count() or 2)
    return max(1, min(n_scored, cap))


def _build_one(payload: tuple) -> dict:
    """Module-level worker (picklable for ProcessPoolExecutor): build one slice."""
    sl, meta = payload
    return _build_deterministic(sl, meta)


def partition_windows(preds: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """{window_number: slice} in ONE pass over the frame.

    Replaces per-landmark ``filter(window_number == t)`` loops (landmark,
    leadtime curve, leadtime trajectory each scanned the full frame per
    landmark — up to ~500 full scans for task 4). ``maintain_order=True`` keeps
    each slice's row order identical to the filter it replaces, so the seeded
    bootstraps see the same rows in the same order.
    """
    parts = preds.partition_by("window_number", as_dict=True,
                               maintain_order=True)
    return {(k[0] if isinstance(k, tuple) else k): v for k, v in parts.items()}


def resolve_n_landmarks(meta: dict, n_landmarks: int | None) -> int:
    """Explicit override wins; otherwise the task's META['landmark_horizon'].

    The horizon is a landmark COUNT in the task's window_unit (30 days for the
    daily tasks, 168 hours for hourly sepsis) — it lives in the task's META,
    never hardcoded here.
    """
    if n_landmarks is not None:
        return n_landmarks
    h = meta.get("landmark_horizon")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(
            "META['landmark_horizon'] (positive int, landmarks in the task's "
            "window_unit) is required for the continuous report; got "
            f"{h!r} for task {meta.get('name')!r}.")
    return h


def build_landmark_report(preds: pl.DataFrame, meta: dict,
                          n_landmarks: int | None = None,
                          max_workers: int | None = None,
                          parts: dict[int, pl.DataFrame] | None = None
                          ) -> dict[str, list[dict]]:
    """Per-landmark episodic bundle, stacked into one list per pillar.

    The scored landmarks are built on a ProcessPoolExecutor (they are independent
    and bootstrap with fixed seeds, so the output is byte-identical to a serial
    run regardless of worker count). Pass ``max_workers=1`` to force serial.
    ``parts`` (from :func:`partition_windows`) lets the continuous composer share
    one partition pass across landmark/leadtime instead of re-filtering.
    """
    n_landmarks = resolve_n_landmarks(meta, n_landmarks)
    window_unit = meta.get("window_unit")
    label_col = meta["label_column"]
    if parts is None:
        parts = partition_windows(preds)
    empty = preds.clear()

    # 1) classify every landmark in the parent. Each job is
    #    (lm, tag, kind, slice) — forward slice = everyone still at risk at
    #    hospitalization time lm+1 (windows only exist pre-event/pre-discharge,
    #    so it is the honest deployed-model risk set, never anchored on outcome).
    jobs = []
    for lm in range(n_landmarks):
        sl = parts.get(lm + 1, empty)
        tag = {"landmark_index": lm, "hospitalization_time": lm + 1,
               "window_unit": window_unit, "n": sl.height}
        if sl.height < MIN_LANDMARK_N:
            kind = "suppressed"
        elif sl[label_col].n_unique() < 2:
            # single-class risk set (e.g. early hours of a windowed task where no
            # event falls inside the label horizon yet): AUROC/calibration/DCA
            # are undefined — skip the builders and emit a compact marker.
            kind = "undefined"
        else:
            kind = "score"
        jobs.append((lm, tag, kind, sl if kind == "score" else None))

    # 2) build the scored slices, in parallel across landmarks (process pool —
    #    the bootstrap loops are GIL-bound, so threads would not scale).
    scored = [(lm, sl) for lm, tag, kind, sl in jobs if kind == "score"]
    reps: dict[int, dict] = {}
    workers = max_workers if max_workers is not None else _worker_count(len(scored))
    # honor an explicit max_workers; otherwise only parallelize larger runs.
    use_pool = bool(scored) and workers > 1 and (
        max_workers is not None or len(scored) >= _MIN_PARALLEL)
    if use_pool:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for (lm, _sl), rep in zip(
                        scored, ex.map(_build_one,
                                       [(sl, meta) for _, sl in scored])):
                    reps[lm] = rep
        except Exception:            # pool unavailable → deterministic serial path
            reps = {lm: _build_deterministic(sl, meta) for lm, sl in scored}
    else:
        reps = {lm: _build_deterministic(sl, meta) for lm, sl in scored}

    # 3) assemble in landmark order (identical layout to the serial version).
    out: dict[str, list[dict]] = {p: [] for p in _PILLARS}
    for lm, tag, kind, _sl in jobs:
        if kind == "suppressed":
            stub = {**tag, "n": f"<{MIN_LANDMARK_N}", "suppressed": True}
            for p in _PILLARS:
                out[p].append(dict(stub))
        elif kind == "undefined":
            stub = {**tag, "undefined": True,
                    "reason": "single label class at this hospitalization time"}
            for p in _PILLARS:
                out[p].append(dict(stub))
        else:
            rep = reps[lm]
            for p in _PILLARS:
                payload = rep[p]
                payload.setdefault("metadata", {}).update(tag)
                out[p].append({**tag, **payload})
    return out
