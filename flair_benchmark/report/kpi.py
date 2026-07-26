"""KPI scoreboard for the continuous report -> kpi.json.

One compact block of headline numbers, aggregated from the ALREADY-BUILT
section dicts (peak / landmark / leadtime) — pure aggregation, no access to the
predictions, so the scoreboard can never disagree with the per-section JSONs.

  pooled_dynamic_auroc : Heagerty & Zheng (2005) style time-varying AUC — the
        per-window (incident/dynamic) AUCs pooled into one global concordance
        with Mann-Whitney pair weights. Within window t the identity
        concordant pairs = AUC(t) * cases_t * controls_t gives

            AUC_pooled = sum_t AUC(t) * w_t / sum_t w_t ,  w_t = cases_t * controls_t

        with cases_t = n_t * prevalence_t. It pools over the FULL STAY — every
        forward window whose risk set still holds >= MIN_CELL_COUNT stays — NOT
        the capped landmark DISPLAY horizon (7d/30d): the displayed landmark
        curve is a show-only view, while this headline describes the whole stay.
        Deep/sparse windows fall below the floor and drop out (near-zero pair
        weight anyway). The CI is a STAY-LEVEL pooled bootstrap: one stay-resample
        per replicate, every window's AUC and pair weight recomputed on that
        resample, pooled, percentile (2.5, 97.5) over replicates — honest to the
        within-stay correlation across windows. (Same formula the FLAIR paper
        pipeline uses; single-sourced here so papers can read kpi.json.)
  peak         : stay-peak screening headline (AUROC/AUPRC, operating point,
                 NNE = 1/PPV — alarms per true case).
  calibration  : calibration at the earliest SCORED hospitalization time
                 (`at_hospitalization_time` names it — landmark 1 itself can be
                 suppressed/single-class) + the slope range across landmarks.
  leadtime     : pct of event stays ever caught + median first-alert
                 hospitalization time (how early in the stay the model fires).
  operating_points : the clinical-threshold confusion headline (sensitivity /
                 specificity / FPR / PPV / NNE) + FPR across the three thresholds,
                 plus the intervention sub-analysis highlights (positive fire
                 rate, median hours-before-treatment, treated-negative median Δ,
                 untreated misfire rate) when the task declares an intervention.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark.report._stats import BOOT_REPS, BOOT_SEED, MIN_CELL_COUNT

_METHOD_NOTE = ("pooled dynamic AUC per Heagerty & Zheng 2005 (incident/dynamic"
                " concordance, pair-weighted across ALL FORWARD windows of the"
                " full stay — patients still at risk at each hospitalization"
                " day/hour, risk set >= MIN_CELL_COUNT; NOT capped at the display"
                " landmark horizon); landmarking per van Houwelingen. Pooled CI"
                " is a stay-level pooled bootstrap (one stay resample per"
                " replicate; all window AUCs + pair weights recomputed per"
                " replicate).")


def build_kpi(peak: dict[str, dict], landmarks: dict[str, list[dict]],
              leadtime: dict, operating_points: dict | None = None,
              preds: pl.DataFrame | None = None,
              meta: dict | None = None) -> dict:
    """Headline scoreboard from the built continuous sections.

    ``preds``/``meta`` (the raw continuous frame + task META) enable the
    stay-level pooled bootstrap CI; without them the scoreboard stays pure
    aggregation and the pooled CI is null.
    """
    disc_lms = landmarks.get("discrimination", [])
    out: dict = {"metadata": {
        "horizon": len(disc_lms),
        "window_unit": leadtime.get("metadata", {}).get("window_unit"),
        "method": _METHOD_NOTE,
    }}

    # The pooled dynamic AUROC describes the FULL STAY, not the capped display
    # horizon: pool over every forward window with >= MIN_CELL_COUNT at risk,
    # recomputed from preds. Fall back to the (capped) displayed landmarks only
    # when preds/meta are absent (a pure-aggregation unit call).
    if preds is not None and meta is not None:
        pool = _full_stay_pool_windows(preds, meta)
        out["metadata"]["pooled_auroc_scope"] = "full_stay"
        out["metadata"]["pooled_auroc_n_windows"] = len(pool)
        out["pooled_dynamic_auroc"] = _pooled_dynamic_auroc(pool)[0]
        out["pooled_dynamic_auroc_ci"] = _pooled_auroc_stay_bootstrap_ci(
            preds, meta, pool)
    else:
        out["metadata"]["pooled_auroc_scope"] = "display_horizon"
        out["metadata"]["pooled_auroc_n_windows"] = len(_scored(disc_lms))
        out["pooled_dynamic_auroc"] = _pooled_dynamic_auroc(disc_lms)[0]
        out["pooled_dynamic_auroc_ci"] = None
    out["peak"] = _peak_block(peak.get("discrimination", {}))
    out["calibration"] = _calibration_block(landmarks.get("calibration", []))
    out["leadtime"] = _leadtime_block(leadtime)
    out["operating_points"] = _operating_points_block(operating_points or {})
    return out


def _scored(entries: list[dict]) -> list[dict]:
    """Landmark entries that carry metrics (skip suppressed/undefined stubs)."""
    return [e for e in entries
            if not e.get("suppressed") and not e.get("undefined")
            and e.get("metrics")]


def _full_stay_pool_windows(preds: pl.DataFrame, meta: dict) -> list[dict]:
    """Per-window discrimination over the FULL STAY, for the pooled dynamic AUROC.

    The displayed landmark curve is capped at META['landmark_horizon'] (7d/30d) —
    a show-only view. The pooled dynamic AUROC, however, must describe the whole
    stay, so it pools over EVERY forward window (hospitalization_time == window_
    number) whose risk set still holds >= MIN_CELL_COUNT stays and both label
    classes — not just the first ``horizon`` windows. Deep/sparse windows fall
    below the floor and drop out (they would carry near-zero Mann-Whitney pair
    weight anyway).

    Each entry mirrors a scored landmark discrimination entry
    ({hospitalization_time, metrics:{auroc}, prevalence, n}) so the pooled point
    estimate and the stay-bootstrap CI consume it unchanged. AUROC/prevalence use
    discrimination.py's exact conventions (roc_auc_score rounded 4dp, prevalence
    rounded 6dp), so within the display horizon these windows are bit-identical to
    the landmark curve — the pool only EXTENDS past it.
    """
    from sklearn.metrics import roc_auc_score

    label_col = meta["label_column"]
    parts = preds.partition_by("window_number", as_dict=True, maintain_order=True)
    out: list[dict] = []
    for key, sl in parts.items():
        t = key[0] if isinstance(key, tuple) else key
        n = sl.height
        if n < MIN_CELL_COUNT:
            continue
        y = sl[label_col].to_numpy().astype(int)
        cases = int(y.sum())
        if cases == 0 or cases == n:          # single class -> AUROC undefined
            continue
        p = sl["y_prob"].to_numpy().astype(float)
        out.append({
            "hospitalization_time": int(t),
            "metrics": {"auroc": round(float(roc_auc_score(y, p)), 4)},
            "prevalence": round(cases / n, 6),
            "n": n,
        })
    out.sort(key=lambda e: e["hospitalization_time"])
    return out


def _pooled_dynamic_auroc(entries: list[dict]) -> tuple[float | None,
                                                        list[float] | None]:
    """Pair-weighted pooled point estimate over the scored landmarks.

    (The CI is NOT derived here — averaging per-landmark CI bounds has no
    coverage guarantee, and substituting zero-width intervals for missing
    landmark CIs shrank the pool as landmarks got LESS stable. The honest CI
    is the stay-level pooled bootstrap below.)
    """
    num = den = 0.0
    for e in _scored(entries):
        auc = e["metrics"].get("auroc")
        prev = e.get("prevalence")
        n = e.get("n")
        if auc is None or prev is None or not isinstance(n, int):
            continue
        cases = n * prev
        w = cases * (n - cases)      # Mann-Whitney comparable pairs at t
        if w <= 0:
            continue
        num += auc * w
        den += w
    if den == 0:
        return None, None
    return round(num / den, 4), None


def _pooled_auroc_stay_bootstrap_ci(preds: pl.DataFrame, meta: dict,
                                    entries: list[dict],
                                    reps: int = BOOT_REPS,
                                    seed: int = BOOT_SEED) -> list[float] | None:
    """Stay-level pooled bootstrap CI for the pooled dynamic AUROC.

    One stay resample per replicate; within the replicate every scored
    landmark's AUC and pair weight are recomputed on the resampled cohort and
    pooled with the same Mann-Whitney weights as the point estimate. This
    respects the within-stay correlation across landmarks (the same stays
    appear at every time they are at risk), which per-landmark CIs cannot.

    Resampling uses sample WEIGHTS (a stay drawn k times counts k-fold via
    sklearn's sample_weight) — identical to materializing repeated rows.
    Landmarks that are suppressed/undefined in the published report are
    excluded, so the CI describes exactly the pooled point estimate's set.
    Single-class landmarks within a replicate are skipped for that replicate.
    """
    from sklearn.metrics import roc_auc_score

    scored_times = {e["hospitalization_time"] for e in _scored(entries)
                    if e["metrics"].get("auroc") is not None}
    if not scored_times:
        return None

    label_col = meta["label_column"]
    # per scored landmark: (stay codes, labels, probs) — one row per stay.
    stays = preds["hospitalization_join_id"].unique().sort()
    code_of = {s: i for i, s in enumerate(stays.to_list())}
    n_stays = len(code_of)
    slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in sorted(scored_times):
        sl = preds.filter(pl.col("window_number") == t)
        codes = np.array([code_of[s] for s in
                          sl["hospitalization_join_id"].to_list()])
        slices.append((codes,
                       sl[label_col].to_numpy().astype(int),
                       sl["y_prob"].to_numpy().astype(float)))

    rng = np.random.default_rng(seed)
    pooled: list[float] = []
    for _ in range(reps):
        mult = np.bincount(rng.integers(0, n_stays, n_stays),
                           minlength=n_stays).astype(float)
        num = den = 0.0
        for codes, y, p in slices:
            w = mult[codes]
            wsum = float(w.sum())
            if wsum <= 0:
                continue
            cases = float(w[y == 1].sum())
            controls = wsum - cases
            pairs = cases * controls
            if pairs <= 0:
                continue                      # single-class in this resample
            auc = roc_auc_score(y, p, sample_weight=w)
            num += auc * pairs
            den += pairs
        if den > 0:
            pooled.append(num / den)

    if len(pooled) < reps // 2:
        return None
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _peak_block(disc: dict) -> dict | None:
    m = disc.get("metrics")
    if not m:
        return None
    op = m.get("operating_point") or {}
    return {
        "auroc": m.get("auroc"), "auroc_ci": m.get("auroc_ci"),
        "auprc": m.get("auprc"), "auprc_ci": m.get("auprc_ci"),
        "sensitivity": op.get("sensitivity"),
        "specificity": op.get("specificity"),
        "ppv": op.get("ppv"),
        "flag_rate": op.get("flag_rate"),
        "nne": op.get("nne"), "nne_ci": op.get("nne_ci"),
    }


def _calibration_block(entries: list[dict]) -> dict | None:
    scored = _scored(entries)
    if not scored:
        return None
    nearest = min(scored, key=lambda e: e.get("hospitalization_time", 0))
    m = nearest["metrics"]
    slopes = [e["metrics"]["slope"] for e in scored
              if e["metrics"].get("slope") is not None]
    return {
        "at_hospitalization_time": nearest.get("hospitalization_time"),
        "slope": m.get("slope"), "slope_ci": m.get("slope_ci"),
        "intercept": m.get("intercept"), "intercept_ci": m.get("intercept_ci"),
        "ece": m.get("ece"),
        "brier": m.get("brier"), "brier_ci": m.get("brier_ci"),
        "o_e_ratio": m.get("o_e_ratio"),
        "slope_range": ([round(min(slopes), 4), round(max(slopes), 4)]
                        if slopes else None),
    }


def _leadtime_block(leadtime: dict) -> dict | None:
    m = leadtime.get("metrics")
    if not m:
        return None
    return {
        "pct_caught": m.get("pct_caught"),
        "median_first_alert_hospitalization_time":
            m.get("median_first_alert_hospitalization_time"),
        "median_first_alert_hospitalization_time_ci":
            m.get("median_first_alert_hospitalization_time_ci"),
    }


def _operating_points_block(op: dict) -> dict | None:
    """Clinical-threshold headline + FPR sweep + encounter burden + sub-analysis.

    Reads the built (continuous, peak-surface) operating_points section. The
    clinical row is flagged, never positional: exactly one is_clinical row exists
    in any non-empty sweep (tasks._validate_taxonomy guarantees the policy lands on
    the grid). ``alert_burden`` is the encounter-level headline; ``sub_analyses``
    is None for tasks without an intervention/treatment column.
    """
    by_thr = op.get("by_threshold") or []
    if not by_thr:
        return None
    t0 = (op.get("metadata") or {}).get("clinical_threshold")
    clinical = next((e for e in by_thr if e.get("is_clinical")), None)
    if clinical is None:
        return None
    m = clinical.get("metrics") or {}
    block = {
        "clinical_threshold": t0,
        "sensitivity": m.get("sensitivity"),
        "specificity": m.get("specificity"),
        "fpr": m.get("fpr"),
        "ppv": m.get("ppv"),
        "nne": m.get("nne"), "nne_ci": m.get("nne_ci"),
        "fpr_by_threshold": [{"threshold": e.get("threshold"),
                              "fpr": (e.get("metrics") or {}).get("fpr")}
                             for e in by_thr],
        "alert_burden": clinical.get("alert_burden"),
        "selected_thresholds": _selected_digest(
            op.get("data_driven_thresholds") or []),
        "sub_analyses": None,
    }
    sub = clinical.get("sub_rates")
    timing = op.get("intervention_timing") or []
    if sub or timing:
        clin_timing = next((x for x in timing if x.get("is_clinical")), {}) or {}
        block["sub_analyses"] = {
            "positive_share": (sub or {}).get("positive_share"),
            "treated_negative_share": (sub or {}).get("treated_negative_share"),
            "untreated_misfire_share": (sub or {}).get("untreated_misfire_share"),
            "n_alarms": (sub or {}).get("n_alarms"),
            "median_hours_before_treatment_positive":
                clin_timing.get("median_hours_before_treatment_positive"),
        }
    return block


def _selected_digest(locators: list[dict]) -> dict:
    """Compact scoreboard digest: where Youden & F1 land (design doc §10 Q5)."""
    out: dict = {}
    for rule in ("youden_j", "f_beta_max"):
        loc = next((locator for locator in locators
                    if locator.get("rule") == rule), None)
        if loc and loc.get("threshold") is not None:
            ma = loc.get("metrics_at") or {}
            out[rule] = {"threshold": loc.get("threshold"),
                         "sensitivity": ma.get("sensitivity"), "ppv": ma.get("ppv")}
    return out
