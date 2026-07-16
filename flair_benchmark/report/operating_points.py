"""Operating points → operating_points.json.

The one self-contained decision-point view: everything that depends on WHERE the
decision threshold is put. Both report modes sweep the SAME fixed, task- and
site-invariant grid of 99 thresholds (0.01 … 0.99, ``_sweep.SWEEP_GRID``); the
shipped clinical policy is one flagged row (``is_clinical``) inside it — the
"clinical intrusion" marker — not a grid centre. There are no per-task offsets and
no dropped-offset bookkeeping: the grid is a constant, so two sites' sweeps line up
row-for-row and pool without interpolation (design doc §2).

ONE SURFACE, ONE SWEEP. Episodic preds are already one row per stay; continuous
preds collapse to the PEAK surface (max risk per stay — the hospitalization-level,
one-alert-per-stay evaluation used for real-time alert models). Both therefore reduce
to a single ``(label, prob)`` array pair and run the identical vectorized sweep
(``_sweep``) + data-driven locators (``_locators``). The two public builders differ
only in what they attach on top.

TWO BASES (see report/_basis.py):
  ALARM basis (``y_prob >= t``, BOTH directions) drives the confusion matrix and
  sensitivity/specificity/PPV — "how well do we DETECT the event", comparable across
  tasks regardless of ``clinical_direction``.
  ACTION / DECISION basis (direction-aware) drives the episodic net-benefit block
  (identical to dca.json by construction) and the intervention timing/rates.

WHAT EACH MODE ADDS:
  episodic  : a direction-aware ``decision`` block per row (net benefit == dca.json)
              and per-threshold subgroup ``fairness`` across all 99 thresholds.
  continuous: an encounter-level ``alert_burden`` block per row (frac of stays ever
              alerting, PPV, 1-PPV false-alarm fraction, NNE — no patient-day
              denominator), an ``operating_budget`` view (sensitivity
              at a fixed encounter alert-rate), and — only when the task declares an
              ``intervention`` column — per-threshold treatment ``sub_rates`` plus a
              single median lead-time scalar at the named thresholds. Net benefit is
              deliberately OMITTED here: DCA on the peak (max-of-many) surface is the
              peak-calibration trap (docs §2), so it lives in dca.json's landmarks.

Privacy: every confusion / burden count passes ``suppress()``; a rate derived from
fewer than ``MIN_CELL_COUNT`` flagged/acted stays is nulled with a reason.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from flair_benchmark._constants import SELECTION_DEFAULTS
from flair_benchmark.report._basis import (
    action_expr,
    axis_threshold,
    net_benefit_all,
    to_decision_basis,
)
from flair_benchmark.report._ci import scalar_ci
from flair_benchmark.report._confusion import nne_from_ppv, r1
from flair_benchmark.report._locators import locate_all, locate_band
from flair_benchmark.report._suppress import MIN_CELL_COUNT, suppress
from flair_benchmark.report._sweep import (
    RATE_KEYS,
    SWEEP_GRID,
    arm_fire_rate,
    band_ci,
    nb_curve,
    sweep_counts,
    sweep_metrics,
)
from flair_benchmark.report.discrimination import _auroc
from flair_benchmark.report.fairness import (
    MIN_SUBGROUP_N,
    SUBGROUP_COLS,
    _bin_age,
    encode_levels,
    gap_triplet,
)

_BLOCK = "hospitalization_join_id"
_r1 = r1

# a threshold is on the grid iff it is a multiple of the step; the clinical policy
# always is (tasks._validate_taxonomy guards it), so exactly one row is_clinical.
_GRID_META = {"min": 0.01, "max": 0.99, "step": 0.01, "n": len(SWEEP_GRID)}


# ---------------------------------------------------------------- shared sweep core


def _selection_params(meta: dict) -> dict:
    """Selection-rule constants: defaults, overridable per task via META."""
    params = dict(SELECTION_DEFAULTS)
    for key in params:
        if meta.get(key) is not None:
            params[key] = meta[key]
    return params


def _clinical_index(t0: float) -> int:
    """The single grid row whose threshold equals the clinical policy."""
    return int(np.argmin(np.abs(SWEEP_GRID - round(float(t0), 2))))


def _confusion_row(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                   tn: np.ndarray, i: int) -> dict:
    """FULL alert x label 2x2 at grid row i — all four cells, suppressed."""
    return {
        "n": int(tp[i] + fp[i] + fn[i] + tn[i]),
        "tp": suppress(int(tp[i])), "fp": suppress(int(fp[i])),
        "fn": suppress(int(fn[i])), "tn": suppress(int(tn[i])),
    }


def _metrics_row(m: dict[str, np.ndarray], band: dict[str, list], i: int,
                 n_flagged: int) -> dict:
    """ALARM-basis rates + CI band + derived NNE at grid row i.

    NNE / its CI invert PPV (nne_from_ppv). PPV and NNE are nulled when fewer than
    MIN_CELL_COUNT stays are flagged — a handful of flagged stays gives an unstable,
    disclosive precision (same floor dca.py applies to its curve).
    """
    def val(k: str) -> float | None:
        v = m[k][i]
        return None if np.isnan(v) else _r1(float(v))

    ppv = val("ppv")
    ppv_ci = band["ppv"][i]
    if n_flagged < MIN_CELL_COUNT:
        ppv, ppv_ci = None, None
    nne, nne_ci = nne_from_ppv(ppv, ppv_ci)
    out: dict = {k: val(k) for k in RATE_KEYS}
    out["ppv"] = ppv
    out.update({f"{k}_ci": band[k][i] for k in RATE_KEYS})
    out["ppv_ci"] = ppv_ci
    out["nne"] = nne
    out["nne_ci"] = nne_ci
    fr = m["flag_rate"][i]
    out["flag_rate"] = None if np.isnan(fr) else round(float(fr), 4)
    return out


def _base_metadata(meta: dict, t0: float, surface: str) -> dict:
    return {
        "clinical_threshold": t0,
        "clinical_direction": meta.get("clinical_direction", "above"),
        "grid": dict(_GRID_META),
        "surface": surface,
        "clinical_index": _clinical_index(t0),
        "min_cell_count": MIN_CELL_COUNT,
        "selection_params": _json_params(_selection_params(meta)),
        "data_driven_ci": (
            "data_driven_thresholds CIs are SELECTION-AWARE — each rule's cutpoint is "
            "re-selected on every bootstrap resample, so threshold_ci and metrics_at "
            "*_ci include cutpoint-selection variance and run wider than the "
            "fixed-threshold by_threshold band. The clinical entry is not fitted, so "
            "its CI is the plain fixed-threshold band (threshold_ci null)."),
    }


def _json_params(params: dict) -> dict:
    """selection_params with the tuple target grid as a JSON list."""
    out = dict(params)
    out["alert_rate_targets"] = list(params["alert_rate_targets"])
    return out


def _degenerate_payload(md: dict, reason: str) -> dict:
    """The empty-sweep payload for a non-binary / thresholdless task."""
    md["reason"] = reason
    return {"by_threshold": [], "data_driven_thresholds": []}


# =============================================================== EPISODIC SCORECARD


def build_episodic_operating_points(preds: pl.DataFrame, meta: dict) -> dict:
    """Every threshold-dependent episodic metric, swept over the 99-point grid.

    THE scorecard. discrimination.json, dca.json and fairness.json are
    threshold-FREE by construction; every threshold-dependent number lives here.
    Each row carries the alarm-basis confusion + metrics, a direction-aware
    ``decision`` block whose net benefit equals dca.json at ``dca_threshold``, and
    (when the frame has demographics) per-threshold subgroup fairness.
    """
    t0 = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    label_col = meta["label_column"]
    md = _base_metadata(meta, t0, "stay") if t0 is not None else {}
    md.update({
        "flag_rule": "y_prob >= threshold (alarm convention)",
        "net_benefit_basis": (
            "decision basis — identical to dca.json. decision.net_benefit is the "
            "net benefit of the ACT at that threshold, read at decision.dca_threshold "
            "(1-t for a 'below' task); NOT derivable from this row's alarm-basis TP/FP."),
    })
    out: dict = {"task_type": meta["task_type"], "metadata": md}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t0 is None:
        out.update(_degenerate_payload(
            md, "the operating-points scorecard requires a binary task with y_prob "
            "and a clinical_threshold"))
        return out

    label = (preds[label_col].to_numpy() > 0).astype(int)
    prob = preds["y_prob"].to_numpy().astype(float)
    n = len(label)
    out["prevalence"] = round(float(label.mean()), 6) if n else None

    tp, fp, fn, tn = sweep_counts(label, prob)
    m = sweep_metrics(tp, fp, fn, tn)
    auc = _auroc(label, prob)

    # decision basis: transform once; the net-benefit curve rides the same seeded
    # resample stream as the alarm rates (one band bootstrap for the whole row).
    below = direction == "below"
    yt_dec, yp_dec = to_decision_basis(label, prob, direction)
    axis_grid = np.array([axis_threshold(float(t), direction) for t in SWEEP_GRID])

    def _epi_compute(lab: np.ndarray, arrs: dict) -> dict:
        ctp, cfp, cfn, ctn = sweep_counts(lab, arrs["alarm"])
        res = {k: v for k, v in sweep_metrics(ctp, cfp, cfn, ctn).items()
               if k in RATE_KEYS}
        res["net_benefit"] = nb_curve(arrs["ytd"], arrs["ypd"], axis_grid, below)
        return res

    band_keys = RATE_KEYS + ("net_benefit",)
    band = band_ci(label, {"alarm": prob, "ytd": yt_dec, "ypd": yp_dec},
                   _epi_compute, band_keys)

    nb_point = nb_curve(yt_dec, yp_dec, axis_grid, below)
    prev_dec = float(yt_dec.mean()) if n else None
    nb_all = [None if prev_dec is None else net_benefit_all(prev_dec, float(t))
              for t in axis_grid]

    fair = _fairness_by_threshold(preds, meta, [round(float(t), 2) for t in SWEEP_GRID])
    ci = _clinical_index(t0)

    rows = []
    for i in range(len(SWEEP_GRID)):
        n_flagged = int(tp[i] + fp[i])
        row: dict = {
            "threshold": round(float(SWEEP_GRID[i]), 2),
            "is_clinical": i == ci,
            "confusion": _confusion_row(tp, fp, fn, tn, i),
            "metrics": _metrics_row(m, band, i, n_flagged),
            "decision": _decision_row(tp, fp, fn, tn, i, direction, axis_grid[i],
                                      nb_point[i], band["net_benefit"][i], nb_all[i]),
        }
        if n_flagged < MIN_CELL_COUNT:
            row["suppressed"] = True
            row["suppressed_reason"] = (
                f"fewer than {MIN_CELL_COUNT} stays flagged at this threshold")
        if fair is not None:
            row["fairness"] = fair[i]
        rows.append(row)

    params = _selection_params(meta)
    loc_ci = locate_band(label, prob, SWEEP_GRID, auc, params)
    out["by_threshold"] = rows
    out["data_driven_thresholds"] = locate_all(
        m, SWEEP_GRID, auc, out["prevalence"], round(float(t0), 2), "stay",
        params, band=band, ci=loc_ci, counts=(tp, fp))
    return out


def _decision_row(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray,
                  i: int, direction: str, dca_t: float, nb: float | None,
                  nb_ci: list | None, nb_all: float | None) -> dict:
    """Direction-aware act taken at row i + the decision curve's verdict there.

    Net benefit is on the DECISION basis at ``dca_threshold`` — the same number as
    dca.json's curve — so it is NOT derivable from this row's alarm-basis TP/FP.
    """
    n = int(tp[i] + fp[i] + fn[i] + tn[i])
    n_ge = int(tp[i] + fp[i])                       # stays with p >= t (fired)
    if direction == "below":                        # acted = p < t
        n_acted, pos_acted, pos_not = n - n_ge, int(fn[i]), int(tp[i])
        acts_on = "y_prob < threshold"
    else:                                           # acted = p >= t
        n_acted, pos_acted, pos_not = n_ge, int(tp[i]), int(fn[i])
        acts_on = "y_prob >= threshold"
    n_not = n - n_acted
    nb_v = None if nb is None or np.isnan(nb) else round(float(nb), 6)
    useful = (None if nb_v is None or nb_all is None
              else bool(nb_v > max(nb_all, 0.0)))
    out = {
        "acts_on": acts_on,
        "n_acted": suppress(n_acted),
        "act_rate": round(n_acted / n, 4) if n else None,
        "event_rate_in_acted": round(pos_acted / n_acted, 4) if n_acted else None,
        "event_rate_in_not_acted": round(pos_not / n_not, 4) if n_not else None,
        "dca_threshold": round(float(dca_t), 6),
        "net_benefit": nb_v,
        "net_benefit_ci": nb_ci,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": 0.0,
        "clinically_useful": useful,
    }
    if n_acted < MIN_CELL_COUNT:                    # net benefit derives from this arm
        out["net_benefit"] = None
        out["net_benefit_ci"] = None
        out["clinically_useful"] = None
        out["suppressed_reason"] = (
            f"fewer than {MIN_CELL_COUNT} stays acted on at this threshold")
    return out


# ============================================================= CONTINUOUS (PEAK)


def build_operating_points(preds: pl.DataFrame, meta: dict) -> dict:
    """Peak-surface sweep + encounter alert burden + optional intervention rates.

    Confusion, metrics, alert burden, locators and (task4) treatment sub-rates all
    ride the one-row-per-stay peak collapse. Net benefit is NOT reported here (peak
    DCA trap — it lives in dca.json's landmarks).
    """
    t0 = meta.get("clinical_threshold")
    direction = meta.get("clinical_direction", "above")
    iv = meta.get("intervention")
    window_unit = meta.get("window_unit")
    label_col = meta["label_column"]
    md = _base_metadata(meta, t0, "peak") if t0 is not None else {}
    md.update({
        "intervention": iv, "window_unit": window_unit,
        "unit_of_analysis": "encounter (peak)",
        "net_benefit_omitted_reason": (
            "net benefit is not reported on the peak (max-of-many) surface — the "
            "peak-calibration trap; the decision curve lives in dca.json's landmarks"),
    })
    out: dict = {"metadata": md}
    if meta["task_type"] != "binary" or "y_prob" not in preds.columns or t0 is None:
        md["reason"] = ("operating points require a binary task with y_prob and a "
                        "clinical_threshold")
        out.update({"by_threshold": [], "data_driven_thresholds": [],
                    "operating_budget": [], "intervention_timing": [],
                    "intervention_offsets": None})
        return out

    iv_col = iv["column"] if iv else None
    has_dttm = "prediction_dttm" in preds.columns
    if not iv_col:
        md["sub_analyses_reason"] = (
            "no intervention/treatment column configured; sub-analyses omitted")

    stay = _collapse_stays(preds, meta, iv_col, has_dttm)
    label = stay["label"].to_numpy().astype(int)
    alarm_peak = stay["alarm_peak"].to_numpy().astype(float)
    action_peak = stay["action_peak"].to_numpy().astype(float)
    treated = (stay["_treat_dttm"].is_not_null().to_numpy() if iv_col
               else np.zeros(len(label), dtype=bool))
    n = len(label)
    out["prevalence"] = round(float(label.mean()), 6) if n else None

    tp, fp, fn, tn = sweep_counts(label, alarm_peak)
    m = sweep_metrics(tp, fp, fn, tn)
    auc = _auroc(label, alarm_peak)

    def _cont_compute(lab: np.ndarray, arrs: dict) -> dict:
        ctp, cfp, cfn, ctn = sweep_counts(lab, arrs["alarm"])
        return {k: v for k, v in sweep_metrics(ctp, cfp, cfn, ctn).items()
                if k in RATE_KEYS}

    band = band_ci(label, {"alarm": alarm_peak}, _cont_compute, RATE_KEYS)

    # point-estimate arm masks + fire rates (direction-aware, one collapse) for §4.4.
    arms = _arm_masks(label, treated) if iv_col else None
    sub_point = ({k: arm_fire_rate(action_peak[mask], SWEEP_GRID, direction)
                  for k, mask in arms.items()} if iv_col else None)

    ci = _clinical_index(t0)
    rows = []
    for i in range(len(SWEEP_GRID)):
        n_flagged = int(tp[i] + fp[i])
        row: dict = {
            "threshold": round(float(SWEEP_GRID[i]), 2),
            "is_clinical": i == ci,
            "confusion": _confusion_row(tp, fp, fn, tn, i),
            "metrics": _metrics_row(m, band, i, n_flagged),
            "alert_burden": _alert_burden_row(m, band, tp, fp, i, n),
        }
        if iv_col:
            row["sub_rates"] = _sub_rates_row(sub_point, arms, i)
        rows.append(row)

    params = _selection_params(meta)
    loc_ci = locate_band(label, alarm_peak, SWEEP_GRID, auc, params)
    out["by_threshold"] = rows
    out["data_driven_thresholds"] = locate_all(
        m, SWEEP_GRID, auc, out["prevalence"], round(float(t0), 2), "peak",
        params, band=band, ci=loc_ci, counts=(tp, fp))
    out["operating_budget"] = _operating_budget(m, params)

    if iv_col and has_dttm:
        named = _named_thresholds(t0, out["data_driven_thresholds"])
        out["intervention_timing"] = [
            _timing_at(preds, meta, t, iv_col, label_col, direction, t0)
            for t in named]
        out["intervention_offsets"] = _intervention_offsets(stay, window_unit)
    else:
        out["intervention_timing"] = []
        out["intervention_offsets"] = None
    return out


def _collapse_stays(preds: pl.DataFrame, meta: dict, iv_col: str | None,
                    has_dttm: bool) -> pl.DataFrame:
    """One row per stay, threshold-independent (design doc §4.1 peak collapse).

    ``alarm_peak`` = max risk (confusion / detection). ``action_peak`` = the
    direction-aware action scalar — max for an "above" task, min for "below" — so
    "did the stay ever ACT at t" is a single searchsorted (arm_fire_rate). The
    trailing .sort(block) is load-bearing: group_by row order is nondeterministic
    and the seeded bootstrap resamples by row index (see _ci's caller contract).
    """
    block = _BLOCK
    label_col = meta["label_column"]
    direction = meta.get("clinical_direction", "above")
    action_agg = (pl.col("y_prob").min() if direction == "below"
                  else pl.col("y_prob").max())
    aggs = [(pl.col(label_col).max() > 0).cast(pl.Int8).alias("label"),
            pl.col("y_prob").max().alias("alarm_peak"),
            action_agg.alias("action_peak")]
    if has_dttm:
        aggs.append(pl.col("prediction_dttm").min().alias("_stay_start_dttm"))
    if iv_col is not None:
        aggs.append(pl.col(iv_col).min().alias("_treat_dttm"))
    return preds.group_by(block).agg(aggs).sort(block)


def _arm_masks(label: np.ndarray, treated: np.ndarray) -> dict[str, np.ndarray]:
    """The three intervention arms as boolean masks over the stay rows."""
    return {
        "positive": label == 1,
        "treated_negative": (label == 0) & treated,
        "untreated_negative": (label == 0) & (~treated),
    }


def _alert_burden_row(m: dict[str, np.ndarray], band: dict[str, list],
                      tp: np.ndarray, fp: np.ndarray, i: int, n: int) -> dict:
    """Encounter-level alarm burden at row i (§4.2).

    One alert per hospitalization on the peak surface — no patient-day denominator.
    """
    ppv = None if np.isnan(m["ppv"][i]) else _r1(float(m["ppv"][i]))
    n_alerting = int(tp[i] + fp[i])
    if n_alerting < MIN_CELL_COUNT:
        ppv = None
    nne, _ = nne_from_ppv(ppv, band["ppv"][i])
    fr = m["flag_rate"][i]
    return {
        "frac_stays_alerting": None if np.isnan(fr) else round(float(fr), 4),
        "ppv": ppv,
        "false_alarm_fraction": None if ppv is None else round(1.0 - ppv, 4),
        "nne": nne,
        "n_stays": n,
        "n_stays_alerting": suppress(n_alerting),
    }


def _sub_rates_row(sub_point: dict, arms: dict[str, np.ndarray], i: int) -> dict:
    """Per-threshold ALARM COMPOSITION — each arm's SHARE of all firing stays (§4.4).

    One shared denominator: the single set of stays that alarm at t. The three arms
    partition every stay, so their shares sum to 1 (a "what is my alarm made of"
    read — real events vs treated non-events vs pure misfires). An arm below
    MIN_CELL_COUNT is nulled (HIPAA cell suppression), which breaks the sum. The
    per-arm CATCH rate (sensitivity / FPR) still lives in ``metrics``.
    """
    sizes, counts = {}, {}
    for arm in ("positive", "treated_negative", "untreated_negative"):
        sz = int(arms[arm].sum())
        r = sub_point[arm][i]
        sizes[arm] = sz
        counts[arm] = 0 if (sz == 0 or np.isnan(r)) else int(round(float(r) * sz))
    n_fired = sum(counts.values())

    def share(arm: str) -> float | None:
        if sizes[arm] < MIN_CELL_COUNT or n_fired == 0:
            return None
        return round(counts[arm] / n_fired, 4)

    return {
        "positive_share": share("positive"),
        "treated_negative_share": share("treated_negative"),
        "untreated_misfire_share": share("untreated_negative"),
        "n_alarms": suppress(n_fired),
    }


def _operating_budget(m: dict[str, np.ndarray], params: dict) -> list[dict]:
    """Sensitivity / PPV / NNE at a fixed encounter alert-rate (§4.3).

    Score scales differ between models, so "at threshold 0.3" is meaningless; the
    honest comparison is at equal workload = the encounter flag-rate. Invert the
    sweep: for each target, the grid row whose frac_stays_alerting is nearest.
    """
    fr = m["flag_rate"]
    if np.all(np.isnan(fr)):
        return []
    out = []
    for a in params["alert_rate_targets"]:
        i = int(np.nanargmin(np.abs(fr - a)))
        ppv = None if np.isnan(m["ppv"][i]) else _r1(float(m["ppv"][i]))
        out.append({
            "target_flag_rate": a,
            "threshold": round(float(SWEEP_GRID[i]), 2),
            "sensitivity": None if np.isnan(m["sensitivity"][i]) else _r1(float(m["sensitivity"][i])),
            "ppv": ppv,
            "nne": nne_from_ppv(ppv, None)[0],
        })
    return out


def _named_thresholds(t0: float, locators: list[dict]) -> list[float]:
    """Clinical + each located cutpoint, de-duplicated and sorted (§4.4)."""
    named = {round(float(t0), 2)}
    for loc in locators:
        if loc.get("threshold") is not None:
            named.add(round(float(loc["threshold"]), 2))
    return sorted(named)


def _timing_at(preds: pl.DataFrame, meta: dict, t: float, iv_col: str,
               label_col: str, direction: str, t0: float) -> dict:
    """Single median hours-before-treatment for the POSITIVE arm at threshold t (§4.4).

    A timing descriptor of an operating point, not a selection signal — one NUMBER
    (no distribution, no histogram, no CI). Scored on the blue-line population:
    label==1 (ever-positive) AND fired AND an attached treatment time — which for an
    intervention task is definitional (the label implies the treatment), so this is
    exactly the blue arm of the composition plot. positive = alarm fired early.
    """
    block = _BLOCK
    action = action_expr(t, direction)
    per = (preds.group_by(block).agg(
        (pl.col(label_col).max() > 0).cast(pl.Int8).alias("label"),
        pl.col("prediction_dttm").filter(action).min().alias("first_alert_dttm"),
        pl.col(iv_col).min().alias("_treat_dttm"),
    ).sort(block))
    hb = ((pl.col("_treat_dttm") - pl.col("first_alert_dttm"))
          .dt.total_seconds() / 3600)

    def median(frame: pl.DataFrame) -> float | None:
        fired = frame.filter(pl.col("first_alert_dttm").is_not_null()
                             & pl.col("_treat_dttm").is_not_null())
        if fired.height < MIN_CELL_COUNT:
            return None
        vals = fired.select(hb.alias("d"))["d"].to_numpy().astype(float)
        return round(float(np.median(vals)), 2)

    pos = per.filter(pl.col("label") == 1)
    return {
        "threshold": round(float(t), 2),
        "is_clinical": round(float(t), 2) == round(float(t0), 2),
        "median_hours_before_treatment_positive": median(pos),
    }


def _intervention_offsets(stay: pl.DataFrame,
                          window_unit: str | None) -> dict | None:
    """Mean ± SD treatment hospitalization time (the red reference line).

    Threshold-independent, computed once on the stay collapse. Places treatment on
    the forward landmark axis (the stay's first prediction window sits at 1),
    anchored on the stay START — never on the event or outcome.
    """
    if "_stay_start_dttm" not in stay.columns or "_treat_dttm" not in stay.columns:
        return None
    treated = stay.filter(pl.col("_treat_dttm").is_not_null())
    if treated.height < MIN_CELL_COUNT:
        return None
    hours_per_window = 24.0 if window_unit == "day" else 1.0
    off = (treated.select(
        ((pl.col("_treat_dttm") - pl.col("_stay_start_dttm")).dt.total_seconds()
         / 3600 / hours_per_window + 1).alias("o"))["o"].to_numpy().astype(float))
    return {
        "n": treated.height,
        "window_unit": window_unit,
        "mean_hospitalization_time": round(float(off.mean()), 2),
        "sd": round(float(off.std(ddof=1)), 2) if len(off) > 1 else None,
    }


def _rate_ci(indicator: np.ndarray) -> list[float] | None:
    """Stay-bootstrap 95% CI of a proportion (mean of a 0/1 indicator)."""
    return scalar_ci(indicator, stat=np.mean)


# ------------------------------------------------- per-threshold subgroup equity


def _fairness_by_threshold(preds: pl.DataFrame, meta: dict,
                           thresholds: list[float]) -> list[dict] | None:
    """Subgroup positive-rate / TPR / FPR + parity gaps, at EVERY grid threshold.

    Returns one dict per threshold (aligned with ``thresholds``), or None when the
    frame carries no demographics.

    Gaps are max-minus-min across subgroup levels, as in fairness.py, with its
    pinned-levels guard: the level set is fixed from the FULL data and a resample
    that loses a level is discarded whole (rather than shrinking the gap toward
    zero over a smaller group set). One resample index per replicate is reused
    across every threshold and column, so the gaps are mutually coherent.
    """
    df = _bin_age(preds)
    cols = [c for c in SUBGROUP_COLS if c in df.columns]
    if not cols:
        return None

    label = (df[meta["label_column"]].to_numpy() > 0).astype(int)
    prob = df["y_prob"].to_numpy().astype(float)
    preds_at = [prob >= t for t in thresholds]

    pinned: dict[str, tuple[np.ndarray, list[str]]] = {}
    for col in cols:
        values = df[col].to_numpy()
        levels = [lv for lv in dict.fromkeys(v for v in values if v is not None)
                  if int((values == lv).sum()) >= MIN_SUBGROUP_N]
        if len(levels) < 2:
            continue
        pinned[col] = (encode_levels(values, levels), [str(lv) for lv in levels])
    if not pinned:
        return None

    per_threshold: list[dict] = []
    for pred in preds_at:
        subgroups, parity = {}, {}
        for col, (codes, levels) in pinned.items():
            subgroups[col] = {
                lv: _level_rates(label[codes == k], pred[codes == k])
                for k, lv in enumerate(levels)
            }
            dp, tg, fg = _gaps(codes, label, pred, len(levels))
            parity[col] = {
                "reference_group": max(
                    levels, key=lambda lv: (int((codes == levels.index(lv)).sum()), lv)),
                "demographic_parity_gap": _r1(dp),
                "equalized_odds_tpr_gap": _r1(tg),
                "equalized_odds_fpr_gap": _r1(fg),
            }
        per_threshold.append({"subgroups": subgroups, "parity": parity})

    from flair_benchmark.report._ci import BOOT_REPS, BOOT_SEED
    from flair_benchmark.report._confusion import pct
    rng = np.random.default_rng(BOOT_SEED)
    n = len(label)
    samples: dict[tuple[int, str, str], list[float]] = {}
    for _ in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        lab_b = label[idx]
        preds_b = [pred[idx] for pred in preds_at]
        codes_b = {col: codes[idx] for col, (codes, _lv) in pinned.items()}
        for ti, pred_b in enumerate(preds_b):
            for col, (_codes, levels) in pinned.items():
                dp, tg, fg = _gaps(codes_b[col], lab_b, pred_b, len(levels))
                for name, val in (("demographic_parity_gap", dp),
                                  ("equalized_odds_tpr_gap", tg),
                                  ("equalized_odds_fpr_gap", fg)):
                    if val is not None:
                        samples.setdefault((ti, col, name), []).append(val)

    for ti, block in enumerate(per_threshold):
        for col in block["parity"]:
            for name in ("demographic_parity_gap", "equalized_odds_tpr_gap",
                         "equalized_odds_fpr_gap"):
                vals = samples.get((ti, col, name), [])
                block["parity"][col][f"{name}_ci"] = pct(vals, BOOT_REPS)
    return per_threshold


def _level_rates(label: np.ndarray, pred: np.ndarray) -> dict:
    """One subgroup level at one threshold: who gets flagged, at what error rates."""
    pos = label == 1
    neg = label == 0
    return {
        "n": int(len(label)),
        "positive_rate": _r1(float(pred.mean())) if len(pred) else None,
        "tpr": _r1(float(pred[pos].mean())) if pos.any() else None,
        "fpr": _r1(float(pred[neg].mean())) if neg.any() else None,
    }


# parity-gap arithmetic is single-sourced in fairness.py (gap_triplet) — the
# per-threshold sweep here and fairness.json must agree by construction.
_gaps = gap_triplet
