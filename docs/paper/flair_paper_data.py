#!/usr/bin/env python3
"""Single source of truth for every number in the FLAIR manuscript.

Two kinds of data live here and must never be confused:

  * REAL  -- observed FLAIR v2 baseline artifacts. MIMIC-CLIF is the internal
            holdout; Rush is the first external reporting site. These are the
            ONLY non-fabricated results in the paper.
  * SYNTH -- six illustrative sites (C-H) that project what a full CLIF
            consortium would look like. They are NOT collected data. Every
            record carries ``illustrative=True`` and a ``source`` tag, and every
            figure/table that renders them is required to label them as such.

The AUROC values for the synthetic sites are fixed (author-chosen to bracket the
observed MIMIC-internal / Rush-external anchors while preserving the real
task-difficulty ordering T1>T2>T3~=T4>T5 and keeping sepsis/readmission the most
fragile). Confidence intervals are derived deterministically from site size, so
there is no run-to-run variation and PDF/DOCX/figures cannot disagree.

Import this module from every figure script and from build_docx_v3.py. Run it
directly to dump ``sites_mock.json`` for non-Python consumers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Task metadata (all REAL: from flair_benchmark _constants.TASK_POLICY and the
# v2 MIMIC-CLIF cohort snapshot).
# ---------------------------------------------------------------------------
TASKS = [
    # id,   short,                       mode,       threshold, direction, prevalence(stay), grid
    ("T1", "ICU daily mortality",        "landmark", 0.95, "above", 0.134, "daily 07:00"),
    ("T2", "LTACH discharge",            "landmark", 0.50, "above", 0.065, "daily 07:00"),
    ("T3", "Extubation failure 24 h",    "episodic", 0.15, "below", 0.054, "at extubation"),
    ("T4", "Sepsis (CDC ASE) 6 h",       "peak",     0.10, "above", 0.063, "hourly"),
    ("T5", "Unplanned ICU readmission",  "episodic", 0.20, "above", 0.093, "at ICU discharge"),
]
TASK_IDS = [t[0] for t in TASKS]

# ---------------------------------------------------------------------------
# REAL observed AUROC and 95% CIs (FLAIR v2 baseline artifacts).
# ---------------------------------------------------------------------------
REAL = {
    "mimic": {
        "label": "MIMIC-CLIF",
        "archetype": "Public development set (MIMIC-IV, Boston quaternary)",
        "observed": True,
        "role": "internal holdout",
        "stays": 72778,
        "female_pct": 43.2,
        "age_median": 67,
        "sofa_median": 5,
        "auroc": {"T1": 0.864, "T2": 0.789, "T3": 0.696, "T4": 0.693, "T5": 0.641},
        "ci": {
            "T1": (0.856, 0.872), "T2": (0.775, 0.803), "T3": (0.665, 0.719),
            "T4": (0.678, 0.709), "T5": (0.626, 0.658),
        },
    },
    "rush": {
        "label": "Rush",
        "archetype": "Urban academic medical center (Chicago)",
        "observed": True,
        "role": "external (observed)",
        "stays": 13839,
        "female_pct": 47.5,
        "age_median": 65,
        "sofa_median": 4,
        "auroc": {"T1": 0.748, "T2": 0.658, "T3": 0.565, "T4": 0.407, "T5": 0.509},
        "ci": {
            "T1": (0.735, 0.763), "T2": (0.622, 0.691), "T3": (0.522, 0.605),
            "T4": (0.391, 0.421), "T5": (0.489, 0.531),
        },
    },
}

# Full MIMIC-CLIF internal metrics (REAL, Table S -- supplement).
MIMIC_FULL_METRICS = {
    # task: (mode, n, prevalence, auprc, cal_slope, cal_int, brier)
    "T1": ("landmark", 18195, 0.134, 0.587, 1.19, -1.24, 0.111),
    "T2": ("landmark", 18195, 0.065, 0.288, 1.01, -1.77, 0.096),
    "T3": ("episodic", 6693, 0.054, 0.134, 0.45, -1.86, 0.070),
    "T4": ("peak", 19943, 0.063, 0.134, None, None, None),
    "T5": ("episodic", 16806, 0.093, 0.248, 0.68, -1.89, 0.151),
}

# ---------------------------------------------------------------------------
# SYNTHETIC illustrative sites C-H (NOT observed data).
# AUROC values are fixed; CIs derived from site size below.
# ---------------------------------------------------------------------------
_SYNTH_SPEC = [
    # key, label, archetype, stays, female_pct, age, sofa, {task: auroc}
    ("siteC", "Site C", "Northeast quaternary academic", 24610, 44.1, 63, 6,
     {"T1": 0.828, "T2": 0.726, "T3": 0.651, "T4": 0.641, "T5": 0.598}),
    ("siteD", "Site D", "Midwest urban safety-net", 11205, 46.3, 58, 5,
     {"T1": 0.771, "T2": 0.671, "T3": 0.598, "T4": 0.501, "T5": 0.548}),
    ("siteE", "Site E", "Southern community-academic", 18470, 48.2, 64, 5,
     {"T1": 0.804, "T2": 0.702, "T3": 0.622, "T4": 0.573, "T5": 0.572}),
    ("siteF", "Site F", "West Coast academic", 22400, 44.9, 63, 5,
     {"T1": 0.812, "T2": 0.712, "T3": 0.634, "T4": 0.612, "T5": 0.586}),
    ("siteG", "Site G", "Rural referral center", 8720, 45.0, 69, 4,
     {"T1": 0.762, "T2": 0.663, "T3": 0.583, "T4": 0.462, "T5": 0.531}),
    ("siteH", "Site H", "Mixed academic-community", 15600, 47.1, 65, 5,
     {"T1": 0.789, "T2": 0.684, "T3": 0.607, "T4": 0.559, "T5": 0.557}),
]

# Per-task reference CI half-widths anchored to the observed Rush report
# (external, realistic widths), scaled by sqrt(n_rush / n_site) for each site.
_RUSH_HALFWIDTH = {t: (REAL["rush"]["ci"][t][1] - REAL["rush"]["ci"][t][0]) / 2 for t in TASK_IDS}
_RUSH_STAYS = REAL["rush"]["stays"]


def _synth_ci(stays: int, task: str, auroc: float) -> tuple[float, float]:
    """Deterministic 95% CI half-width scaled by site size; clamped to [0,1]."""
    hw = _RUSH_HALFWIDTH[task] * math.sqrt(_RUSH_STAYS / stays)
    lo = max(0.0, round(auroc - hw, 3))
    hi = min(1.0, round(auroc + hw, 3))
    return lo, hi


def _synth_auprc(task: str, auroc: float) -> float:
    """Illustrative AUPRC that degrades with AUROC, floored at prevalence."""
    prev = dict((t[0], t[5]) for t in TASKS)[task]
    ref_auprc = MIMIC_FULL_METRICS[task][3]
    ref_auroc = REAL["mimic"]["auroc"][task]
    scale = max(0.0, (auroc - 0.5)) / max(1e-6, (ref_auroc - 0.5))
    return round(max(prev, ref_auprc * scale), 3)


def _build_synth() -> dict:
    out = {}
    for key, label, arch, stays, fpct, age, sofa, aurocs in _SYNTH_SPEC:
        ci = {t: _synth_ci(stays, t, aurocs[t]) for t in TASK_IDS}
        auprc = {t: _synth_auprc(t, aurocs[t]) for t in TASK_IDS}
        out[key] = {
            "label": label,
            "archetype": arch,
            "observed": False,
            "illustrative": True,
            "source": "synthetic_pending_federated_run",
            "role": "external (illustrative)",
            "stays": stays,
            "female_pct": fpct,
            "age_median": age,
            "sofa_median": sofa,
            "auroc": dict(aurocs),
            "ci": ci,
            "auprc": auprc,
        }
    return out


SYNTH = _build_synth()

# Ordered site list used everywhere (observed first, then illustrative C-H).
SITE_ORDER = ["mimic", "rush", "siteC", "siteD", "siteE", "siteF", "siteG", "siteH"]
ALL_SITES = {**REAL, **SYNTH}

# Consortium palette (matches build_roc_figures.py / FLAIR website).
COLOR = {
    "mimic": "#5b86a6",   # blue  -- observed internal
    "rush": "#b9744e",    # terra -- observed external
    "synth": "#8c8c88",   # grey  -- illustrative
    "panel": "#f7f7f6", "grid": "#e7e7e4", "diag": "#c9c9c4",
    "muted": "#6b6b6b", "ink": "#1a1a1a", "band": "#f0e3da",
}


def external_sites() -> list[str]:
    """All external sites (observed Rush + illustrative C-H), in order."""
    return [s for s in SITE_ORDER if s != "mimic"]


def external_median(task: str) -> float:
    vals = sorted(ALL_SITES[s]["auroc"][task] for s in external_sites())
    n = len(vals)
    mid = n // 2
    return round((vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2), 3)


def internal_external_gap(task: str) -> float:
    return round(REAL["mimic"]["auroc"][task] - external_median(task), 3)


def consortium_total_stays() -> int:
    return sum(ALL_SITES[s]["stays"] for s in SITE_ORDER)


def external_total_stays() -> int:
    return sum(ALL_SITES[s]["stays"] for s in external_sites())


def fmt_ci(site: str, task: str) -> str:
    lo, hi = ALL_SITES[site]["ci"][task]
    return f"{ALL_SITES[site]['auroc'][task]:.3f} ({lo:.3f}-{hi:.3f})"


def dump_json(path: Path | None = None) -> Path:
    path = path or (HERE / "sites_mock.json")
    payload = {
        "note": "MIMIC-CLIF and Rush are observed; sites C-H are illustrative "
                "projections pending federated runs and are NOT collected data.",
        "tasks": [
            {"id": t[0], "name": t[1], "mode": t[2], "threshold": t[3],
             "direction": t[4], "prevalence_mimic": t[5], "grid": t[6]}
            for t in TASKS
        ],
        "sites": {s: ALL_SITES[s] for s in SITE_ORDER},
        "external_median": {t: external_median(t) for t in TASK_IDS},
        "internal_external_gap": {t: internal_external_gap(t) for t in TASK_IDS},
        "consortium_total_stays": consortium_total_stays(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    p = dump_json()
    print(f"wrote {p}")
    print("\nExternal AUROC medians and internal->external gaps:")
    for tid, short, *_ in TASKS:
        print(f"  {tid} {short:28s} median={external_median(tid):.3f} "
              f"gap={internal_external_gap(tid):+.3f}")
    print(f"\nConsortium total stays: {consortium_total_stays():,} "
          f"(external only: {external_total_stays():,}) across {len(SITE_ORDER)} sites")
