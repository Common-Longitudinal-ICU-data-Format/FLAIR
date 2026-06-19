"""Build the website's public `data/` from raw per-site result dumps in `runs_data/`.

Each site drops its flair-baseline **non-PHI upload** folder into `runs_data/`:

    runs_data/<SITE>_baseline_non_phi_for_upload/<task>/
        table1.json
        codes.parquet                 (ignored here)
        report/
            discrimination.json
            calibration.json          (episodic + landmark only)
            dca.json                  (episodic + landmark only)
            fairness.json
            leadtime.json             (landmark only)
            viz/*.png

This script is the ONLY bridge between those raw dumps and what the public website
serves. It is deterministic and idempotent: it wipes and rewrites `data/reports/`
and `data/leaderboard.json` every run. Nothing from `runs_data/` is served directly
— the public site only ever reads the processed `data/`.

Run:  python scripts/build_data.py   (from the flair_website/ directory)

The per-task report mode (episodic / peak / landmark) is read from each pillar's
`metadata.report_mode`. Pillars are mode-shaped:
  * episodic / peak : flat  {metrics, curves}
  * landmark        : {metadata, landmarks:[12 × {metrics, curves}]} + leadtime.json
Peak omits calibration + dca (the peak-calibration trap).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUNS = BASE / "runs_data"
DATA = BASE / "data"
REPORTS = DATA / "reports"

UPLOAD_SUFFIX = "_baseline_non_phi_for_upload"

# Baseline model identity (the only model the backend ships today). Future models
# would be discovered the same way from differently-named dump folders.
MODEL = {
    "id": "flair-xgboost-baseline",
    "name": "FLAIR XGBoost baseline",
    "owner": "FLAIR Working Group",
    "type": "count-featurization + XGBoost",
    "github_url": "https://github.com/clif-consortium/FLAIR",
    "is_baseline": True,
    "split": {
        "type": "Random",
        "ratio": "75/25",
        "grain": "hospitalization (hospitalization_join_id)",
        "detail": "deterministic 75/25 hold-out grouped by hospitalization; MIMIC dates de-identified",
    },
}

# Friendly site labels; default to KEY.upper() when unlisted.
SITE_LABELS = {"mimic": "MIMIC-CLIF"}


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.is_file() else None


def _headline_landmark(pillar: dict) -> dict | None:
    """The landmark scored closest to each stay's endpoint (landmark_index 0)."""
    lms = pillar.get("landmarks") or []
    if not lms:
        return None
    return min(lms, key=lambda lm: lm.get("landmark_index", 0))


def _cell(disc: dict, calib: dict | None, landmark: bool) -> dict:
    """The leaderboard cell: headline metrics for one (model, site, task)."""
    if landmark:
        d = _headline_landmark(disc) or {}
        c = _headline_landmark(calib) if calib else None
        n_test = d.get("n")
        prevalence = d.get("prevalence")
        dm = d.get("metrics", {})
        cm = (c or {}).get("metrics", {}) if c else {}
    else:
        n_test = (disc.get("metadata") or {}).get("n_stays")
        prevalence = disc.get("prevalence")
        dm = disc.get("metrics", {})
        cm = (calib or {}).get("metrics", {}) if calib else {}

    return {
        "n_test": n_test,
        "prevalence": prevalence,
        "auroc": dm.get("auroc"),
        "auroc_ci": dm.get("auroc_ci"),
        "auprc": dm.get("auprc"),
        "cal_slope": cm.get("slope") if cm else None,
        "cal_intercept": cm.get("intercept") if cm else None,
        "brier": cm.get("brier") if cm else None,
    }


def _meta(disc: dict, leadtime: dict | None) -> dict:
    m = disc.get("metadata") or {}
    wu = (leadtime or {}).get("metadata", {}).get("window_unit") if leadtime else None
    return {
        "generated_at": m.get("generated_at"),
        "flair_version": m.get("flair_version"),
        "split": m.get("split"),
        "report_mode": m.get("report_mode"),
        "clinical_threshold": m.get("evaled_at"),
        "window_unit": wu,
    }


def build(runs: Path = RUNS, data_dir: Path = DATA) -> dict:
    if not runs.is_dir():
        sys.exit(f"no runs_data/ at {runs} — nothing to build")
    reports = data_dir / "reports"

    # Fresh reports tree every run (idempotent; drops anything stale).
    if reports.exists():
        shutil.rmtree(reports)
    reports.mkdir(parents=True, exist_ok=True)

    sites: dict[str, str] = {}
    scores: dict[str, dict] = {}
    flair_version = None

    site_dirs = sorted(d for d in runs.iterdir()
                       if d.is_dir() and d.name.endswith(UPLOAD_SUFFIX))
    if not site_dirs:
        sys.exit(f"no '*{UPLOAD_SUFFIX}' folders in {runs}")

    for site_dir in site_dirs:
        site = site_dir.name[: -len(UPLOAD_SUFFIX)]
        sites[site] = SITE_LABELS.get(site, site.upper())

        for task_dir in sorted(p for p in site_dir.iterdir() if p.is_dir()):
            task = task_dir.name
            rep = task_dir / "report"
            disc = _load(rep / "discrimination.json")
            if disc is None:
                print(f"  [skip] {site}/{task}: no discrimination.json")
                continue
            calib = _load(rep / "calibration.json")
            dca = _load(rep / "dca.json")
            fairness = _load(rep / "fairness.json")
            leadtime = _load(rep / "leadtime.json")
            table1 = _load(task_dir / "table1.json")

            mode = (disc.get("metadata") or {}).get("report_mode", "episodic")
            landmark = mode == "landmark"
            flair_version = flair_version or (disc.get("metadata") or {}).get("flair_version")

            # Leaderboard cell.
            scores.setdefault(task, {})[site] = _cell(disc, calib, landmark)

            # Combined report bundle (pillars copied verbatim).
            bundle = {
                "mode": mode,
                "landmark": landmark,
                "task": task,
                "site": site,
                "meta": _meta(disc, leadtime),
                "discrimination": disc,
                "calibration": calib,
                "dca": dca,
                "fairness": fairness,
                "leadtime": leadtime,
                "table1": table1,
            }
            out_dir = reports / MODEL["id"] / site / task
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report.json").write_text(json.dumps(bundle, indent=2, default=str))

            # Copy sanity-check PNGs (non-PHI, not served in v1 but kept for completeness).
            viz_src = rep / "viz"
            if viz_src.is_dir():
                shutil.copytree(viz_src, out_dir / "viz")

    board = {
        "_note": ("Built from runs_data/ by scripts/build_data.py. Real MIMIC-CLIF "
                  "numbers; more sites/models appear as they drop result folders."),
        "flair_version": flair_version,
        "sites": [{"key": k, "label": v} for k, v in sorted(sites.items())],
        "models": [{**MODEL, "scores": scores}],
    }
    (data_dir / "leaderboard.json").write_text(json.dumps(board, indent=2, default=str) + "\n")
    return board


if __name__ == "__main__":
    b = build()
    n_cells = sum(len(s) for s in b["models"][0]["scores"].values())
    print(f"built leaderboard: {len(b['sites'])} site(s), "
          f"{len(b['models'][0]['scores'])} task(s), {n_cells} cell(s)")
    print(f"sites: {[s['key'] for s in b['sites']]}")
