"""Read-only access to the committed model-centric leaderboard snapshot + task
catalog, and the per-task model ranking that respects each site's opt-in and
per-field visibility (read from the DB)."""
from __future__ import annotations

import json
import math
from functools import lru_cache

from .config import settings


def _finite(o):
    """Replace NaN/Inf with None so the payload is strict-JSON serializable."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_finite(x) for x in o]
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    return o


@lru_cache(maxsize=1)
def _tasks_raw() -> dict:
    return json.loads((settings.data_dir / "tasks.json").read_text())


@lru_cache(maxsize=1)
def _board_raw() -> dict:
    return json.loads((settings.data_dir / "leaderboard.json").read_text())


def tasks() -> list[dict]:
    return _tasks_raw()["tasks"]


def task_by_id(task_id: str) -> dict | None:
    return next((t for t in tasks() if t["id"] == task_id), None)


def models() -> list[dict]:
    return _board_raw()["models"]


def board_sites() -> list[dict]:
    """Public site columns baked into leaderboard.json by scripts/build_data.py.
    The public site reads these instead of the (PII-bearing) DB."""
    return _board_raw().get("sites", [])


def model_by_id(model_id: str) -> dict | None:
    return next((m for m in models() if m["id"] == model_id), None)


def snapshot_site_keys() -> list[str]:
    """Distinct site keys appearing in any model's scores (mimic first)."""
    keys: set[str] = set()
    for m in models():
        for tscores in m["scores"].values():
            keys.update(tscores.keys())
    ordered = sorted(keys, key=lambda k: (k != "mimic", k))
    return ordered


def model_meta(model: dict) -> dict:
    return {k: model[k] for k in ("id", "name", "owner", "type", "github_url",
                                  "is_baseline", "split")
            if k in model}


def _weighted_pool(rows: list[dict]) -> dict | None:
    """Sample-size (n_test) weighted mean AUROC + pooled CI over included sites."""
    rows = [r for r in rows if r and r.get("auroc") is not None]
    if not rows:
        return None
    total = sum(r["n_test"] for r in rows) or 1
    auroc = sum(r["auroc"] * r["n_test"] for r in rows) / total
    lo = sum(r["auroc_ci"][0] * r["n_test"] for r in rows) / total
    hi = sum(r["auroc_ci"][1] * r["n_test"] for r in rows) / total
    return {
        "auroc": round(auroc, 4),
        "auroc_ci": [round(lo, 4), round(hi, 4)],
        "n_sites": len(rows),
        "n_test": total,
    }


def _visible_sites(sites: list[dict] | None) -> list[dict]:
    # Every site baked into leaderboard.json is intended-public by construction
    # (the parser only writes public, cell-suppressed aggregates).
    return list(sites if sites is not None else board_sites())


def _site_label(site: dict) -> str:
    return site.get("label") or site["key"].upper()


def public_leaderboard_by_task(sites: list[dict] | None = None) -> dict:
    """Per task: site columns + models ranked best-first by pooled AUROC, each
    with its per-site AUROC cells. Site columns come from leaderboard.json."""
    visible = _visible_sites(sites)
    columns = [
        {"key": s["key"], "label": _site_label(s), "identified": True}
        for s in visible
    ]
    site_keys = [s["key"] for s in visible]

    by_task = {}
    for t in tasks():
        rows = []
        for m in models():
            tscores = m["scores"].get(t["id"], {})
            cells = {}
            pool_rows = []
            for key in site_keys:
                sc = tscores.get(key)
                if not sc:
                    continue
                cells[key] = {"auroc": sc["auroc"], "auroc_ci": sc["auroc_ci"],
                              "n_test": sc["n_test"],
                              "has_report": report_exists(m["id"], key, t["id"])}
                pool_rows.append(sc)
            pooled = _weighted_pool(pool_rows)
            if pooled is None:
                continue
            rows.append({**model_meta(m), "pooled": pooled, "cells": cells})
        rows.sort(key=lambda r: r["pooled"]["auroc"], reverse=True)
        by_task[t["id"]] = rows

    return {"columns": columns, "tasks": tasks(), "by_task": by_task}


def _report_path(model_id: str, site_key: str, task_id: str):
    return settings.data_dir / "reports" / model_id / site_key / task_id / "report.json"


def report_exists(model_id: str, site_key: str, task_id: str) -> bool:
    return _report_path(model_id, site_key, task_id).is_file()


def get_report(model_id: str, site_key: str, task_id: str) -> dict | None:
    p = _report_path(model_id, site_key, task_id)
    if not p.is_file():
        return None
    return _finite(json.loads(p.read_text()))


def task_models(task_id: str, sites: list[dict] | None = None) -> list[dict]:
    """Per-task ranked models (public view), each annotated with which visible
    sites have a full report bundle available."""
    visible = _visible_sites(sites)
    cols = [{"key": s["key"], "label": _site_label(s)} for s in visible]
    out = []
    for m in models():
        tscores = m["scores"].get(task_id, {})
        pool_rows, cells, site_avail = [], {}, []
        for s, c in zip(visible, cols):
            sc = tscores.get(s["key"])
            if not sc:
                continue
            cells[s["key"]] = sc
            pool_rows.append(sc)
            site_avail.append({**c, "has_report": report_exists(m["id"], s["key"], task_id)})
        pooled = _weighted_pool(pool_rows)
        if pooled is None:
            continue
        out.append({**model_meta(m), "pooled": pooled, "sites": site_avail})
    out.sort(key=lambda r: r["pooled"]["auroc"], reverse=True)
    return out


def site_publicly_visible(site_key: str, sites: list[dict] | None = None) -> bool:
    return any(s["key"] == site_key for s in _visible_sites(sites))


def site_model_scores(site_key: str) -> list[dict]:
    """For the portal 'my scores': per model, this site's own per-task numbers."""
    out = []
    for m in models():
        rows = []
        for t in tasks():
            rows.append({"task": t, "score": m["scores"].get(t["id"], {}).get(site_key)})
        out.append({"model": model_meta(m), "rows": rows})
    return out
