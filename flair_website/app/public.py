"""Public, no-login, READ-ONLY pages: landing, task catalog, model-driven
leaderboard, the JSON the leaderboard reads, and per-report JSON.

This module never touches the database. Every site/model/metric it serves comes
from the committed, pre-built `data/` snapshot (see scripts/build_data.py), so the
public deployment needs no DB, no sessions, and has no write surface."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .data import (board_sites, get_report, models, public_leaderboard_by_task,
                   task_by_id, task_models, tasks)
from .templates import templates

router = APIRouter()


def _stats() -> dict:
    return {"n_tasks": len(tasks()), "n_models": len(models()),
            "n_sites": max(len(board_sites()), 1)}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"tasks": tasks(), "stats": _stats()}
    )


def _task_leaderboard(request: Request):
    """Merged Task Leaderboard: all tasks listed, each expandable to its ranking
    and per-model detailed reports. Served at both /leaderboard and /tasks."""
    return templates.TemplateResponse(
        request, "task_leaderboard.html", {"board": public_leaderboard_by_task()}
    )


@router.get("/tasks", response_class=HTMLResponse)
def task_catalog(request: Request):
    return _task_leaderboard(request)


@router.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(request: Request, task_id: str):
    t = task_by_id(task_id)
    if t is None:
        return HTMLResponse("Task not found", status_code=404)
    return templates.TemplateResponse(
        request, "task_detail.html",
        {"task": t, "task_models": task_models(task_id)},
    )


@router.get("/api/report/{model_id}/{site_key}/{task_id}")
def api_report(model_id: str, site_key: str, task_id: str):
    # Everything under data/ is intended-public; gate purely on the file existing.
    report = get_report(model_id, site_key, task_id)
    if report is None:
        return JSONResponse({"error": "no report"}, status_code=404)
    return JSONResponse(report)


@router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request):
    return _task_leaderboard(request)


@router.get("/api/leaderboard")
def api_leaderboard():
    return JSONResponse(public_leaderboard_by_task())
