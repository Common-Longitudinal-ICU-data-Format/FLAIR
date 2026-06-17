"""Public, no-login pages: landing, task catalog, model-driven leaderboard, the
JSON the leaderboard reads, and the public model-submission form."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import auth, db
from .data import (get_report, models, public_leaderboard_by_task,
                   site_publicly_visible, task_by_id, task_models, tasks)
from .templates import templates

router = APIRouter()


def _stats() -> dict:
    n_sites = sum(1 for s in db.list_sites() if s["opted_in"])
    return {"n_tasks": len(tasks()), "n_models": len(models()), "n_sites": max(n_sites, 1)}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"tasks": tasks(), "stats": _stats()}
    )


@router.get("/tasks", response_class=HTMLResponse)
def task_catalog(request: Request):
    return templates.TemplateResponse(request, "tasks.html", {"tasks": tasks()})


@router.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(request: Request, task_id: str):
    t = task_by_id(task_id)
    if t is None:
        return HTMLResponse("Task not found", status_code=404)
    return templates.TemplateResponse(
        request, "task_detail.html",
        {"task": t, "task_models": task_models(task_id, db.list_sites())},
    )


@router.get("/api/report/{model_id}/{site_key}/{task_id}")
def api_report(model_id: str, site_key: str, task_id: str):
    # Only expose a site's full report if that site is publicly visible.
    if not site_publicly_visible(site_key, db.list_sites()):
        return JSONResponse({"error": "not available"}, status_code=404)
    report = get_report(model_id, site_key, task_id)
    if report is None:
        return JSONResponse({"error": "no report"}, status_code=404)
    return JSONResponse(report)


@router.get("/leaderboard", response_class=HTMLResponse)
def leaderboard(request: Request):
    board = public_leaderboard_by_task(db.list_sites())
    return templates.TemplateResponse(request, "leaderboard.html", {"board": board})


@router.get("/api/leaderboard")
def api_leaderboard():
    return JSONResponse(public_leaderboard_by_task(db.list_sites()))


# ---- model submission (public, no login) ----------------------------------

@router.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request, error: str | None = None):
    return templates.TemplateResponse(
        request, "submit.html", {"tasks": tasks(), "error": error, "form": {}}
    )


@router.post("/submit")
async def submit_create(request: Request):
    form = await request.form()
    if not auth.check_csrf(request, form.get("csrf")):
        return templates.TemplateResponse(
            request, "submit.html",
            {"tasks": tasks(), "error": "Session expired, try again.", "form": {}},
            status_code=400,
        )

    model_name = (form.get("model_name") or "").strip()
    owner_email = (form.get("owner_email") or "").strip()
    github_url = (form.get("github_url") or "").strip()
    selected_tasks = [t["id"] for t in tasks() if form.get(f"task_{t['id']}") == "on"]

    errors = []
    if not model_name:
        errors.append("Model name is required.")
    if "@" not in owner_email:
        errors.append("A valid contact email is required.")
    if not (github_url.startswith("http://") or github_url.startswith("https://")):
        errors.append("A GitHub (or code) URL is required.")
    if not selected_tasks:
        errors.append("Select at least one task to run.")
    if errors:
        return templates.TemplateResponse(
            request, "submit.html",
            {"tasks": tasks(), "error": " ".join(errors), "form": dict(form),
             "selected": selected_tasks},
            status_code=400,
        )

    all_sites = form.get("target_all") == "on"
    target_sites = "all" if all_sites else [
        s.strip() for s in (form.get("target_sites") or "").split(",") if s.strip()
    ] or "all"

    sub_id = db.create_submission(
        model_name=model_name,
        model_type=(form.get("model_type") or "").strip(),
        owner_name=(form.get("owner_name") or "").strip(),
        owner_email=owner_email,
        is_site_lead=form.get("is_site_lead") == "on",
        github_url=github_url,
        docker_url=(form.get("docker_url") or "").strip(),
        model_url=(form.get("model_url") or "").strip(),
        tasks=selected_tasks,
        target_sites=target_sites,
        wants_site_reg=form.get("wants_site_reg") == "on",
        reg_site_name=(form.get("reg_site_name") or "").strip(),
        reg_abbrev=(form.get("reg_abbrev") or "").strip(),
        reg_pi=(form.get("reg_pi") or "").strip(),
        notes=(form.get("notes") or "").strip(),
    )
    return templates.TemplateResponse(
        request, "submit_thanks.html", {"sub_id": sub_id, "model_name": model_name}
    )
