"""Benchmark-committee console. Admin-only (user.role == 'admin').

Three surfaces:
  - /admin                      review queue of model run requests
  - /admin/submission/{id}      one request: links, MIMIC-CLIF audit, processing
                                lifecycle, and "post run to all sites"
  - /admin/runs                 orchestration 'own view': run × site status matrix
                                + each site's charted preferences
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import auth, db
from .config import settings
from .data import task_by_id
from .templates import templates

router = APIRouter()


def _gate(request: Request):
    return auth.is_admin(request)


def _task_labeler():
    def label(tid: str) -> str:
        t = task_by_id(tid)
        return f"T{t['n']} · {t['display_name']}" if t else tid
    return label


@router.get("/admin", response_class=HTMLResponse)
def queue(request: Request):
    if not _gate(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "admin/queue.html", {"submissions": db.list_submissions()}
    )


@router.get("/admin/submission/{sub_id}", response_class=HTMLResponse)
def submission_detail(request: Request, sub_id: int):
    if not _gate(request):
        return RedirectResponse("/login", status_code=302)
    sub = db.get_submission(sub_id)
    if not sub:
        return HTMLResponse("Submission not found", status_code=404)
    return templates.TemplateResponse(
        request, "admin/submission.html",
        {"sub": sub, "task_label": _task_labeler()},
    )


@router.get("/admin/submission/{sub_id}/audit")
def submission_audit(request: Request, sub_id: int):
    """Stream the stored MIMIC-CLIF audit bundle. Admin-gated; the path comes
    from the DB (not the request) and is confined to uploads_dir."""
    if not _gate(request):
        return RedirectResponse("/login", status_code=302)
    sub = db.get_submission(sub_id)
    if not sub or not sub.get("audit_report_path"):
        return HTMLResponse("No audit report", status_code=404)
    root = settings.uploads_dir.resolve()
    path = (root / sub["audit_report_path"]).resolve()
    if root not in path.parents or not path.is_file():
        return HTMLResponse("No audit report", status_code=404)
    return FileResponse(path, filename=path.name)


@router.post("/admin/submission/{sub_id}")
async def submission_action(request: Request, sub_id: int):
    if not _gate(request):
        return RedirectResponse("/login", status_code=302)
    sub = db.get_submission(sub_id)
    if not sub:
        return HTMLResponse("Submission not found", status_code=404)
    form = await request.form()
    if not auth.check_csrf(request, form.get("csrf")):
        return RedirectResponse(f"/admin/submission/{sub_id}", status_code=302)

    action = form.get("action")
    notes = (form.get("review_notes") or "").strip()
    if action in ("reviewing", "processing", "approve", "reject"):
        status = {"reviewing": "reviewing", "processing": "processing",
                  "approve": "approved", "reject": "rejected"}[action]
        db.update_submission_status(sub_id, status, notes)
    elif action == "post":
        db.push_submission_to_sites(sub)
        db.update_submission_status(sub_id, "posted", notes or sub.get("review_notes", ""))
    return RedirectResponse(f"/admin/submission/{sub_id}", status_code=302)


@router.get("/admin/runs", response_class=HTMLResponse)
def runs(request: Request):
    if not _gate(request):
        return RedirectResponse("/login", status_code=302)
    grid = db.orchestration_grid()
    # Per-site preference + participation summary, charted in the template.
    accepting = {"accepted", "running", "results-shared"}
    site_stats = []
    for s in grid["sites"]:
        responded = acted = 0
        for req in grid["requests"]:
            resp = grid["cell"].get(req["id"], {}).get(s["id"])
            if resp:
                responded += 1
                if resp["status"] in accepting:
                    acted += 1
        site_stats.append({
            "site": s,
            "responded": responded,
            "accept_rate": (acted / responded) if responded else None,
        })
    return templates.TemplateResponse(
        request, "admin/runs.html",
        {"grid": grid, "site_stats": site_stats, "task_label": _task_labeler()},
    )
