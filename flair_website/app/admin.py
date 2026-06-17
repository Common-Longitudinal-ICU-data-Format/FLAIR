"""Benchmark-committee review queue. Admin-only (user.role == 'admin'). Lists
model submissions, lets the committee approve/reject and push an approved
submission to sites as run requests carrying the model + its links."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from . import auth, db
from .data import task_by_id
from .templates import templates

router = APIRouter()


def _gate(request: Request):
    return auth.is_admin(request)


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


def _task_labeler():
    def label(tid: str) -> str:
        t = task_by_id(tid)
        return f"T{t['n']} · {t['display_name']}" if t else tid
    return label


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
    if action == "reviewing":
        db.update_submission_status(sub_id, "reviewing", notes)
    elif action == "approve":
        db.update_submission_status(sub_id, "approved", notes)
    elif action == "reject":
        db.update_submission_status(sub_id, "rejected", notes)
    elif action == "push":
        db.push_submission_to_sites(sub)
        db.update_submission_status(sub_id, "pushed", notes or sub.get("review_notes", ""))
    return RedirectResponse(f"/admin/submission/{sub_id}", status_code=302)
