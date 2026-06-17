"""Authenticated orchestration portal: login/logout, dashboard (run status +
requests), the site's own unblinded scores, artifacts, and the editable profile
with opt-in + per-field visibility switches."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from . import auth, db
from .data import site_model_scores, tasks
from .templates import templates

router = APIRouter()


# ---- auth -----------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: str | None = None):
    if auth.current_site(request):
        return RedirectResponse("/portal", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": error})


@router.post("/login")
def login_submit(request: Request, email: str = Form(...), password: str = Form(...),
                 csrf: str = Form("")):
    if not auth.check_csrf(request, csrf):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Session expired, try again."}, status_code=400
        )
    user, err = auth.authenticate(email, password)
    if err:
        return templates.TemplateResponse(
            request, "login.html", {"error": err}, status_code=401
        )
    auth.login_session(request, user)
    return RedirectResponse("/portal", status_code=302)


@router.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    if auth.check_csrf(request, csrf):
        auth.logout_session(request)
    return RedirectResponse("/", status_code=302)


# ---- gated pages ----------------------------------------------------------

def _gate(request: Request):
    return auth.current_site(request)


@router.get("/portal", response_class=HTMLResponse)
def dashboard(request: Request):
    site = _gate(request)
    if not site:
        return RedirectResponse("/login", status_code=302)
    status = db.get_run_status(site["id"])
    rows = [{"task": t, "status": status.get(t["id"], "not-run")} for t in tasks()]
    task_label = {t["id"]: f"T{t['n']} · {t['display_name']}" for t in tasks()}
    return templates.TemplateResponse(
        request, "portal/dashboard.html",
        {"site": site, "rows": rows,
         "requests": db.list_run_requests(site["id"]), "task_label": task_label},
    )


@router.get("/portal/scores", response_class=HTMLResponse)
def scores(request: Request):
    site = _gate(request)
    if not site:
        return RedirectResponse("/login", status_code=302)
    model_scores = site_model_scores(site["key"])
    return templates.TemplateResponse(
        request, "portal/scores.html", {"site": site, "model_scores": model_scores}
    )


@router.get("/portal/artifacts", response_class=HTMLResponse)
def artifacts(request: Request):
    site = _gate(request)
    if not site:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "portal/artifacts.html", {"site": site, "tasks": tasks()}
    )


@router.get("/portal/profile", response_class=HTMLResponse)
def profile_form(request: Request, saved: bool = False):
    site = _gate(request)
    if not site:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(
        request, "portal/profile.html", {"site": site, "saved": saved}
    )


@router.post("/portal/profile")
async def profile_save(request: Request):
    site = _gate(request)
    if not site:
        return RedirectResponse("/login", status_code=302)
    form = await request.form()
    if not auth.check_csrf(request, form.get("csrf")):
        return RedirectResponse("/portal/profile", status_code=302)

    visibility = {
        "identity": form.get("vis_identity") == "on",
        "metrics": form.get("vis_metrics") == "on",
        "fairness": form.get("vis_fairness") == "on",
        "calibration": form.get("vis_calibration") == "on",
    }
    db.update_profile(
        site["id"],
        site_name=(form.get("site_name") or "").strip(),
        abbreviation=(form.get("abbreviation") or "").strip(),
        pi_name=(form.get("pi_name") or "").strip(),
        contact=(form.get("contact") or "").strip(),
        opted_in=form.get("opted_in") == "on",
        visibility=visibility,
    )
    return RedirectResponse("/portal/profile?saved=1", status_code=302)
