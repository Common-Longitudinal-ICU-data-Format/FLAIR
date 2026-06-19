"""Model run-request form (GET + POST). DB-backed and write-bearing, so it is
mounted ONLY in internal mode (FLAIR_PUBLIC_MODE=false), never on the public site.

A request carries the model's code/image/weights links, a MIMIC-CLIF attestation,
and an uploaded `flair build-report` audit bundle the committee verifies before
posting the run to all sites."""
from __future__ import annotations

import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from . import auth, db
from .config import settings
from .data import tasks
from .templates import templates

router = APIRouter()

# Accepted audit-bundle extensions and a hard size cap (internal upload).
AUDIT_EXTS = (".json", ".zip", ".tar.gz", ".tgz")
AUDIT_MAX_BYTES = 25 * 1024 * 1024
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(filename: str) -> str:
    """basename only + conservative character whitelist (no path traversal)."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return _SAFE.sub("_", base) or "audit"


def _render(request: Request, *, error: str, form: dict, selected: list[str], code=400):
    return templates.TemplateResponse(
        request, "submit.html",
        {"tasks": tasks(), "error": error, "form": form, "selected": selected},
        status_code=code,
    )


@router.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request, error: str | None = None):
    return templates.TemplateResponse(
        request, "submit.html", {"tasks": tasks(), "error": error, "form": {}}
    )


@router.post("/submit")
async def submit_create(request: Request):
    form = await request.form()
    if not auth.check_csrf(request, form.get("csrf")):
        return _render(request, error="Session expired, try again.", form={}, selected=[])

    model_name = (form.get("model_name") or "").strip()
    owner_email = (form.get("owner_email") or "").strip()
    github_url = (form.get("github_url") or "").strip()
    selected_tasks = [t["id"] for t in tasks() if form.get(f"task_{t['id']}") == "on"]
    tested_mimic = form.get("tested_mimic") == "on"
    audit = form.get("audit_report")  # UploadFile | None

    # str-only echo for re-rendering the form on error (drop the UploadFile).
    form_echo = {k: v for k, v in form.items() if isinstance(v, str)}

    errors = []
    if not model_name:
        errors.append("Model name is required.")
    if "@" not in owner_email:
        errors.append("A valid contact email is required.")
    if not (github_url.startswith("http://") or github_url.startswith("https://")):
        errors.append("A GitHub (or code) URL is required.")
    if not selected_tasks:
        errors.append("Select at least one task to run.")
    if not tested_mimic:
        errors.append("Confirm the model was run on MIMIC-CLIF with `flair build-report`.")

    audit_bytes = b""
    audit_name = ""
    filename = getattr(audit, "filename", "") or ""
    if not filename:
        errors.append("Attach the MIMIC-CLIF audit report bundle.")
    else:
        if not filename.lower().endswith(AUDIT_EXTS):
            errors.append("Audit report must be a .json, .zip, or .tar.gz file.")
        else:
            audit_bytes = await audit.read()
            if len(audit_bytes) > AUDIT_MAX_BYTES:
                errors.append("Audit report exceeds the 25 MB limit.")
            elif not audit_bytes:
                errors.append("The audit report file is empty.")
            else:
                audit_name = _safe_name(filename)

    if errors:
        return _render(request, error=" ".join(errors), form=form_echo,
                       selected=selected_tasks)

    # Committee always posts approved runs to all sites; no submitter targeting.
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
        target_sites="all",
        wants_site_reg=form.get("wants_site_reg") == "on",
        reg_site_name=(form.get("reg_site_name") or "").strip(),
        reg_abbrev=(form.get("reg_abbrev") or "").strip(),
        reg_pi=(form.get("reg_pi") or "").strip(),
        notes=(form.get("notes") or "").strip(),
        tested_mimic=True,
    )

    audit_dir = settings.uploads_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rel = f"audit/{sub_id}-{audit_name}"
    (settings.uploads_dir / rel).write_bytes(audit_bytes)
    db.set_audit_report_path(sub_id, rel)

    return templates.TemplateResponse(
        request, "submit_thanks.html", {"sub_id": sub_id, "model_name": model_name}
    )
