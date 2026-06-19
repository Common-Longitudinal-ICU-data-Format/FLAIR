"""FLAIR 101 — public, no-login documentation pages.

Hand-built Jinja pages that render the same content as the repo's docs/101/*.md
(keep the two in sync). Each route passes a `page` key so the shared docs sidebar
can highlight the active entry."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .templates import templates

router = APIRouter()

# (page key = template name, URL, sidebar label) — also the prev/next ordering.
PAGES = [
    ("index", "/docs", "Overview"),
    ("clif", "/docs/clif", "CLIF data layer"),
    ("tasks", "/docs/tasks", "Tasks & task maker"),
    ("features", "/docs/features", "Feature engineering"),
    ("reports", "/docs/reports", "Building reports"),
    ("baseline", "/docs/baseline", "Reference baseline"),
]


def _render(request: Request, page: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, f"docs/{page}.html", {"page": page, "pages": PAGES}
    )


@router.get("/docs", response_class=HTMLResponse)
def docs_index(request: Request):
    return _render(request, "index")


@router.get("/docs/clif", response_class=HTMLResponse)
def docs_clif(request: Request):
    return _render(request, "clif")


@router.get("/docs/tasks", response_class=HTMLResponse)
def docs_tasks(request: Request):
    return _render(request, "tasks")


@router.get("/docs/features", response_class=HTMLResponse)
def docs_features(request: Request):
    return _render(request, "features")


@router.get("/docs/reports", response_class=HTMLResponse)
def docs_reports(request: Request):
    return _render(request, "reports")


@router.get("/docs/baseline", response_class=HTMLResponse)
def docs_baseline(request: Request):
    return _render(request, "baseline")
