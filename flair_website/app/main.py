"""FLAIR website entrypoint.

Public mode (default, FLAIR_PUBLIC_MODE=true): a READ-ONLY site that serves only the
pre-built `data/` snapshot — public + docs routers, static assets, no database, no
sessions, no write endpoints. Nothing a visitor can reach mutates state or exposes
anything beyond the committed public aggregates.

Internal mode (FLAIR_PUBLIC_MODE=false): additionally mounts the DB-backed management
app (portal + admin + submit) behind sessions, for running the consortium internally.

  uv run uvicorn app.main:app
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import docs, public
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.public_mode:
        from .db import init_db
        init_db()
    yield


app = FastAPI(title="FLAIR Benchmark", docs_url=None, redoc_url=None, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

# Always-on, read-only.
app.include_router(public.router)
app.include_router(docs.router)

# Internal-only: sessions + DB-backed write surface (portal, admin, submit).
if not settings.public_mode:
    from starlette.middleware.sessions import SessionMiddleware

    from . import admin, portal, submit
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        https_only=settings.secure_cookie,
        same_site="lax",
    )
    app.include_router(portal.router)
    app.include_router(admin.router)
    app.include_router(submit.router)
