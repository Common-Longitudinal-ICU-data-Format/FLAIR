"""FLAIR website entrypoint. Mounts static assets, the session middleware, and
the public + portal routers. `uv run uvicorn app.main:app`."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import admin, public, portal
from .config import settings
from .db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FLAIR Benchmark", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    https_only=settings.secure_cookie,
    same_site="lax",
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

app.include_router(public.router)
app.include_router(portal.router)
app.include_router(admin.router)
