"""Password hashing (bcrypt), session login bound to a (user, site) pair, and
a per-session CSRF token. No external auth service — credentials live in the DB."""
from __future__ import annotations

import secrets

import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse

from . import db
from .config import settings


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash)
    except (ValueError, TypeError):
        return False


def login_session(request: Request, user: dict) -> None:
    """Bind the session to the user AND their site → verifiable site↔lead link."""
    request.session["user_id"] = user["id"]
    request.session["site_id"] = user["site_id"]
    request.session["email"] = user["email"]


def logout_session(request: Request) -> None:
    request.session.clear()


def authenticate(email: str, password: str) -> tuple[dict | None, str | None]:
    """Returns (user, error). Throttles after settings.max_login_fails fails."""
    user = db.get_user_by_email(email)
    if not user:
        return None, "Invalid email or password."
    if user["fail_count"] >= settings.max_login_fails:
        return None, "Account locked after too many attempts. Contact the coordinating center."
    if not verify_password(password, user["password_hash"]):
        db.record_login_fail(user["id"])
        return None, "Invalid email or password."
    db.reset_login_fail(user["id"])
    return user, None


def current_site(request: Request) -> dict | None:
    """The logged-in site, or None. Used to gate portal routes."""
    site_id = request.session.get("site_id")
    if not site_id:
        return None
    return db.get_site(site_id)


def require_site(request: Request) -> dict | RedirectResponse:
    site = current_site(request)
    if site is None:
        return RedirectResponse("/login", status_code=302)
    return site


def current_user(request: Request) -> dict | None:
    uid = request.session.get("user_id")
    return db.get_user(uid) if uid else None


def is_admin(request: Request) -> bool:
    u = current_user(request)
    return bool(u and u.get("role") == "admin")


# ---- CSRF -----------------------------------------------------------------

def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


def check_csrf(request: Request, submitted: str | None) -> bool:
    expected = request.session.get("csrf")
    return bool(expected and submitted and secrets.compare_digest(expected, submitted))
