"""Thin sqlite3 layer (stdlib, no ORM). Holds writable site state: profiles,
the one lead user per site, per-field visibility, opt-in, and run status.
Metric numbers themselves live in the static snapshot (see data.py)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings

# Default visibility — every field group is a switch the site controls.
DEFAULT_VISIBILITY = {
    "identity": False,   # show real site name / abbreviation / PI publicly
    "metrics": True,     # include this site's AUROC column on the public board
    "fairness": True,
    "calibration": True,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    key           TEXT UNIQUE NOT NULL,
    anon_label    TEXT NOT NULL,
    site_name     TEXT DEFAULT '',
    abbreviation  TEXT DEFAULT '',
    pi_name       TEXT DEFAULT '',
    contact       TEXT DEFAULT '',
    opted_in      INTEGER NOT NULL DEFAULT 0,
    visibility    TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id       INTEGER NOT NULL REFERENCES sites(id),
    email         TEXT UNIQUE NOT NULL,
    password_hash BLOB NOT NULL,
    role          TEXT NOT NULL DEFAULT 'lead',
    fail_count    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS run_status (
    site_id    INTEGER NOT NULL REFERENCES sites(id),
    task       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'not-run',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (site_id, task)
);
CREATE TABLE IF NOT EXISTS submissions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name     TEXT NOT NULL,
    model_type     TEXT DEFAULT '',
    owner_name     TEXT DEFAULT '',
    owner_email    TEXT NOT NULL,
    is_site_lead   INTEGER NOT NULL DEFAULT 0,
    github_url     TEXT DEFAULT '',
    docker_url     TEXT DEFAULT '',
    model_url      TEXT DEFAULT '',
    tasks          TEXT NOT NULL DEFAULT '[]',
    target_sites   TEXT NOT NULL DEFAULT '"all"',
    wants_site_reg INTEGER NOT NULL DEFAULT 0,
    reg_site_name  TEXT DEFAULT '',
    reg_abbrev     TEXT DEFAULT '',
    reg_pi         TEXT DEFAULT '',
    notes          TEXT DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'submitted',
    review_notes   TEXT DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    reviewed_at    TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS run_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER REFERENCES submissions(id),
    site_id       INTEGER REFERENCES sites(id),   -- NULL = all sites
    model_name    TEXT NOT NULL,
    owner_name    TEXT DEFAULT '',
    github_url    TEXT DEFAULT '',
    docker_url    TEXT DEFAULT '',
    model_url     TEXT DEFAULT '',
    tasks         TEXT NOT NULL DEFAULT '[]',
    note          TEXT DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'open',
    requested_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _row_to_site(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    vis = dict(DEFAULT_VISIBILITY)
    try:
        vis.update(json.loads(d.get("visibility") or "{}"))
    except json.JSONDecodeError:
        pass
    d["visibility"] = vis
    d["opted_in"] = bool(d["opted_in"])
    return d


def list_sites() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sites ORDER BY anon_label").fetchall()
    return [_row_to_site(r) for r in rows]


def get_site(site_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sites WHERE id = ?", (site_id,)).fetchone()
    return _row_to_site(row) if row else None


def get_site_by_key(key: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sites WHERE key = ?", (key,)).fetchone()
    return _row_to_site(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    return dict(row) if row else None


def create_site(key: str, anon_label: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sites (key, anon_label, visibility) VALUES (?, ?, ?)",
            (key, anon_label, json.dumps(DEFAULT_VISIBILITY)),
        )
        return cur.lastrowid


def create_user(site_id: int, email: str, password_hash: bytes, role: str = "lead") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (site_id, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (site_id, email.lower(), password_hash, role),
        )
        return cur.lastrowid


def get_user(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def update_profile(site_id: int, *, site_name: str, abbreviation: str, pi_name: str,
                   contact: str, opted_in: bool, visibility: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE sites SET site_name=?, abbreviation=?, pi_name=?, contact=?,
               opted_in=?, visibility=? WHERE id=?""",
            (site_name, abbreviation, pi_name, contact, int(opted_in),
             json.dumps(visibility), site_id),
        )


def get_run_status(site_id: int) -> dict[str, str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT task, status FROM run_status WHERE site_id = ?", (site_id,)
        ).fetchall()
    return {r["task"]: r["status"] for r in rows}


def set_run_status(site_id: int, task: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO run_status (site_id, task, status, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(site_id, task) DO UPDATE SET status=excluded.status,
               updated_at=datetime('now')""",
            (site_id, task, status),
        )


def _row_to_request(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["tasks"] = json.loads(d.get("tasks") or "[]")
    except json.JSONDecodeError:
        d["tasks"] = []
    return d


def list_run_requests(site_id: int | None = None) -> list[dict]:
    """All requests, or those a given site should see (its own + the all-sites ones)."""
    with get_conn() as conn:
        if site_id is None:
            rows = conn.execute(
                "SELECT * FROM run_requests ORDER BY requested_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM run_requests WHERE site_id IS NULL OR site_id = ? "
                "ORDER BY requested_at DESC", (site_id,)
            ).fetchall()
    return [_row_to_request(r) for r in rows]


# ---- submissions ----------------------------------------------------------

def _row_to_submission(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("tasks", "target_sites"):
        try:
            d[k] = json.loads(d.get(k) or ("[]" if k == "tasks" else '"all"'))
        except (json.JSONDecodeError, TypeError):
            d[k] = [] if k == "tasks" else "all"
    d["is_site_lead"] = bool(d["is_site_lead"])
    d["wants_site_reg"] = bool(d["wants_site_reg"])
    return d


def create_submission(**f) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO submissions
               (model_name, model_type, owner_name, owner_email, is_site_lead,
                github_url, docker_url, model_url, tasks, target_sites,
                wants_site_reg, reg_site_name, reg_abbrev, reg_pi, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f["model_name"], f.get("model_type", ""), f.get("owner_name", ""),
             f["owner_email"], int(f.get("is_site_lead", False)),
             f.get("github_url", ""), f.get("docker_url", ""), f.get("model_url", ""),
             json.dumps(f.get("tasks", [])), json.dumps(f.get("target_sites", "all")),
             int(f.get("wants_site_reg", False)), f.get("reg_site_name", ""),
             f.get("reg_abbrev", ""), f.get("reg_pi", ""), f.get("notes", "")),
        )
        return cur.lastrowid


def list_submissions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM submissions ORDER BY created_at DESC").fetchall()
    return [_row_to_submission(r) for r in rows]


def get_submission(sub_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM submissions WHERE id = ?", (sub_id,)).fetchone()
    return _row_to_submission(row) if row else None


def update_submission_status(sub_id: int, status: str, review_notes: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE submissions SET status=?, review_notes=?, reviewed_at=datetime('now') "
            "WHERE id=?", (status, review_notes, sub_id),
        )


def push_submission_to_sites(sub: dict) -> int:
    """Create run_requests from an approved submission. One global row when
    target_sites is 'all', otherwise one per named site key. Returns rows made."""
    targets = sub.get("target_sites")
    site_ids: list[int | None]
    if targets == "all" or not targets:
        site_ids = [None]
    else:
        site_ids = []
        for key in targets:
            s = get_site_by_key(key)
            site_ids.append(s["id"] if s else None)
        site_ids = site_ids or [None]

    with get_conn() as conn:
        for sid in site_ids:
            conn.execute(
                """INSERT INTO run_requests
                   (submission_id, site_id, model_name, owner_name, github_url,
                    docker_url, model_url, tasks, note)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (sub["id"], sid, sub["model_name"], sub.get("owner_name", ""),
                 sub.get("github_url", ""), sub.get("docker_url", ""),
                 sub.get("model_url", ""), json.dumps(sub.get("tasks", [])),
                 f"Submitted by {sub.get('owner_name') or sub['owner_email']}"),
            )
    return len(site_ids)


def record_login_fail(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET fail_count = fail_count + 1 WHERE id = ?", (user_id,))


def reset_login_fail(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET fail_count = 0 WHERE id = ?", (user_id,))
