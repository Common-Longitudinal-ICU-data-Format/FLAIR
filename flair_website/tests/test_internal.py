"""Internal (FLAIR_PUBLIC_MODE=false) management app: run-request upload + audit,
committee orchestration, and site accept/reject lifecycle.

Builds the internal app in-process against a temp DB + uploads dir, so the public
read-only fixtures in conftest are untouched."""
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

TASK = "task1_icu_daily_mortality"


@pytest.fixture
def internal(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "public_mode", False)
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")

    from app import admin, auth, db, docs, portal, public, submit
    from app.templates import templates as tpl
    monkeypatch.setitem(tpl.env.globals, "public_mode", False)

    db.init_db()
    # one committee admin + one site with a lead
    csite = db.create_site("committee", "—")
    db.create_user(csite, "admin@flair", auth.hash_password("pw"), role="admin")
    sid = db.create_site("mimic", "A")
    db.create_user(sid, "lead@mimic", auth.hash_password("pw"), role="lead")

    application = FastAPI()
    application.add_middleware(SessionMiddleware, secret_key="test", same_site="lax")
    for r in (public.router, docs.router, portal.router, admin.router, submit.router):
        application.include_router(r)
    return TestClient(application), db, auth, sid


def _csrf(client, path="/login"):
    m = re.search(r'name="csrf" value="([^"]+)"', client.get(path).text)
    assert m, f"no csrf token on {path}"
    return m.group(1)


def _login(client, email):
    csrf = _csrf(client, "/login")
    r = client.post("/login", data={"email": email, "password": "pw", "csrf": csrf},
                    follow_redirects=False)
    assert r.status_code == 302
    return csrf


# ---- submit / audit upload -------------------------------------------------

def test_submit_requires_attestation_and_audit(internal):
    client, db, _auth, _sid = internal
    csrf = _csrf(client, "/submit")
    base = {"csrf": csrf, "model_name": "M", "owner_email": "a@b.c",
            "github_url": "https://github.com/x/y", f"task_{TASK}": "on"}
    # no attestation, no file → rejected, nothing written
    r = client.post("/submit", data=base)
    assert r.status_code == 400
    assert "mimic" in r.text.lower() and "audit" in r.text.lower()
    assert db.list_submissions() == []


def test_submit_with_audit_stores_file(internal):
    client, db, _auth, _sid = internal
    csrf = _csrf(client, "/submit")
    r = client.post(
        "/submit",
        data={"csrf": csrf, "model_name": "M", "owner_email": "a@b.c",
              "github_url": "https://github.com/x/y", f"task_{TASK}": "on",
              "tested_mimic": "on"},
        files={"audit_report": ("report.json", b'{"auroc": 0.9}', "application/json")},
    )
    assert r.status_code == 200
    subs = db.list_submissions()
    assert len(subs) == 1
    sub = subs[0]
    assert sub["tested_mimic"] is True
    assert sub["audit_report_path"].startswith("audit/")
    from app.config import settings
    assert (settings.uploads_dir / sub["audit_report_path"]).is_file()


def test_submit_rejects_bad_extension(internal):
    client, db, _auth, _sid = internal
    csrf = _csrf(client, "/submit")
    r = client.post(
        "/submit",
        data={"csrf": csrf, "model_name": "M", "owner_email": "a@b.c",
              "github_url": "https://github.com/x/y", f"task_{TASK}": "on",
              "tested_mimic": "on"},
        files={"audit_report": ("evil.exe", b"x", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert db.list_submissions() == []


# ---- site accept/reject lifecycle ------------------------------------------

def _post_a_run(db):
    sub_id = db.create_submission(model_name="M", owner_email="a@b.c",
                                  github_url="https://github.com/x/y", tasks=[TASK],
                                  target_sites="all", tested_mimic=True)
    db.push_submission_to_sites(db.get_submission(sub_id))
    return db.list_run_requests(None)[0]["id"]


def test_site_accept_then_lifecycle_with_anonymity(internal):
    client, db, _auth, sid = internal
    rid = _post_a_run(db)
    csrf = _login(client, "lead@mimic")

    client.post(f"/portal/runs/{rid}/respond",
                data={"csrf": csrf, "action": "accept", "anonymous": "on"})
    resp = db.get_run_response(rid, sid)
    assert resp["status"] == "accepted" and resp["anonymous"] is True

    # anonymity carries forward when no checkbox is sent
    client.post(f"/portal/runs/{rid}/respond", data={"csrf": csrf, "action": "running"})
    resp = db.get_run_response(rid, sid)
    assert resp["status"] == "running" and resp["anonymous"] is True

    client.post(f"/portal/runs/{rid}/respond", data={"csrf": csrf, "action": "results"})
    assert db.get_run_response(rid, sid)["status"] == "results-shared"


def test_site_reject(internal):
    client, db, _auth, sid = internal
    rid = _post_a_run(db)
    csrf = _login(client, "lead@mimic")
    client.post(f"/portal/runs/{rid}/respond", data={"csrf": csrf, "action": "reject"})
    assert db.get_run_response(rid, sid)["status"] == "rejected"


def test_respond_unknown_request_is_ignored(internal):
    client, db, _auth, sid = internal
    _login(client, "lead@mimic")
    csrf = _csrf(client, "/portal")
    client.post("/portal/runs/9999/respond", data={"csrf": csrf, "action": "accept"})
    assert db.get_run_response(9999, sid) is None


# ---- committee orchestration + audit gating --------------------------------

def test_admin_runs_matrix_renders(internal):
    client, db, _auth, _sid = internal
    _post_a_run(db)
    _login(client, "admin@flair")
    r = client.get("/admin/runs")
    assert r.status_code == 200
    assert "orch-matrix" in r.text and "Site" in r.text


def test_audit_download_admin_gated(internal):
    client, db, _auth, _sid = internal
    # create a submission with a real stored audit file
    sub_id = db.create_submission(model_name="M", owner_email="a@b.c",
                                  github_url="https://github.com/x/y", tasks=[TASK],
                                  target_sites="all", tested_mimic=True)
    from app.config import settings
    (settings.uploads_dir / "audit").mkdir(parents=True, exist_ok=True)
    rel = f"audit/{sub_id}-r.json"
    (settings.uploads_dir / rel).write_bytes(b'{"x":1}')
    db.set_audit_report_path(sub_id, rel)

    # anonymous → redirected to login, no file leak
    r = client.get(f"/admin/submission/{sub_id}/audit", follow_redirects=False)
    assert r.status_code == 302

    _login(client, "lead@mimic")  # site lead is not admin
    assert client.get(f"/admin/submission/{sub_id}/audit",
                      follow_redirects=False).status_code == 302

    _login(client, "admin@flair")  # re-login swaps the session to the admin
    ok = client.get(f"/admin/submission/{sub_id}/audit")
    assert ok.status_code == 200 and ok.content == b'{"x":1}'
