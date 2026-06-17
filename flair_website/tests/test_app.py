import re

from app import db
from tests.conftest import (ADMIN_EMAIL, ADMIN_PW, LEAD_EMAIL, LEAD_PW, SITE_KEY)


def _csrf(client, path):
    html = client.get(path).text
    m = re.search(r'name="csrf" value="([^"]+)"', html)
    assert m, f"no csrf token in form at {path}"
    return m.group(1)


def _login(client, email, pw):
    csrf = _csrf(client, "/login")
    return client.post("/login", data={"email": email, "password": pw, "csrf": csrf},
                       follow_redirects=False)


def test_public_pages_ok(client):
    for path in ["/", "/tasks", "/tasks/task1_icu_daily_mortality",
                 "/leaderboard", "/api/leaderboard", "/submit"]:
        assert client.get(path).status_code == 200, path


def test_unknown_task_404(client):
    assert client.get("/tasks/nope").status_code == 404


def test_task_page_lists_models(client):
    html = client.get("/tasks/task1_icu_daily_mortality").text
    assert "Models" in html and "full report" in html


def test_report_api_real_bundle(client):
    r = client.get("/api/report/flair-xgboost-baseline/mimic/task1_icu_daily_mortality")
    assert r.status_code == 200
    j = r.json()
    assert "curves" in j and "fairness" in j and j["curves"]["roc"]["auroc"] > 0.5


def test_report_api_missing_is_404(client):
    assert client.get("/api/report/flair-xgboost-baseline/mimic/nope").status_code == 404
    # a model with no bundle
    assert client.get("/api/report/demo-tabpfn-icu/mimic/task1_icu_daily_mortality").status_code == 404


def test_report_api_hidden_site_404(client):
    # opt mimic out → its report must not be exposed publicly
    site = db.get_site_by_key(SITE_KEY)
    db.update_profile(site["id"], site_name=site["site_name"], abbreviation=site["abbreviation"],
                      pi_name=site["pi_name"], contact=site["contact"], opted_in=False,
                      visibility=site["visibility"])
    code = client.get("/api/report/flair-xgboost-baseline/mimic/task1_icu_daily_mortality").status_code
    db.update_profile(site["id"], site_name=site["site_name"], abbreviation=site["abbreviation"],
                      pi_name=site["pi_name"], contact=site["contact"], opted_in=True,
                      visibility=site["visibility"])
    assert code == 404


def test_leaderboard_model_driven(client):
    data = client.get("/api/leaderboard").json()
    assert "by_task" in data and len(data["tasks"]) == 5
    rows = data["by_task"]["task1_icu_daily_mortality"]
    assert len(rows) >= 1
    # ranked by pooled AUROC descending
    aurocs = [r["pooled"]["auroc"] for r in rows]
    assert aurocs == sorted(aurocs, reverse=True)
    # the baseline is one of the models
    assert any(r["id"] == "flair-xgboost-baseline" for r in rows)


def test_site_appears_and_anonymized(client):
    cols = client.get("/api/leaderboard").json()["columns"]
    col = next(c for c in cols if c["key"] == SITE_KEY)
    assert col["identified"] is False and col["label"] == "A"


def test_optout_hides_site(client):
    site = db.get_site_by_key(SITE_KEY)
    db.update_profile(site["id"], site_name=site["site_name"], abbreviation=site["abbreviation"],
                      pi_name=site["pi_name"], contact=site["contact"], opted_in=False,
                      visibility=site["visibility"])
    cols = client.get("/api/leaderboard").json()["columns"]
    assert all(c["key"] != SITE_KEY for c in cols)
    db.update_profile(site["id"], site_name=site["site_name"], abbreviation=site["abbreviation"],
                      pi_name=site["pi_name"], contact=site["contact"], opted_in=True,
                      visibility=site["visibility"])


def test_submit_creates_submission(client):
    before = len(db.list_submissions())
    csrf = _csrf(client, "/submit")
    r = client.post("/submit", data={
        "csrf": csrf, "model_name": "My GBM", "owner_email": "me@uni.edu",
        "github_url": "https://github.com/me/gbm", "target_all": "on",
        "task_task1_icu_daily_mortality": "on",
    })
    assert r.status_code == 200 and "received" in r.text.lower()
    assert len(db.list_submissions()) == before + 1


def test_submit_validation_rejects_missing(client):
    csrf = _csrf(client, "/submit")
    r = client.post("/submit", data={"csrf": csrf, "model_name": "", "owner_email": "bad"})
    assert r.status_code == 400


def test_admin_requires_admin_role(client):
    # a site lead is not a committee admin → bounced to /login
    _login(client, LEAD_EMAIL, LEAD_PW)
    assert client.get("/admin", follow_redirects=False).status_code == 302


def test_admin_queue_and_push(client):
    a = TestClient_login(client)  # fresh admin session
    assert client.get("/admin").status_code == 200
    sub = db.list_submissions()[-1]  # the seeded "Test Model"
    n_before = len(db.list_run_requests())
    csrf = _csrf(client, f"/admin/submission/{sub['id']}")
    client.post(f"/admin/submission/{sub['id']}", data={"csrf": csrf, "action": "push"},
                follow_redirects=False)
    assert len(db.list_run_requests()) == n_before + 1
    # the pushed request (all sites → site_id NULL) is visible to the mimic site
    site = db.get_site_by_key(SITE_KEY)
    seen = db.list_run_requests(site["id"])
    assert any(r["model_name"] == sub["model_name"] for r in seen)


def TestClient_login(client):
    csrf = _csrf(client, "/login")
    client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PW, "csrf": csrf})
    return client
