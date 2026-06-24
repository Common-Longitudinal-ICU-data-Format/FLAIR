"""Public read-only deployment surface: pages render from data/, no DB, no write
endpoints, no PII leaks."""

PUBLIC_PAGES = ["/", "/tasks", "/tasks/task1_icu_daily_mortality", "/leaderboard",
                "/api/leaderboard", "/docs", "/docs/clif", "/docs/reports"]

# Routes that only exist in the internal (DB-backed) app — must be absent in public.
WRITE_SURFACE = ["/submit", "/login", "/admin", "/admin/runs",
                 "/admin/submission/1/audit", "/portal", "/portal/scores"]


def test_public_pages_ok(client):
    for path in PUBLIC_PAGES:
        assert client.get(path).status_code == 200, path


def test_write_and_admin_surface_absent(client):
    for path in WRITE_SURFACE:
        assert client.get(path).status_code == 404, f"GET {path} should not exist in public mode"
    assert client.post("/submit", data={"model_name": "x"}).status_code == 404
    assert client.post("/login", data={"email": "a@b.c"}).status_code == 404


def test_unknown_task_404(client):
    assert client.get("/tasks/nope").status_code == 404


def test_report_api_new_pillar_shape(client):
    # landmark task → has the landmark pillars + leadtime
    r = client.get("/api/report/flair-xgboost-baseline/mimic/task1_icu_daily_mortality")
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "landmark" and j["landmark"] is True
    assert j["discrimination"]["landmarks"] and j["leadtime"]
    assert j["discrimination"]["landmarks"][0]["metrics"]["auroc"] > 0.5
    # peak task → no calibration / dca / leadtime
    p = client.get("/api/report/flair-xgboost-baseline/mimic/task4_sepsis_abx_6h").json()
    assert p["mode"] == "peak"
    assert p["calibration"] is None and p["dca"] is None and p["leadtime"] is None


def test_report_api_missing_is_404(client):
    assert client.get("/api/report/flair-xgboost-baseline/mimic/nope").status_code == 404
    assert client.get("/api/report/no-such-model/mimic/task1_icu_daily_mortality").status_code == 404


def test_leaderboard_real_numbers(client):
    data = client.get("/api/leaderboard").json()
    assert len(data["tasks"]) == 5
    assert [c["key"] for c in data["columns"]] == ["mimic", "rush"]
    rows = data["by_task"]["task1_icu_daily_mortality"]
    assert any(r["id"] == "flair-xgboost-baseline" for r in rows)
    base = rows[0]
    assert base["pooled"]["auroc"] > 0.5
    assert base["split"]["ratio"] == "75/25"


def test_rush_site_present(client):
    data = client.get("/api/leaderboard").json()
    # rush carries real numbers and a renderable report bundle
    cell = data["by_task"]["task1_icu_daily_mortality"][0]["cells"]["rush"]
    assert cell and cell["auroc"] > 0.5
    r = client.get("/api/report/flair-xgboost-baseline/rush/task1_icu_daily_mortality")
    assert r.status_code == 200 and r.json()["mode"] == "landmark"


def test_results_carry_report_mode(client):
    data = client.get("/api/leaderboard").json()
    assert all(t.get("report_mode") for t in data["tasks"])
    mode = {t["id"]: t["report_mode"] for t in data["tasks"]}
    assert mode["task1_icu_daily_mortality"] == "landmark"
    assert mode["task4_sepsis_abx_6h"] == "peak"


def test_leaderboard_cells_have_report_flag(client):
    # merged Task Leaderboard needs to know which (model, site, task) has a report
    data = client.get("/api/leaderboard").json()
    cells = data["by_task"]["task1_icu_daily_mortality"][0]["cells"]
    assert cells["mimic"]["has_report"] is True
    assert cells["rush"]["has_report"] is True


def test_task_leaderboard_merged_page(client):
    # /tasks and /leaderboard render the same merged accordion page
    for path in ("/leaderboard", "/tasks"):
        html = client.get(path).text
        assert "Task Leaderboard" in html, path
        assert "ICU Daily In-Hospital Mortality" in html, path  # all tasks listed
        assert "outcome" in html and "intervention" in html, path  # class pills


def test_no_pii_or_inspect_leak(client):
    blobs = [client.get("/").text, client.get("/leaderboard").text,
             client.get("/api/leaderboard").text]
    for b in blobs:
        low = b.lower()
        for bad in ("pi_name", "owner_email", "password", "contact", "anon_label"):
            assert bad not in low, f"leaked {bad!r}"
    # site identity word never appears on the task page either
    assert "encounter" not in client.get("/tasks/task1_icu_daily_mortality").text.lower()
