"""Point the app at a throwaway SQLite DB BEFORE any app module imports
(config reads env at import time), then seed one known site + lead."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["FLAIR_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["FLAIR_SECRET_KEY"] = "test-secret-key-not-for-prod"
os.environ["FLAIR_SECURE_COOKIE"] = "false"

import pytest
from fastapi.testclient import TestClient

from app import auth, db
from app.main import app

SITE_KEY = "mimic"          # present in the leaderboard snapshot
LEAD_EMAIL = "lead@test.flair"
LEAD_PW = "correct-horse"
ADMIN_EMAIL = "admin@test.flair"
ADMIN_PW = "admin-horse"


@pytest.fixture(scope="session", autouse=True)
def _seed():
    db.init_db()
    site_id = db.create_site(SITE_KEY, "A")
    db.create_user(site_id, LEAD_EMAIL, auth.hash_password(LEAD_PW))
    # opted in + metrics visible so it appears on the public board
    s = db.get_site(site_id)
    db.update_profile(site_id, site_name="Test Site", abbreviation="TST",
                      pi_name="Dr. Test", contact=LEAD_EMAIL, opted_in=True,
                      visibility={**s["visibility"], "metrics": True})
    # committee admin on a hidden site
    cid = db.create_site("committee", "—")
    db.create_user(cid, ADMIN_EMAIL, auth.hash_password(ADMIN_PW), role="admin")
    # one example submission to review/push
    db.create_submission(model_name="Test Model", owner_email="x@y.z",
                         github_url="https://github.com/x/y",
                         tasks=["task1_icu_daily_mortality"], target_sites="all")
    yield


@pytest.fixture
def client():
    return TestClient(app)
