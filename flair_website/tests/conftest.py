"""Test fixtures for the PUBLIC, read-only deployment (the shipped surface).

The app defaults to public mode: no DB, no sessions, no write routes — it serves
only the committed data/ snapshot. So these fixtures need no database seeding."""
import os

os.environ.setdefault("FLAIR_SECRET_KEY", "test-secret-not-for-prod")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)
