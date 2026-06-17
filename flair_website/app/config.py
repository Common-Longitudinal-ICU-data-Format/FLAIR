"""Runtime settings, all overridable by environment variables (FLAIR_*)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLAIR_", env_file=".env", extra="ignore")

    # Signs the session cookie. MUST be overridden in production.
    secret_key: str = "dev-insecure-secret-change-me"
    # SQLite file for site profiles / users / run status.
    db_path: str = str(BASE_DIR / "flair_website.db")
    # Mark the session cookie Secure (set true behind HTTPS).
    secure_cookie: bool = False
    # Lock a login after this many consecutive failures.
    max_login_fails: int = 8

    data_dir: Path = BASE_DIR / "data"
    static_dir: Path = BASE_DIR / "static"
    templates_dir: Path = BASE_DIR / "templates"


settings = Settings()
