"""Shared Jinja2 environment. Exposes csrf_token + current_site as globals so
every template can render a CSRF field and the nav login state."""
from __future__ import annotations

from fastapi.templating import Jinja2Templates

from .auth import csrf_token, current_site, is_admin
from .config import settings

templates = Jinja2Templates(directory=str(settings.templates_dir))
templates.env.globals["csrf_token"] = csrf_token
templates.env.globals["current_site"] = current_site
templates.env.globals["is_admin"] = is_admin
templates.env.globals["public_mode"] = settings.public_mode
# Bump on any static asset change to bust the browser cache of /static/css/site.css.
templates.env.globals["asset_ver"] = "10"
