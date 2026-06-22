# Developer guide — run, test, deploy

[← guides index](README.md) · see also the top-level [`../README.md`](../README.md)

The FLAIR website is one FastAPI app with two faces: a READ-ONLY **public** site (what deploys),
and a DB-backed **internal** management app (run orchestration). One codebase, one env var
(`FLAIR_PUBLIC_MODE`) flips between them.

## Stack

- **FastAPI** + **Jinja2** templates + hand-written CSS + a little vanilla JS (inline-SVG charts
  in `static/js/report.js`).
- **SQLite** (stdlib `sqlite3`, no ORM) for writable internal state — internal mode only.
- Metric numbers come from a committed **static snapshot** (`data/`), not the DB.
- **uv** for deps. Self-contained: own `uv` env, one `Dockerfile`.

## Two modes (`FLAIR_PUBLIC_MODE`, default `true`)

| | Public (`true`) | Internal (`false`) |
|---|---|---|
| Mounted | `public` + `docs` + `/static` | + `portal` + `admin` + `submit` |
| Sessions / DB | none | SessionMiddleware + SQLite |
| Writes | **none** | submit, accept/reject, profile, admin |
| Uploads | none | audit bundles in `uploads/` |
| Deploys publicly | ✅ | ✗ (run behind the consortium) |

The public site cannot mutate anything or read the DB. That's the security boundary — keep it.

## Run

```bash
cd flair_website
uv sync

# public read-only (no DB needed)
uv run uvicorn app.main:app --reload                 # http://127.0.0.1:8000

# internal management app
uv run python -m app.seed --demo                     # demo sites + admin + example run
FLAIR_PUBLIC_MODE=false uv run uvicorn app.main:app --reload
#   committee:  admin@flair       / demo   → /admin, /admin/runs
#   site lead:  lead@mimic.flair  / demo   → /portal
```

Port 8000 stuck? `lsof -ti tcp:8000 | xargs kill -9`, or `--port 8001`.

## Data model (internal)

`app/db.py` — five tables plus the orchestration additions:

- `sites`, `users`, `run_status` — site profiles, one lead/admin per row, per-task status.
- `submissions` — run requests. New: `tested_mimic`, `audit_report_path`.
- `run_requests` — a posted run. `site_id IS NULL` = broadcast to all sites.
- `run_responses` — **(request × site)** lifecycle:
  `pending → rejected | accepted → running → results-shared`, plus an `anonymous` flag.
  One broadcast `run_requests` row collects one `run_responses` row per site.

Schema evolves via `_migrate()` (idempotent `ALTER TABLE ADD COLUMN`, guarded by
`PRAGMA table_info`), run inside `init_db()`. A dev DB is disposable; on a real deployment,
`_migrate` adds new columns in place.

Key helpers: `set_run_response`, `responses_for_site`, `orchestration_grid` (the matrix feed),
`push_submission_to_sites` (posts a run).

## Audit uploads

`app/submit.py` accepts a multipart `audit_report`, sanitizes the filename (basename +
char-whitelist, no traversal), whitelists extensions (`.json/.zip/.tar.gz`), caps size (25 MB),
and writes to `settings.uploads_dir/audit/<sub_id>-<name>`. Served back only through the
admin-gated `GET /admin/submission/{id}/audit`, which resolves the DB-stored path and confirms
it's inside `uploads_dir` before streaming. `uploads/` is gitignored and dockerignored — never
shipped, never served by StaticFiles.

## Build the public snapshot from site dumps

Sites' non-PHI baseline folders go in `runs_data/`; the parser turns them into `data/`:

```bash
uv run python scripts/build_data.py     # runs_data/** → data/leaderboard.json + data/reports/**
```

`runs_data/` is gitignored and never ships; only processed `data/` is committed and served.
Add a site = drop its folder, re-run the parser. Report **mode** (episodic / peak / landmark) is
read per pillar from `metadata.report_mode`.

## Test

```bash
uv run pytest                # 17 tests
```

- `tests/test_app.py` — public surface: pages render from `data/`, write/admin routes 404, no
  PII/inspect leak.
- `tests/test_internal.py` — builds the internal app in-process against a temp DB + uploads dir
  (mutates the `settings` singleton, leaves the public fixtures untouched): audit upload +
  attestation, site accept/reject/lifecycle + anonymity, `/admin/runs` matrix, admin-gated audit
  download.
- `tests/test_build_data.py` — the `runs_data/ → data/` parser.

## Deploy (public)

```bash
docker build -t flair-website .
docker run -p 8000:8000 \
    -e FLAIR_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))") \
    -e FLAIR_SECURE_COOKIE=true -v flair-data:/data flair-website
```

Public deploy needs no DB. For an internal deployment, additionally set `FLAIR_PUBLIC_MODE=false`,
mount a volume for the SQLite DB and `uploads/`, and front it with auth/VPN — it is not for the
open internet.

## Cache-busting

Bump `asset_ver` in `app/templates.py` on any `static/` change so browsers reload CSS/JS
(`?v={{ asset_ver }}`).

## File map

| Area | Files |
|---|---|
| Wiring / config | `app/main.py`, `app/config.py`, `app/templates.py` |
| Auth / DB | `app/auth.py`, `app/db.py`, `app/seed.py` |
| Public | `app/public.py`, `app/data.py`, `app/docs.py` |
| Internal | `app/submit.py`, `app/portal.py`, `app/admin.py` |
| Templates | `templates/{submit,login}.html`, `templates/portal/*`, `templates/admin/*` |
| Front-end | `static/css/site.css`, `static/js/report.js` |
| Snapshot / ingest | `data/`, `runs_data/`, `scripts/build_data.py` |

## Conventions

- Term is **hospitalization**, never "encounter" — anywhere user-facing.
- Public site stays read-only. Any new write route is internal-mode only and behind sessions.
- Only aggregate, cell-suppressed (n<10) numbers ever reach `data/` or the wire.
