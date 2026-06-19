# FLAIR website

Public benchmark site + private federated **orchestration portal** for the FLAIR
ICU benchmark. Self-contained: its own `uv` environment, deployable on any
service (Fly / Render / Railway / a plain VPS) from one `Dockerfile`.

- **Public** (no login): landing, task catalog, **model-driven leaderboard** (pick a task →
  models ranked by pooled AUROC; expand a model → per-site table whose **columns are sites**,
  add/remove sites and the pooled number recomputes). Sites shown anonymized (`A`, `B`, …)
  unless revealed. The public site is READ-ONLY: to propose a model, visitors follow a
  "Submit via GitHub" link — the run-request form lives in the internal app (below).
  Each **task page** lists the models evaluated on it and renders a full per-(model, site)
  report — ROC/PR, calibration, decision curve, operating point, fairness, temporal drift,
  encounter, cohort, TRIPOD-AI — as interactive inline-SVG charts re-rendered from the site's
  exported JSON (`data/reports/<model>/<site>/<task>.json`); combos without a bundle show
  "pending". Reports are only exposed for publicly-visible (opted-in) sites.
- **Run request** (`/submit`, internal): a model run request — code link (required), model +
  docker links, the tasks to run, a **"tested on MIMIC-CLIF" attestation**, and an **uploaded
  `flair build-report` audit bundle**. The committee audits these numbers before any site runs
  the model. Approved runs always go to **all** sites (no per-submitter targeting). Public
  visitors instead see a "Submit via GitHub" link — the public site has no write surface.
- **Committee** (admin login): `/admin` processing queue — move a request through
  reviewing → processing → approved/rejected, download its MIMIC-CLIF audit, then **post the run
  to all sites**. `/admin/runs` is the committee's own orchestration view: a **run × site status
  matrix** (pending / accepted / rejected / running / results-shared, with an `anon` marker) and
  each site's **charted preferences** (opt-in, per-field visibility, accept rate).
- **Portal** (per-site login, one lead per site): posted runs the site can **accept or reject**,
  then progress **running → results shared**, choosing to **stay anonymous**. Results themselves
  go to the committee **out-of-band** (the committee emails how to share data); in-portal results
  upload is **coming soon**. Plus the site's own unblinded scores, opt-in/out of the public board,
  editable profile, and a visibility switch on every field.

No PHI ever touches this app — it serves aggregate, cell-suppressed (n<10) numbers only.

## Stack

FastAPI + Jinja templates + a little vanilla JS. SQLite (stdlib) for writable site
state (sites, users, submissions, run requests). Metric numbers come from a committed static
snapshot (`data/leaderboard.json`, keyed model → task → site) — the `flair-xgboost-baseline`
model on site `mimic` carries the real MIMIC-CLIF numbers; the other model and all `demo_*`
sites are synthetic UI filler. Replace `data/leaderboard.json` with real per-site scorecards
at launch.

> The SQLite schema is created on startup; a dev DB is disposable. If you change the schema on
> an already-deployed DB, reseed or migrate it manually.

## Two modes

The site runs in one of two modes (`FLAIR_PUBLIC_MODE`, default `true`):

- **Public (default)** — a READ-ONLY site. Only the public + docs pages are mounted; it
  serves the committed `data/` snapshot with **no database, no sessions, and no write
  endpoints**. This is what deploys publicly. Nothing a visitor can reach mutates state or
  exposes anything beyond the public aggregates in `data/`.
- **Internal (`FLAIR_PUBLIC_MODE=false`)** — additionally mounts the DB-backed management app
  (portal + admin + submit) behind sessions, for running the consortium internally.

## Develop (public site)

```bash
cd flair_website
uv sync
uv run uvicorn app.main:app --reload     # public read-only, no DB needed
# http://127.0.0.1:8000
```

## Build the data from site dumps

Sites drop their flair-baseline **non-PHI upload** folder into `runs_data/`:

```
runs_data/<site>_baseline_non_phi_for_upload/<task>/
    table1.json
    report/{discrimination,calibration,dca,fairness,leadtime}.json   # mode-dependent
    report/viz/*.png
```

Then regenerate the public `data/` snapshot:

```bash
uv run python scripts/build_data.py      # → data/leaderboard.json + data/reports/**
```

`runs_data/` is gitignored and never ships; only the processed `data/` is committed and
served. Adding a site = drop its folder and re-run the parser. The per-task report **mode**
(episodic / peak / landmark) is read from each pillar's `metadata.report_mode`.

## Internal management app

```bash
cp .env.example .env          # set FLAIR_SECRET_KEY
uv run python -m app.seed --demo    # demo sites + committee admin + example submission
FLAIR_PUBLIC_MODE=false uv run uvicorn app.main:app --reload
#   site lead:  lead@mimic.flair / demo
#   committee:  admin@flair      / demo   (→ /admin review queue)
```

Create a committee admin against a real deployment:

```bash
uv run python -m app.seed --admin --email committee@flair.org --password '...'
```

Seed a real site + lead:

```bash
uv run python -m app.seed --key rush --email lead@rush.org --password '...' \
    --site-name "Rush University Medical Center" --abbrev RUMC --pi "Dr. Doe"
```

The site `--key` must match a key in `data/scorecard_snapshot.json` to join profile ↔ numbers.

## Test

```bash
uv run pytest
```

## Deploy

```bash
docker build -t flair-website .
docker run -p 8000:8000 -e FLAIR_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))") \
    -e FLAIR_SECURE_COOKIE=true -v flair-data:/data flair-website
```

Set `FLAIR_SECRET_KEY` (required), `FLAIR_SECURE_COOKIE=true` behind HTTPS, and mount a
volume at `/data` so the SQLite DB persists. `PORT` is honored if the platform sets it.

## Update the leaderboard

Drop new/updated site folders into `runs_data/` and re-run `python scripts/build_data.py`
(see "Build the data" above); commit the regenerated `data/` and redeploy. Edit
`data/tasks.json` if task META changes.

## FLAIR 101 docs

The on-site docs (`/docs`, served by `app/docs.py` from `templates/docs/*.html`) are a
hand-built web rendering of the repo's library docs at `../docs/101.md` + `../docs/101/*.md`.
The markdown is the canonical reference; keep the two in sync when either changes.
