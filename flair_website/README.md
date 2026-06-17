# FLAIR website

Public benchmark site + private federated **orchestration portal** for the FLAIR
ICU benchmark. Self-contained: its own `uv` environment, deployable on any
service (Fly / Render / Railway / a plain VPS) from one `Dockerfile`.

- **Public** (no login): landing, task catalog, **model-driven leaderboard** (pick a task →
  models ranked by pooled AUROC; expand a model → per-site table whose **columns are sites**,
  add/remove sites and the pooled number recomputes), and a **model submission form**
  (`/submit`) — model + github/docker/weights links, tasks, target sites, whether you're a
  site lead, optional site registration. Sites shown anonymized (`A`, `B`, …) unless revealed.
  Each **task page** lists the models evaluated on it and renders a full per-(model, site)
  report — ROC/PR, calibration, decision curve, operating point, fairness, temporal drift,
  encounter, cohort, TRIPOD-AI — as interactive inline-SVG charts re-rendered from the site's
  exported JSON (`data/reports/<model>/<site>/<task>.json`); combos without a bundle show
  "pending". Reports are only exposed for publicly-visible (opted-in) sites.
- **Committee** (admin login): `/admin` review queue — approve / reject a submission, and
  **push** an approved one to sites (creates the run requests sites see, carrying the links).
- **Portal** (per-site login, one lead per site): run requests (model + links + tasks), run
  status, artifacts, the site's own unblinded scores per model, opt-in/out of the public
  board, editable site profile, and a visibility switch on every field.

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

## Develop

```bash
cd flair_website
uv sync
cp .env.example .env          # set FLAIR_SECRET_KEY
uv run python -m app.seed --demo    # demo sites + committee admin + example submission
uv run uvicorn app.main:app --reload
# http://127.0.0.1:8000
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

Replace `data/leaderboard.json` (models → task → site → scorecard values) with real numbers,
and `data/tasks.json` if task META changes, then redeploy.
