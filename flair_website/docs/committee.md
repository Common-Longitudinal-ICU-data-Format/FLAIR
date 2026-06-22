# Committee guide — audit requests, orchestrate runs

[← guides index](README.md)

You are the benchmark committee (admin role). You audit incoming model run requests against their
MIMIC-CLIF numbers, post approved runs to every site, and track who runs what. You also tell
sites how to return results.

## Sign in

Go to **`/login`** with your admin credentials. Two committee surfaces appear in the nav:

- **Queue** (`/admin`) — incoming run requests.
- **Run orchestration** (`/admin/runs`) — the run × site status board.

## 1. The review queue (`/admin`)

Every run request, newest first. Columns: model, owner, task count, **Audit** (ok / missing),
status, date. The Audit column is your first filter — a request with no attestation or no audit
bundle is not postable.

Open a request to see its detail page.

## 2. Auditing a request (`/admin/submission/{id}`)

The detail page gives you:

- **Model & links** — code / docker / weights, tasks requested, "posts to all sites".
- **MIMIC-CLIF audit** — the attestation flag and a **Download audit report** link. Pull the
  bundle, check the submitter's reported numbers are sane and reproducible.
- **Submitter** — contact, whether they're an existing lead, site-registration ask (with a
  ready-to-run `app.seed` command if they want onboarding).

### Lifecycle buttons

```
submitted ─► reviewing/processing ─► approved ─► POSTED to all sites
                              └────► rejected
```

- **Move to processing** — you're actively auditing it.
- **Approve** — audit passed, not yet dispatched.
- **Reject** — declined (record why in *Review notes*).
- **Post run to all sites** — dispatches it. Every site immediately sees it on their dashboard.
  Do this only after the audit checks out — there's a warning banner if attestation/audit is
  missing, but it won't hard-stop you.

> Posting always targets **all** participating sites. There is no per-site targeting by design —
> the consortium runs a model everywhere or not at all.

## 3. Orchestration board (`/admin/runs`) — your "own view"

Two panels:

### Run × site matrix
Rows = posted runs, columns = sites. Each cell is that site's response:

| Badge | Meaning |
|---|---|
| `pending` | site hasn't responded |
| `rejected` | site declined |
| `accepted` | site will run it |
| `running` | site is running it |
| `results-shared` | site returned aggregate results |
| `anon` marker | site chose to stay anonymous for this run |

### Site preferences (charted)
Per site: public/private, per-field visibility (identity / metrics / calibration / fairness), and
an **accept-rate** bar over the runs they've responded to. Use it to see who participates and
what each site exposes publicly.

## 4. Returning results — your job to coordinate

In-portal results upload is **coming soon**. Until then, **you** send each running site the
instructions to share their cell-suppressed aggregate bundle securely (your SFTP / encrypted
drop / whatever the consortium agreed). When a bundle arrives:

1. Drop it into `runs_data/<site>_baseline_non_phi_for_upload/...` (see [dev.md](dev.md)).
2. Re-run the parser to refresh the public snapshot.
3. The site's number appears on the leaderboard (subject to their visibility switches).

## Provisioning a new site

When a submitter or a new center asks to join, create their login:

```bash
uv run python -m app.seed --key rush --email lead@rush.org --password '...' \
    --site-name "Rush University Medical Center" --abbrev RUMC --pi "Dr. Doe"
```

The `--key` must match the site's key in the metric snapshot so profile ↔ numbers join.

## Quick reference

- Queue: `/admin` · Board: `/admin/runs`
- Audit **before** you post. Download the bundle; don't trust the checkbox alone.
- Post = all sites, always.
- Results come back out-of-band — you email the how; then ingest via `runs_data/` + the parser.
