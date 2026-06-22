# Site-lead guide — run posted models at your site

[← guides index](README.md)

You are the lead for a participating site (one lead per site). The committee posts approved
models to you; you decide whether to run them, execute them on your local CLIF data, and report
aggregate results back. **No PHI leaves your site** — the code travels, the data does not.

## Sign in

Go to **`/login`** with the credentials the committee provisioned for your site. You land on the
**portal dashboard** (`/portal`). Your four tabs:

| Tab | What it's for |
|---|---|
| **Dashboard** | posted runs + your accept/reject/lifecycle controls; task status |
| **My scores** | your site's own *unblinded* scores per model × task |
| **Artifacts** | model/task code to download; results return (coming soon) |
| **Profile & visibility** | site details, public opt-in, per-field visibility switches |

## Run requests — accept, reject, or run

Each posted run is a card with the model's **code / docker / weights** links and the tasks to
run. Your controls depend on where the run is:

```
pending ──accept──► accepted ──Mark running──► running ──Mark results shared──► results-shared
   │
   └──reject──► rejected         (you can re-accept a rejected run later)
```

- **Accept & run** — you intend to run it. Tick **"stay anonymous"** here if you want your
  results attributed to your anonymous label (`A`, `B`, …) rather than your site name. The
  anonymity choice carries through the rest of the lifecycle.
- **Reject** — you decline this model (incompatible, governance, capacity — your call).
- **Mark running** → **Mark results shared** — move the run forward as you go. The committee
  watches these states on their orchestration board.

## Returning results

**For now, results go to the committee out-of-band.** When you finish a run, the committee emails
you instructions on how to share the cell-suppressed aggregate bundle securely. The portal's
in-portal upload (with automatic schema + n<10 suppression checks) is **coming soon** — the
Artifacts tab shows it as a disabled stub.

What you share is **only** aggregate, cell-suppressed (n<10) JSON. Never raw rows, never PHI.

## My scores

The **My scores** tab shows your site's real, unblinded numbers per model and task — AUROC, CI,
AUPRC, calibration slope, Brier. These are *yours*; what the public sees is controlled by your
visibility switches.

## Profile & visibility — staying anonymous on the board

Two layers of anonymity, independent:

1. **Per-run anonymity** (above) — how results are attributed when you send them back.
2. **Public board visibility** (Profile tab):
   - **Opt in / out** — whether your site appears on the public leaderboard at all.
   - **Reveal identity** — off → you show as your anonymous label; on → real name/abbrev/PI.
   - **Metrics / calibration / fairness** — per-field switches for what's exposed publicly.

Default is conservative: private until you opt in, identity hidden even when you do.

## Quick reference

- Sign in: `/login` · Dashboard: `/portal`
- One run, three outcomes: **reject**, or **accept → running → results-shared**.
- Anonymity: per-run (results) *and* per-field (public board) — separate toggles.
- Results sharing today = **out-of-band**, committee emails how. Upload = coming soon.
