# Submitter guide — request a model run

[← guides index](README.md)

You built a model and want it evaluated across the FLAIR consortium. You submit **one run
request**. The committee audits it against MIMIC-CLIF, then posts it to every site. You never
touch site data — sites run your code locally and only aggregate numbers come back.

## Before you submit — run it on MIMIC-CLIF first

The committee will not post a model it cannot audit. So:

1. Get your model running through the `flair` library on **MIMIC-CLIF** (the public CLIF
   dataset).
2. Produce the audit bundle:
   ```bash
   flair build-report ...        # see the FLAIR 101 → reports docs
   ```
   This yields the aggregate scorecard JSON (and optionally a zipped report bundle).
3. Keep that file — you attach it to the request.

## What a run request needs

| Field | Required | Notes |
|---|---|---|
| Model name | ✅ | shown on the leaderboard |
| Code / GitHub URL | ✅ | must be `http(s)://`; how sites get your code |
| Model type / description | — | e.g. "gradient-boosted trees", "LSTM" |
| Docker image | — | `docker.io/org/image:tag` — strongly recommended for reproducible runs |
| Model weights / link | — | HuggingFace, Zenodo, etc. |
| Tasks to run | ✅ | pick ≥1 of the five FLAIR tasks |
| **Tested on MIMIC-CLIF** | ✅ | the attestation checkbox |
| **Audit report bundle** | ✅ | the `flair build-report` file, `.json` / `.zip` / `.tar.gz`, ≤ 25 MB |
| Your name / contact email | email ✅ | how the committee reaches you |
| Register a site | — | tick if you also want to onboard a site |

There is **no site picker** — once approved, a run goes to **all** participating sites.

## Submit

Open the internal app and go to **`/submit`** ("Request a model run"). Fill the form, attach the
audit bundle, submit. You land on a confirmation with your request number.

> Public visitors don't see this form — they get a "Submit via GitHub" link. The form is part
> of the internal app. If you don't have access, send your request via GitHub and the committee
> enters it.

## What happens next

1. **Review queue** — the committee sees your request and downloads your audit report.
2. **Processing** — they sanity-check your MIMIC-CLIF numbers and the code/image.
3. **Posted** — approved → dispatched to all sites. Each site chooses to run or reject it.
4. **Results** — sites share aggregate results with the committee out-of-band; the committee
   folds them into the leaderboard. (Sites may stay anonymous.)

## Tips

- **Ship a Docker image.** Sites run on locked-down infrastructure; a single reproducible image
  is the difference between "ran in a day" and "never ran."
- **Make the code self-contained** against the FLAIR task cohort contract (see FLAIR 101 →
  tasks). If it needs the network or a private registry at runtime, sites will reject it.
- **Your audit numbers are the contract.** If a site's numbers diverge wildly from your
  MIMIC-CLIF report, expect questions.
