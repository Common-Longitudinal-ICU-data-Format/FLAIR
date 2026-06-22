# FLAIR orchestration — operator guides

How the FLAIR benchmark site moves a model from "someone proposes it" to "every site has run
it." One guide per role; pick yours.

| You are… | You want to… | Guide |
|---|---|---|
| **A model author** | propose a model for federated evaluation | [submitter.md](submitter.md) |
| **A site lead** | run (or reject) posted models at your site | [site-lead.md](site-lead.md) |
| **The benchmark committee** | audit requests and orchestrate runs across sites | [committee.md](committee.md) |
| **A developer / operator** | run, test, and deploy this app | [dev.md](dev.md) |

## The flow in one picture

```
 author                committee                       sites
   │                       │                             │
   │  run request          │                             │
   │  (code+model+docker,   │                             │
   │   MIMIC-CLIF attest +  │                             │
   │   audit upload)        │                             │
   ├──────────────────────►│  review queue               │
   │                       │  reviewing → processing      │
   │                       │  (download + audit numbers)  │
   │                       │                             │
   │                       │  approve → POST run to ALL ──┤──► each site sees it
   │                       │                             │     accept / reject
   │                       │                             │     running → results-shared
   │                       │                             │     (stay anonymous optional)
   │                       │  /admin/runs matrix ◄────────┤     tracks every site
   │                       │                             │
   │                       │  results returned OUT-OF-BAND (committee emails how)
   │                       │  in-portal upload = coming soon
```

## Hard rules (apply to everyone)

- The **public** site is READ-ONLY: leaderboard + tasks + docs, no login, no writes, no DB.
  All workflows below run on the **internal** app (`FLAIR_PUBLIC_MODE=false`).
- **No PHI ever leaves a site.** Only aggregate, cell-suppressed (n<10) numbers move.
- A model reaches sites **only after** the committee audits its MIMIC-CLIF report.
- The term is **hospitalization**, never "encounter."
