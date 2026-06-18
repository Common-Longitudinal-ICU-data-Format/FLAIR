# Three Ways to Score a "Continuous" Task

> A walk-through for someone new to the FLAIR report. Goal: understand **why** a model that predicts every hour (or every day) can't be scored the naive way, the **three** ways to fix it, and **which metric needs what** (a ranking? a real probability? a threshold?).

------------------------------------------------------------------------

## 0. First, two kinds of task

- **Single-window task** (e.g. "will this patient be readmitted?") → the model makes **one prediction per hospital stay**. One row = one patient. Nothing to worry about. Score it directly.
- **Continuous task** (e.g. "today, will this ICU patient die this admission?") → the model makes a prediction **every day (or hour)**. One patient = **many rows**. This is where it gets tricky, and this whole note is about it.

If you remember one thing: **single-window is easy because a row already *is* a patient. Continuous is hard because a patient is many rows.**

------------------------------------------------------------------------

## 1. The one problem (why naive scoring lies)

In a continuous task the **label is the eventual outcome, copied onto every row**. If a patient dies, *every* daily row for that patient is labelled "1".

So:

- A patient who stays **10 days and dies** → contributes **10 rows labelled 1**.
- A patient who stays **3 days and survives** → contributes **3 rows labelled 0**.

If you dump all rows into one bucket and compute a score, the long-staying, sicker patients are **counted more times than they should be**. The numbers come out **too optimistic**, and the apparent "how common is death" (prevalence) gets pushed up. The patient who lingers dominates; the quick survivor barely registers. That distortion is the thing all three "ways" below are trying to handle.

------------------------------------------------------------------------

## 2. The three ways (with a kitchen analogy)

Think of each patient stay as a **basket of daily risk scores**. The three ways differ in *what you put on the scale* before you weigh.

| Way | What you score | Kitchen analogy | One row per… |
|------------------|------------------|------------------|------------------|
| **1. Row-pooled** | every single daily row, all mixed together | dump every apple from every basket into one pile and judge the pile | row (stay counted many times) |
| **2. Stay-peak** | one number per patient = their **highest** risk | take only the **ripest apple** from each basket | stay |
| **3. Landmark** | one number per patient **at a fixed time point**, repeated across time points → a **curve** | line baskets up by "day 3 apple", judge those; then "day 2 apple", judge those… | stay, *per time point* |

Plain words:

1.  **Row-pooled** — the naive default. Easy, but it's the one that lies (Section 1).
2.  **Stay-peak** — collapse each patient to their **worst moment**. This is exactly how an alarm works in real life: the alert fires the *first time* risk crosses the line, i.e. at the peak. (The Kamran 2024 NEJM-AI sepsis paper uses exactly this — they call it the "hospitalization-level max score".) In FLAIR this is `stay_frame` → `auroc_stay_peak`.
3.  **Landmark** — freeze the clock at "day 3 of stay" (or "3 windows before the outcome"), take each patient's single row at that moment, score it; repeat for every time point. You get a **curve of skill over time**, not one number. In FLAIR this is `landmark_table` → `landmark_avg_auroc`.

------------------------------------------------------------------------

## 3. The key idea: each metric needs a *different ingredient*

This is the part most people miss. The three report metrics don't all need the same thing from your model's output:

| Metric | What it actually needs | Plain meaning |
|------------------------|------------------------|------------------------|
| **Discrimination** (AUROC / AUPRC) | only a **ranking** | "are sicker patients scored higher than healthier ones?" Order is all that matters. The exact number doesn't. |
| **Calibration** (slope, O:E, Brier, ECE) | a **true probability** | "when the model says 30%, do 30% actually have the event?" The number must *mean* something. |
| **DCA / net benefit** | a **true probability + a threshold + one decision per patient** | "if we treat everyone above risk t, do we help more than we harm?" Needs an honest risk *and* a cut-off *and* counting each patient once. |

Now run each of the 3 ways through each metric:

|   | Row-pooled | Stay-peak | Landmark |
|------------------|------------------|------------------|------------------|
| **Discrimination** | ✅ works, but **too optimistic** (long stays counted many times) | ✅ works — and it's the real-world "alarm" view | ✅ works — cleanest; gives a skill-over-time curve |
| **Calibration** | ⚠️ **meaningless** — mixes "30% on day 1" with "30% on day 7", which have different real event rates | ❌ **broken** — the peak is *not* a probability (see Section 4) | ✅ **the only honest way** — at a fixed time the number is a real probability |
| **DCA** | ❌ **wrong** — one patient counted as many treat/no-treat decisions | ⚠️ right **count** (one decision per patient) but the threshold axis is off (peak isn't calibrated) | ✅ **correct** — honest probability, one decision per patient, at the deployment time |

**Why discrimination is the easy one:** it only cares about *order*. Shuffle the actual numbers up or down however you like — as long as sicker \> healthier, AUROC is unchanged. So all three ways "work"; they just answer slightly different questions. **No probability needed, no threshold needed.**

**Why calibration and DCA are the hard ones:** they need the number to literally *be* a probability. And that's where stay-peak quietly breaks.

------------------------------------------------------------------------

## 4. The "peak-calibration trap" (worked numbers)

This is the single most important gotcha. **You can rank with the peak, but you can never calibrate with it.**

Take a patient whose true risk is **10% on each of 10 days** (an honest, well-calibrated model).

- Their **peak** score across 10 days ≈ **0.30** (the max of ten \~0.10 draws lands much higher than 0.10 — taking the biggest of many numbers is *upward biased*).
- But their **actual** chance of the event is still about **10%**.

So if you build a calibration plot on peak scores, you'll see:

```         
   model said:     ~30%   (the peak)
   reality:        ~10%   (events actually happened)
   → "over-predicting by 3x"
```

The model isn't broken — **the peak just isn't a probability**. Max-of-many is always inflated. That's why Section 3 marks stay-peak calibration as ❌. Great for "who is highest risk" (ranking), useless for "what *is* the risk" (calibration).

DCA inherits the same problem on its threshold axis: "treat if risk \> 0.20" assumes 0.20 is a real risk. Feed it inflated peaks and the cut-off no longer means what it says.

------------------------------------------------------------------------

## 5. "Do I even need probabilities, or just thresholds?" — a decision guide

Use this to pick what to compute:

- **I only want to compare models / rank patients.** → You need **discrimination only**. Ranking is enough — **no calibrated probability, no threshold required.** Any of the 3 ways works (landmark is the cleanest, stay-peak is the most deployment-like).

- **I want to show a clinician an honest risk number ("this patient: 18%").** → You need **calibration**. The number must be a true probability → **landmark, at a fixed time, on calibrated outputs.** Stay-peak and row-pooled will mislead.

- **I want to decide "is it worth acting at threshold t?"** → You need **DCA / net benefit**. That needs **a real probability + the threshold + one decision per patient** → **landmark at the deployment time.** Stay-peak gets the counting right but the threshold wrong; row-pooled gets both wrong.

Quick mental model: **ranking → cheap (any way). probability → expensive (landmark only). threshold decisions → expensive (landmark only).**

------------------------------------------------------------------------

## 5b. Which analyses need a *fixed threshold* handed in up front?

A **threshold** is the cut-off where the model turns a risk number into an action ("alert / treat if risk ≥ 0.20"). Some analyses can't run without one; most don't need it at all. In FLAIR the fixed threshold lives in the task definition: `META.clinical_threshold` (plus `clinical_direction` = `above`/`below`, which says whether you act on **high** risk or **low** risk).

| Analysis | Needs a fixed threshold given up front? | Why |
|------------------------|------------------------|------------------------|
| **Discrimination** (AUROC, AUPRC, landmark) | ❌ No | pure ranking — sweeps *all* thresholds at once |
| **Calibration** (slope, O:E, Brier, ECE) | ❌ No | judges the probability itself, no cut-off involved |
| **DCA curve** (the whole net-benefit sweep) | ❌ No | it *sweeps* every threshold 0.01–0.99 by design |
| **DCA clinical marker** ("useful at *our* operating point?") | ✅ **Yes** | marks net benefit *at* the provided threshold |
| **Operating point** (`threshold.fixed`: sens/spec/PPV/NPV at t) | ✅ **Yes** | "fixed" literally means *at the clinical threshold* |
| **Clinical-utility pillar** (tripod `clinical_utility`) | ✅ **Yes** | net benefit + classification **at the threshold** |
| **Encounter / alarm** (stay-sensitivity, alarms/day, lead-time) | ✅ **Yes** | an alarm = "peak ≥ threshold"; no cut-off → no alarm |
| **Threshold `varying`** (the operating curve) | ❌ No | it builds the curve *across* a grid of thresholds |
| **Threshold `selected`** (Youden-J, max-F1, target-sens) | ❌ No | it *discovers* a threshold from the data |

**Plain version:** - **Ranking and probability questions** ("who's higher?", "is 30% really 30%?") → **no threshold needed.** - **Decision questions** ("if we alert at 20%, how many do we catch and how many false alarms?") → **a threshold must be provided** — there's no decision without a cut-off. - The **`varying` + `selected`** blocks are the escape hatch: they let you explore and *pick* a threshold from the data when the task hasn't fixed one. (Use them to *choose*; use `clinical_threshold` to *report at the clinically chosen* point.)

**What happens if no threshold is provided?** The ranking/probability parts still run fine. The decision parts (`threshold.fixed`, `clinical_utility`, `encounter`) **return null with a reason** rather than inventing a cut-off — you still get the `varying` curve to choose from.

> ⚠️ The threshold also sets the **direction** of the stay-peak reduction (Section 2): `above` → peak = **max** risk (act on high), `below` → peak = **min** risk (e.g. extubate when failure risk is low). So a wrong/absent `clinical_direction` doesn't just mislabel the action — it changes which row represents the stay.

------------------------------------------------------------------------

## 6. Recommendation — what's right, and what FLAIR does today

| Metric | ✅ Right way | ⚠️ Acceptable | ❌ Wrong | What FLAIR ships now |
|---------------|---------------|---------------|---------------|---------------|
| **Discrimination** | landmark (skill-over-time) | stay-peak (deployment) | — (row-pooled is OK but optimistic) | ✅ all three: `auroc` (row), `auroc_stay_peak`, `landmark_avg_auroc` |
| **Calibration** | landmark, per time point | — | stay-peak, row-pooled | ❌ still **row-pooled** — known gap |
| **DCA** | landmark at deployment time | stay-peak (counts only) | row-pooled | ❌ still **row-pooled** — known gap |

So FLAIR already does **discrimination** the right way (three views side by side, plus an audit that shows `n_rows` vs `n_stays` and `prevalence_row` vs `prevalence_stay` so you can *see* the skew). **Calibration and DCA are honestly still row-pooled** — and those are exactly the two that stay-peak *can't* rescue, so fixing them properly means computing them **per landmark** on calibrated probabilities.

------------------------------------------------------------------------

## 7. When is each needed? (by task shape)

| Task shape | Example | What to use |
|------------------------|------------------------|------------------------|
| **Single-window** (one prediction per stay) | extubation failure, readmission | Nothing special — row == patient. Compute everything once, directly. All three "ways" are identical here. |
| **Continuous, fixed outcome** (label = eventual outcome) | daily ICU mortality (task1/2) | **Outcome-anchored landmark** — line patients up by "windows before death/discharge". Plus stay-peak for the alarm view. |
| **Continuous, onset + horizon** (label = "event in next X hours") | sepsis-in-6h (task4) | **Dynamic-risk-set landmark** — at each hour, score only patients still admitted & event-free, label = "onset in next 6h". This is the textbook form (Kamran et al., NEJM AI 2024). |

------------------------------------------------------------------------

## 7b. How real benchmarks handle the same problem

You're not the first to hit "many rows per patient". Two well-known benchmarks solve it in **different** ways — and comparing them teaches one more idea.

### EHRSHOT (Stanford, 15 EHR tasks)

EHRSHOT has the **exact same** multiplicity: it makes **many labels per patient** (e.g. thrombocytopenia ≈ **33 labels per patient** — one per lab draw). What they do (`ehrshot/7_eval_finetune.py`):

- **Metrics:** AUROC, AUPRC, **Brier** only. **No** calibration curve / ECE, **no** DCA, **no** thresholds, **no** subgroups. Pure discrimination (+ Brier).
- **Point estimate:** **row-pooled** — every label thrown in together (our "Way 1").
- **The clever bit — the confidence interval:** they bootstrap by **resampling unique patients**, not labels (patient-stratified / clustered bootstrap, 1000 reps). A patient picked twice has *all* their labels weighted ×2. This keeps the **error bar honest** even though labels within a patient are correlated.

### Kamran et al. (NEJM AI 2024, sepsis / Epic ESM)

- **One score per hospitalization = the MAX score before a cutoff** = our **stay-peak** (Way 2).
- Discrimination only (AUROC), 1000-bootstrap CI, PPV/sensitivity at the alert threshold.
- Extra idea: the cutoff is **"before the clinician acted"** — a time anchor, like our landmark.
- The follow-up paper (Kamran et al., *NEJM AI* 2024) makes the point sharper still: when they evaluated sepsis models **only on predictions made before treatment began**, AUROC fell from **0.62 to 0.47** — a coin flip. Scoring a model *after* the clinician already acted is **label leakage in disguise**, and a landmark/hourly curve that looks great may just be reading the chart back to itself. (See §7d.)

### MIMIC-III benchmark (Harutyunyan et al., *Sci Data* 2019)

The most-cited public ICU benchmark — and a clean illustration that the unit of analysis **changes task by task**:

- **In-hospital mortality** and **phenotyping** → one row per ICU stay (single-window — easy, row == patient).
- **Decompensation** ("will this patient die in the next 24 h?") and **length-of-stay** → predicted **every hour from hour 4 onward** (the continuous case). \~**2.9 million** training instances, **positive rate only \~2 %**.

For the hourly tasks they **micro-average AUROC/AUPRC over all patient-hours pooled together** — exactly our **Way 1 (row-pooled)**, with **no** patient-clustering correction. Long stays dominate and prevalence is diluted to \~2 %. This is precisely the skew of Section 1, shipped as a community-standard benchmark — so the naive way isn't a strawman; it's what most people actually publish.

### HiRID-ICU-Benchmark & YAIB (the modern ICU benchmarks)

- **HiRID-ICU-Benchmark** (Yèche et al., *NeurIPS* 2021) splits tasks into **"dynamic/online"** (score at every \~5-min step — circulatory/respiratory failure within 12 h) vs **"single-point"** (mortality-at-24h). Online tasks are scored by **pooling every timestep**, with **AUPRC emphasized** because positives are only \~4 % of \~14 M timesteps. Same pooled-row unit as MIMIC.
- **YAIB — "Yet Another ICU Benchmark"** (van de Water et al., *ICLR* 2024) is a multi-center framework, and its most useful finding for us is a *meta* one: **"the choice of dataset, cohort definition, and preprocessing have a major impact on prediction performance — often more so than model class."** Concretely, flipping the mortality exclusion window from \>30 h to \>24 h moved AUROC by \~3 points (≈84 → 87). **How you define the unit moves the number more than which model you pick** — which is the whole reason this note exists.

### PhysioNet/CinC Challenge 2019 — a *fourth* way

Reyna et al. (*Crit Care Med* 2020) ran the big public sepsis-prediction challenge and **refused to collapse the hourly stream to a single AUROC at all**. Instead they invented a **clinical utility score** $U(s,t)$, evaluated at *every* hour and summed, then normalized so 1 = perfect and 0 = always-negative:

- maximum reward for firing **\~6 h before** sepsis onset,
- a graded reward window from \~12 h before to \~3 h after onset,
- a growing **penalty for late** predictions, a small penalty for firing **too early**,
- a small penalty for **every false alarm** on a non-septic patient (an explicit alarm-fatigue cost).

This is a philosophy beyond our three ways: instead of picking a unit and computing AUROC, **bake timeliness straight into a custom score**. It's the de-facto standard whenever *when* you fire is the entire point — and it's the closest thing the field has to a purpose-built metric for hourly-firing models.

### The new idea: pooling breaks **two** things, not one

| Problem from pooling correlated rows | Symptom | Who fixes it |
|------------------------|------------------------|------------------------|
| **Biased point estimate** | AUROC/prevalence too optimistic (long stays over-counted) | **stay-peak** or **landmark** (1 row/patient) |
| **Over-confident error bar** | confidence interval too *narrow* — 33 correlated labels look like 33 independent patients | **patient-clustered bootstrap** (EHRSHOT) |

These are **independent**. EHRSHOT fixes the *error bar* but keeps a *pooled point estimate*. FLAIR's stay-peak/landmark fix the *point estimate*. The fully-correct recipe does **both**: collapse to one honest number per patient (landmark/stay-peak) **and** bootstrap by patient.

### Benchmark comparison

|   | Prediction unit | Point estimate | Error bar | Calibration | DCA / threshold |
|------------|------------|------------|------------|------------|------------|
| **MIMIC-III (decompensation)** | per patient-hour | row-pooled | none (rows iid) | none | none |
| **HiRID / YAIB (online tasks)** | per timestep | row-pooled (AUPRC) | seed std only | none | none |
| **PhysioNet 2019** | per patient-hour | custom **utility score** (timeliness-weighted) | bootstrap | none | built into utility |
| **EHRSHOT** | per-label (many/patient) | row-pooled | patient-clustered bootstrap ✅ | Brier only (pooled) | none |
| **Kamran/ESM** | per-hospitalization | stay-peak (max) ✅ | bootstrap | none | PPV/sens at threshold |
| **FLAIR (now)** | row + stay-peak + landmark | landmark/stay-peak ✅ | bootstrap (landmark CI) | row-pooled ❌ | row-pooled ❌ |
| **FLAIR (ideal)** | landmark | landmark ✅ | patient-clustered ✅ | landmark ✅ | landmark ✅ |

**Read the top three rows as the field's default and the bottom four as the frontier:** most published benchmarks (MIMIC, HiRID, YAIB) **pool patient-hours and stop at discrimination** — no calibration, no DCA, no clustering. PhysioNet sidesteps the unit problem with a bespoke utility score. Only EHRSHOT fixes the error bar, and only FLAIR even attempts calibration/DCA in continuous form. The "ideal" row is where none of them are yet.

**Takeaways:** 1. Everyone agrees the *unit* problem is real — they just stop at different points. 2. EHRSHOT shows a piece FLAIR is missing: even a pooled number needs a **patient-clustered bootstrap** so the error bar isn't fake-tight. 3. Nobody except FLAIR even attempts continuous-correct **calibration + DCA** — which is exactly why those two are the hard, still-open part (Section 6).

------------------------------------------------------------------------

## 7c. What do the official reporting guidelines say? (TRIPOD+AI)

Short version: **they don't cover this case.** The continuous / many-predictions-per-patient situation is a genuine **gap in the official guidelines** — which is *why* FLAIR has to define its own recipe.

- **TRIPOD+AI** (BMJ 2024, the 27-item checklist FLAIR's `tripod_ai.json` is named after): its only time-related items are **8a** "define the outcome and its **time horizon**" (the prediction window) and **20a** "follow-up time" (study length). There is **no** item on the *unit of analysis* when one patient contributes many predictions, no clustering, no landmarking, no time-updated/dynamic models, no calibration-over-time. It implicitly assumes **one prediction per patient.**
- **TRIPOD-Cluster** (BMJ 2023) — the guideline you'd expect to handle it — explicitly **opts out**: \> "repeated measurements are clustered within individuals … requires special efforts … which is \> **beyond the scope of TRIPOD-Cluster**." It addresses *between*-cluster heterogeneity (sites, hospitals), **not** within-patient repeated predictions.
- **PROBAST / PROBAST+AI** (Wolff et al., *Ann Intern Med* 2019; Moons et al., *BMJ* 2025) — the *risk-of-bias* companion to TRIPOD. Its **analysis domain** flags exactly the failure modes this note is about: too few events per variable, mishandled missing data, and — pointedly — **failing to assess calibration**. But it too assumes one prediction per patient: there is **no signalling question** for within-patient repeated predictions, so a row-pooled continuous model can sail through PROBAST while committing every sin in Section 1.

**Implication:** there is *no* official standard for how to report a model that fires every hour/day. The three-ways framework here (stay-peak, landmark, patient-clustered bootstrap) is FLAIR filling that gap — the row-multiplicity skew, the peak-calibration trap, and the still-open continuous calibration + DCA problem are precisely the issues the guidelines leave unspecified.

------------------------------------------------------------------------

## 7d. Two cautionary tales: what happened when continuous sepsis models met the real world

The abstract problem (Section 1) isn't abstract. Two of the most famous deployed sepsis models — one celebrated, one notorious — are case studies in *exactly* the evaluation choices this note is about. They are the best argument for why the unit of analysis, calibration, and alert burden aren't pedantry.

### Tale 1 — TREWS (Johns Hopkins / Bayesian Health): the celebrated one

**What it is.** TREWS ("Targeted Real-time Early Warning System") grew out of the 2015 **TREWScore** (Henry et al., *Sci Transl Med* 2015) — a Cox-style time-varying risk score for septic shock that hit **AUC 0.83** and flagged patients a median **\~28 h before** shock onset, beating MEWS (AUC 0.73). It fires **continuously**.

**What they did to evaluate it.** The headline is a **prospective, multi-site** study across **5 hospitals**, monitoring **590,736 encounters** (Adams et al., *Nat Med* 2022), with a companion adoption paper (Henry et al., *Nat Med* 2022). Two things are worth copying and one is worth scrutinizing:

- **They collapsed the hourly stream to one unit per encounter.** Sensitivity ("**82 %** of sepsis encounters were flagged"), confirmation rate, mortality, and time-to-antibiotics are all reported **per hospitalization**, not per prediction-hour. This is our stay-level reduction (Section 2), done by necessity.
- **They measured a clinical outcome, not just AUC.** Encounters where a provider **confirmed the alert within 3 h** had an **adjusted \~18.7 % relative reduction in in-hospital mortality** (absolute −3.3 %), and antibiotics **\~1.85 h sooner**. That's an *impact* claim — the gold standard the methodology literature keeps asking for (§7f).
- **The catch — the comparison wasn't randomized.** The contrast is *provider responded promptly* vs *didn't*. A preprint critique (Nemati et al., "TREWS or FALSE?", *medRxiv* 2022 — **not peer-reviewed**) argues this is **confounding by indication / "convenient controls"**: prompt confirmation may just mark patients who already had a more attentive care team, so some of the "benefit" could be the team, not the alert. The authors rebut that adding process-of-care variables didn't move the result. The lesson stands regardless: **an as-treated adoption contrast is not the same as a randomized intervention**, and a continuous model's outcome claim is only as clean as its comparison group.

**Takeaway:** TREWS is what *good* looks like — per-encounter unit, prospective deployment, an outcome endpoint, an honest lead-time definition — and even it gets challenged on the one thing retrospective AUC can never settle: causality.

### Tale 2 — the Epic Sepsis Model (ESM): the notorious one

**What it is.** A proprietary in-EHR model scored **every 15 minutes** on inpatients, deployed at **hundreds** of US hospitals — for years, largely **without independent validation**.

**What independent validation found** (Wong et al., *JAMA Intern Med* 2021; **27,697 patients / 38,455 hospitalizations**):

| What Epic reported | What external validation measured |
|------------------------------------|------------------------------------|
| AUC **0.76–0.83** (vendor documentation) | **AUC 0.63** (95 % CI 0.62–0.64) |
| — | Sensitivity **33 %** at the recommended threshold |
| — | Alerts on **18 %** of all hospitalizations; **number-needed-to-evaluate ≈ 109** |
| — | **Missed 67 %** of sepsis cases; added only **7 %** sensitivity beyond patients clinicians had *already* treated |

**Why the gap?** The accompanying editorial (Habib, Lin & Grant, "The Epic Sepsis Model Falls Short", *JAMA Intern Med* 2021) names the causes, and every one is a theme from this note:

1.  **Label leakage.** The model's features reportedly included **billing codes and antibiotic-order data** — signals that only exist *after* a clinician already suspected sepsis. The 33 % → 7 % sensitivity collapse (once you exclude already-treated patients) is the fingerprint: it was partly **predicting the clinician, not the disease**. This is the same trap Kamran et al. quantified (§7b).
2.  **No site-specific recalibration** — a model trained on 3 health systems, deployed unchanged on a 4th.
3.  **Prevalence + alert burden.** At \~7 % prevalence and modest discrimination, PPV is low and **alert fatigue becomes the dominant real-world harm** — a first-class evaluation axis, not a footnote.

Epic **retooled** the model in 2022 (gradient-boosted, with site-specific tuning), and a later prospective multi-site study reports much-improved encounter-level AUROC — but with **per-site variation** wide enough to reinforce the moral: **trust your own prospective number, not the vendor's.**

### The third example worth knowing — Sepsis Watch (Duke)

Sendak et al. (*JMIR Med Inform* 2020) deployed a recurrent-network sepsis model (Futoma et al., *MLHC* 2017) live in the Duke ED, scoring **hourly**. Their distinctive move: they treated the **alert rate as a design constraint up front** — tuning toward **≤4 high-risk alerts per hour** so the tool stayed usable. It's the clearest example of **"work created per true positive" as a first-class metric**, designed in rather than discovered after launch.

### What these tales teach (mapped to this note)

| Real-world failure | Section it illustrates | FLAIR's answer |
|------------------------------|-----------------------|------------------------|
| Long stays over-counted → optimistic AUC | §1, §2 | stay-peak + landmark reductions |
| "Predicting the clinician" via post-hoc labels | §7b (Kamran), §7d (ESM) | label timing / pre-treatment evaluation (continuous rebuild) |
| Vendor AUC ≠ deployed AUC | §7d (ESM) | independent, local computation |
| Alert fatigue ignored | §7d (ESM, Sepsis Watch) | encounter/alarm pillar (alarms/day, NNE) |
| Outcome claim from a non-randomized contrast | §7d (TREWS) | (open) — needs prospective/randomized design |

------------------------------------------------------------------------

## 7e. What does the FDA want?

Short version: as of the **2022 Clinical Decision Support (CDS) final guidance**, a sepsis score that **fires automatically** is almost certainly a **regulated medical device**, and the FDA's expectations for it line up strikingly well with the metrics in this note.

### Is an hourly sepsis score even a "device"? (Usually yes.)

The CDS final guidance (FDA/CDRH, Sept 2022) lays out **four criteria**; software must meet **all four** to be *exempt* (Non-Device CDS). Two of them sink most continuous risk models:

- **Criterion 3 — no "specific output or directive."** FDA reads this strictly: software that outputs a **risk probability or risk score** for a specific condition **fails** this criterion. A sepsis score *is* a specific output.
- **Criterion 4 — the clinician can independently review the basis.** A model that **fires automatically under time pressure** with a non-inspectable basis fails this — it's the **automation-bias** concern. (Continuous models also often fail Criterion 1 by processing raw signals/IVD data.)

So a continuously-firing sepsis risk score is a **device (SaMD)**, not exempt CDS. This is why the regulatory status of the three tales differs sharply:

| Tool | FDA status |
|------------------------------------|------------------------------------|
| **Prenosis Sepsis ImmunoScore** | **De Novo authorized, Apr 2024** — first AI sepsis diagnostic; created a new device class (**21 CFR 880.6316**, product code **SAK**) |
| **TREWS / Bayesian Health** | **510(k) cleared** (K250680), found substantially equivalent to the Prenosis predicate *(exact clearance date reported in secondary press — treat as unconfirmed)* |
| **Epic Sepsis Model** | **Never FDA-reviewed** — deployed in-EHR in the historical regulatory gray area |

### The substantive bar: GMLP, calibration, subgroups, monitoring

Three FDA documents define *what good looks like*:

- **Good Machine Learning Practice (GMLP)** — 10 guiding principles (FDA + Health Canada + MHRA, 2021). The load-bearing ones here: **#3 data representative of the intended population**, **#7 evaluate the human-AI *team*** (not the model alone — echoing Sepsis Watch), **#8 testing under clinically relevant conditions on an independent test set**, and **#10 monitor deployed models for performance degradation and retraining risk.**
- **Predetermined Change Control Plan (PCCP)** — final guidance Dec 2024. Lets a manufacturer **pre-authorize future retraining/recalibration** and the validation each will get, so a model can be **updated without a new submission**. This is the regulatory answer to **calibration drift** (§7f): the FDA expects you to *plan* for the model to decay and be refit.
- **The evaluation expectations that follow** map almost one-to-one onto FLAIR's pillars: **discrimination *and* calibration** (not AUC alone — the ESM is the cautionary tale FDA reviewers cite), **subgroup / bias analysis**, **prospective clinical validity**, **human-AI team / alert burden**, **transparency of inputs and limitations**, and **ongoing performance monitoring**.

**Why this matters for FLAIR:** the report already computes the discrimination + calibration + DCA + fairness quartet the FDA effectively asks for. The **known gaps** (continuous calibration, continuous DCA, per-site/local validation, drift monitoring) are precisely the items GMLP #8/#10 and the PCCP press on — so closing them isn't gold-plating, it's the regulated bar.

------------------------------------------------------------------------

## 7f. How do you show a model is *truly* better? (beyond AUROC)

This is the question behind the whole note. The methodology literature is blunt: **a higher AUROC does not mean a better model.** Here is the canon, and how FLAIR's metrics already embody most of it.

### 1. AUROC is insensitive — and reclassification metrics don't rescue it

- **Cook (*Circulation* 2007)** showed the c-statistic barely moves even when a genuinely useful new marker meaningfully changes individual risks. Small AUC gains can hide real clinical value — or hide its absence.
- The fix people reached for — **NRI / IDI** (Pencina et al., *Stat Med* 2008) — turned out to be **worse than the disease**. A series of critiques (**Hilden & Gerds 2014; Kerr et al. 2014; Pepe et al. 2015**) showed NRI is **not a proper scoring rule**, is routinely miscomputed, and has an **unacceptably high false-positive rate** — it can declare a *useless* marker "significant" even on independent data. **Don't use NRI to claim improvement.**

### 2. Calibration is the other half of the model — and it has a hierarchy

- **Van Calster et al., "Calibration: the Achilles heel of predictive analytics" (*BMC Med* 2019)** is the flagship "AUROC is not enough" reference: a model can discriminate well yet systematically over- or under-state risk, leading to real harm at the bedside.
- Their **calibration hierarchy** (*J Clin Epidemiol* 2016) gives the vocabulary FLAIR should target: **mean** (average predicted = observed) → **weak** (slope = 1, intercept = 0) → **moderate** (a flexible calibration *curve* matches across the whole range) → **strong** (utopian). Reporting only slope/intercept (weak) can mask local miscalibration that a **moderate-level calibration plot** exposes. This is exactly why Section 4's "peak-calibration trap" matters: stay-peak can't even reach *mean* calibration.

### 3. Clinical usefulness = net benefit, not accuracy

- **Decision Curve Analysis** (Vickers & Elkin, *Med Decis Making* 2006; Vickers, Van Calster & Steyerberg, *BMJ* 2016; step-by-step guide, *Diagn Progn Res* 2019) puts benefits and harms on **one scale** via the threshold odds, and asks the only question that matters clinically: **does acting on this model beat "treat all" and "treat none" at the thresholds clinicians actually use?** FLAIR computes this (§7h) — but, per Section 6, still row-pooled.

### 4. "Validated" is not a stamp — it's a ladder, and it leaks over time

- **Steyerberg & Harrell (*J Clin Epidemiol* 2016)** define the ladder: **internal** (bootstrap optimism) → **internal-external** (leave-one-site/period-out) → **external** (new setting), with **geographic and temporal** validation probing transportability. **Riley et al. (*Stat Med* 2021)** give the minimum events needed to validate calibration and net benefit with any precision (often **≥100–200 events**, frequently more).
- **Van Calster et al., "There is no such thing as a validated prediction model" (*BMC Med* 2023)** is the title to remember: performance is **context-specific**, so "validated" only ever means "validated *here, now*."
- And it **decays**: **Davis et al. (*JAMIA* 2017)** documented **calibration drift** in AKI models over 9 years (discrimination held; calibration rotted), and **Finlayson et al., "The Clinician and Dataset Shift in AI" (*NEJM* 2021)** catalogues why — case-mix change, practice change, EHR upgrades, even the model's own deployment. This is the empirical basis for the FDA's monitoring/PCCP demands (§7e).

### 5. The final arbiter is prospective, not retrospective

- Retrospective AUC **systematically overstates** deployable performance. The vivid demonstration: a **"silent trial"** (run the model live but hide its output) where a model with retrospective **AUROC ≈ 0.90 fell to ≈ 0.50** in real time. The COVID-19 model review (**Wynants et al., *BMJ* 2020**) applied PROBAST to hundreds of models and found **nearly all at high risk of bias with optimistic reported performance.** The TREWS prospective study (§7d) is the contrast — flawed comparison and all, it at least measured a **real outcome on real future patients.**

### The one-line standard

> A model is **truly better** only when it shows **(1)** maintained discrimination, **(2)** *moderate-level* calibration, and **(3)** positive net benefit at clinically relevant thresholds — demonstrated under **(4)** internal-external + external/temporal validation with adequate sample size, reported per **(5)** TRIPOD+AI and clean against **(6)** PROBAST+AI, with **(7)** a plan for drift, and ideally **(8)** prospective / silent-mode (or randomized) confirmation. AUROC alone, and NRI, prove none of this.

FLAIR's report already covers (1), (2), (3), and (5) for single-window tasks. For **continuous** tasks, (2) and (3) are the known row-pooled gaps (Section 6); (4), (7), and (8) are organizational, not just code.

------------------------------------------------------------------------

## 7g. Where landmarking comes from (the statistical lineage)

FLAIR's "landmark" view (Section 2, Way 3) isn't homegrown — it's a 50-year-old idea from survival analysis, which is reassuring: the hard parts are already worked out.

- **Landmarking** itself: **van Houwelingen (*Scand J Stat* 2007)** and the book **van Houwelingen & Putter, *Dynamic Prediction in Clinical Survival Analysis* (2011)**. The core move — refit/score at each fixed time $s$ using only the patients still at risk at $s$, then slide $s$ forward — is *exactly* FLAIR's "freeze the clock at day 3, score everyone present, repeat." The book's "landmark supermodel" is the formal way to turn the per-landmark points into a **smooth skill-over-time curve**.
- **The AUROC-at-a-time-point** FLAIR plots is the **time-dependent ROC**: **Heagerty, Lumley & Pepe (*Biometrics* 2000)** (cumulative/dynamic form) and **Heagerty & Zheng (*Biometrics* 2005)** (the **incident-case / dynamic-control** $\text{AUC}(t)$ — the precise object behind "AUC this hour, among patients still at risk this hour"). **Blanche et al. (*Stat Med* 2013)** add censoring-correct estimators with **confidence intervals** and a two-marker comparison test — i.e. how to put honest error bars on each point of the curve.
- **Competing risks.** For an ICU-mortality model, **discharge alive is a competing event** — a i tpatient leaves the risk set without dying. Treating discharge as ordinary censoring **overstates** mortality risk and biases the curve. **Fine & Gray (*JASA* 1999)** (subdistribution hazard / cumulative incidence) is the standard correction, and Blanche et al. supply the matching competing-risks $\text{AUC}(t)$.
- **The error bar, again.** The patient-clustered bootstrap FLAIR needs (Section 7b) is the classical **cluster bootstrap** — **Field & Welsh (*JRSS-B* 2007)**, textbook treatment in **Davison & Hinkley (1997)**: resample whole *patients* (with their entire trajectory), never rows.

The dynamic-risk-set landmark in §7h (the "not yet in code" one) is just this lineage specialized to a moving risk set with an onset+horizon label — i.e. Heagerty-Zheng's incident/dynamic AUC, computed at each hour over the patients still admitted and event-free.

------------------------------------------------------------------------

## 7h. The math, every symbol defined

Simple formulas for **every method FLAIR actually computes today** (the episodic bundle: discrimination, calibration, DCA, fairness), plus the bootstrap CI and the three-ways reductions. The dynamic-risk-set landmark (the rigorous "Way 3" form) is listed last and flagged **not yet in code**.

### Shared symbols

| Symbol | Meaning |
|------------------------------------|------------------------------------|
| $N$ | number of scored units (in episodic mode, one per stay) |
| $i$ | index of a unit, $i = 1\dots N$ |
| $y_i \in \{0,1\}$ | true label (1 = event happened) |
| $p_i \in [0,1]$ | model's predicted probability (`y_prob`) |
| $t$ | a decision threshold; act when $p_i \ge t$ |
| $\hat y_i = \mathbb{1}\{p_i \ge t\}$ | the 0/1 decision at threshold $t$ |
| $\pi = \frac1N\sum_i y_i$ | prevalence (event rate) |
| $\mathbb{1}\{\cdot\}$ | indicator: 1 if true, else 0 |

**Confusion counts at** $t$: $TP=\sum_i \hat y_i y_i$, $FP=\sum_i \hat y_i(1-y_i)$, $FN=\sum_i(1-\hat y_i)y_i$, $TN=\sum_i(1-\hat y_i)(1-y_i)$.

### 1. Discrimination (`discrimination.json`)

**AUROC** — Mann–Whitney form (chance a random positive outranks a random negative), with $\mathcal{P}=\{i:y_i=1\}$, $\mathcal{N}=\{j:y_j=0\}$ and tie term $\psi(a,b)=\mathbb{1}\{a>b\}+\tfrac12\mathbb{1}\{a=b\}$:

$$\widehat{\text{AUROC}}=\frac{1}{|\mathcal P|\,|\mathcal N|}\sum_{i\in\mathcal P}\sum_{j\in\mathcal N}\psi(p_i,p_j)$$

**AUPRC** — area under precision–recall; **lift** $=\widehat{\text{AUPRC}}/\pi$ (1 = no skill). **Operating point at** $t$:

$$\text{sens}=\frac{TP}{TP+FN},\quad \text{spec}=\frac{TN}{TN+FP},\quad \text{PPV}=\frac{TP}{TP+FP},\quad \text{NPV}=\frac{TN}{TN+FN},\quad F_1=\frac{2TP}{2TP+FP+FN}$$

### 2. Calibration (`calibration.json`)

Bin the $p_i$ into $B=10$ equal-width bins; bin $b$ has $n_b$ units, mean prediction $\bar p_b$, observed rate $\bar y_b$.

**ECE** (gap between predicted and observed, size-weighted):

$$\text{ECE}=\sum_{b=1}^{B}\frac{n_b}{N}\,\bigl|\bar y_b-\bar p_b\bigr|$$

**Brier** (mean squared error of probabilities): $\text{Brier}=\frac1N\sum_i (p_i-y_i)^2$.

**Calibration slope/intercept** — fit $\operatorname{logit}(P(y=1))=\alpha+\beta\,\operatorname{logit}(p_i)$; perfect = $\beta{=}1,\ \alpha{=}0$ ($\beta$ = slope, $\alpha$ = intercept). **O:E** $=\bigl(\frac1N\sum y_i\bigr)/\bigl(\frac1N\sum p_i\bigr)$ (observed ÷ expected).

### 3. Decision-curve analysis (`dca.json`)

Net benefit trades true positives against false positives weighted by the threshold odds $\frac{t}{1-t}$:

$$\text{NB}_{\text{model}}(t)=\frac{TP}{N}-\frac{FP}{N}\cdot\frac{t}{1-t},\qquad \text{NB}_{\text{all}}(t)=\pi-(1-\pi)\frac{t}{1-t},\qquad \text{NB}_{\text{none}}=0$$

A model is "useful" at $t$ when $\text{NB}_{\text{model}}(t)>\max(\text{NB}_{\text{all}}(t),0)$.

### 4. Fairness (`fairness.json`)

For subgroup $g$, let $r_g$ = positive (alarm) rate, $\text{TPR}_g$, $\text{FPR}_g$ at the clinical threshold. Gaps = max − min across groups:

$$\text{DP gap}=\max_g r_g-\min_g r_g,\quad \text{EO}_{\text{TPR}}=\max_g\text{TPR}_g-\min_g\text{TPR}_g,\quad \text{EO}_{\text{FPR}}=\max_g\text{FPR}_g-\min_g\text{FPR}_g$$

(DP = demographic parity, EO = equalized odds.)

### 5. Bootstrap 95% CI (`_ci.py`)

Resample units with replacement $B{=}500$ times; recompute the statistic $\hat\theta^{(b)}$ each time; CI = the 2.5th and 97.5th percentiles:

$$\text{CI}_{95\%}=\bigl[\,Q_{2.5\%}(\hat\theta^{(1)}\dots\hat\theta^{(B)}),\ Q_{97.5\%}(\dots)\,\bigr]$$

In episodic mode one row = one stay, so resampling rows **is** a patient-clustered bootstrap.

### 6. The three-ways reductions

Let $S$ index stays and $\mathcal{I}_s$ = the rows belonging to stay $s$.

- **Row-pooled (Way 1):** use all rows as-is — $\{(p_i,y_i)\}_{i=1}^{N}$.
- **Stay-peak (Way 2):** one row per stay, $\ p_s^{\max}=\max_{i\in\mathcal I_s}p_i$, $\ y_s=\max_{i\in\mathcal I_s}y_i$. (FLAIR `stay_frame`.) Ranks fine; **not a probability** → no calibration.
- **Landmark (Way 3):** one row per stay *per landmark time*, then a curve $\widehat{\text{AUROC}}(t)$ over $t$.

### 7. Dynamic-risk-set landmark — **NOT yet in code**

The rigorous Way 3 for onset+horizon tasks (sepsis/task4; the image you shared). Extra symbols: $T_h$ = event-onset time of stay $h$, $D_h$ = discharge time, $\tau$ = horizon (6 h). At landmark time $t$:

$$\mathcal R(t)=\{h: T_h>t \text{ and } D_h>t\}\quad(\text{still admitted, event-free})$$ $$Y_h(t)=\mathbb{1}\{t<T_h\le t+\tau\}\quad(\text{onset within next }\tau)$$ $$\widehat{\text{AUROC}}(t)=\frac{\sum_{i,j\in\mathcal R(t)}Y_i(t)\,[1-Y_j(t)]\,\psi\!\left(p_i(t),p_j(t)\right)}{\bigl(\sum_{i\in\mathcal R(t)}Y_i(t)\bigr)\bigl(\sum_{j\in\mathcal R(t)}[1-Y_j(t)]\bigr)}$$

This is **landmark (Way 3)** specialized to a moving risk set. It belongs to the **continuous rebuild** (the deleted `_eval.landmark_table` was the first cut); the current episodic code does not compute it.

------------------------------------------------------------------------

## 7i. References & key citations

Grouped by the section that uses them. DOIs/PMIDs verified against publisher / PubMed / PMC pages during the literature search; a few items flagged where only secondary sources were reachable.

**Deployed sepsis models (§7d)** - Henry KE, Hager DN, Pronovost PJ, Saria S. "A targeted real-time early warning score (TREWScore) for septic shock." *Sci Transl Med* 2015;7(299):299ra122. doi:10.1126/scitranslmed.aab3719 · PMID 26246167. - Adams R, et al. "Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis." *Nat Med* 2022;28(7):1455–1460. doi:10.1038/s41591-022-01894-0 · PMID 35864252. - Henry KE, et al. "Factors driving provider adoption of the TREWS … and its effects on sepsis treatment timing." *Nat Med* 2022;28(7):1447–1454. doi:10.1038/s41591-022-01895-z. - Nemati S, et al. "Randomized Clinical Trials or Convenient Controls: TREWS or FALSE?" *medRxiv* 2022. doi:10.1101/2022.08.08.22278526. *(preprint — not peer-reviewed)* - Wong A, Otles E, Donnelly JP, et al. "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients." *JAMA Intern Med* 2021;181(8):1065–1070. doi:10.1001/jamainternmed.2021.2626 · PMID 34152373. - Habib AR, Lin AL, Grant RW. "The Epic Sepsis Model Falls Short—The Importance of External Validation." *JAMA Intern Med* 2021;181(8):1040–1041. doi:10.1001/jamainternmed.2021.3333 · PMID 34152360. - Sendak MP, et al. "Real-World Integration of a Sepsis Deep Learning Technology Into Routine Clinical Care (Sepsis Watch)." *JMIR Med Inform* 2020;8(7):e15182. doi:10.2196/15182 · PMID 32673244. - Futoma J, et al. "An Improved Multi-Output Gaussian Process RNN … for Early Sepsis Detection." *MLHC* 2017 (PMLR v68); arXiv:1708.05894.

**Benchmarks (§7b)** - Wornow M, et al. "EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models." *NeurIPS Datasets & Benchmarks* 2023; arXiv:2307.02028. - Harutyunyan H, et al. "Multitask learning and benchmarking with clinical time series data." *Sci Data* 2019;6:96. doi:10.1038/s41597-019-0103-9. - Yèche H, et al. "HiRID-ICU-Benchmark — A Comprehensive ML Benchmark on High-resolution ICU Data." *NeurIPS Datasets & Benchmarks* 2021; arXiv:2111.08536. - van de Water R, et al. "Yet Another ICU Benchmark (YAIB)." *ICLR* 2024; arXiv:2306.05109. - Reyna MA, et al. "Early Prediction of Sepsis from Clinical Data: The PhysioNet/CinC Challenge 2019." *Crit Care Med* 2020;48(2):210–217. doi:10.1097/CCM.0000000000004145 · PMID 31850926. - Kamran F, et al. "Evaluation of Sepsis Prediction Models before Onset of Treatment." *NEJM AI* 2024;1(3). doi:10.1056/AIoa2300032.

**Regulatory (§7e)** - FDA/CDRH. "Clinical Decision Support Software — Final Guidance." Sept 2022. - FDA, Health Canada, MHRA. "Good Machine Learning Practice for Medical Device Development: Guiding Principles." Oct 2021. - FDA. "Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions — Final Guidance." Dec 2024. - FDA. De Novo DEN230036 (Prenosis Sepsis ImmunoScore), authorized Apr 2024; classification 21 CFR 880.6316, product code SAK. 510(k) K250680 (Bayesian Health / TREWS).

**Proving "truly better" (§7f)** - Cook NR. "Use and misuse of the ROC curve in risk prediction." *Circulation* 2007;115(7):928–935. doi:10.1161/CIRCULATIONAHA.106.672402. - Pencina MJ, et al. "Evaluating the added predictive ability of a new marker (NRI/IDI)." *Stat Med* 2008;27(2):157–172. doi:10.1002/sim.2929. - Hilden J, Gerds TA. "Do not rely on IDI and NRI." *Stat Med* 2014;33(19):3405–3414. doi:10.1002/sim.5804. · Kerr KF, et al. *Epidemiology* 2014;25(1):114–121. doi:10.1097/EDE.0000000000000018. · Pepe MS, et al. "The NRI: A Misleading Measure…" *Stat Biosci* 2015;7(2):282–295. doi:10.1007/s12561-014-9118-0. - Van Calster B, et al. "Calibration: the Achilles heel of predictive analytics." *BMC Med* 2019;17:230. doi:10.1186/s12916-019-1466-7. · "A calibration hierarchy…" *J Clin Epidemiol* 2016;74:167–176. doi:10.1016/j.jclinepi.2015.12.005. - Vickers AJ, Elkin EB. "Decision curve analysis." *Med Decis Making* 2006;26(6):565–574. doi:10.1177/0272989X06295361. · Vickers AJ, Van Calster B, Steyerberg EW. *BMJ* 2016;352:i6. doi:10.1136/bmj.i6. · "Step-by-step guide to DCA." *Diagn Progn Res* 2019;3:18. doi:10.1186/s41512-019-0064-7. - Steyerberg EW, Harrell FE. "Prediction models need appropriate internal, internal-external, and external validation." *J Clin Epidemiol* 2016;69:245–247. doi:10.1016/j.jclinepi.2015.04.005. · Riley RD, et al. "Minimum sample size for external validation…" *Stat Med* 2021;40(19):4230–4251. doi:10.1002/sim.9025. · Van Calster B, et al. "There is no such thing as a validated prediction model." *BMC Med* 2023;21:70. doi:10.1186/s12916-023-02779-w. - Davis SE, et al. "Calibration drift … for acute kidney injury." *JAMIA* 2017;24(6):1052–1061. doi:10.1093/jamia/ocx030. · Finlayson SG, et al. "The Clinician and Dataset Shift in AI." *NEJM* 2021;385(3):283–286. doi:10.1056/NEJMc2104626. - Wynants L, et al. "Prediction models for diagnosis and prognosis of covid-19." *BMJ* 2020;369:m1328. doi:10.1136/bmj.m1328.

**Reporting / risk-of-bias standards (§7c)** - Collins GS, et al. "TRIPOD statement." *BMJ* 2015;350:g7594. doi:10.1136/bmj.g7594. · "TRIPOD+AI." *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378. - Wolff RF, et al. "PROBAST." *Ann Intern Med* 2019;170(1):51–58. doi:10.7326/M18-1376. · Moons KGM, et al. "PROBAST+AI." *BMJ* 2025;388:e082505. doi:10.1136/bmj-2024-082505.

**Landmarking & dynamic prediction (§7g)** - van Houwelingen HC. "Dynamic Prediction by Landmarking in Event History Analysis." *Scand J Stat* 2007;34(1):70–85. doi:10.1111/j.1467-9469.2006.00529.x. · van Houwelingen HC, Putter H. *Dynamic Prediction in Clinical Survival Analysis.* Chapman & Hall/CRC, 2011. - Heagerty PJ, Lumley T, Pepe MS. "Time-Dependent ROC Curves for Censored Survival Data." *Biometrics* 2000;56(2):337–344. doi:10.1111/j.0006-341X.2000.00337.x. · Heagerty PJ, Zheng Y. "Survival Model Predictive Accuracy and ROC Curves." *Biometrics* 2005;61(1):92–105. doi:10.1111/j.0006-341X.2005.030814.x. · Blanche P, et al. *Stat Med* 2013;32(30):5381–5397. doi:10.1002/sim.5958. - Fine JP, Gray RJ. "A Proportional Hazards Model for the Subdistribution of a Competing Risk." *JASA* 1999;94(446):496–509. doi:10.1080/01621459.1999.10474144. - Field CA, Welsh AH. "Bootstrapping Clustered Data." *JRSS-B* 2007;69(3):369–390. doi:10.1111/j.1467-9868.2007.00593.x. · Davison AC, Hinkley DV. *Bootstrap Methods and their Application.* Cambridge, 1997.

------------------------------------------------------------------------

## 8. One-line glossary

- **Ranking** — putting patients in order of risk; you don't care about the exact number, only who's higher.
- **Probability (calibrated)** — a number that means what it says: "30%" ⇒ 30 out of 100 such patients have the event.
- **Threshold** — the cut-off where you act ("treat if risk \> 20%").
- **Net benefit / DCA** — a score for "is acting at this threshold worth the false alarms?"
- **Row-pooled** — score every prediction row together (naive; over-counts long stays).
- **Stay-peak** — collapse each patient to their highest risk (the alarm trigger).
- **Landmark** — score patients one row each *at a fixed time point*, giving a skill-over-time curve.
- **Risk set** — at a given time, the patients still in play (still admitted, event not yet happened).
- **Patient-clustered bootstrap** — resample whole *patients* (not rows) to build the confidence interval, so correlated rows within a patient don't fake a tighter error bar (EHRSHOT's trick).
- **Calibration hierarchy** — mean → weak (slope/intercept) → moderate (the full calibration *curve*) → strong (utopian). "Moderate" is the bar a clinical model should clear; slope alone (weak) hides local miscalibration.
- **Label leakage** — using features/labels that only exist *after* the clinician acted (billing codes, antibiotic orders), so the model "predicts" what was already known. Sank the Epic Sepsis Model.
- **Confounding by indication** — when the patients who got the intervention differ systematically from those who didn't, so an apparent benefit may be the patients, not the tool (the TREWS-critique objection).
- **Calibration drift / dataset shift** — a model's probabilities go stale as case-mix, practice, or the EHR changes; discrimination can hold while calibration rots. Why the FDA wants monitoring + a recalibration plan (PCCP).
- **Silent trial** — run the model live but hide its output, to see real-time performance before trusting it; retrospective AUROC routinely overstates it (0.90 → \~0.50 in one famous case).
- **GMLP / PCCP** — FDA's "Good Machine Learning Practice" principles, and the "Predetermined Change Control Plan" that pre-authorizes future retraining/recalibration.

------------------------------------------------------------------------

### TL;DR

- **Discrimination** only needs *order* → all 3 ways work; landmark is cleanest, stay-peak is the alarm view, row-pooled is optimistic.
- **Calibration & DCA** need a *real probability* → only **landmark** is honest. **Peak is not a probability** (max-of-many is inflated), so it can rank but never calibrate.
- FLAIR does discrimination all three ways today; **calibration and DCA are still row-pooled (the known gap)**.
- **The field mostly stops at pooled discrimination** (MIMIC, HiRID, YAIB); only EHRSHOT fixes the error bar and only PhysioNet's utility score bakes in timeliness. Nobody else does continuous calibration/DCA — that's FLAIR's frontier (§7b).
- **The real-world stakes are settled law:** the Epic Sepsis Model failed external validation (AUC 0.83→0.63, 67 % of sepsis missed, label leakage); TREWS shows the right shape (per-encounter unit, prospective, an outcome endpoint) but can't fully escape confounding without randomization (§7d).
- **The FDA effectively asks for FLAIR's exact quartet** — discrimination *and* calibration, subgroup/bias, DCA, plus prospective evidence and drift monitoring (§7e).
- **"Truly better" ≠ higher AUROC.** It means moderate calibration + positive net benefit + external/temporal validation + prospective confirmation, reported per TRIPOD+AI. AUROC alone and NRI prove none of it (§7f).