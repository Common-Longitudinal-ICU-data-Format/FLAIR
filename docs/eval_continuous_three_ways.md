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

### The new idea: pooling breaks **two** things, not one

| Problem from pooling correlated rows | Symptom | Who fixes it |
|------------------------|------------------------|------------------------|
| **Biased point estimate** | AUROC/prevalence too optimistic (long stays over-counted) | **stay-peak** or **landmark** (1 row/patient) |
| **Over-confident error bar** | confidence interval too *narrow* — 33 correlated labels look like 33 independent patients | **patient-clustered bootstrap** (EHRSHOT) |

These are **independent**. EHRSHOT fixes the *error bar* but keeps a *pooled point estimate*. FLAIR's stay-peak/landmark fix the *point estimate*. The fully-correct recipe does **both**: collapse to one honest number per patient (landmark/stay-peak) **and** bootstrap by patient.

### Benchmark comparison

|   | Prediction unit | Point estimate | Error bar | Calibration | DCA / threshold |
|------------|------------|------------|------------|------------|------------|
| **EHRSHOT** | per-label (many/patient) | row-pooled | patient-clustered bootstrap ✅ | Brier only (pooled) | none |
| **Kamran/ESM** | per-hospitalization | stay-peak (max) ✅ | bootstrap | none | PPV/sens at threshold |
| **FLAIR (now)** | row + stay-peak + landmark | landmark/stay-peak ✅ | bootstrap (landmark CI) | row-pooled ❌ | row-pooled ❌ |
| **FLAIR (ideal)** | landmark | landmark ✅ | patient-clustered ✅ | landmark ✅ | landmark ✅ |

**Takeaways:** 1. Everyone agrees the *unit* problem is real — they just stop at different points. 2. EHRSHOT shows a piece FLAIR is missing: even a pooled number needs a **patient-clustered bootstrap** so the error bar isn't fake-tight. 3. Nobody except FLAIR even attempts continuous-correct **calibration + DCA** — which is exactly why those two are the hard, still-open part (Section 6).

------------------------------------------------------------------------

## 7c. What do the official reporting guidelines say? (TRIPOD+AI)

Short version: **they don't cover this case.** The continuous / many-predictions-per-patient situation is a genuine **gap in the official guidelines** — which is *why* FLAIR has to define its own recipe.

- **TRIPOD+AI** (BMJ 2024, the 27-item checklist FLAIR's `tripod_ai.json` is named after): its only time-related items are **8a** "define the outcome and its **time horizon**" (the prediction window) and **20a** "follow-up time" (study length). There is **no** item on the *unit of analysis* when one patient contributes many predictions, no clustering, no landmarking, no time-updated/dynamic models, no calibration-over-time. It implicitly assumes **one prediction per patient.**
- **TRIPOD-Cluster** (BMJ 2023) — the guideline you'd expect to handle it — explicitly **opts out**: \> "repeated measurements are clustered within individuals … requires special efforts … which is \> **beyond the scope of TRIPOD-Cluster**." It addresses *between*-cluster heterogeneity (sites, hospitals), **not** within-patient repeated predictions.

**Implication:** there is *no* official standard for how to report a model that fires every hour/day. The three-ways framework here (stay-peak, landmark, patient-clustered bootstrap) is FLAIR filling that gap — the row-multiplicity skew, the peak-calibration trap, and the still-open continuous calibration + DCA problem are precisely the issues the guidelines leave unspecified.

------------------------------------------------------------------------

## 7d. The math, every symbol defined

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

------------------------------------------------------------------------

### TL;DR

- **Discrimination** only needs *order* → all 3 ways work; landmark is cleanest, stay-peak is the alarm view, row-pooled is optimistic.
- **Calibration & DCA** need a *real probability* → only **landmark** is honest. **Peak is not a probability** (max-of-many is inflated), so it can rank but never calibrate.
- FLAIR does discrimination all three ways today; **calibration and DCA are still row-pooled (the known gap)**.