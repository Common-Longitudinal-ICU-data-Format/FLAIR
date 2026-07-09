# Episodic vs Continuous Evaluation, In Simple Terms

This note explains what `flair build-report` should do for two task shapes:

1.  **Episodic tasks**: one prediction per hospital stay.
2.  **Continuous tasks**: many predictions per hospital stay, such as daily ICU mortality or hourly sepsis risk.

The main rule is simple: **do not score continuous rows as if they were independent patients.** A 10-day stay would count 10 times, while a 2-day stay would count 2 times. That makes the numbers look better or worse for the wrong reason.

## 1. Episodic Tasks: Basic TRIPOD+AI Is Enough

For an episodic task, one row already means one patient stay.

Examples:

| Task | Prediction shape | Why it is simple |
|------------------------|------------------------|------------------------|
| Readmission | one score per stay | the stay is counted once |
| Extubation failure | one score at extubation | the patient has one decision point |
| 24-hour ICU mortality | one score at a fixed time | one patient, one score, one label |

Because the unit is already clean, FLAIR can run the normal report directly:

| Report file | Plain question |
|------------------------------------|------------------------------------|
| `discrimination.json` | Does the model rank high-risk patients above low-risk patients? |
| `calibration.json` | When the model says 30 percent, is the real event rate about 30 percent? |
| `dca.json` | At the clinical threshold, is acting on the model worth the false alarms? |
| `fairness.json` | Does performance look similar across groups? |

This is the basic TRIPOD+AI setup. TRIPOD+AI asks for clear outcome definition, discrimination, calibration, clinical usefulness, and subgroup reporting. That works well when one prediction equals one patient.

For episodic tasks, there is no special landmark or peak logic. The build report can just score the table as-is, because rows and patients line up.

Current FLAIR behavior:

| Mode | Use when | Output |
|------------------------|------------------------|------------------------|
| `episodic` | one prediction per stay | discrimination, calibration, DCA, fairness |

## 2. Continuous Tasks: Two Modes Are Needed

For a continuous task, the model predicts again and again during the same stay.

Examples:

| Task | Prediction shape | Problem |
|------------------------|------------------------|------------------------|
| Daily ICU mortality | one score per ICU day | long stays create more rows |
| Sepsis in next 6 hours | one score per hour | one patient can create many alert chances |
| Deterioration warning score | repeated score while admitted | the report must respect hospital workflow |

If we pool every row, the report no longer answers a patient-level question. It answers a row-level question. That is usually not what the clinician, site, or regulator wants.

So FLAIR uses two continuous modes because they answer two different real questions.

| Mode | Plain question | Best for | What FLAIR reports |
|------------------|------------------|------------------|------------------|
| `peak` | Did this patient ever get high risk? | screening, alerts, burden | discrimination and fairness |
| `landmark` | At this fixed time, is the risk number honest? | calibration, DCA, lead-time curves | discrimination, calibration, DCA, fairness per landmark, plus `leadtime.json` |

**Why `peak` exists**

Peak mode collapses each stay to one number: the highest risk score during the stay.

That matches an alarm workflow. If the model crosses the alert threshold at any point, the patient gets flagged. For that question, the peak score is useful.

Peak is good for:

- AUROC or AUPRC as an alarm-screening view.
- Sensitivity, PPV, and number needed to evaluate at an alert threshold.
- Fairness of alert behavior across groups.
- Comparing how much alert burden different models create.

Peak is not good for calibration or DCA.

Reason: the maximum of many scores is not a probability. If a patient has ten daily predictions around 10 percent, the highest one may be much larger than 10 percent just by chance. That peak can rank patients, but it does not mean, "this patient has exactly this risk." Calibration and DCA need a real probability, so FLAIR should not run them on peak scores.

**Why `landmark` exists**

Landmark mode freezes time and scores each patient once at that time.

Example for daily mortality:

| Landmark             | What gets scored                           |
|----------------------|--------------------------------------------|
| 0 windows before end | each stay's last prediction                |
| 1 window before end  | each stay's prediction one window earlier  |
| 2 windows before end | each stay's prediction two windows earlier |

At each landmark, every stay contributes at most one row. That makes the table look like an episodic task again, so the four normal pillars are valid at that landmark.

Landmark is needed when we want:

- A calibrated risk number.
- Net benefit or DCA at a threshold.
- A curve showing performance as the outcome gets closer.
- Lead time, meaning how early the model starts warning.

Current FLAIR behavior:

| Mode | Use when | Output |
|------------------------|------------------------|------------------------|
| `peak` | continuous screening or alert evaluation | discrimination and fairness only |
| `landmark` | continuous probability and decision evaluation | all four pillars per landmark, plus lead-time |

**Can landmarking be used for sepsis?**

Yes, but sepsis needs the right kind of landmark.

For sepsis, the question is usually: "At this hour, will sepsis happen in the next X hours?"

The honest setup is a dynamic risk-set landmark:

| Step | Plain meaning |
|------------------------------------|------------------------------------|
| Pick an hour | for example, 8 AM or hour 12 of admission |
| Keep only eligible patients | still admitted, not already septic, not already treated |
| Score each patient once | one row per patient at that hour |
| Label the row | 1 if sepsis starts within the next X hours, otherwise 0 |
| Repeat | move to the next hour and build a curve |

This matters because scoring after antibiotics, cultures, or treatment orders can create leakage. The model may look good because it is reading what the clinician already did, not because it predicted sepsis early.

This is the point of the Kamran sepsis evaluation in `docs/paper/lit/AIoa2300032.pdf`: they evaluated sepsis predictions before treatment began. That is the safer clinical question.

For sepsis build reports, both modes are useful:

| Mode | Sepsis use |
|------------------------------------|------------------------------------|
| `peak` | "Would this model ever alert during the stay, and how many alerts would clinicians see?" |
| `landmark` | "At this hour, among patients still at risk, is the risk probability honest and useful?" |

**What other papers have done**

Different benchmarks solve the repeated-row problem in different ways.

| Paper or benchmark | What they did | Lesson for FLAIR |
|------------------------|------------------------|------------------------|
| MIMIC-III decompensation | pooled patient-hours | common, but long stays can dominate |
| HiRID ICU benchmark | pooled frequent time steps | useful for ranking benchmarks, weak for patient-level decisions |
| EHRSHOT | pooled repeated labels but bootstrapped by patient | patient-clustered uncertainty is important |
| PhysioNet/CinC 2019 sepsis | used a custom timeliness utility score | early warning may need timing-aware utility, not just AUROC |
| Kamran/Epic ESM sepsis | used max score per hospitalization before treatment | close to FLAIR `peak`, with attention to leakage |
| Zhang/Parker JAMA heart transplant score | used time-varying clinical data, discrete-time survival, time-dependent AUC, and calibration | close to FLAIR `landmark`: score at defined times, then check ranking and calibration |
| Mayampurath/Parker clinical-utility EWS paper | used census-based utility over concurrently hospitalized patients | possible future build-report add-on for real hospital operations |

**Zhang/Parker paper: why it matters here**

Local PDF: `docs/paper/lit/jama_zhang_2024_oi_230162_1736960263.11982.pdf`.

Paper: Zhang KC, Narang N, Jasseron C, et al. "Development and Validation of a Risk Score Predicting Death Without Transplant in Adult Heart Transplant Candidates." JAMA. 2024.

This is not an ICU sepsis paper, but it is very relevant to continuous reporting.

What they did:

| Item | Plain version |
|------------------------------------|------------------------------------|
| Clinical problem | rank heart transplant candidates by urgency |
| Data shape | patient status and lab values change over time |
| Modeling setup | discrete-time survival with repeated time intervals |
| Main outcome | death without transplant within 6 weeks |
| Evaluation | incident/dynamic time-dependent AUC and calibration plots |
| Fairness check | calibration by sex and race/ethnicity |
| Clinical output | a 50-point urgency score for allocation |

Why it matters for FLAIR:

- It shows that continuous clinical scores should be evaluated at defined time points, not by blindly pooling all rows.
- It checks both ranking and calibration, not just AUROC.
- It turns model output into a clinical score where calibration matters because score differences affect allocation.
- It supports adding stronger time-dependent evaluation summaries to continuous build reports.

Can we add this idea to `build-report`?

Yes, mostly through landmark mode. FLAIR already has the right structure: one row per stay per landmark, then AUROC, calibration, DCA, and fairness at each landmark.

Useful additions inspired by Zhang/Parker would be:

| Possible build-report addition | Why useful | Difficulty |
|------------------------|------------------------|------------------------|
| Time-dependent AUC summary across landmarks | gives one readable summary of the curve | low to medium |
| Calibration-by-group over landmarks | checks whether risk numbers stay honest by sex/race/ethnicity | medium |
| Score-distribution plot by outcome at each landmark | shows how much the model separates groups | low |
| Export a compact clinical-score table | useful if a model is turned into points or bands | medium |

These are additions to landmark reporting. They do not replace `peak`, because Zhang/Parker is about calibrated ranking for allocation, not alert firing.

**Clinical-utility EWS paper: why it matters here**

Local PDF: `docs/paper/lit/CCMED-D-26-00584.pdf`.

Paper draft: Mayampurath A, Wu P, Carey K, Strutz S, Afshar M, Edelson DP, Churpek MM, Parker WF. "Evaluating the Clinical Utility of Early Warning Scores." Critical Care Medicine manuscript draft.

This paper is directly relevant to build reports because it asks a practical hospital question: "If this early warning score is deployed, is it useful under real staffing and census conditions?"

Standard DCA asks whether a threshold has positive net benefit. That is useful, but it assumes a clean patient-level table and a threshold-based decision.

The clinical-utility paper adds an operational view:

| Idea | Plain version |
|------------------------------------|------------------------------------|
| Census-based utility | evaluate patients who are in the hospital at the same time |
| Cost values | give different values to TP, FP, FN, and TN |
| Treat-none comparator | compare the model to no deployed warning system |
| Perfect-model comparator | show how much possible benefit the model captures |
| Threshold focus | evaluate at the operating threshold used for alerts |
| Calibration-independent utility | can compare models without requiring all scores to be on the same probability scale |

Why this is useful for FLAIR:

- DCA tells us whether a probability threshold is worth using.
- Census-based utility tells us whether an alerting system is useful under real hospital workload.
- It can represent false negatives and false positives with different costs.
- It can evaluate days or hours when the hospital has many or few deterioration events.
- It is closer to how rapid response teams and bedside staff experience the model.

Can we add this to `build-report`?

Yes, but it should be a new optional output, not a replacement for DCA.

A reasonable future file would be `clinical_utility.json`.

It would need these inputs:

| Needed input | Why |
|------------------------------------|------------------------------------|
| prediction time | to know who was in the hospital at the same time |
| hospitalization ID | to avoid double-counting patients incorrectly |
| event time or label horizon | to decide TP, FP, FN, TN at each evaluation time |
| operational threshold | to know when the model alerts |
| cost values | to define how bad false alarms and missed events are |
| census sampling rule | for example, evaluate every day at 8 AM or every hour |

Recommended build-report path:

| Phase | What to add |
|------------------------------------|------------------------------------|
| Now | keep DCA as the standard TRIPOD+AI clinical-utility metric |
| Next | strengthen alert-burden summaries in `peak` mode, such as alerts per day alongside NNE |
| Later | add optional `clinical_utility.json` for census-based utility when prediction times and cost values are available |

This keeps the report honest. DCA remains the standard probability-based decision metric. Census-based utility becomes the operations metric for early warning systems.

**Bottom line**

For episodic tasks, basic TRIPOD+AI reporting is enough because one row already equals one stay.

For continuous tasks, FLAIR needs two modes:

| Mode | Why it exists |
|------------------------------------|------------------------------------|
| `peak` | best for alert screening and burden: did the stay ever cross the line? |
| `landmark` | best for honest probabilities and decisions: what was the risk at a fixed time? |

Use `peak` when the report is about alarms. Use `landmark` when the report is about calibrated risk, DCA, sepsis risk sets, or time-dependent performance.

The Zhang/Parker paper supports stronger time-dependent landmark reporting. The clinical-utility EWS paper supports a future optional build-report output for census-based utility, especially for real-time warning systems.