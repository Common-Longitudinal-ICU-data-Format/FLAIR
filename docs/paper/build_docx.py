#!/usr/bin/env python3
"""Build main_v2.docx (Google-Docs uploadable) from the corrected section
.tex files. Prose is parsed straight from sections/*.tex so the Word draft
stays in lockstep with the LaTeX; the five tables and the figures are rendered
natively (LaTeX tables/TikZ can't be auto-converted faithfully).

Run: ../../.venv/bin/python build_docx.py   (needs python-docx + figures/)
"""
from __future__ import annotations
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = Path(__file__).resolve().parent
SEC = HERE / "sections"
FIG = HERE / "figures"

# ---------------------------------------------------------------- LaTeX → text
_SUBS = [
    (r"\\noindent", ""), (r"\\clearpage", ""), (r"\\medskip", ""),
    (r"\\todo\{([^}]*)\}", r"[TODO: \1]"),
    (r"\\est\{([^}]*)\}", r"\1"),
    (r"\\textbf\{([^}]*)\}", r"\1"), (r"\\textit\{([^}]*)\}", r"\1"),
    (r"\\emph\{([^}]*)\}", r"\1"), (r"\\texttt\{([^}]*)\}", r"\1"),
    (r"\\citep\[[^\]]*\]\{([^}]*)\}", r"[\1]"),
    (r"\\citep\{([^}]*)\}", r"[\1]"), (r"\\citet\{([^}]*)\}", r"\1"),
    (r"\\cite\{([^}]*)\}", r"[\1]"),
    (r"\\S\\ref\{[^}]*\}", "the relevant section"),
    (r"\\ref\{[^}]*\}", "[ref]"), (r"\\label\{[^}]*\}", ""),
    (r"\\S(?=[\\~ ])", "Section "),
    (r"\\%", "%"), (r"\\&", "&"), (r"\\_", "_"),
    (r"\\\{", "{"), (r"\\\}", "}"), (r"\\#", "#"),
    (r"\\in\b", "∈"), (r"\\pm\b", "±"),
    (r"\\ge\b", "≥"), (r"\\le\b", "≤"), (r"\\times\b", "×"),
    (r"\\rightarrow", "→"), (r"\\to\b", "→"),
    (r"\\infty\b", "∞"), (r"\\sim\b", "~"),
    (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\\text\{([^}]*)\}", r"\1"),
    (r"_\{([^}]*)\}", r"\1"), (r"\^\{([^}]*)\}", r"\1"),
    (r"\\,", " "), (r"\\ ", " "), (r"\\\\", " "),
    (r"\$", ""), (r"\{,\}", ","), (r"\{", ""), (r"\}", ""),
    (r"~", " "),
]

def latex_to_text(s: str) -> str:
    # drop full-line comments and inline comments (protect escaped \%)
    s = s.replace(r"\%", "\x00")
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("\x00", r"\%")
    s = s.replace("---", "—").replace("--", "–")
    for pat, rep in _SUBS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s)   # collapse newlines too — else Word/Docs shows hard breaks
    return s

# ---------------------------------------------------------------- doc helpers
def add_heading(doc, text, level):
    h = doc.add_heading(text.strip(), level=level)
    return h

def add_para(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
    p.add_run(text.strip())
    return p

def emit_prose(doc, block: str):
    """Render a prose chunk: split into paragraphs, route headings."""
    # neutralize list environments; turn items into blank-separated bullets
    block = re.sub(r"\\begin\{(itemize|enumerate)\}(\[[^\]]*\])?", "", block)
    block = re.sub(r"\\end\{(itemize|enumerate)\}", "", block)
    block = re.sub(r"\\item\s+", "\n\n• ", block)
    # pull \section/\subsection/\paragraph as structural markers first
    lines = block.split("\n")
    buff: list[str] = []

    def flush():
        if not buff:
            return
        raw = "\n".join(buff).strip()
        buff.clear()
        if not raw:
            return
        for para in re.split(r"\n\s*\n", raw):
            t = latex_to_text(para).strip()
            if t:
                add_para(doc, t)

    for ln in lines:
        m = re.match(r"\s*\\section\*?\{(.+?)\}", ln)
        if m:
            flush(); add_heading(doc, latex_to_text(m.group(1)), 1); continue
        m = re.match(r"\s*\\subsection\*?\{(.+?)\}", ln)
        if m:
            flush(); add_heading(doc, latex_to_text(m.group(1)), 2); continue
        m = re.match(r"\s*\\paragraph\{(.+?)\}(.*)", ln)
        if m:
            flush()
            rest = latex_to_text(m.group(2)).strip()
            add_para(doc, rest, bold_lead=latex_to_text(m.group(1)).strip())
            continue
        buff.append(ln)
    flush()

def add_table(doc, header, rows, caption):
    cap = doc.add_paragraph()
    cr = cap.add_run(caption)
    cr.bold = True; cr.font.size = Pt(9)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = h
        for run in c.paragraphs[0].runs:
            run.bold = True; run.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for run in cells[i].paragraphs[0].runs:
                run.font.size = Pt(8)
    doc.add_paragraph()

def add_figure(doc, images, caption, width=5.8):
    for img in images:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        path = FIG / img
        if path.exists():
            p.add_run().add_picture(str(path), width=Inches(width))
        else:
            p.add_run(f"[missing figure: {img}]")
    cap = doc.add_paragraph()
    cr = cap.add_run(caption); cr.italic = True; cr.font.size = Pt(9)
    doc.add_paragraph()

# ---------------------------------------------------------------- table data
TABLES = {
    "tab:consortium-sites": dict(
        caption="Table 1. Federated consortium cohort skeleton (one suppressed "
                "table1.json per site, T1 cohort). MIMIC-CLIF is the public development "
                "entry point; Rush is the first reporting external site. Remaining sites "
                "report at submission, so their cells read TBD. Cell suppression at n<10 "
                "site by site.",
        header=["", "MIMIC-CLIF", "Rush", "Site C", "Site D"],
        rows=[["ICU stays", "72,778", "13,839", "TBD", "TBD"],
              ["Female (%)", "43.2%", "47.5%", "TBD", "TBD"],
              ["Age, median (yr)", "67", "65", "TBD", "TBD"],
              ["SOFA hosp. max (med.)", "5", "4", "TBD", "TBD"]]),
    "tab:mimic-snapshot": dict(
        caption="Table 2. MIMIC-CLIF cohort snapshot per task, written as a suppressed "
                "table1.json by the FLAIR baseline (flair-baseline train; flair_benchmark "
                "v0.1.0; HIPAA cell-suppression at n<10). Counts are per stitched encounter "
                "(hospitalization_join_id).",
        header=["", "T1 mort.", "T2 LTACH", "T3 ext-fail", "T4 sepsis", "T5 readmit"],
        rows=[["ICU encounters", "64,408", "64,408", "26,768", "80,600", "67,235"],
              ["Decision rows", "198,125", "198,125", "26,771", "14,347,321", "67,250"],
              ["Positive rate (row)", "19.2%", "14.3%", "5.4%", "0.2%", "9.2%"],
              ["Stays ever positive", "13.0%", "6.3%", "5.4%", "7.2%", "9.2%"],
              ["Sex: F / M", "27,795/36,613", "27,795/36,613", "10,031/16,737", "35,760/44,840", "29,199/38,036"],
              ["Age 65+ share", "55.3%", "55.3%", "54.1%", "54.8%", "54.1%"],
              ["SOFA hosp. max (med., IQR)", "5 (2–8)", "5 (2–8)", "7 (5–10)", "4 (2–7)", "4 (2–7)"],
              ["Charlson (median, IQR)", "2 (0–4)", "2 (0–4)", "2 (0–4)", "2 (0–4)", "2 (0–4)"]]),
    "tab:baseline": dict(
        caption="Table 3. Federated leaderboard: XGBoost count-featurization baseline, "
                "headline mode-matched AUROC (deployed, stay-level) with 1,000-iteration "
                "bootstrap 95% CI. MIMIC-CLIF is the internal 75/25 hold-out; Rush is the "
                "first external site (MIMIC-trained model on a 25% hold-out of Rush data); "
                "remaining site columns read TBD. Full internal detail in Table 5.",
        header=["Task", "Mode", "MIMIC-CLIF AUROC (95% CI)", "Rush AUROC (95% CI)", "Site C", "Site D"],
        rows=[["T1, ICU daily mortality", "landmark", "0.864 (0.856–0.872)", "0.748 (0.735–0.763)", "TBD", "TBD"],
              ["T2, LTAC discharge", "landmark", "0.789 (0.775–0.803)", "0.658 (0.622–0.691)", "TBD", "TBD"],
              ["T3, Extubation failure 24 h", "episodic", "0.696 (0.665–0.719)", "0.565 (0.522–0.605)", "TBD", "TBD"],
              ["T4, Sepsis 6 h", "peak", "0.693 (0.678–0.709)", "0.407 (0.391–0.421)", "TBD", "TBD"],
              ["T5, Unplanned ICU readmission", "episodic", "0.641 (0.626–0.658)", "0.509 (0.489–0.531)", "TBD", "TBD"]]),
    "tab:comparators": dict(
        caption="Table 4. FLAIR alongside adjacent federated infrastructures.",
        header=["", "OHDSI/LEGEND", "MedPerf", "FeTS", "OpenSAFELY", "FLAIR"],
        rows=[["Data model", "OMOP", "task-specific", "imaging (brain MRI)", "TPP / EMIS", "CLIF"],
              ["Federation unit", "SQL / coefficients", "container", "training gradients", "analytic code", "container / model code"],
              ["Modern DL CDS", "no", "yes (imaging)", "yes (training only)", "limited", "yes (longitudinal EHR)"],
              ["Hosted-model portal", "no", "yes", "yes", "no", "yes"],
              ["PR-extensible tasks", "no", "partial", "no", "n/a", "yes"],
              ["Sites (current)", "many", "~10s", "71", "English NHS", "12+"]]),
    "tab:supp-baseline-full": dict(
        caption="Table 5. XGBoost count-featurization baseline: full mode-matched per-task "
                "metrics on the MIMIC-CLIF internal test split (75/25 hold-out on "
                "hospitalization_join_id). n is stays scored; prevalence is the mode-matched "
                "event rate; AUROC/AUPRC carry 1,000-iteration bootstrap 95% CIs.",
        header=["Task", "Mode", "n", "Prev.", "AUROC (95% CI)", "AUPRC", "Cal. slope", "Cal. int.", "Brier"],
        rows=[["T1, ICU daily mortality", "landmark", "18,195", "0.134", "0.864 (0.856–0.872)", "0.587", "1.19", "−1.24", "0.111"],
              ["T2, LTAC discharge", "landmark", "18,195", "0.065", "0.789 (0.775–0.803)", "0.288", "1.01", "−1.77", "0.096"],
              ["T3, Extubation failure 24 h", "episodic", "6,693", "0.054", "0.696 (0.665–0.719)", "0.134", "0.45", "−1.86", "0.070"],
              ["T4, Sepsis 6 h", "peak", "19,943", "0.063", "0.693 (0.678–0.709)", "0.134", "—", "—", "—"],
              ["T5, Unplanned ICU readmission", "episodic", "16,806", "0.093", "0.641 (0.626–0.658)", "0.248", "0.68", "−1.89", "0.151"]]),
}

FIGURES = {
    "fig:workflow": dict(
        images=["figure1_flair_workflow.png"], width=6.2,
        caption="Figure 1. The FLAIR model-to-data loop. Patient-level data never leaves any "
                "site; only a model crosses in and an aggregate report crosses out. Develop "
                "centrally on MIMIC-CLIF with flair-baseline train; ship the approved model "
                "folder to sites; each site runs flair-baseline infer locally and returns only "
                "the non-PHI, mode-keyed report bundle; the Portal pools per-site reports into "
                "the leaderboard (per-site forest plot is the headline)."),
    "fig:supp-roc-pr": dict(
        images=["roc_overlay_t1.png", "roc_overlay_t2.png", "roc_overlay_t3.png",
                "roc_overlay_t4.png", "roc_overlay_t5.png"], width=4.2,
        caption="Figure S1. Per-task ROC, all sites overlaid — MIMIC-CLIF (blue) and Rush "
                "(terracotta) on one axes per task, legend AUROC per site, dashed diagonal = "
                "chance. External curves sit below MIMIC on every task; the Rush sepsis (T4) "
                "curve crosses below chance (AUROC 0.41). Landmark tasks (T1, T2) pool all 12 "
                "lead-time risk sets into one curve (mean TPR on the shared FPR grid), so the "
                "pooled AUROC is lower than the lead-0 value in Table 3."),
    "fig:supp-calibration": dict(
        images=["landmark_calib_t1.png", "landmark_calib_t2.png", "calibration_t3.png", "calibration_t5.png"], width=4.2,
        caption="Figure S2. Reliability (calibration) curves. Landmark tasks (T1, T2) are close "
                "to calibrated in slope; episodic tasks (T3, T5) have slope <1. Peak (T4) omits "
                "calibration by design (the peak-trap)."),
    "fig:supp-dca": dict(
        images=["dca_t3.png", "dca_t5.png"], width=4.5,
        caption="Figure S3. Decision curve analysis (episodic tasks T3, T5): net benefit vs. "
                "treat-all / treat-none near the rare-event decision region."),
    "fig:supp-landmark": dict(
        images=["landmark_skill_t1.png", "landmark_skill_t2.png", "leadtime_sens_t1.png", "landmark_risk_t1.png"], width=4.5,
        caption="Figure S4. Landmark lead-time views (T1, T2): AUROC at each lead-time landmark, "
                "share of event stays alarmed by lead time, and mean predicted-risk trajectory "
                "by outcome."),
    "fig:supp-fairness": dict(
        images=["fairness_auroc_t3.png", "fairness_auroc_t4.png", "fairness_auroc_t5.png"], width=4.5,
        caption="Figure S5. Subgroup AUROC (fairness) across sex, race, ethnicity, and age bins "
                "with 95% CIs; subgroups with n<10 are suppressed."),
}

# ---------------------------------------------------------------- block parsing
BLOCK_RE = re.compile(
    r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", re.DOTALL)

def route_block(doc, block: str):
    m = re.search(r"\\label\{([^}]*)\}", block)
    label = m.group(1) if m else None
    if label in TABLES:
        td = TABLES[label]; add_table(doc, td["header"], td["rows"], td["caption"])
    elif label in FIGURES:
        fd = FIGURES[label]; add_figure(doc, fd["images"], fd["caption"], fd.get("width", 5.8))
    # unknown floats silently skipped

def emit_section(doc, path: Path):
    text = path.read_text()
    pos = 0
    for m in BLOCK_RE.finditer(text):
        emit_prose(doc, text[pos:m.start()])
        route_block(doc, m.group(0))
        pos = m.end()
    emit_prose(doc, text[pos:])

# ---------------------------------------------------------------- abstract
def emit_abstract(doc, path: Path):
    text = path.read_text()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    body = m.group(1) if m else text
    add_heading(doc, "Abstract", 1)
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        lead = re.match(r"\\noindent\\textbf\{([^}]*)\}\s*(.*)", para, re.DOTALL)
        if lead:
            add_para(doc, latex_to_text(lead.group(2)), bold_lead=latex_to_text(lead.group(1)).strip())
        else:
            t = latex_to_text(para).strip()
            if t:
                add_para(doc, t)

# ---------------------------------------------------------------- main
def main():
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    title = doc.add_heading(
        "FLAIR: A Federated Benchmark for ICU Clinical Decision Support Across "
        "Multiple Academic Medical Centers Using the Common Longitudinal ICU Format", 0)
    sub = doc.add_paragraph("CLIF Consortium · FLAIR Working Group")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    emit_abstract(doc, SEC / "00_abstract.tex")
    for fn in ["01_introduction", "02_methods", "03_datasets", "04_benchmarks",
               "05_protocols", "07_discussion"]:
        emit_section(doc, SEC / f"{fn}.tex")

    add_heading(doc, "Required statements", 1)
    add_para(doc, "Source code, task definitions, Dockerfiles, and the report-generation "
             "library are released under Apache 2.0 at github.com/clif-consortium/FLAIR. "
             "MIMIC-CLIF is distributed via PhysioNet; consortium data are accessible only by "
             "running submitted code at each site through the FLAIR Portal.",
             bold_lead="Data and Code Availability.")
    add_para(doc, "Each consortium site obtained independent institutional review board "
             "approval; central protocol coordination is provided by the University of Chicago.",
             bold_lead="IRB.")
    for label in ["Funding.", "Author Contributions.", "Conflicts of Interest.", "AI-Use Disclosure."]:
        add_para(doc, "[TODO]", bold_lead=label)
    add_para(doc, "See references.bib / main_v2.pdf for the full reference list.",
             bold_lead="References.")

    doc.add_page_break()
    emit_section(doc, SEC / "08_supplement.tex")

    out = HERE / "main_v2.docx"
    doc.save(str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
