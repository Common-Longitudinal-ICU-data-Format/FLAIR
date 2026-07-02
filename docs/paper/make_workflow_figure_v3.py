#!/usr/bin/env python3
"""Generate the v3 FLAIR workflow figure as a PNG.

The output is intentionally raster-first so the same asset can be used in
Word and LaTeX without relying on Graphviz, Inkscape, or TikZ.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "figure1_flair_workflow_v3.png"

W, H = 2200, 1300
BG = "#ffffff"
INK = "#142033"
MUTED = "#5d667a"
BLUE = "#3358d4"
BLUE_LIGHT = "#eef2ff"
TEAL = "#168a7a"
TEAL_LIGHT = "#e8f7f4"
ORANGE = "#c97612"
ORANGE_LIGHT = "#fff3e2"
VIOLET = "#7d3fc7"
VIOLET_LIGHT = "#f2ebff"
GRAY = "#e8ebf1"
GRAY_DARK = "#5a6475"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(46, True)
F_SUB = font(24)
F_BAND = font(22, True)
F_HEAD = font(25, True)
F_BODY = font(20)
F_SMALL = font(17)
F_TINY = font(15)


def text(draw: ImageDraw.ImageDraw, xy, s, fill=INK, fnt=F_BODY, anchor=None):
    draw.text(xy, s, fill=fill, font=fnt, anchor=anchor)


def wrapped(draw, box, title, body, fill, outline, width_chars=28):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    y = y1 + 22
    text(draw, (x1 + 24, y), title, fill=outline, fnt=F_HEAD)
    y += 40
    for line in body.split("\n"):
        for part in wrap(line, width=width_chars) or [""]:
            text(draw, (x1 + 24, y), part, fill=INK if part else MUTED, fnt=F_BODY)
            y += 28
        y += 4


def arrow(draw, start, end, color, width=8):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - direction * 24, y2 - 15), (x2 - direction * 24, y2 + 15)]
    else:
        direction = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - 15, y2 - direction * 24), (x2 + 15, y2 - direction * 24)]
    draw.polygon(pts, fill=color)


def badge(draw, center, label, color):
    x, y = center
    draw.ellipse((x - 23, y - 23, x + 23, y + 23), fill=color)
    text(draw, (x, y + 1), label, fill="white", fnt=font(20, True), anchor="mm")


def site_card(draw, box, label):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=16, fill=ORANGE_LIGHT, outline=ORANGE, width=3)
    draw.rectangle((x1, y1, x2, y1 + 44), fill=ORANGE)
    text(draw, ((x1 + x2) / 2, y1 + 23), label, fill="white", fnt=font(19, True), anchor="mm")
    cx = (x1 + x2) / 2
    draw.ellipse((cx - 38, y1 + 68, cx + 38, y1 + 94), fill="#d99030", outline=None)
    draw.rectangle((cx - 38, y1 + 81, cx + 38, y1 + 118), fill="#d99030")
    draw.ellipse((cx - 38, y1 + 103, cx + 38, y1 + 131), fill="#c17621", outline=None)
    text(draw, (cx, y1 + 153), "local CLIF tables", fill=MUTED, fnt=F_SMALL, anchor="mm")
    draw.ellipse((cx - 15, y1 + 178, cx + 15, y1 + 208), outline="#8a4b04", width=4)
    draw.arc((cx - 38, y1 + 203, cx + 38, y1 + 268), 200, 340, fill="#8a4b04", width=4)
    draw.rounded_rectangle((cx - 54, y2 - 48, cx + 54, y2 - 18), radius=15, fill="white", outline=ORANGE, width=2)
    text(draw, (cx, y2 - 33), "HIPAA-safe", fill=ORANGE, fnt=F_TINY, anchor="mm")


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    text(draw, (W / 2, 48), "The FLAIR Model-to-Data Workflow", fill=INK, fnt=F_TITLE, anchor="mm")
    text(
        draw,
        (W / 2, 92),
        "Code and frozen models travel to CLIF sites; only aggregate, cell-suppressed reports return.",
        fill=MUTED,
        fnt=F_SUB,
        anchor="mm",
    )

    draw.rounded_rectangle((45, 122, 910, 172), radius=24, fill=BLUE)
    text(draw, (72, 147), "PUBLIC SIDE: Model Owner + FLAIR Portal + Leaderboard", fill="white", fnt=F_BAND)
    draw.rounded_rectangle((980, 122, 1120, 172), radius=18, fill=GRAY_DARK)
    text(draw, (1050, 147), "BOUNDARY", fill="white", fnt=F_BAND, anchor="mm")
    draw.rounded_rectangle((1188, 122, 2155, 172), radius=24, fill=ORANGE)
    text(draw, (1215, 147), "SITE SIDE: CLIF data stay local; no patient-level rows leave", fill="white", fnt=F_BAND)

    wrapped(
        draw,
        (55, 205, 520, 430),
        "1. Develop on MIMIC-CLIF",
        "Model Owner builds against public CLIF data.\nflair load-task -> cohort\nfeature extraction -> predictions\nflair build-report -> reports",
        BLUE_LIGHT,
        BLUE,
        33,
    )
    wrapped(
        draw,
        (560, 205, 910, 430),
        "2. Review and approve",
        "Benchmark Committee checks reproducibility, clinical face validity, task fit, and security before dispatch.",
        BLUE_LIGHT,
        BLUE,
        25,
    )
    wrapped(
        draw,
        (55, 485, 520, 675),
        "FLAIR Portal",
        "Registers tasks, models, sites, versions.\nDistributes approved artifacts.\nVerifies hashes.\nCollects aggregate reports.",
        TEAL_LIGHT,
        TEAL,
        35,
    )
    wrapped(
        draw,
        (55, 715, 520, 935),
        "Public leaderboard",
        "Per-site forest plots are the headline signal.\nPooled metrics are shown but de-emphasized.\nCross-site variation is visible.",
        TEAL_LIGHT,
        TEAL,
        36,
    )

    # Barrier
    draw.rounded_rectangle((980, 205, 1120, 935), radius=12, fill="#f8fafc", outline=GRAY_DARK, width=3)
    for i in range(-150, 800, 36):
        draw.line((970, 205 + i, 1130, 365 + i), fill="#b3bac7", width=3)
    text(draw, (1050, 365), "CODE /", fill=BLUE, fnt=font(22, True), anchor="mm")
    text(draw, (1050, 395), "MODEL IN", fill=BLUE, fnt=font(22, True), anchor="mm")
    text(draw, (1050, 600), "PRIVACY", fill=GRAY_DARK, fnt=font(30, True), anchor="mm")
    text(draw, (1050, 635), "BOUNDARY", fill=GRAY_DARK, fnt=font(30, True), anchor="mm")
    text(draw, (1050, 815), "AGGREGATE", fill=ORANGE, fnt=font(20, True), anchor="mm")
    text(draw, (1050, 842), "JSON OUT", fill=ORANGE, fnt=font(20, True), anchor="mm")

    # Site silo background
    draw.rounded_rectangle((1175, 205, 2160, 935), radius=24, fill="#fff9f0", outline=ORANGE, width=4)
    text(draw, (1205, 232), "Institutional CLIF silos: Site Leads/Data Stewards run CLIF-WorkBench locally", fill=ORANGE, fnt=F_BAND)

    x0, y0 = 1215, 270
    card_w, card_h = 205, 270
    gap_x, gap_y = 38, 48
    n = 1
    for row in range(2):
        for col in range(4):
            site_card(draw, (x0 + col * (card_w + gap_x), y0 + row * (card_h + gap_y), x0 + col * (card_w + gap_x) + card_w, y0 + row * (card_h + gap_y) + card_h), f"Site {n}")
            n += 1
    draw.rounded_rectangle((x0, 880, x0 + 710, 930), radius=22, fill="white", outline=ORANGE, width=3)
    text(draw, (x0 + 355, 905), "Sites 9-N run the same approved artifact locally", fill=ORANGE, fnt=font(20, True), anchor="mm")

    # Arrows and labels
    arrow(draw, (520, 318), (560, 318), BLUE)
    arrow(draw, (910, 318), (980, 318), BLUE)
    arrow(draw, (1120, 318), (1175, 430), BLUE)

    arrow(draw, (1175, 760), (1120, 820), ORANGE)
    arrow(draw, (980, 820), (520, 575), ORANGE)
    text(draw, (548, 596), "cell-suppressed aggregate JSONs only", fill=ORANGE, fnt=F_SMALL)
    arrow(draw, (290, 675), (290, 715), TEAL)

    # Report bundle
    draw.rounded_rectangle((565, 720, 910, 935), radius=18, fill=VIOLET_LIGHT, outline=VIOLET, width=3)
    text(draw, (590, 754), "Report bundle", fill=VIOLET, fnt=F_HEAD)
    bundle = [
        "table1 + code registry",
        "discrimination + CIs",
        "calibration + DCA",
        "fairness/subgroups",
        "lead time for landmark tasks",
        "audit hash + package version",
    ]
    y = 792
    for item in bundle:
        text(draw, (596, y), f"- {item}", fill=INK, fnt=F_BODY)
        y += 29

    # Number badges
    badge(draw, (72, 205), "1", BLUE)
    badge(draw, (577, 205), "2", BLUE)
    badge(draw, (72, 485), "3", TEAL)
    badge(draw, (1175, 205), "4", ORANGE)
    badge(draw, (72, 715), "5", TEAL)

    # Task catalog
    yb = 1010
    text(draw, (60, yb), "Task catalog grows by community PR after MIMIC-CLIF reproducibility, clinical review, and site-governance review.", fill=MUTED, fnt=F_SMALL)
    labels = [
        "T1 ICU mortality",
        "T2 LTACH discharge",
        "T3 extubation failure",
        "T4 ASE sepsis 6h",
        "T5 ICU readmission",
        "+ community task PR",
    ]
    x = 55
    widths = [250, 290, 320, 270, 300, 300]
    for i, label in enumerate(labels):
        outline = BLUE if i < 5 else ORANGE
        draw.rounded_rectangle((x, 1060, x + widths[i], 1116), radius=24, fill="white", outline=outline, width=3)
        text(draw, (x + widths[i] / 2, 1088), label, fill=outline, fnt=font(20, True), anchor="mm")
        x += widths[i] + 28

    # Legend
    draw.rounded_rectangle((55, 1180, 430, 1238), radius=24, fill="white", outline=GRAY_DARK, width=2)
    text(draw, (80, 1210), "PHI artifacts remain local", fill=GRAY_DARK, fnt=F_SMALL)
    draw.rounded_rectangle((470, 1180, 845, 1238), radius=24, fill="white", outline=ORANGE, width=2)
    text(draw, (495, 1210), "Suppression at n < 10", fill=ORANGE, fnt=F_SMALL)
    draw.rounded_rectangle((885, 1180, 1260, 1238), radius=24, fill="white", outline=TEAL, width=2)
    text(draw, (910, 1210), "Leaderboard shows sites", fill=TEAL, fnt=F_SMALL)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
