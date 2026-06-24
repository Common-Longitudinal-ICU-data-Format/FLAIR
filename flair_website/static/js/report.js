// Full per-(model, site, task) report, re-rendered from the exported pillar JSON
// (scripts/build_data.py). Pure inline-SVG charts — no external libs, no images.
// Mode-aware: episodic/peak tasks are flat; landmark tasks carry a per-lead-time
// `landmarks[]` array on each pillar + a leadtime pillar, surfaced via a selector.
(function () {
  // Default render target is #report (task-detail page). The merged Task
  // Leaderboard has no #report and instead calls window.FlairReport.render(...)
  // with its own per-row host element.
  let panel = document.getElementById("report");

  // ---- tiny SVG chart kit -------------------------------------------------
  const COL = { blue: "#5b86a6", purple: "#8a72ad", sage: "#5e9b78", terra: "#b9744e", grey: "#9a9a96", ink: "#1a1a1a" };

  function ticks(lo, hi, n = 5) {
    const out = [];
    for (let i = 0; i <= n; i++) out.push(lo + ((hi - lo) * i) / n);
    return out;
  }
  const fmt = (v) => (v == null || !isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));

  let _clipSeq = 0;
  function chart(cfg) {
    const W = 380, H = 300, P = { l: 52, r: 16, t: 28, b: 44 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const [x0, x1] = cfg.xDomain, [y0, y1] = cfg.yDomain;
    const sx = (x) => P.l + ((x - x0) / (x1 - x0 || 1)) * iw;
    const sy = (y) => P.t + ih - ((y - y0) / (y1 - y0 || 1)) * ih;
    const clipId = `clip${_clipSeq++}`;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img">`;
    s += `<defs><clipPath id="${clipId}"><rect x="${P.l}" y="${P.t}" width="${iw}" height="${ih}"/></clipPath></defs>`;
    s += `<text x="${W / 2}" y="16" text-anchor="middle" class="c-title">${cfg.title}</text>`;
    ticks(y0, y1).forEach((t) => {
      const y = sy(t);
      s += `<line x1="${P.l}" y1="${y}" x2="${W - P.r}" y2="${y}" class="c-grid"/>`;
      s += `<text x="${P.l - 6}" y="${y + 3}" text-anchor="end" class="c-tick">${fmt(t)}</text>`;
    });
    const xt = cfg.xTicks || ticks(x0, x1);
    const xtf = cfg.xTickFmt || fmt;
    xt.forEach((t) => {
      const x = sx(t);
      s += `<text x="${x}" y="${H - P.b + 16}" text-anchor="middle" class="c-tick">${xtf(t)}</text>`;
    });
    s += `<text x="${P.l + iw / 2}" y="${H - 6}" text-anchor="middle" class="c-axis">${cfg.xLabel || ""}</text>`;
    s += `<text transform="translate(13 ${P.t + ih / 2}) rotate(-90)" text-anchor="middle" class="c-axis">${cfg.yLabel || ""}</text>`;
    // Everything data-mapped is clipped to the inner plot rect so nothing spills out.
    let g = `<g clip-path="url(#${clipId})">`;
    (cfg.bands || []).forEach((b) => {
      const bx0 = sx(b.x0), bx1 = sx(b.x1);
      g += `<rect x="${Math.min(bx0, bx1).toFixed(1)}" y="${P.t}" width="${Math.abs(bx1 - bx0).toFixed(1)}" height="${ih}" fill="${b.color}" opacity="0.16"/>`;
    });
    if (cfg.diagonal) g += `<line x1="${sx(x0)}" y1="${sy(y0)}" x2="${sx(x1)}" y2="${sy(y1)}" class="c-diag"/>`;
    (cfg.hlines || []).forEach((h) => {
      g += `<line x1="${P.l}" y1="${sy(h.y)}" x2="${W - P.r}" y2="${sy(h.y)}" stroke="${h.color}" stroke-dasharray="4 3" stroke-width="1"/>`;
    });
    (cfg.vlines || []).forEach((v) => {
      g += `<line x1="${sx(v.x)}" y1="${P.t}" x2="${sx(v.x)}" y2="${P.t + ih}" stroke="${v.color}" stroke-dasharray="3 3" stroke-width="1"/>`;
    });
    const finite = (p) => p && isFinite(p[0]) && isFinite(p[1]);
    (cfg.series || []).forEach((ser) => {
      const pts = (ser.pts || []).filter(finite);
      if (pts.length > 1) {
        const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
        g += `<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="2"${ser.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
      }
      if (ser.markers) pts.forEach((p) => {
        g += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${ser.r || 3}" fill="${ser.color}"/>`;
      });
    });
    s += g + `</g>`;
    s += `<rect x="${P.l}" y="${P.t}" width="${iw}" height="${ih}" class="c-frame"/></svg>`;
    if (cfg.legend) {
      s += `<div class="c-legend">` + cfg.legend.map((l) =>
        `<span><i style="background:${l.color}"></i>${l.label}</span>`).join("") + `</div>`;
    }
    return `<div class="chart-box">${s}</div>`;
  }

  function barChart(cfg) {
    const W = 380, H = 300, P = { l: 52, r: 16, t: 28, b: 64 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const max = cfg.max || Math.max(0.0001, ...cfg.values.filter((v) => isFinite(v)));
    const bw = iw / cfg.values.length;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart"><text x="${W / 2}" y="16" text-anchor="middle" class="c-title">${cfg.title}</text>`;
    cfg.values.forEach((v, i) => {
      const fv = isFinite(v) ? v : 0;
      const h = (fv / max) * ih, x = P.l + i * bw + 4, y = P.t + ih - h;
      s += `<rect x="${x}" y="${y}" width="${bw - 8}" height="${h}" fill="${cfg.color || COL.blue}"/>`;
      s += `<text x="${x + (bw - 8) / 2}" y="${y - 4}" text-anchor="middle" class="c-tick">${isFinite(v) ? fmt(v) : "—"}</text>`;
      s += `<text x="${x + (bw - 8) / 2}" y="${P.t + ih + 12}" text-anchor="end" class="c-tick" transform="rotate(-28 ${x + (bw - 8) / 2} ${P.t + ih + 12})">${esc(cfg.labels[i])}</text>`;
    });
    s += `<rect x="${P.l}" y="${P.t}" width="${iw}" height="${ih}" class="c-frame"/></svg>`;
    return `<div class="chart-box">${s}</div>`;
  }

  const zip = (a, b) => (a || []).map((v, i) => [v, (b || [])[i]]);
  // Fit a y-domain to the data so nothing clips out of view. Pass the same series
  // arrays handed to chart(); lo/hi cap the axis (e.g. AUROC ∈ [0,1]).
  function fitY(seriesPts, { lo = -Infinity, hi = Infinity, pad = 0.04, fallback = [0, 1] } = {}) {
    const ys = seriesPts.flat().map((p) => p && p[1]).filter((v) => v != null && isFinite(v));
    if (!ys.length) return fallback;
    return [Math.max(lo, Math.min(...ys) - pad), Math.min(hi, Math.max(...ys) + pad)];
  }
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const none = (msg) => `<p class="muted">${msg}</p>`;
  const ci = (a) => (a && a.length === 2 ? ` <span class="ci">(${fmt(a[0])}–${fmt(a[1])})</span>` : "");
  const row = (k, v) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`;

  // ---- lead-time geometry -------------------------------------------------
  // Display day is 1-indexed (no "day 0"): nearest landmark = 1 day before outcome.
  // Charts read left→right as a countdown TOWARD the prediction window: 12 days on
  // the left, 1 day (the prediction window) on the right.
  const dayOf = (lm) => ((lm.windows_before_end ?? lm.landmark_index) || 0) + 1;
  const LEAD_X = { xDomain: [12, 1], xLabel: "days before outcome",
                   xTicks: [12, 10, 8, 6, 4, 2, 1], xTickFmt: (t) => `${Math.round(t)}` };
  // Gradient by lead-time: nearest (1 day) dark, earliest (12 days) light.
  const lmColor = (li, n) => `hsl(208, 48%, ${28 + (li / Math.max(1, n - 1)) * 50}%)`;
  const leadLegend = () =>
    `<div class="lm-legend"><span>1 day</span><i></i><span>12 days before outcome</span></div>`;

  // ---- landmark state + pillar accessor -----------------------------------
  const state = { li: 0, all: false };
  // Return the active node of a pillar: the selected landmark for landmark tasks,
  // else the flat pillar itself.
  function node(pillar) {
    if (!pillar) return null;
    if (Array.isArray(pillar.landmarks)) {
      return pillar.landmarks.find((l) => l.landmark_index === state.li) || pillar.landmarks[0] || null;
    }
    return pillar;
  }

  // ---- per-tab renderers --------------------------------------------------
  function discrimination(r) {
    if (r.landmark && state.all) {
      const lms = r.discrimination.landmarks || [];
      const n = lms.length;
      const roc = chart({
        title: "ROC — all lead-times", xLabel: "false positive rate", yLabel: "true positive rate",
        xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
        series: lms.map((l, i) => ({ pts: zip((l.curves || {}).roc?.fpr, (l.curves || {}).roc?.tpr), color: lmColor(l.landmark_index, n) })),
      });
      const pr = chart({
        title: "Precision–Recall — all lead-times", xLabel: "recall", yLabel: "precision",
        xDomain: [0, 1], yDomain: [0, 1],
        series: lms.map((l) => ({ pts: zip((l.curves || {}).pr?.recall, (l.curves || {}).pr?.precision), color: lmColor(l.landmark_index, n) })),
      });
      const aucPts = [{ pts: lms.map((l) => [dayOf(l), (l.metrics || {}).auroc]), color: COL.blue, markers: true }];
      const auc = chart({
        ...LEAD_X, title: "AUROC by lead-time", yLabel: "AUROC",
        yDomain: fitY(aucPts.map((s) => s.pts), { lo: 0, hi: 1, fallback: [0.5, 1] }),
        series: aucPts,
      });
      return `${leadLegend()}<div class="chart-grid">${roc}${pr}${auc}</div>`;
    }
    const d = node(r.discrimination);
    if (!d || !d.curves) return none("No discrimination data.");
    const m = d.metrics || {}, c = d.curves;
    const roc = chart({
      title: "ROC", xLabel: "false positive rate", yLabel: "true positive rate",
      xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
      series: [{ pts: zip(c.roc.fpr, c.roc.tpr), color: COL.blue }],
    });
    const prev = r.discrimination.prevalence ?? d.prevalence;
    const pr = chart({
      title: "Precision–Recall", xLabel: "recall", yLabel: "precision",
      xDomain: [0, 1], yDomain: [0, 1],
      hlines: prev != null ? [{ y: prev, color: COL.grey }] : [],
      series: [{ pts: zip(c.pr.recall, c.pr.precision), color: COL.sage }],
    });
    const tbl = `<table class="rep-tbl"><tbody>
      ${row("AUROC", fmt(m.auroc) + ci(m.auroc_ci))}
      ${row("AUPRC", fmt(m.auprc) + ci(m.auprc_ci))}
      ${row("AUPRC lift over prevalence", "×" + fmt(m.auprc_lift))}
      ${row("Prevalence", fmt(prev))}</tbody></table>`;
    return `<div class="chart-grid">${roc}${pr}</div>${tbl}`;
  }

  function calibration(r) {
    if (!r.calibration) return none("Calibration omitted for peak-mode screening (the peak-calibration trap).");
    if (r.landmark && state.all) {
      const lms = r.calibration.landmarks || [];
      const n = lms.length;
      const rel = chart({
        title: "Reliability (smoothed) — all lead-times", xLabel: "predicted probability", yLabel: "observed frequency",
        xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
        series: lms.map((l) => { const sm = (l.curves || {}).smoothed || {}; return { pts: zip(sm.pred_grid, sm.obs_smoothed), color: lmColor(l.landmark_index, n) }; }),
      });
      const slopePts = [{ pts: lms.map((l) => [dayOf(l), (l.metrics || {}).slope]), color: COL.purple, markers: true }];
      const slope = chart({
        ...LEAD_X, title: "Calibration slope by lead-time", yLabel: "slope",
        yDomain: fitY([...slopePts.map((s) => s.pts), [[0, 1]]], { lo: 0, fallback: [0, 2] }),
        hlines: [{ y: 1, color: COL.grey }],
        series: slopePts,
      });
      return `${leadLegend()}<div class="chart-grid">${rel}${slope}</div>`;
    }
    const cal = node(r.calibration);
    if (!cal) return none("No calibration data.");
    const m = cal.metrics || {}, rc = (cal.curves || {}).reliability || {}, sm = (cal.curves || {}).smoothed || {};
    const rel = chart({
      title: "Reliability", xLabel: "predicted probability", yLabel: "observed frequency",
      xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
      series: [
        { pts: zip(sm.pred_grid, sm.obs_smoothed), color: COL.purple },
        { pts: zip(rc.bin_pred_prob, rc.bin_true_prob), color: COL.blue, markers: true },
      ],
      legend: [{ color: COL.blue, label: "binned" }, { color: COL.purple, label: "smoothed" }],
    });
    const tbl = `<table class="rep-tbl"><tbody>
      ${row("Calibration slope", fmt(m.slope) + ci(m.slope_ci))}
      ${row("Calibration intercept", fmt(m.intercept) + ci(m.intercept_ci))}
      ${row("ECE", fmt(m.ece) + ci(m.ece_ci))}
      ${row("Brier", fmt(m.brier) + ci(m.brier_ci))}
      ${row("Observed / expected", fmt(m.o_e_ratio))}</tbody></table>`;
    return `<div class="chart-grid">${rel}</div>${tbl}`;
  }

  function decision(r) {
    if (!r.dca) return none("Decision curve omitted for peak-mode screening.");
    if (r.landmark && state.all) {
      const lms = r.dca.landmarks || [];
      const n = lms.length;
      const all = lms.flatMap((l) => (l.curves || {}).net_benefit_model || []).filter(isFinite);
      const yHi = Math.min(1, Math.max(0.0001, ...all) * 1.05);
      const yLo = Math.max(-1, Math.min(all.length ? Math.min(...all) - 0.02 : 0, -0.5 * yHi));
      const ch = chart({
        title: "Decision curve (model net benefit) — all lead-times", xLabel: "threshold probability", yLabel: "net benefit",
        xDomain: (lms[0] && (lms[0].curves || {}).focus_range) || [0, 1], yDomain: [yLo, yHi],
        hlines: [{ y: 0, color: COL.grey }],
        series: lms.map((l) => ({ pts: zip((l.curves || {}).thresholds, (l.curves || {}).net_benefit_model), color: lmColor(l.landmark_index, n) })),
      });
      return `${leadLegend()}<div class="chart-grid">${ch}</div>`;
    }
    const d = node(r.dca);
    if (!d) return none("No decision-curve data.");
    const c = d.curves || {}, m = d.metrics || {};
    const th = c.thresholds || [];
    const xr = c.focus_range || [Math.min(...th), Math.max(...th)];
    const model = (c.net_benefit_model || []).filter(isFinite);
    const allY = [...model, ...(c.net_benefit_all || []).filter(isFinite)];
    // Scale to the readable region: treat-all dives steeply negative as threshold→1,
    // so anchor the floor to the model curve and let treat-all clip at the bottom.
    const yHi = Math.min(1, Math.max(0.0001, ...allY) * 1.05);
    const yLo = Math.max(-1, Math.min(model.length ? Math.min(...model) - 0.02 : 0, -0.5 * yHi));
    const nbAll = Array.isArray(c.net_benefit_all) ? zip(th, c.net_benefit_all) : [];
    const ct = m.clinical_threshold;
    const nb = m.net_benefit_at_clinical_threshold || {};
    const ur = m.useful_range || {};
    const intervals = Array.isArray(m.all_useful_intervals) && m.all_useful_intervals.length
      ? m.all_useful_intervals
      : (ur.low != null && ur.high != null ? [[ur.low, ur.high]] : []);
    const bands = intervals.filter((iv) => iv && iv.length === 2).map((iv) => ({ x0: iv[0], x1: iv[1], color: COL.sage }));
    const opPt = ct != null && nb.model != null ? [{ pts: [[ct, nb.model]], color: COL.terra, markers: true, r: 4 }] : [];
    const ch = chart({
      title: "Decision curve (net benefit)", xLabel: "threshold probability", yLabel: "net benefit",
      xDomain: xr, yDomain: [yLo, yHi],
      bands,
      hlines: [{ y: 0, color: COL.grey }],
      vlines: ct != null ? [{ x: ct, color: COL.terra }] : [],
      series: [
        { pts: zip(th, c.net_benefit_model), color: COL.blue },
        { pts: nbAll, color: COL.grey, dash: true },
        ...opPt,
      ],
      legend: [
        { color: COL.blue, label: "model" }, { color: COL.grey, label: "treat all" },
        { color: COL.grey, label: "treat none" }, { color: COL.sage, label: "useful range" },
        ...(ct != null ? [{ color: COL.terra, label: `clinical threshold (${fmt(ct)})` }] : []),
      ],
    });
    const tbl = `<table class="rep-tbl"><tbody>
      ${row("Net benefit @ clinical threshold", fmt(nb.model) + ci(nb.model_ci))}
      ${row("Clinically useful", m.clinically_useful ? "yes" : "no")}
      ${row("Useful threshold range", ur.low != null ? `${fmt(ur.low)} – ${fmt(ur.high)}` : "—")}</tbody></table>`;
    return `<div class="chart-grid">${ch}</div>${tbl}`;
  }

  function operating(r) {
    if (r.landmark && state.all) {
      const lms = r.discrimination.landmarks || [];
      const op = (l) => (l.metrics || {}).operating_point || {};
      const ch = chart({
        ...LEAD_X, title: "Operating metrics by lead-time", yLabel: "rate", yDomain: [0, 1],
        series: [
          { pts: lms.map((l) => [dayOf(l), op(l).sensitivity]), color: COL.blue, markers: true },
          { pts: lms.map((l) => [dayOf(l), op(l).specificity]), color: COL.sage, markers: true },
          { pts: lms.map((l) => [dayOf(l), op(l).ppv]), color: COL.terra, markers: true },
        ],
        legend: [{ color: COL.blue, label: "sensitivity" }, { color: COL.sage, label: "specificity" }, { color: COL.terra, label: "PPV" }],
      });
      return `<p class="count">Operating metrics at the clinical threshold, across lead-times (right = nearest the prediction window).</p><div class="chart-grid">${ch}</div>`;
    }
    const d = node(r.discrimination);
    const op = (d && d.metrics && d.metrics.operating_point) || null;
    if (!op) return none("No operating point.");
    const tbl = `<table class="rep-tbl"><tbody>
      ${row("Threshold", `${fmt(op.threshold)} (${esc(op.direction || "")})`)}
      ${row("Sensitivity", fmt(op.sensitivity) + ci(op.sensitivity_ci))}
      ${row("Specificity", fmt(op.specificity) + ci(op.specificity_ci))}
      ${row("PPV", fmt(op.ppv) + ci(op.ppv_ci))}
      ${row("NPV", fmt(op.npv) + ci(op.npv_ci))}
      ${row("F1", fmt(op.f1) + ci(op.f1_ci))}
      ${row("Flag rate", fmt(op.flag_rate))}
      ${row("TP / FP / FN / TN", `${op.tp} / ${op.fp} / ${op.fn} / ${op.tn}`)}</tbody></table>`;
    return `<p class="count">Confusion + operating metrics at the task's fixed clinical threshold.</p>${tbl}`;
  }

  function fairness(r) {
    if (r.landmark && state.all) {
      const lms = r.fairness.landmarks || [];
      const dims = Object.keys((lms[0] || {}).subgroups || {});
      let out = `<p class="count">Per-subgroup AUROC across lead-times (right = nearest the prediction window).</p><div class="chart-grid">`;
      const palette = [COL.blue, COL.sage, COL.terra, COL.purple, COL.grey, COL.ink, "#c0496b"];
      for (const dim of dims) {
        const levels = Object.keys((lms[0].subgroups[dim]) || {});
        const fSeries = levels.map((lv, k) => ({
          pts: lms.map((l) => [dayOf(l), (((l.subgroups || {})[dim] || {})[lv] || {}).auroc]),
          color: palette[k % palette.length], markers: true,
        }));
        out += chart({
          ...LEAD_X, title: dim.replace("_category", "").replace(/_/g, " ") + " — AUROC", yLabel: "AUROC",
          yDomain: fitY(fSeries.map((s) => s.pts), { lo: 0, hi: 1, fallback: [0.5, 1] }),
          series: fSeries,
          legend: levels.map((lv, k) => ({ color: palette[k % palette.length], label: lv })),
        });
      }
      return out + "</div>";
    }
    const f = node(r.fairness);
    if (!f || !f.subgroups) return none("No fairness data.");
    let out = "<p class=\"count\">Per-subgroup AUROC, with equity gaps vs the reference group at the clinical threshold.</p><div class=\"chart-grid\">";
    const parityRows = [];
    for (const [dim, levels] of Object.entries(f.subgroups)) {
      const names = Object.keys(levels), vals = names.map((n) => levels[n].auroc);
      out += barChart({
        title: dim.replace("_category", "").replace(/_/g, " ") + " — AUROC",
        labels: names, values: vals, color: COL.blue, max: 1,
      });
      const p = (f.parity || {})[dim];
      if (p) parityRows.push(`<tr><td>${esc(dim.replace("_category", "").replace(/_/g, " "))}</td>
        <td class="num">${fmt(p.demographic_parity_gap)}</td>
        <td class="num">${fmt(p.equalized_odds_tpr_gap)}</td>
        <td class="num">${fmt(p.equalized_odds_fpr_gap)}</td>
        <td class="muted">${esc(p.reference_group || "")}</td></tr>`);
    }
    out += "</div>";
    if (parityRows.length) out += `<table class="rep-tbl"><thead><tr><th>Subgroup</th>
      <th class="num">Demographic parity gap</th><th class="num">Eq. odds TPR gap</th>
      <th class="num">Eq. odds FPR gap</th><th>Reference</th></tr></thead><tbody>${parityRows.join("")}</tbody></table>`;
    return out;
  }

  function leadtime(r) {
    if (!r.leadtime) return none("Lead-time analysis applies to landmark (continuous) tasks only.");
    const lms = (r.discrimination && r.discrimination.landmarks) || [];
    const skillPts = [{ pts: lms.map((l) => [dayOf(l), (l.metrics || {}).auroc]), color: COL.blue, markers: true }];
    const skill = chart({
      ...LEAD_X, title: "Skill over lead-time", yLabel: "AUROC",
      yDomain: fitY(skillPts.map((s) => s.pts), { lo: 0, hi: 1, fallback: [0.5, 1] }),
      series: skillPts,
    });
    const cv = r.leadtime.curve || [];
    const sens = chart({
      ...LEAD_X, title: "Detection sensitivity by lead-time", yLabel: "sensitivity", yDomain: [0, 1],
      series: [{ pts: cv.map((c) => [dayOf(c), c.sensitivity]), color: COL.sage, markers: true }],
    });
    const tr = r.leadtime.trajectory || [];
    const traj = chart({
      ...LEAD_X, title: "Mean risk trajectory by outcome", yLabel: "mean predicted risk", yDomain: [0, 1],
      series: [
        { pts: tr.map((t) => [dayOf(t), t.mean_prob_positive]), color: COL.terra },
        { pts: tr.map((t) => [dayOf(t), t.mean_prob_negative]), color: COL.blue },
      ],
      legend: [{ color: COL.terra, label: "eventual positive" }, { color: COL.blue, label: "eventual negative" }],
    });
    const unit = (r.meta && r.meta.window_unit) || "day";
    const mt = r.leadtime.metrics || {};
    const tbl = `<table class="rep-tbl"><tbody>
      ${row("Positive stays caught", `${mt.n_caught} / ${mt.n_positive_stays} (${fmt(mt.pct_caught)})`)}
      ${row(`Median lead (${unit}s)`, fmt(mt.median_lead_windows) + ci(mt.median_lead_ci))}</tbody></table>`;
    return `<p class="count">Charts read left→right toward the prediction window (12 days before on the left, 1 day on the right).</p><div class="chart-grid">${skill}${sens}${traj}</div>${tbl}`;
  }

  function cohort(r) {
    const t = r.table1;
    if (!t) return none("No cohort table.");
    let out = `<p class="count">n = ${(t.n_total || 0).toLocaleString()} hospitalizations (cell suppression at n&lt;${(t.metadata || {}).min_cell_count || 10}).</p>`;
    const cont = Object.entries(t.continuous || {});
    if (cont.length) {
      out += `<table class="rep-tbl"><thead><tr><th>Characteristic</th>
        <th class="num">Median (IQR)</th><th class="num">Mean ± SD</th></tr></thead><tbody>`;
      for (const [k, v] of cont) {
        out += `<tr><td>${esc(k.replace(/_/g, " "))}</td>
          <td class="num">${fmt(v.median)} (${fmt(v.q25)}–${fmt(v.q75)})</td>
          <td class="num">${fmt(v.mean)} ± ${fmt(v.std)}</td></tr>`;
      }
      out += `</tbody></table>`;
    }
    for (const [dim, groups] of Object.entries(t.categorical || {})) {
      out += `<h4>${esc(dim.replace("_category", "").replace(/_/g, " "))}</h4><table class="rep-tbl"><tbody>`;
      for (const [g, n] of Object.entries(groups)) out += `<tr><td>${esc(g)}</td><td class="num">${typeof n === "number" ? n.toLocaleString() : esc(n)}</td></tr>`;
      out += `</tbody></table>`;
    }
    return out;
  }

  function buildTabs(r) {
    const t = [["Discrimination", discrimination]];
    if (r.calibration) t.push(["Calibration", calibration]);
    if (r.dca) t.push(["Decision curve", decision]);
    t.push(["Operating point", operating], ["Fairness", fairness]);
    if (r.leadtime) t.push(["Lead-time", leadtime]);
    t.push(["Cohort", cohort]);
    return t;
  }

  function render(report, label, targetEl, opts) {
    opts = opts || {};
    const host = targetEl || panel;
    if (!host) return;
    state.li = 0;
    state.all = false;
    const TABS = buildTabs(report);
    const meta = report.meta || {};
    let prov = `<div class="count rep-prov">mode: <strong>${esc(report.mode || "—")}</strong>`;
    if (meta.split) prov += ` · split: ${esc(meta.split)}`;
    if (meta.clinical_threshold != null) prov += ` · clinical threshold: ${fmt(meta.clinical_threshold)}`;
    prov += `</div>`;

    let sel = "";
    if (report.landmark) {
      const n = ((report.discrimination && report.discrimination.landmarks) || []).length || 12;
      // Slider raw 1..n maps left→right toward the prediction window:
      // left = n days before (earliest), right = 1 day before (the prediction window).
      // landmark_index li = n - raw  (raw=n → li=0 = 1 day; raw=1 → li=n-1 = n days).
      sel = `<div class="rep-landmark">
        <label class="lm-all"><input type="checkbox" class="lm-all-cb"> Overlay all ${n} lead-times on one plot</label>
        <div class="lm-scrub">
          <span class="lm-cap">${n} d</span>
          <input type="range" class="lm-slider" min="1" max="${n}" value="${n}" step="1"
                 aria-label="lead-time in days before outcome">
          <span class="lm-cap">1 d · prediction window</span>
        </div>
        <span class="lm-now muted count">showing: 1 day before outcome</span>
      </div>`;
    }

    host.innerHTML =
      `<div class="report-head"><h3>${esc(label)}</h3><button class="rep-close" aria-label="close">✕</button></div>` +
      prov + sel +
      `<div class="rep-tabs">` +
      TABS.map((t, i) => `<button class="rep-tab${i ? "" : " active"}" data-i="${i}">${t[0]}</button>`).join("") +
      `</div><div class="rep-body"></div>`;
    host.hidden = false;
    const body = host.querySelector(".rep-body");
    let active = 0;
    const show = (i) => { active = i; body.innerHTML = TABS[i][1](report); };
    show(0);
    host.querySelectorAll(".rep-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        host.querySelectorAll(".rep-tab").forEach((x) => x.classList.remove("active"));
        tab.classList.add("active");
        show(Number(tab.dataset.i));
      });
    });
    const lmAll = host.querySelector(".lm-all-cb");
    const lmSlider = host.querySelector(".lm-slider");
    const lmNow = host.querySelector(".lm-now");
    const nLm = lmSlider ? Number(lmSlider.max) : 12;
    const updateNow = () => {
      if (!lmNow) return;
      lmNow.textContent = state.all ? "showing: all lead-times overlaid"
        : `showing: ${state.li + 1} day${state.li === 0 ? "" : "s"} before outcome`;
    };
    if (lmAll) lmAll.addEventListener("change", (e) => {
      state.all = e.target.checked;
      if (lmSlider) lmSlider.disabled = state.all;
      updateNow(); show(active);
    });
    if (lmSlider) lmSlider.addEventListener("input", (e) => {
      state.li = nLm - Number(e.target.value);
      updateNow(); show(active);
    });
    host.querySelector(".rep-close").addEventListener("click", () => {
      host.hidden = true;
      if (typeof opts.onClose === "function") opts.onClose();
    });
    if (opts.scroll !== false) host.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Auto-bind only when the single shared #report panel exists (task-detail page).
  if (panel) {
    document.querySelectorAll(".site-rep[data-model]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        document.querySelectorAll(".site-rep").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        panel.hidden = false;
        panel.innerHTML = `<p class="muted">Loading report…</p>`;
        const { model, site, task, label } = btn.dataset;
        try {
          const res = await fetch(`/api/report/${model}/${site}/${task}`);
          if (!res.ok) throw new Error("not found");
          render(await res.json(), label);
        } catch (e) {
          panel.innerHTML = `<p class="error">Report not available.</p>`;
        }
      });
    });
  }

  window.FlairReport = { render };
})();
