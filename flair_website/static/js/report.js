// Full per-(model, site, task) report, re-rendered from the exported JSON.
// Pure inline-SVG charts — no external libs, no pre-rendered images.
(function () {
  const panel = document.getElementById("report");
  if (!panel) return;

  // ---- tiny SVG chart kit -------------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const COL = { blue: "#5b86a6", purple: "#8a72ad", sage: "#5e9b78", terra: "#b9744e", grey: "#9a9a96", ink: "#1a1a1a" };

  function ticks(lo, hi, n = 5) {
    const out = [];
    for (let i = 0; i <= n; i++) out.push(lo + ((hi - lo) * i) / n);
    return out;
  }
  const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));

  // config: {title, xLabel, yLabel, xDomain, yDomain, series:[{pts,color,dash,markers}],
  //          diagonal, hlines:[{y,color,label}], vlines:[{x,color,label}]}
  function chart(cfg) {
    const W = 380, H = 300, P = { l: 52, r: 16, t: 28, b: 44 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const [x0, x1] = cfg.xDomain, [y0, y1] = cfg.yDomain;
    const sx = (x) => P.l + ((x - x0) / (x1 - x0 || 1)) * iw;
    const sy = (y) => P.t + ih - ((y - y0) / (y1 - y0 || 1)) * ih;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img">`;
    s += `<text x="${W / 2}" y="16" text-anchor="middle" class="c-title">${cfg.title}</text>`;
    // grid + ticks
    ticks(y0, y1).forEach((t) => {
      const y = sy(t);
      s += `<line x1="${P.l}" y1="${y}" x2="${W - P.r}" y2="${y}" class="c-grid"/>`;
      s += `<text x="${P.l - 6}" y="${y + 3}" text-anchor="end" class="c-tick">${fmt(t)}</text>`;
    });
    ticks(x0, x1).forEach((t) => {
      const x = sx(t);
      s += `<text x="${x}" y="${H - P.b + 16}" text-anchor="middle" class="c-tick">${fmt(t)}</text>`;
    });
    // axis labels
    s += `<text x="${P.l + iw / 2}" y="${H - 6}" text-anchor="middle" class="c-axis">${cfg.xLabel || ""}</text>`;
    s += `<text transform="translate(13 ${P.t + ih / 2}) rotate(-90)" text-anchor="middle" class="c-axis">${cfg.yLabel || ""}</text>`;
    if (cfg.diagonal) s += `<line x1="${sx(x0)}" y1="${sy(y0)}" x2="${sx(x1)}" y2="${sy(y1)}" class="c-diag"/>`;
    (cfg.hlines || []).forEach((h) => {
      s += `<line x1="${P.l}" y1="${sy(h.y)}" x2="${W - P.r}" y2="${sy(h.y)}" stroke="${h.color}" stroke-dasharray="4 3" stroke-width="1"/>`;
    });
    (cfg.vlines || []).forEach((v) => {
      s += `<line x1="${sx(v.x)}" y1="${P.t}" x2="${sx(v.x)}" y2="${P.t + ih}" stroke="${v.color}" stroke-dasharray="3 3" stroke-width="1"/>`;
    });
    const finite = (p) => p && isFinite(p[0]) && isFinite(p[1]);
    (cfg.series || []).forEach((ser) => {
      const pts = (ser.pts || []).filter(finite);
      if (pts.length > 1) {
        const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
        s += `<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="2"${ser.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
      }
      if (ser.markers) pts.forEach((p) => {
        s += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="3" fill="${ser.color}"/>`;
      });
    });
    // frame
    s += `<rect x="${P.l}" y="${P.t}" width="${iw}" height="${ih}" class="c-frame"/>`;
    s += `</svg>`;
    // legend
    if (cfg.legend) {
      s += `<div class="c-legend">` + cfg.legend.map((l) =>
        `<span><i style="background:${l.color}"></i>${l.label}</span>`).join("") + `</div>`;
    }
    return `<div class="chart-box">${s}</div>`;
  }

  function barChart(cfg) {
    const W = 380, H = 300, P = { l: 52, r: 16, t: 28, b: 56 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const max = Math.max(1, ...cfg.values);
    const bw = iw / cfg.values.length;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="chart"><text x="${W / 2}" y="16" text-anchor="middle" class="c-title">${cfg.title}</text>`;
    cfg.values.forEach((v, i) => {
      const h = (v / max) * ih, x = P.l + i * bw + 4, y = P.t + ih - h;
      s += `<rect x="${x}" y="${y}" width="${bw - 8}" height="${h}" fill="${cfg.color || COL.blue}"/>`;
      s += `<text x="${x + (bw - 8) / 2}" y="${y - 4}" text-anchor="middle" class="c-tick">${v}</text>`;
      s += `<text x="${x + (bw - 8) / 2}" y="${P.t + ih + 14}" text-anchor="middle" class="c-tick" transform="rotate(20 ${x + (bw - 8) / 2} ${P.t + ih + 14})">${cfg.labels[i]}</text>`;
    });
    s += `<rect x="${P.l}" y="${P.t}" width="${iw}" height="${ih}" class="c-frame"/></svg>`;
    return `<div class="chart-box">${s}</div>`;
  }

  const zip = (a, b) => a.map((v, i) => [v, b[i]]);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  // ---- per-tab renderers --------------------------------------------------
  function discrimination(r) {
    const c = r.curves; if (!c) return "<p class='muted'>No curves.</p>";
    const roc = chart({
      title: `ROC · AUROC ${c.roc.auroc.toFixed(3)} (${c.roc.auroc_ci.map((x) => x.toFixed(3)).join("–")})`,
      xLabel: "False positive rate", yLabel: "True positive rate",
      xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
      series: [{ pts: zip(c.roc.grid_fpr, c.roc.tpr_at_grid), color: COL.blue }],
    });
    const pr = chart({
      title: `Precision–Recall · AUPRC ${c.pr.auprc.toFixed(3)} (${c.pr.auprc_lift}× lift)`,
      xLabel: "Recall", yLabel: "Precision", xDomain: [0, 1], yDomain: [0, 1],
      hlines: [{ y: c.pr.baseline, color: COL.grey }],
      series: [{ pts: zip(c.pr.recall, c.pr.precision), color: COL.purple }],
      legend: [{ color: COL.purple, label: "model" }, { color: COL.grey, label: `prevalence ${c.pr.baseline.toFixed(3)}` }],
    });
    return `<div class="chart-grid">${roc}${pr}</div>`;
  }

  function calibration(r) {
    const c = r.curves; if (!c) return "<p class='muted'>No calibration.</p>";
    const ce = c.calibration_extended, sm = ce && ce.smoothed;
    const series = [];
    if (sm) series.push({ pts: zip(sm.pred_grid, sm.obs_smoothed), color: COL.sage });
    series.push({ pts: zip(c.calibration.bin_pred_prob, c.calibration.bin_true_prob), color: COL.terra, markers: true });
    const maxv = Math.max(1, ...c.calibration.bin_pred_prob, ...c.calibration.bin_true_prob);
    const dom = Math.min(1, Math.ceil(maxv * 10) / 10);
    const cal = chart({
      title: `Reliability · slope ${ce.slope} · intercept ${ce.intercept} · Brier ${ce.brier}`,
      xLabel: "Predicted probability", yLabel: "Observed frequency",
      xDomain: [0, dom], yDomain: [0, dom], diagonal: true, series,
      legend: [{ color: COL.terra, label: "binned" }, { color: COL.sage, label: "logistic recalibration" }],
    });
    return `<div class="chart-grid">${cal}</div><p class="count">ECE ${c.calibration.ece} · slope &lt;1 means over-confident scores.</p>`;
  }

  function decision(r) {
    const d = r.dca; if (!d) return "<p class='muted'>No decision-curve data.</p>";
    const [lo, hi] = d.focus_range || [0, 1];
    const idx = d.thresholds.map((t, i) => i).filter((i) => d.thresholds[i] >= lo && d.thresholds[i] <= hi);
    const xs = idx.map((i) => d.thresholds[i]);
    const model = xs.map((t, k) => [t, d.net_benefit_model[idx[k]]]);
    const all = xs.map((t, k) => [t, d.net_benefit_all[idx[k]]]);
    const ymax = Math.max(...model.map((p) => p[1]), ...all.map((p) => p[1]), 0);
    const ymin = Math.min(...model.map((p) => p[1]), ...all.map((p) => p[1]), 0);
    const ch = chart({
      title: `Decision curve · net benefit`, xLabel: "Threshold probability", yLabel: "Net benefit",
      xDomain: [lo, hi], yDomain: [ymin, ymax * 1.05 || 0.01],
      hlines: [{ y: 0, color: COL.grey }], vlines: [{ x: d.clinical_threshold, color: COL.terra }],
      series: [{ pts: model, color: COL.blue }, { pts: all, color: COL.grey }],
      legend: [{ color: COL.blue, label: "model" }, { color: COL.grey, label: "treat all / none" }, { color: COL.terra, label: "clinical threshold" }],
    });
    const u = d.useful_range || {};
    return `<div class="chart-grid">${ch}</div><p class="count">Clinically useful: <strong>${d.clinically_useful ? "yes" : "no"}</strong>. Useful threshold band ${u.low}–${u.high}.</p>`;
  }

  function operating(r) {
    const th = r.threshold_analysis; if (!th) return "<p class='muted'>No threshold analysis.</p>";
    const lm = th.label_metrics;
    const sweep = th.sweep || [];
    const sens = sweep.map((s) => [s.threshold, s.sensitivity]);
    const spec = sweep.map((s) => [s.threshold, s.specificity]);
    const ch = chart({
      title: "Operating curve (sweep)", xLabel: "Threshold", yLabel: "Rate", xDomain: [0, 1], yDomain: [0, 1],
      vlines: [{ x: th.clinical_threshold, color: COL.terra }],
      series: [{ pts: sens, color: COL.blue }, { pts: spec, color: COL.sage }],
      legend: [{ color: COL.blue, label: "sensitivity" }, { color: COL.sage, label: "specificity" }, { color: COL.terra, label: "clinical threshold" }],
    });
    const rows = [["Sensitivity", lm.sensitivity], ["Specificity", lm.specificity], ["PPV", lm.ppv],
      ["NPV", lm.npv], ["F1", lm.f1], ["Accuracy", lm.accuracy], ["Flag rate", lm.flag_rate]];
    const tbl = `<table class="rep-tbl"><thead><tr><th>At clinical threshold ${th.clinical_threshold} (${th.clinical_direction})</th><th class="num">value</th></tr></thead><tbody>` +
      rows.map((x) => `<tr><td>${x[0]}</td><td class="num">${(x[1] ?? 0).toFixed(3)}</td></tr>`).join("") +
      `<tr><td>Confusion (TP/FP/FN/TN)</td><td class="num">${lm.tp}/${lm.fp}/${lm.fn}/${lm.tn}</td></tr></tbody></table>`;
    return `<div class="chart-grid">${ch}</div>${tbl}`;
  }

  function fairness(r) {
    const f = r.fairness; if (!f) return "<p class='muted'>No fairness data.</p>";
    let out = "";
    for (const [dim, groups] of Object.entries(f.subgroups)) {
      const par = (f.parity && f.parity[dim]) || {};
      let t = `<h4>${dim.replace("_category", "").replace("_", " ")}</h4>`;
      t += `<table class="rep-tbl"><thead><tr><th>group</th><th class="num">n</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">TPR</th><th class="num">FPR</th><th class="num">Brier</th></tr></thead><tbody>`;
      for (const [g, v] of Object.entries(groups)) {
        t += `<tr><td>${esc(g)}</td><td class="num">${v.n.toLocaleString()}</td><td class="num">${(v.auroc ?? 0).toFixed(3)}</td><td class="num">${(v.auprc ?? 0).toFixed(3)}</td><td class="num">${(v.tpr ?? 0).toFixed(3)}</td><td class="num">${(v.fpr ?? 0).toFixed(3)}</td><td class="num">${(v.brier ?? 0).toFixed(3)}</td></tr>`;
      }
      t += `</tbody></table>`;
      if (par.equalized_odds_tpr_gap != null)
        t += `<p class="count">Parity gaps — demographic ${par.demographic_parity_gap}, equalized-odds TPR ${par.equalized_odds_tpr_gap}, FPR ${par.equalized_odds_fpr_gap} (ref ${esc(par.reference_group)}).</p>`;
      out += t;
    }
    return out;
  }

  function temporal(r) {
    const t = r.temporal; if (!t || !t.by_window) return "<p class='muted'>No temporal data.</p>";
    const bins = t.by_window.bins || [];
    const pts = bins.map((b) => [Number(b.bin), b.auroc]);
    const ch = chart({
      title: `AUROC by ${t.by_window.unit || "window"} in stay`, xLabel: t.by_window.unit || "window", yLabel: "AUROC",
      xDomain: [pts[0] ? pts[0][0] : 0, pts.length ? pts[pts.length - 1][0] : 1], yDomain: [0.5, 1],
      series: [{ pts, color: COL.blue, markers: true }],
    });
    let traj = "";
    const tb = t.trajectory_by_outcome;
    if (tb && tb.series) {
      const wins = tb.windows.map((_, i) => i + 1);
      const series = tb.series.map((s, i) => ({ pts: zip(wins, s.mean_prob), color: i ? COL.terra : COL.blue }));
      traj = chart({
        title: "Mean predicted risk by window", xLabel: "window", yLabel: "mean predicted prob",
        xDomain: [1, wins.length], yDomain: [0, 1], series,
        legend: tb.series.map((s, i) => ({ color: i ? COL.terra : COL.blue, label: s.outcome })),
      });
    }
    return `<div class="chart-grid">${ch}${traj}</div>`;
  }

  function encounter(r) {
    const e = r.encounter; if (!e || !e.operating_curve) return "<p class='muted'>No encounter-level data (one-per-stay task).</p>";
    const oc = e.operating_curve;
    const oce = chart({
      title: "Encounter operating curve", xLabel: "alarms / patient-day", yLabel: "stay sensitivity",
      xDomain: [0, Math.max(...oc.alarms_per_patient_day) || 1], yDomain: [0, 1],
      series: [{ pts: zip(oc.alarms_per_patient_day, oc.stay_sensitivity), color: COL.blue }],
    });
    const lh = e.lead_time_hist;
    const bars = lh ? barChart({ title: "Lead-time distribution (caught stays)", labels: lh.labels, values: lh.counts, color: COL.sage }) : "";
    const a = e.at_clinical_threshold || {};
    const tbl = `<table class="rep-tbl"><tbody>
      <tr><td>Stay sensitivity</td><td class="num">${(a.stay_sensitivity ?? 0).toFixed(3)}</td></tr>
      <tr><td>Stay specificity</td><td class="num">${(a.stay_specificity ?? 0).toFixed(3)}</td></tr>
      <tr><td>Alarms / patient-day</td><td class="num">${(a.alarms_per_patient_day_silenced ?? 0).toFixed(3)}</td></tr>
      <tr><td>Number needed to alert</td><td class="num">${a.number_needed_to_alert ?? "—"}</td></tr>
      <tr><td>Caught / missed stays</td><td class="num">${a.n_caught}/${a.n_missed}</td></tr></tbody></table>`;
    return `<div class="chart-grid">${oce}${bars}</div>${tbl}`;
  }

  function cohort(r) {
    const t = r.table1; if (!t) return "<p class='muted'>No cohort table.</p>";
    let out = `<p class="count">n = ${t.n_total.toLocaleString()} (cell suppression at n&lt;${(t.metadata || {}).min_cell_count || 10}).</p>`;
    for (const [k, v] of Object.entries(t.continuous || {})) {
      out += `<p><strong>${k.replace(/_/g, " ")}</strong>: median ${v.median} (IQR ${v.q25}–${v.q75}), mean ${v.mean} ± ${v.std}, range ${v.min}–${v.max}.</p>`;
    }
    for (const [dim, groups] of Object.entries(t.categorical || {})) {
      out += `<h4>${dim.replace("_category", "").replace(/_/g, " ")}</h4><table class="rep-tbl"><tbody>`;
      for (const [g, n] of Object.entries(groups)) out += `<tr><td>${esc(g)}</td><td class="num">${typeof n === "number" ? n.toLocaleString() : esc(n)}</td></tr>`;
      out += `</tbody></table>`;
    }
    return out;
  }

  const TABS = [
    ["Discrimination", discrimination], ["Calibration", calibration], ["Decision curve", decision],
    ["Operating point", operating], ["Fairness", fairness], ["Temporal", temporal],
    ["Encounter", encounter], ["Cohort", cohort],
  ];

  function render(report, label) {
    let html = `<div class="report-head"><h3>${esc(label)}</h3>
      <button class="rep-close" aria-label="close">✕</button></div>`;
    html += `<div class="rep-tabs">` +
      TABS.map((t, i) => `<button class="rep-tab${i ? "" : " active"}" data-i="${i}">${t[0]}</button>`).join("") +
      `</div><div class="rep-body"></div>`;
    panel.innerHTML = html;
    panel.hidden = false;
    const body = panel.querySelector(".rep-body");
    const show = (i) => { body.innerHTML = TABS[i][1](report); };
    show(0);
    panel.querySelectorAll(".rep-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        panel.querySelectorAll(".rep-tab").forEach((x) => x.classList.remove("active"));
        tab.classList.add("active");
        show(Number(tab.dataset.i));
      });
    });
    panel.querySelector(".rep-close").addEventListener("click", () => { panel.hidden = true; });
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

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
})();
