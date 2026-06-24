// Merged Task Leaderboard. Each task is an accordion section; open one to see
// its model ranking (pooled AUROC across sites). Expand a model for per-site
// AUROC cells with add/remove-site toggles, plus its full federated report
// rendered inline via window.FlairReport.render (from report.js).
(function () {
  const board = JSON.parse(document.getElementById("board-data").textContent);
  const columns = board.columns;          // [{key,label,identified}]
  const allKeys = columns.map((c) => c.key);

  // selected[taskId][modelId] = Set(siteKey) for the pooled-AUROC recompute
  const selected = {};
  board.tasks.forEach((t) => {
    selected[t.id] = {};
    (board.by_task[t.id] || []).forEach((m) => (selected[t.id][m.id] = new Set(allKeys)));
  });

  const taskById = {};
  board.tasks.forEach((t) => (taskById[t.id] = t));

  const fmt = (x, d) => (x == null ? "—" : x.toFixed(d));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function pool(task, model) {
    const keys = [...selected[task][model.id]].filter((k) => model.cells[k]);
    if (!keys.length) return null;
    let n = 0, a = 0, lo = 0, hi = 0;
    keys.forEach((k) => {
      const c = model.cells[k];
      n += c.n_test; a += c.auroc * c.n_test;
      lo += c.auroc_ci[0] * c.n_test; hi += c.auroc_ci[1] * c.n_test;
    });
    return { auroc: a / n, ci: [lo / n, hi / n], n_sites: keys.length };
  }

  function renderDetail(task, model, container) {
    const cols = columns.filter((c) => model.cells[c.key]);
    if (!cols.length) { container.innerHTML = '<p class="muted">No site reports this model on this task.</p>'; return; }
    const sel = selected[task][model.id];
    const chips = cols.map((c) => {
      const on = sel.has(c.key) ? " on" : "";
      return `<span class="site-toggle${on}" data-site="${c.key}">${esc(c.label)}</span>`;
    }).join("");
    const head = cols.map((c) => `<th class="num">${esc(c.label)}</th>`).join("");
    const cells = cols.map((c) => {
      const on = sel.has(c.key);
      return `<td class="num" style="opacity:${on ? 1 : 0.3}">${on ? fmt(model.cells[c.key].auroc, 3) : "—"}</td>`;
    }).join("");
    const p = pool(task, model);
    const pooledTxt = p ? `${p.auroc.toFixed(3)} <span class="ci">(${p.ci[0].toFixed(3)}–${p.ci[1].toFixed(3)}, ${p.n_sites})</span>` : "—";
    const t = taskById[task] || {};
    const s = model.split;
    const provenance =
      `<div class="count" style="margin:2px 0 12px">` +
      (t.report_mode ? `Report mode: <strong>${esc(t.report_mode)}</strong>` : "") +
      (s ? ` &nbsp;·&nbsp; Split: <strong>${esc(s.type)} ${esc(s.ratio)}</strong> hold-out by ${esc(s.grain)}` : "") +
      `</div>`;
    // per-site full-report buttons (only where a report exists) + permalink
    const repBtns = cols.map((c) => {
      const label = `${model.name} · ${c.label}`;
      if (model.cells[c.key].has_report)
        return `<button type="button" class="site-rep" data-site="${c.key}" data-label="${esc(label)}">${esc(c.label)} · full report</button>`;
      return `<span class="site-rep pending">${esc(c.label)} · pending</span>`;
    }).join("");
    container.innerHTML =
      provenance +
      `<div class="count" style="margin-bottom:8px">Add / remove sites — pooled AUROC recomputes:</div>` +
      `<div class="site-toggles">${chips}</div>` +
      `<table style="margin:0"><thead><tr><th>Metric</th>${head}<th class="num">Pooled</th></tr></thead>` +
      `<tbody><tr><td>AUROC</td>${cells}<td class="num"><strong>${pooledTxt}</strong></td></tr></tbody></table>` +
      `<div class="rep-controls"><div class="site-buttons">${repBtns}</div>` +
      `<a class="more rep-permalink" href="/tasks/${esc(task)}">open full page ↗</a></div>` +
      `<div class="report-panel" hidden></div>`;

    // site add/remove toggles → recompute pooled (rebuilds this detail block)
    container.querySelectorAll(".site-toggle").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        const k = chip.dataset.site;
        sel.has(k) ? sel.delete(k) : sel.add(k);
        renderDetail(task, model, container);
        const pp = pool(task, model);
        const rowCell = container.closest("tr").previousElementSibling.querySelector(".row-pooled");
        if (rowCell) rowCell.textContent = pp ? pp.auroc.toFixed(3) : "—";
      });
    });

    // inline full report → render into this row's own panel
    const host = container.querySelector(".report-panel");
    container.querySelectorAll("button.site-rep").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        container.querySelectorAll("button.site-rep").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        host.hidden = false;
        host.innerHTML = '<p class="muted">Loading report…</p>';
        try {
          const res = await fetch(`/api/report/${model.id}/${btn.dataset.site}/${task}`);
          if (!res.ok) throw new Error("not found");
          const json = await res.json();
          window.FlairReport.render(json, btn.dataset.label, host, {
            scroll: false,
            onClose: () => { btn.classList.remove("active"); },
          });
        } catch (err) {
          host.innerHTML = '<p class="error">Report not available.</p>';
        }
      });
    });
  }

  function renderTask(task, container) {
    if (container.dataset.rendered) return;
    container.dataset.rendered = "1";
    const rows = board.by_task[task] || [];
    if (!rows.length) { container.innerHTML = '<p class="muted">No model reports this task yet.</p>'; return; }
    let html = '<table class="lb"><thead><tr><th>#</th><th>Model</th><th>Owner</th>' +
      '<th class="num">Pooled AUROC</th><th class="num">95% CI</th><th class="num">Sites</th><th></th></tr></thead><tbody>';
    rows.forEach((m, i) => {
      const p = m.pooled;
      const baseline = m.is_baseline ? ' <span class="pill sage">baseline</span>' : "";
      const link = m.github_url ? `<a href="${esc(m.github_url)}" target="_blank" rel="noopener">${esc(m.name)}</a>` : esc(m.name);
      html +=
        `<tr class="lb-model" data-model="${esc(m.id)}">` +
        `<td class="num">${i + 1}</td>` +
        `<td>${link}${baseline}</td>` +
        `<td class="muted">${esc(m.owner || "—")}</td>` +
        `<td class="num"><span class="row-pooled">${p.auroc.toFixed(3)}</span></td>` +
        `<td class="num ci">${p.auroc_ci[0].toFixed(3)}–${p.auroc_ci[1].toFixed(3)}</td>` +
        `<td class="num">${p.n_sites}</td>` +
        `<td><span class="chev">expand ▾</span></td></tr>` +
        `<tr class="lb-detail" data-detail="${esc(m.id)}"><td colspan="7"><div class="inner"></div></td></tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;

    container.querySelectorAll(".lb-model").forEach((tr) => {
      tr.addEventListener("click", () => {
        const mid = tr.dataset.model;
        const model = rows.find((x) => x.id === mid);
        const detail = container.querySelector(`.lb-detail[data-detail="${mid}"]`);
        const inner = detail.querySelector(".inner");
        const opening = !detail.classList.contains("open");
        detail.classList.toggle("open");
        tr.querySelector(".chev").textContent = opening ? "collapse ▴" : "expand ▾";
        if (opening && !inner.dataset.rendered) {
          inner.dataset.rendered = "1";
          renderDetail(task, model, inner);
        }
      });
    });
  }

  // accordion: open a task to lazily render its ranking
  document.querySelectorAll(".task-acc-head").forEach((head) => {
    head.addEventListener("click", () => {
      const acc = head.parentElement;
      const body = acc.querySelector(".task-acc-body");
      const opening = !body.classList.contains("open");
      body.classList.toggle("open");
      head.setAttribute("aria-expanded", opening ? "true" : "false");
      head.classList.toggle("open", opening);
      if (opening) renderTask(body.dataset.task, body);
    });
  });
})();
