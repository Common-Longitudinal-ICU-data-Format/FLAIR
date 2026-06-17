// Model-driven leaderboard. Task tabs switch the ranking table; each row is a
// model ranked by pooled AUROC. Expanding a model reveals its per-site AUROC
// cells with add/remove-site toggle chips that recompute that model's pooled.
(function () {
  const board = JSON.parse(document.getElementById("board-data").textContent);
  const columns = board.columns;          // [{key,label,identified}]
  const allKeys = columns.map((c) => c.key);
  const labelOf = {};
  columns.forEach((c) => (labelOf[c.key] = c.label));

  // selected[taskId][modelId] = Set(siteKey)
  const selected = {};
  board.tasks.forEach((t) => {
    selected[t.id] = {};
    (board.by_task[t.id] || []).forEach((m) => (selected[t.id][m.id] = new Set(allKeys)));
  });

  let activeTask = board.tasks[0] ? board.tasks[0].id : null;
  const boardEl = document.getElementById("board");

  const fmt = (x, d) => (x == null ? "—" : x.toFixed(d));

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
      return `<span class="site-toggle${on}" data-site="${c.key}">${c.label}</span>`;
    }).join("");
    const head = cols.map((c) => `<th class="num">${c.label}</th>`).join("");
    const cells = cols.map((c) => {
      const on = sel.has(c.key);
      return `<td class="num" style="opacity:${on ? 1 : 0.3}">${on ? fmt(model.cells[c.key].auroc, 3) : "—"}</td>`;
    }).join("");
    const p = pool(task, model);
    const pooledTxt = p ? `${p.auroc.toFixed(3)} <span class="ci">(${p.ci[0].toFixed(3)}–${p.ci[1].toFixed(3)}, ${p.n_sites})</span>` : "—";
    container.innerHTML =
      `<div class="count" style="margin-bottom:8px">Add / remove sites — pooled AUROC recomputes:</div>` +
      `<div class="site-toggles">${chips}</div>` +
      `<table style="margin:0"><thead><tr><th>Metric</th>${head}<th class="num">Pooled</th></tr></thead>` +
      `<tbody><tr><td>AUROC</td>${cells}<td class="num"><strong>${pooledTxt}</strong></td></tr></tbody></table>`;
    container.querySelectorAll(".site-toggle").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        const k = chip.dataset.site;
        sel.has(k) ? sel.delete(k) : sel.add(k);
        renderDetail(task, model, container);
        // refresh the row's pooled cell
        const pp = pool(task, model);
        const rowCell = container.closest("tr").previousElementSibling.querySelector(".row-pooled");
        if (rowCell) rowCell.textContent = pp ? pp.auroc.toFixed(3) : "—";
      });
    });
  }

  function renderTask(task) {
    const rows = board.by_task[task] || [];
    if (!rows.length) { boardEl.innerHTML = '<p class="muted">No model reports this task yet.</p>'; return; }
    let html = '<table class="lb"><thead><tr><th>#</th><th>Model</th><th>Owner</th>' +
      '<th class="num">Pooled AUROC</th><th class="num">95% CI</th><th class="num">Sites</th><th></th></tr></thead><tbody>';
    rows.forEach((m, i) => {
      const p = m.pooled;
      const baseline = m.is_baseline ? ' <span class="pill sage">baseline</span>' : "";
      const link = m.github_url ? `<a href="${m.github_url}" target="_blank" rel="noopener">${m.name}</a>` : m.name;
      html +=
        `<tr class="lb-model" data-model="${m.id}">` +
        `<td class="num">${i + 1}</td>` +
        `<td>${link}${baseline}</td>` +
        `<td class="muted">${m.owner || "—"}</td>` +
        `<td class="num"><span class="row-pooled">${p.auroc.toFixed(3)}</span></td>` +
        `<td class="num ci">${p.auroc_ci[0].toFixed(3)}–${p.auroc_ci[1].toFixed(3)}</td>` +
        `<td class="num">${p.n_sites}</td>` +
        `<td><span class="chev">expand ▾</span></td></tr>` +
        `<tr class="lb-detail" data-detail="${m.id}"><td colspan="7"><div class="inner"></div></td></tr>`;
    });
    html += "</tbody></table>";
    boardEl.innerHTML = html;

    boardEl.querySelectorAll(".lb-model").forEach((tr) => {
      tr.addEventListener("click", () => {
        const mid = tr.dataset.model;
        const model = rows.find((x) => x.id === mid);
        const detail = boardEl.querySelector(`.lb-detail[data-detail="${mid}"]`);
        const inner = detail.querySelector(".inner");
        const opening = !detail.classList.contains("open");
        detail.classList.toggle("open");
        tr.querySelector(".chev").textContent = opening ? "collapse ▴" : "expand ▾";
        if (opening) renderDetail(task, model, inner);
      });
    });
  }

  document.querySelectorAll("#task-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll("#task-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      activeTask = tab.dataset.task;
      renderTask(activeTask);
    });
  });

  if (activeTask) renderTask(activeTask);
})();
