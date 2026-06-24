/* FLAIR landing-page motion: the vertical "how it works" flow, scroll reveals,
   and stat count-ups. No libraries. Every effect is guarded by
   prefers-reduced-motion (final static state, no motion). */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var doc = document;

  /* -------- the flowing federation pipeline -------- */
  var howit = doc.querySelector(".howit");
  if (howit) setupFlow(howit);

  function setupFlow(root) {
    var wrap = root.querySelector(".flow-wrap");
    var steps = [].slice.call(root.querySelectorAll(".flow-step"));
    var lead = root.querySelector(".flow-lead");
    var replay = root.querySelector(".howit-replay");
    var STEP = 700; // ms between stations
    var timers = [];
    var played = false;

    function moveLead(step) {
      if (!lead || !wrap) return;
      var sr = step.getBoundingClientRect();
      var wr = wrap.getBoundingClientRect();
      lead.style.top = (sr.top - wr.top + 16) + "px"; // mark centre ≈ 16px below step top
    }

    function allOn() { steps.forEach(function (s) { s.classList.add("on"); }); }

    function play() {
      if (reduce) { allOn(); return; }
      timers.forEach(clearTimeout); timers = [];
      steps.forEach(function (s) { s.classList.remove("on"); });
      if (lead) lead.classList.remove("show");
      void root.offsetWidth; // reflow so re-added .on restarts the animations
      steps.forEach(function (s, i) {
        timers.push(setTimeout(function () {
          s.classList.add("on");
          if (lead) { lead.classList.add("show"); moveLead(s); }
        }, i * STEP));
      });
      // settle: let the lead token rest, then fade it out
      timers.push(setTimeout(function () {
        if (lead) lead.classList.remove("show");
      }, steps.length * STEP + 900));
    }

    if (replay) {
      if (reduce) replay.hidden = true;
      else replay.addEventListener("click", play);
    }

    if (reduce) { allOn(); return; }

    // ambient stream pauses off-screen; guided pass runs once on first view
    if (wrap) wrap.classList.add("paused");
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) {
            if (wrap) wrap.classList.remove("paused");
            if (!played) { played = true; play(); }
          } else if (wrap) {
            wrap.classList.add("paused");
          }
        });
      }, { threshold: 0.2, rootMargin: "0px 0px -10% 0px" });
      io.observe(root);
    } else {
      if (wrap) wrap.classList.remove("paused");
      play();
    }
  }

  /* -------- scroll reveals (stagger) + count-ups -------- */
  var reveals = doc.querySelectorAll("[data-reveal]");

  if (reduce || !("IntersectionObserver" in window)) {
    [].forEach.call(reveals, function (el) { el.classList.add("is-revealed"); });
    runCounts(doc);
  } else {
    var ro = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var el = en.target;
        var kids = el.children;
        for (var i = 0; i < kids.length; i++) {
          kids[i].style.setProperty("--rev-delay", (i * 80) + "ms");
        }
        el.classList.add("is-revealed");
        runCounts(el);
        obs.unobserve(el);
      });
    }, { threshold: 0.15 });
    [].forEach.call(reveals, function (el) { ro.observe(el); });
  }

  function runCounts(scope) {
    [].forEach.call(scope.querySelectorAll("[data-countup]"), function (el) {
      if (el.dataset.counted) return;
      el.dataset.counted = "1";
      var to = Number(el.getAttribute("data-to")) || 0;
      var prefix = el.getAttribute("data-prefix") || "";
      var suffix = el.getAttribute("data-suffix") || "";
      var fmt = function (v) {
        return prefix + Math.round(v).toLocaleString("en-US") + suffix;
      };
      if (reduce) { el.textContent = fmt(to); return; }
      var dur = 1100, start = null;
      function tick(ts) {
        if (start === null) start = ts;
        var p = Math.min(1, (ts - start) / dur);
        el.textContent = fmt(to * (1 - Math.pow(1 - p, 3))); // easeOutCubic
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = fmt(to);
      }
      requestAnimationFrame(tick);
    });
  }
})();
