"""PNG rendering for a 'below'-direction task — the first live one is T3.

The renderers had no test coverage at all, and every live task was "above", where
the alarm axis and the decision axis happen to be the same numbers. That
coincidence hid a real defect: the scorecard's net-benefit panel plotted
decision-basis values against the label-scale grid, so under "below" it came out
mirrored relative to dca.png and the two figures drew their red policy line at
different x for the same policy.
"""

from __future__ import annotations

import pytest
from test_report_extensions import META, make_synth

from flair_benchmark.report._viz import (
    _import_plt,
    _render_dca,
    _render_scorecard,
    render_sanity_pngs,
)
from flair_benchmark.report.dca import _focus_range, build_dca
from flair_benchmark.report.operating_points import build_episodic_operating_points

BELOW = {**META, "clinical_direction": "below"}


def _figures(meta: dict) -> tuple[dict, dict, dict]:
    """Render both figures with a capturing saver; return (figs, ops, dca)."""
    plt = _import_plt()
    df = make_synth(n_stays=900, seed=7)
    ops, dca = build_episodic_operating_points(df, meta), build_dca(df, meta)
    figs: dict = {}

    def _save(fig, name, tight=True):
        figs[name] = fig

    _render_scorecard(ops, dca, _save, plt)
    _render_dca(dca, _save, plt)
    return figs, ops, dca


def _red_vline(ax, ls: str) -> float:
    """x of the single red policy line on an axes (axvline -> xdata [x, x])."""
    xs = [line.get_xdata()[0] for line in ax.lines
          if line.get_color() == "red" and line.get_linestyle() == ls
          and len(set(line.get_xdata())) == 1]
    assert len(xs) == 1, f"expected one red '{ls}' vline, found {xs}"
    return float(xs[0])


def _panel_by_title(fig, title: str):
    return next(ax for ax in fig.axes if ax.get_title() == title)


# ------------------------------------------------------------------ focus range


def test_focus_range_is_the_exact_mirror():
    """The 'below' window is the 'above' rule seen through the complement."""
    for prev, clin in [(0.05, 0.20), (0.12, 0.35), (0.946, 0.90), (0.5, 0.5)]:
        lo, hi = _focus_range(prev, clin, below=True)
        a, b = _focus_range(1.0 - prev, 1.0 - clin, below=False)
        assert (lo, hi) == (pytest.approx(1.0 - b), pytest.approx(1.0 - a))


def test_focus_range_survives_a_high_complement_prevalence():
    """The regression: at prev ~= 0.95 the old rule clamped to [0, 1] and died.

    T3's failure rate is ~5.4%, so on the decision basis P(successful extubation)
    is ~0.946 and 10*prev overshoots 1.0. Mirroring instead of recomputing keeps a
    real zoom, and it must still contain the shipped policy.
    """
    lo, hi = _focus_range(0.946, 0.90, below=True)
    assert (round(lo, 3), hi) == (0.46, 1.0)
    assert lo <= 0.90 <= hi

    # the "above" twin at the same event rate gets the mirrored window
    assert _focus_range(0.054, 0.10, below=False) == [0.0, 0.54]


# ---------------------------------------------------------------- the two axes


def test_scorecard_decision_panel_is_drawn_on_the_decision_axis():
    """Each point must sit at the x its net benefit was evaluated at.

    Asserting the x VALUES alone proves nothing: 1-t maps {0.01..0.99} onto
    itself, so the mirrored panel plots the same x set. The defect is in the
    PAIRING — the old code put row 0.01's net benefit at x=0.01 when it had been
    evaluated at x=0.99. So the assertion has to be on (x, y) together.
    """
    figs, ops, _ = _figures(BELOW)
    ax = _panel_by_title(figs["operating_points.png"], "Net benefit (decision basis)")

    expected = {e["decision"]["dca_threshold"]: e["decision"]["net_benefit"]
                for e in ops["by_threshold"]
                if e["decision"].get("net_benefit") is not None}
    # the KPI trend line _panel draws, not the "act on all" reference beside it
    line = next(ln for ln in ax.lines if ln.get_color() == "#5b86a6")
    xs, ys = line.get_xdata(), line.get_ydata()
    assert {round(float(x), 6): float(y) for x, y in zip(xs, ys)} == pytest.approx(
        expected)
    # ascending on the decision axis — an unsorted 1-t sequence would zig-zag
    assert list(xs) == sorted(xs)


def test_both_figures_mark_the_same_policy():
    """The B1 regression, stated as the thing a reader would notice.

    scorecard.png and dca.png show the same net-benefit quantity. If their red
    policy lines disagree, one of them is drawn in the wrong frame.
    """
    figs, ops, dca = _figures(BELOW)
    card = _red_vline(
        _panel_by_title(figs["operating_points.png"], "Net benefit (decision basis)"),
        "--")
    curve = _red_vline(figs["dca.png"].axes[0], ":")
    assert card == pytest.approx(curve)
    assert card == pytest.approx(dca["metrics"]["clinical_threshold"])
    # and it is the complement of the label-scale policy, not the policy itself
    assert card == pytest.approx(1.0 - META["clinical_threshold"])


def test_above_task_keeps_both_axes_identical():
    """The coincidence that hid the bug must still hold: no change for 'above'."""
    figs, _, dca = _figures(META)
    card = _red_vline(
        _panel_by_title(figs["operating_points.png"], "Net benefit (decision basis)"),
        "--")
    assert card == pytest.approx(META["clinical_threshold"])
    assert card == pytest.approx(_red_vline(figs["dca.png"].axes[0], ":"))


def test_below_bundle_renders_real_pngs(tmp_path):
    df = make_synth(n_stays=600, seed=9)
    reports = {"operating_points": build_episodic_operating_points(df, BELOW),
               "dca": build_dca(df, BELOW)}
    written = render_sanity_pngs(reports, tmp_path)
    names = {p.name for p in written}
    assert {"operating_points.png", "dca.png"} <= names
    for p in written:
        assert p.stat().st_size > 5_000, p.name
