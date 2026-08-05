"""Risk–coverage figure (RQ4) — selective rejection of imagined candidates, oracle-free.

All THREE probed tasks in one main-text figure, in the paper's usual ascending-qubit order:
LiH-4q, LiH-6q, BeH2-8q. LiH-4q is the task where rejection does NOT beat the prediction itself,
and it is shown in the main text rather than exiled to an appendix — the heterogeneity IS the
result. There is no separate appendix version: it would now be a byte-for-byte duplicate.

Curves are recomputed by CALLING `risk_coverage.py` rather than parsing its text output, so the
figure and the table cannot drift and the grid is not limited to the ten rows the table prints.

WHAT THE AXES MEAN
  coverage(tau) = fraction of imagined candidates KEPT after rejecting the worst by each rule.
  risk(tau)     = mean |predicted - real| over the kept candidates, in the WM's own target space
                  (frontier score S). NOT mHa — never label it as an energy error.

SAMPLE (the reason this analysis is credible): the SIGMA-NEUTRAL stratum only, i.e. candidates the
DAgger probe selected by predicted VALUE (`selection in {top, both}`). The disagreement-selected ones
are excluded, so sigma was never used to build the sample. A rule based on sigma is therefore being
tested on data whose sigma distribution it did not shape — conservative by construction.

UNCERTAINTY — what is and is not drawn
  The AURC annotations carry the REAL per-seed spread (n=5 seeds, the experimental unit; each seed
  contributes ~10^3 candidates, so pooling candidates would inflate n by three orders of magnitude).
  No band is drawn around the curves: that was not asked for, and four bands on a single-column panel
  would obscure the crossings that matter. Nothing here is invented — every +- comes from
  `risk_coverage.py`'s per-seed block.

  ⚠ A significance fact that must not be lost in the "-46%" headline: on BeH2-8q the advantage of
  disagreement over the PREDICTED-VALUE rule is NOT significant (paired ΔAURC -0.076,
  95% CI [-0.135, +0.013]), and on LiH-4q it is flatly null (-0.008, [-0.020, +0.003]). The -46%
  and -24% are measured against NO rejection, which is a different and weaker statement. Only
  LiH-6q beats the value rule with a CI that excludes 0 — 1 task in 3. Both non-significant panels
  are tagged in red on the figure itself, so the figure cannot be quoted without the caveat.

WHY LiH-4q EARNS A MAIN-TEXT PANEL
  It is the control that makes the LiH-6q result meaningful. On LiH-4q the value rule is if
  anything slightly better at 50% coverage (-26% vs -24%) and the predicted-best half is the safer
  half — i.e. no over-optimism. On LiH-6q the ordering INVERTS (value +34%, inverse value -34%).
  Showing only the tasks where sigma wins would make the mechanism look general when it is not.

DESIGN (dataviz method)
  form  : risk vs coverage, 3 compared rules + 1 baseline -> multi-line small multiples.
  colour: categorical identity for the three RULES (slots 1/2/7 of the validated reference palette:
          #2a78d6, #eb6834, #4a3aa7). `random` is a BASELINE, not a rule, so it is neutral grey.
          All-pairs validated: worst CVD dE 13.0 (target 8), normal-vision 16.3 (floor 15), lightness
          band / chroma / contrast all pass. The obvious 4-colour choice (adding yellow #eda100)
          FAILED the normal-vision floor at dE 13.7 against aqua and was re-stepped, not excused.
  a11y  : identity never rests on colour alone — line width and dash pattern differ per rule, and
          one shared legend serves all panels. Direct end-labels were tried and removed: all four
          curves converge at coverage=1, so they collided and blew up the tight bbox.
  size  : going from 2 to 3 panels, the TOTAL figure width is held near the old 7.0in instead of
          growing to 10in. A 10in-wide figure set to \\textwidth (5.5in) is scaled 0.55x and every
          label shrinks with it; holding the width and narrowing the panels keeps the on-page font
          size roughly what it was. Fonts are raised ~0.5pt to absorb the remaining difference.

Usage: python analysis/plot_risk_coverage.py
Output: risk_coverage_oraclefree.{pdf,png}  (single main-text figure; no appendix variant)
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import risk_coverage as RC          # SSOT: same loader, same curve(), same GRID, same AURC

OUT = f"{HERE}/outputs/main_results"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]        # ascending qubit count, as in every other table
DISP = {"LiH4q": "LiH (4q)", "LiH6q": "LiH (6q)", "BeH2_8q": "BeH$_2$ (8q)"}
# (key, display, colour, linewidth, dash, z)  — `random` is a baseline, hence neutral grey
# Tasks where the headline % is real but the comparison against the PREDICTED-VALUE rule is not
# significant. Without this the figure alone would overstate; four words is enough to prevent that.
# The bracketed ΔAURC CI that used to ride along here was dropped: at 3 panels it no longer fit the
# panel width, and it is in the caption and in risk_coverage.txt. The tag's job is to stop the
# reader quoting the % as a win over the value rule, and "n.s. vs predicted value" does that.
# It rides UNDER THE TITLE, not inside the panel. Inside, it was competing for the bottom strip with
# the point labels — on LiH-4q the "-26%" landed on it. Above the axes the space is guaranteed free,
# and the caveat is now read at the same moment as the task name, which is where it belongs.
NS_TAG = {"LiH4q": "n.s. vs predicted value", "BeH2_8q": "n.s. vs predicted value"}
# 50%-coverage callouts: (rule, dx_pt, dy_pt, ha, va). The percentage is attached DIRECTLY to a dot
# on its own curve instead of listed in a corner block. The corner block worked at 2 wide panels and
# broke at 3 narrow ones — on BeH2-8q it landed on top of the disagreement curve, because clamping
# the y-axis at 0 (risk cannot be negative) silently ate the space that had been reserved for it.
# Point labels need no reserved band at all, and they remove ~6 lines of small text from the figure.
# Offsets are chosen per panel against the OTHER curves' positions at coverage 0.5; the fit pass
# below then verifies that nothing actually overlaps rather than trusting these by eye.
ANNOT = {
    "LiH4q":   [("disagree", -5, 7, "right", "bottom"),      # blue up-left, orange down-left: the
                ("value", -5, -9, "right", "top")],          # two curves are only 0.01 apart at 50%
    "LiH6q":   [("disagree", 6, 7, "left", "bottom"),        # the inversion is the RQ4 headline, so
                ("value", 6, 6, "left", "bottom"),           # all three rules are labelled here
                ("value_inv", 6, -9, "left", "top")],
    "BeH2_8q": [("disagree", 6, -9, "left", "top")],         # single-number story; below the curve,
}                                                            # where nothing else lives

# Which AURCs appear in each panel's box. `random` is dropped: it is visibly the flat grey line and
# its AURC equals the no-rejection risk by construction, so a printed number adds nothing. The full
# 4-rule AURC table with per-seed spread lives in risk_coverage.txt and is quoted in the caption.
BOX_RULES = {"LiH4q": ["disagree", "value"], "BeH2_8q": ["disagree", "value"],
             "LiH6q": ["disagree", "value", "value_inv"]}
SHORT = {"disagree": "disagreement", "value": "predicted value", "value_inv": "inverse value"}
RULES = [("disagree",  "Ensemble disagreement",      "#2a78d6", 2.6, (None, None), 5),
         ("value",     "Predicted value",            "#eb6834", 1.6, (4, 2),       4),
         ("value_inv", "Inverse value (diagnostic)", "#4a3aa7", 1.5, (1, 2),       3),
         ("random",    "Random rejection",           "#8a8a86", 1.3, (6, 3),       2)]


def curves(mol):
    """-> (grid, {rule: mean curve}, {rule: per-seed AURC array}, n_seeds, n_cand).

    The random baseline uses RC.rule_rng(mol) — the SAME order-independent generator the table uses,
    so the figure reproduces `risk_coverage.txt` exactly. An earlier version seeded its own RNG and
    the BeH2-8q random AURC came out 0.2466 against the table's 0.2310; that mismatch is what exposed
    the order dependence in the table itself.
    """
    runs = [RC.load_run(d, False) for d in sorted(glob.glob(f"{RC.C}/dreamqas/*_{mol}_s*_of"))]
    runs = [r for r in runs if r]
    if not runs:
        return None
    per = {k: [] for k, *_ in RULES}
    ns = []
    for sig, pred, err in runs:
        per["disagree"].append(RC.curve(sig, err, RC.GRID))
        per["value"].append(RC.curve(pred, err, RC.GRID))
        per["value_inv"].append(RC.curve(-pred, err, RC.GRID))
        per["random"].append(RC.curve(pred, err, RC.GRID, rng=RC.rule_rng(mol)))
        ns.append(len(err))
    mean = {k: np.mean(np.asarray(v), 0) for k, v in per.items()}
    au = {k: np.array([RC.aurc(c, RC.GRID) for c in v]) for k, v in per.items()}
    return np.asarray(RC.GRID, float), mean, au, len(runs), int(np.mean(ns))


def at(grid, y, c):
    return float(np.interp(c, grid, y))


def draw(ax, mol, data, label_ref):
    """Draw one panel. Returns (line handles, the artists the fit pass has to keep clear)."""
    grid, mean, au, nseed, ncand = data
    full = mean["random"][-1]                     # risk with no rejection (all rules meet here)
    col = {k: c for k, _, c, *_ in RULES}
    handles = []
    for key, disp, c, lw, dash, z in RULES:
        ln, = ax.plot(grid, mean[key], color=c, lw=lw, zorder=z, label=disp,
                      dashes=dash if dash[0] else (None, None))
        handles.append(ln)
    ax.axvline(0.5, color="#c9c9c5", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axhline(full, color="#bbb", lw=0.8, ls=":", zorder=1)
    # The 0.5 guide carries no text label: it sits exactly on the 0.5 axis tick and is met by the
    # labelled markers, so the reference coverage is unambiguous without another small string.
    inside = []
    for key, dx, dy, ha, va in ANNOT[mol]:
        y50 = at(grid, mean[key], 0.5)
        ax.plot([0.5], [y50], marker="o", ms=5.0, mfc=col[key], mec="white", mew=1.1, zorder=9)
        p50 = round(100 * (y50 - full) / full)
        inside.append(ax.annotate(
            f"{p50:+d}%", (0.5, y50), textcoords="offset points", xytext=(dx, dy),
            ha=ha, va=va, fontsize=7.6, color=col[key], zorder=10,
            fontweight=("bold" if key == "disagree" else "normal")))
    # "no rejection" identifies the horizontal reference line. Placement searches over BOTH x and
    # side-of-the-line, scoring the label's whole footprint against every curve — including `random`,
    # which is the one that matters: it tracks the reference line by construction, so a label pinned
    # above the line lands on it wherever random happens to run high (it did, on LiH-4q). Scoring
    # only the three rules made that collision invisible to the search. x near 0.5 is excluded so the
    # label does not straddle the coverage guide.
    # Only the first panel carries the text. The line means exactly the same thing in all three, and
    # BeH2-8q has no gap wide enough to hold it: every placement there sat on the inverse-value or
    # the random curve. One label for the row is both cleaner and three fewer small strings.
    if label_ref:
        ylo0 = min(mean[k].min() for k, *_ in RULES)
        sp = max(mean[k].max() for k, *_ in RULES) - ylo0
        gap, hh, hw = 0.02 * sp, 0.065 * sp, 0.16      # clearance, label height, half-width (coverage)
        # Candidate centres must leave room for the label's own WIDTH inside the frame — the search
        # is over x, so without this it happily picked a spot whose left half fell off the axis. The
        # x-limits are shared across panels and must not be widened to accommodate one grey string.
        ok = ((grid >= 0.05 + hw + 0.02) & (grid <= 1.04 - hw - 0.02)
              & (np.abs(grid - 0.5) > 0.08))
        best = (-np.inf, 0, 1)
        for i in np.flatnonzero(ok):
            w = np.abs(grid - grid[i]) <= hw           # the x-window the label actually covers
            for side in (1, -1):
                lo, hi = sorted((full + side * gap, full + side * (gap + hh)))
                s = min(min(np.min(np.abs(mean[k][w] - lo)), np.min(np.abs(mean[k][w] - hi)),
                            0.0 if np.any((mean[k][w] > lo) & (mean[k][w] < hi)) else np.inf)
                        for k, *_ in RULES)
                if s > best[0]:
                    best = (s, i, side)
        _, j, side = best
        inside.append(ax.annotate("no rejection", (grid[j], full), textcoords="offset points",
                                  xytext=(0, 4 * side), ha="center", fontsize=6.8,
                                  va=("bottom" if side > 0 else "top"), color="#8a8a86", zorder=8))
    if NS_TAG.get(mol):     # subtitle, above the axes — see the NS_TAG comment for why not inside
        ax.text(0.5, 1.015, NS_TAG[mol], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.6, color="#a01d1d", style="italic")
    ax.set_xlim(0.05, 1.04)
    ylo = min(mean[k].min() for k, *_ in RULES)
    yhi = max(mean[k].max() for k, *_ in RULES)
    span = yhi - ylo
    ax.set_ylim(max(0.0, ylo - span * 0.10), yhi + span * 0.07)     # fit_annotations() refines this
    ax.set_xlabel("Coverage (fraction kept)", fontsize=8.5)
    ax.grid(True, lw=0.35, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)
    # AURC block, with the REAL per-seed spread. Only the rules whose comparison the panel is making.
    lines = [f"AURC (n={nseed})"] + [
        f" {SHORT[k]:<16s}{au[k].mean():.3f}±{au[k].std(ddof=1):.3f}" for k in BOX_RULES[mol]]
    box = ax.text(0.985, 0.975, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                  fontsize=6.2, family="monospace", zorder=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.82", lw=0.6, alpha=0.94))
    return handles, box, inside


MARGIN = 0.025          # axes fractions a data-anchored label must keep from the frame


def fit_annotations(fig, panels):
    """Expand each panel's y-range until the AURC box clears every curve and no label is clipped.

    Hand-tuned data-space reserves were what broke when the figure went from 2 wide panels to 3
    narrow ones — a reserve is a guess at how tall text will render, and on BeH2-8q the
    `max(0.0, ...)` clamp (risk cannot be negative) silently discarded it. Here the artists are
    RENDERED first and their real extents measured, so the clearance is a fact, not an estimate.

    Two constraint kinds, both solved by widening the limits, never by moving a label away from
    what it labels:
      box     anchored in AXES fractions, so its fractional extent is fixed; raise `top` until the
              tallest curve beneath its x-span sits below its lower edge. Single exact solve.
      inside  anchored in DATA, so the fractional extent moves with the limits; nudge and re-measure
              until every "%" label and the "no rejection" tag is fully inside the frame.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for _ in range(8):
        moved = False
        for ax, (grid, mean, *_), box, inside in panels:
            bot, top = ax.get_ylim()
            xlo, xhi = ax.get_xlim()
            bb = box.get_window_extent(r).transformed(ax.transAxes.inverted())
            m = grid >= xlo + bb.x0 * (xhi - xlo)         # only the x-span the box actually covers
            peak = max(mean[k][m].max() for k, *_ in RULES)
            need = bot + (peak + 0.035 * (top - bot) - bot) / max(bb.y0, 1e-3)
            if need > top + 1e-9:
                ax.set_ylim(bot, need)
                top, moved = need, True
            for art in inside:
                bb = art.get_window_extent(r).transformed(ax.transAxes.inverted())
                if bb.y0 < MARGIN:
                    bot -= (MARGIN - bb.y0) * (top - bot) * 1.15
                    ax.set_ylim(bot, top)
                    moved = True
                elif bb.y1 > 1 - MARGIN:
                    top += (bb.y1 - 1 + MARGIN) * (top - bot) * 1.15
                    ax.set_ylim(bot, top)
                    moved = True
        if not moved:
            break
        fig.canvas.draw()


def build(mols, fname, note):
    figs = []
    for m in mols:
        d = curves(m)
        if d is None:
            print(f"  [skip] {m}: no data")
            continue
        figs.append((m, d))
    if not figs:
        return
    # Total width is held near 7.4in for 3 panels rather than 3.3in/panel: see `size` in the module
    # docstring — a 10in figure scaled into a 5.5in column loses more legibility than a narrow panel.
    w = 2.35 if len(figs) >= 3 else 3.3
    fig, AX = plt.subplots(1, len(figs), figsize=(w * len(figs) + 0.4, 3.15), sharey=False)
    AX = np.atleast_1d(AX)
    panels = []
    for i, ((m, d), ax) in enumerate(zip(figs, AX)):
        H, box, inside = draw(ax, m, d, label_ref=(i == 0))
        panels.append((ax, d, box, inside))
        # One pad for every panel, whether or not it carries a red subtitle: a per-panel pad would
        # put the titles at different heights across the row, which reads as a layout defect.
        ax.set_title(f"({chr(97 + i)}) {DISP[m]}", fontsize=9.5, pad=12.5)
    AX[0].set_ylabel("Mean $|$predicted $-$ real$|$\n(frontier score $S$; lower is better)", fontsize=8.5)
    fig.legend(H, [r[1] for r in RULES], loc="lower center", ncol=4, fontsize=7.8,
               frameon=False, bbox_to_anchor=(0.5, 0.012), handlelength=2.6, columnspacing=1.6)
    # No caption block is baked into the figure: a paper caption belongs in LaTeX, and a three-line
    # note here duplicated it while eating a third of the panel height. The recommended caption is
    # printed to stdout instead, so nothing is lost.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fit_annotations(fig, panels)                 # after layout: extents are only real once placed
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[written] {fname}.{{pdf,png}}")
    print("  ---- recommended LaTeX caption (not baked into the figure) ----")
    for ln in note.split("\n"):
        print(f"  {ln}")
    print("  ---------------------------------------------------------------")
    for m, (grid, mean, au, ns, nc) in figs:
        full = mean["random"][-1]
        print(f"  {DISP[m]:12s} n={ns} seeds, {nc} cand/seed · "
              + " · ".join(f"{k} AURC {au[k].mean():.4f}±{au[k].std(ddof=1):.4f}" for k, *_ in RULES))
        print(f"  {'':12s} risk@50%: " + " ".join(
            f"{k}={at(grid, mean[k], 0.5):.4f}({round(100 * (at(grid, mean[k], 0.5) - full) / full):+d}%)"
            for k, *_ in RULES) + f"  | no-rejection {full:.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    note = ("$\\sigma$-neutral stratum (value-selected candidates only), oracle-free Full runs.  "
            "$\\pm$ = per-seed spread over 5 seeds; no curve bands are drawn.\n"
            "Percentages are risk at 50\\% coverage relative to no rejection.  "
            "⚠ Disagreement beats the predicted-value rule on LiH (6q) only\n"
            "(paired $\\Delta$AURC $-0.236$, 95\\% CI $[-0.318, -0.155]$); it is n.s. on "
            "BeH$_2$ (8q) ($-0.076$, $[-0.135, +0.013]$) and null on LiH (4q)\n"
            "($-0.008$, $[-0.020, +0.003]$).  The $-24\\%$/$-46\\%$ are measured against no "
            "rejection, which is a weaker statement.")
    build(MOLS, "risk_coverage_oraclefree", note)


if __name__ == "__main__":
    main()
