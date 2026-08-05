"""WM ranking accuracy as a function of CIRCUIT DEPTH, per molecule (oracle-free).

The operational question this answers: **how deep into the ansatz can the world model's ranking still
be trusted, and does that depend on the molecule?** It is the fact that bounds where imagination is
worth using and where real-VQE verification has to take over.

WHY THIS IS A DEPTH CURVE AND NOT A "HORIZON" CURVE
---------------------------------------------------
The underlying probe (`t1a_action_ranking.py::horizon_fidelity`) is historically named "horizon
fidelity", but under DreamQAS's architecture that name is wrong and so was the claim built on it:
imagined circuit dynamics are EXACT, and the WM's prediction is a pure function of the whole action
sequence (`_encode_seeds` restarts from `wm.init_state` every call, imagine.py:34-44), so predictions
never chain and nothing accumulates across steps. Every prefix is drawn at a fixed depth
sd = num_layers // 4 and H more gates are appended, so the endpoint is simply a circuit of depth
sd + H. fidelity(H) IS accuracy-at-depth. See `oracle_free_horizon_slope.txt` for the retraction of
the "no accumulation of transition error" argument.

WHAT IS PLOTTED
---------------
y = endpoint ranking fidelity: Spearman, ACROSS the 15 held-out prefixes at that depth, between the
WM's predicted log-error and the real post-VQE log-error. Mean +- sample std over 5 seeds.
⚠ This is NOT the within-prefix action-ranking rho (which ranks ~10 sibling actions of ONE prefix and
lives in `oracle_free_t1a.txt`). Different chance level, different question — never mix them.

left  panel: absolute circuit depth — the number the reader can act on ("trust it to depth ~15").
right panel: depth / num_layers — the fair cross-molecule view, since the ansatz cap differs
             (LiH-4q 40 gates; LiH-6q and BeH2-8q 50).

⚠ MEASURABILITY GATE (added 2026-07-28 — an earlier version of this figure was WRONG without it)
------------------------------------------------------------------------------------------------
A cross-circuit Spearman is only defined if the 15 prefixes actually differ in real post-VQE error.
On BeH2-8q they do not: the actor's rollout is already at the ansatz plateau 2.175 mHa (exactly the
value the main table reports for Full) by circuit depth 4, only 3 distinct endpoint levels exist
across the 15 prefixes, and 3 of 5 seeds have EXACTLY zero cross-prefix variance. BeH2-10q plateaus
at depth 5 with 1/5 usable seeds. Their near-zero fidelities are therefore NOT "the world model ranks
at chance" — there was nothing to rank. Drawing them as a low line asserts a finding that the data
cannot support, so they are excluded from the axes and named in the exclusion box instead.
Gate: a molecule is plotted only if >= 3 of 5 seeds have cross-prefix spread > 1e-6, measured on the
advfid traces (`plateau_diagnostic.py`). The plateau is a property of molecule+policy, so the verdict
transfers to the horizon endpoints even though those store no raw values — stated as the inference it is.

OTHER KNOWN LIMITS (printed under the figure, and they are real)
----------------------------------------------------------------
* 4 depth points, all reached from ONE start depth. The start-depth sweep (`horizon_sd{10,15,20}`)
  found no detectable difference between reaching a depth via different H (paired p = 0.06-0.82,
  n=25), i.e. consistent with depth alone explaining the curve.
* n = 15 prefixes per (seed, depth); the per-point spread is large. Read the trend, not a point.

Design notes (dataviz method): form = change over a continuous variable, 3 categorical series ->
multi-line, two views of the same data. Colour = categorical identity, slots 1-3 of the validated
reference palette (#2a78d6, #eb6834, #1baf7a), checked ALL-PAIRS with the ported validator: worst CVD
dE 9.2 (target 8), normal-vision 24.0 (floor 15). The aqua slot WARNs on contrast (2.74 < 3.0), which
per the method obliges visible labels — hence the direct end-labels, which also carry identity without
relying on colour, alongside distinct markers.

Usage: python analysis/plot_wm_accuracy_by_depth.py
Outputs: wm_accuracy_by_depth.{pdf,png}
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/outputs/main_results"
OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CKPT = "final"   # the DEPLOYED model — "can we trust it" is a question about the end of training
HS = [1, 5, 10, 15]
# (display, hex, marker, num_layers, sd = num_layers//4)
MOL = {
    "LiH4q":   ("LiH (4q)",     "#2a78d6", "o", 40, 10),
    "BeH2":    ("BeH$_2$ (6q)", "#eb6834", "s", 50, 12),
    "LiH6q":   ("LiH (6q)",     "#1baf7a", "^", 50, 12),
    "BeH2_8q": ("BeH$_2$ (8q)", "#eda100", "D", 50, 12),
    "BeH2_10q":("BeH$_2$ (10q)", "#4a3aa7", "v", 50, 12),
}
MIN_USABLE = 3      # seeds with a non-degenerate ranking target required to plot a molecule at all


def per_seed(mol):
    """-> [seed, H] fidelity matrix from the stored probe; incomplete runs are skipped."""
    rows = []
    for f in sorted(glob.glob(f"{OF}/dreamqas/gru_energy_surrogate_{mol}_s*_of/t1a_probe.json")):
        hz = json.load(open(f)).get(CKPT, {}).get("horizon", {})
        row = [hz.get(str(H), {}).get("fidelity") for H in HS]
        if all(v is not None and np.isfinite(v) for v in row):
            rows.append([float(v) for v in row])
    return np.asarray(rows, float)


def usable_seeds(mol):
    """Seeds whose cross-prefix ranking target actually varies. A Spearman computed on a degenerate
    target is undefined, not low — see plateau_diagnostic.py."""
    k = 0
    for f in sorted(glob.glob(f"{OF}/dreamqas/gru_energy_surrogate_{mol}_s*_of/t1a_probe.json")):
        d = json.load(open(f)).get(CKPT, {}).get("advfid")
        if not d or not d.get("traces"):
            continue
        ends = np.array([t["phi_true"][-1] for t in d["traces"]], float)
        if ends.std() > 1e-6:
            k += 1
    return k


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    excluded = []
    for mol, (disp, col, mk, nl, sd) in MOL.items():
        M = per_seed(mol)
        u = usable_seeds(mol)
        if u < MIN_USABLE:
            # Degenerate ranking target: the real error is pinned at the ansatz plateau, so a Spearman
            # over these prefixes is undefined, not low. Naming it is honest; drawing it is not.
            excluded.append(f"{disp} ({u}/5 usable)")
            print(f"  [excluded] {mol}: only {u}/5 seeds have a non-degenerate ranking target")
            continue
        if len(M) < 2:
            print(f"  [skip] {mol}: {len(M)} evaluated seeds")
            continue
        mu, sderr = M.mean(0), M.std(0, ddof=1)
        depth = np.array([sd + H for H in HS], float)
        for ax, x in ((axA, depth), (axB, depth / nl)):
            ax.errorbar(x, mu, yerr=sderr, color=col, marker=mk, ms=8, mew=1.4, mfc=col, mec="white",
                        lw=2.0, capsize=3, elinewidth=1.2, zorder=4)
            ax.annotate(disp, (x[-1], mu[-1]), textcoords="offset points", xytext=(7, -1),
                        fontsize=8.5, color=col, va="center", zorder=6)
        print(f"  {mol:9s} n={len(M)} usable={u}/5 depth {depth.astype(int).tolist()}  fidelity "
              + " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(mu, sderr)))
    if excluded:
        axA.text(0.02, 0.03, "not measurable — ansatz plateau:\n" + "\n".join("  · " + e for e in excluded),
                 transform=axA.transAxes, fontsize=7.6, color="#a01d1d", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fdf0ef", ec="#e0b4b0", lw=0.8), zorder=7)
    for ax, lab, note in ((axA, "circuit depth (gates)", "ansatz caps: 40 (4q) / 50 (others)"),
                          (axB, "relative depth  (depth / ansatz cap)", "fair cross-molecule view")):
        ax.axhline(0.0, color="#999", ls=":", lw=0.9, zorder=1)
        ax.set_xlabel(lab, fontsize=10)
        ax.grid(True, ls="-", lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(note, fontsize=8.5, color="#666")
        ax.margins(x=0.16)
    axA.set_ylabel("WM endpoint ranking fidelity\n(Spearman across 15 held-out prefixes)", fontsize=9.5)
    axA.set_ylim(-0.15, 1.02)
    fig.suptitle("How deep can the world model's ranking be trusted?  (oracle-free, FINAL checkpoint)",
                 fontsize=11)
    fig.text(0.5, 0.012,
             "mean$\\pm$std over 5 seeds  ·  $n$=15 prefixes per (seed, depth)  ·  "
             "molecules whose true error is pinned at the ansatz plateau are excluded, not plotted low  ·  "
             "start-depth sweep found no path effect (paired $p$=0.06-0.82)",
             ha="center", va="bottom", fontsize=6.9, color="#666")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/wm_accuracy_by_depth.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[written] wm_accuracy_by_depth.{{pdf,png}} in {OUT}")


if __name__ == "__main__":
    main()
