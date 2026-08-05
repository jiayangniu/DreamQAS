"""Appendix figure: pairwise ranking fidelity (Acc_pair) vs training, five molecules.

Acc_pair is the WM's *gate* statistic, logged every `wm_refresh_every` iters into `fidelity.jsonl`
by `Runner.fidelity_check`: 50k uniformly-drawn PAIRS of held-out replay prefixes, scored by
    mean( sign(pred_i - pred_j) == sign(ref_i - ref_j) )   over pairs with distinct reference,
ref = true error (canonical) or raw energy (oracle-free — bijectively the same order). Chance = 0.5;
imagination is enabled only while Acc_pair >= cfg.fidelity_tau (0.70).

NOT the same as the T1.a within-prefix Spearman: that ranks the <=10 legal next actions of ONE prefix
(siblings differing by a single gate, chance = 0), which is the decision the agent actually faces and
is far harder. Do not present one as a proxy for the other.

Panels: two figures.
  A  oracle-free Full vs No-imag, 5 molecules  -> the paper's training signal.
  B  canonical component ablations, 5 molecules -> development evidence (the ablation grid exists only
     in campaign_v1). Missing cells are drawn as an explicit gap, never silently omitted.

Cross-seed MEDIAN + IQR band (same convention as the speed figures). x = real training episodes.

ABLATION COMPARABILITY (verified 2026-07-27, load-bearing for figure B — do not drop this note):
all four ablations are ONE ablation set, i.e. they share the reward function and the code era.
Checked, not assumed: (i) 13 reward-related config fields (reward, reward_kind, err_floor_mHa,
tail_w_*, maxk_*, pot_head, popart, ...) take a SINGLE value across every variant x molecule;
(ii) `oracle_free`/`score_margin_mHa` are ABSENT from every config -> all predate the oracle-free
reward work; (iii) all 25 noDAG runs share an identical 79-key config schema; (iv) each ablation
differs from Full in exactly its own mechanism flag (noUNC flips pessimism_beta AND imag_conf_tau,
which together ARE the uncertainty mechanism; `assert_full` is a launcher guard, not training
behaviour); (v) only two commits touched phase2_surrogate during the campaign — 9cd89d6 predates
every run, and 8ee90af only replaced `x.std(0)` with `_ens_std(x)`, which returns exactly `x.std(0)`
for K>1; every campaign run has K=3, so that change is a no-op here.
CAVEAT to keep in the caption: the BeH2-6q noDAG seeds were backfilled 2026-07-18 on a DIFFERENT
node than that molecule's Full seeds (2026-07-14). Reward and code are equivalent; the hardware
difference shows up as seed-level noise, not as a systematic reward change.

Usage: python analysis/plot_fidelity_curves.py [--out outputs/main_results]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CV = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH (4q)", "BeH2": "BeH$_2$ (6q)", "LiH6q": "LiH (6q)",
        "BeH2_8q": "BeH$_2$ (8q)", "BeH2_10q": "BeH$_2$ (10q)"}
TAU = 0.70                    # cfg.fidelity_tau — the imagination gate
COMMON_EPS = 15000            # locked common budget (LiH6q is the only 2x-budget task)

# (label, glob template, colour, linestyle)
SET_A = [("DreamQAS (Full)", f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of", "#d62728", "-"),
         ("No-imag", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag", "#ff7f0e", "-")]
# SET_C — the ORACLE-FREE component ablations. Completed 2026-07-28 and STRICTLY BETTER than SET_B:
# same training signal as the paper's method (SET_B is canonical = development evidence), complete
# 5 molecules x 5 seeds on all four arms (SET_B's BeH2-6q is missing -DIR and -uncertainty), and one
# code era throughout. Cite this one; SET_B stays only as the canonical sensitivity reference.
SET_C = [("Full", f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of", "#d62728", "-"),
         ("−DAgger", f"{OF}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag", "#1f77b4", "--"),
         ("−DIR reweight", f"{OF}/ablations/gru_energy_surrogate_{{m}}_s*_of_noDIR", "#2ca02c", "--"),
         ("−uncertainty", f"{OF}/ablations/gru_energy_surrogate_{{m}}_s*_of_noUNC", "#9467bd", "--"),
         ("−imagination", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag", "#8c564b", ":")]
SET_B = [("Full", f"{CV}/dreamqas/gru_energy_surrogate_{{m}}_s*_q", "#d62728", "-"),
         ("−DAgger", f"{CV}/ablations/gru_energy_surrogate_{{m}}_s*_ab_noDAG", "#1f77b4", "--"),
         ("−DIR reweight", f"{CV}/ablations/gru_energy_surrogate_{{m}}_s*_ab_noDIR", "#2ca02c", "--"),
         ("−uncertainty", f"{CV}/ablations/gru_energy_surrogate_{{m}}_s*_ab_noUNC", "#9467bd", "--"),
         ("−imagination", f"{CV}/ablations/gru_energy_none_{{m}}_s*_ab_noimag", "#8c564b", ":")]


def seed_curves(pattern, mol):
    """-> list of (episodes[], acc_pair[]) per seed."""
    out = []
    for d in sorted(glob.glob(pattern.format(m=mol))):
        f = f"{d}/fidelity.jsonl"
        if not os.path.exists(f):
            continue
        x, y = [], []
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("pairwise") is None:
                continue
            x.append(float(r.get("n_eps", r["iter"] * 4)))
            y.append(float(r["pairwise"]))
        if len(y) > 3:
            out.append((np.array(x), np.array(y)))
    return out


def _smooth(y, w):
    """Rolling median over w logged points (each point = wm_refresh_every iters = 80 episodes).
    Acc_pair is recomputed each refresh on a SHIFTING recent-validation slice, so the raw trace
    oscillates; the rolling median shows the trend without inventing one. w=1 disables it."""
    if w <= 1 or len(y) < w:
        return y
    return np.array([np.median(y[max(0, i - w + 1):i + 1]) for i in range(len(y))])


def aggregate(curves, smooth=1):
    if len(curves) < 2:
        return None
    L = min(len(y) for _x, y in curves)
    X = np.array([x[:L] for x, _y in curves])
    Y = np.array([_smooth(y[:L], smooth) for _x, y in curves])
    return np.median(X, 0), np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0), len(curves)


def draw(ax, mol, spec, smooth=1):
    handles, missing = {}, []
    for label, pat, col, ls in spec:
        agg = aggregate(seed_curves(pat, mol), smooth)
        if agg is None:
            missing.append(label)
            continue
        x, y, lo, hi, ns = agg
        ln, = ax.plot(x, y, color=col, ls=ls, lw=1.5, zorder=4, label=label)
        ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0, zorder=1)
        handles[label] = ln
    ax.axhline(TAU, color="gray", ls=":", lw=1, zorder=2)
    ax.axhline(0.5, color="k", ls="-", lw=0.8, alpha=0.35, zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(0.42, 1.02)
    ax.set_title(DISP[mol], fontsize=11, fontweight="bold")
    ax.set_xlabel("training episodes", fontsize=9)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    if missing:                       # never silently drop a variant — state the gap on the panel
        ax.text(0.03, 0.03, "not run: " + ", ".join(missing), transform=ax.transAxes,
                fontsize=6.5, color="#b00", va="bottom", ha="left")
    return handles


def build(spec, fname, title, out, smooth=1):
    fig, axes = plt.subplots(1, 5, figsize=(19.5, 3.5), sharey=True)
    handles = {}
    for ax, mol in zip(axes, MOLS):
        handles.update(draw(ax, mol, spec, smooth))
    axes[0].set_ylabel("pairwise ranking fidelity  Acc$_{pair}$", fontsize=10)
    axes[0].annotate("imagination gate $\\tau$=0.70", xy=(0.03, TAU), xycoords=("axes fraction", "data"),
                     fontsize=7, color="gray", va="bottom")
    axes[0].annotate("chance", xy=(0.03, 0.5), xycoords=("axes fraction", "data"),
                     fontsize=7, color="k", alpha=0.55, va="bottom")
    hs = [handles[k] for k, *_ in spec if k in handles]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=len(hs), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.14, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/{fname}.{ext}", dpi=(160 if ext == "png" else None), bbox_inches="tight")
    plt.close(fig)
    print(f"[written] {fname}.{{png,pdf}} in {out}")


def coverage(spec, label):
    print(f"\n--- seed coverage, {label} (blank = variant not run on that molecule)")
    print(f"{'variant':18s}" + "".join(f"{DISP[m].replace('$','').replace('_',''):>13}" for m in MOLS))
    for lab, pat, _c, _ls in spec:
        print(f"{lab:18s}" + "".join(f"{len(seed_curves(pat, m)):>13}" for m in MOLS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=5,
                    help="rolling-median window in logged points (1 = raw); 5 pts = 400 episodes")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "outputs", "main_results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sm = f" · rolling median {a.smooth} pts" if a.smooth > 1 else ""
    build(SET_A, "fidelity_curves_oraclefree",
          "Pairwise ranking fidelity vs training — oracle-free runs (cross-seed median $\\pm$ IQR, 5 seeds" + sm + ")",
          a.out, a.smooth)
    build(SET_B, "fidelity_curves_ablations",
          "Pairwise ranking fidelity vs training — WM component ablations, canonical campaign "
          "(development evidence; cross-seed median $\\pm$ IQR, 5 seeds" + sm + ")", a.out, a.smooth)
    build(SET_C, "fidelity_curves_ablations_oraclefree",
          "Pairwise ranking fidelity vs training — WM component ablations, ORACLE-FREE "
          "(cross-seed median $\\pm$ IQR, 5 seeds" + sm + ")", a.out, a.smooth)
    coverage(SET_A, "oracle-free")
    coverage(SET_C, "oracle-free ablations")
    coverage(SET_B, "canonical ablations (reference only)")


if __name__ == "__main__":
    main()
