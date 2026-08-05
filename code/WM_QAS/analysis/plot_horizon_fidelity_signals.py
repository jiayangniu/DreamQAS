"""Appendix figure — WM ranking accuracy vs CIRCUIT DEPTH, under both training signals.

⚠ This is NOT a horizon figure, despite the historical name of the underlying probe. Imagined circuit
dynamics are exact and the WM re-encodes the whole action sequence for every prediction (imagine.py
_encode_seeds), so the "H-step endpoint" is simply a circuit of depth sd+H. Depth is the x variable;
H is shown only as the knob that produced it. The canonical Figure 3a argument -- "ranking does not
degrade with H, therefore no step-wise accumulation of learned-transition error" -- is an
architectural tautology (accumulation is impossible here) and is RETRACTED, not supported.

Puts the oracle-free and canonical curves on the SAME axes, per task, so the reader can see how much
of the apparent difference is real. It is LESS than it looks:

    canonical    pooled slope -0.0062 +- 0.0213 /H, p=0.28   -> n.s. vs 0
    oracle-free  pooled slope -0.0185 +- 0.0224 /H, p=0.0064 -> significant vs 0
    BETWEEN the two arms: Welch t, p=0.134 -> NOT DISTINGUISHABLE (Cohen's d 0.56; ~50 seeds/arm
    would be needed for 80% power, we have 15).

⚠ The figure must therefore NOT be captioned "the two signals reach opposite conclusions" — that is
the difference-of-significance fallacy, and an earlier draft of these docs made exactly that error.
What it shows is a within-arm trend on the oracle-free side that the canonical arm is too noisy to
confirm or refute.

⚠ Second limit, in the caption too: H is PERFECTLY COLLINEAR with endpoint circuit depth. Every
prefix is drawn at a fixed depth sd = num_layers // 4 (LiH-4q 10; LiH-6q / BeH2-8q 12), so the
endpoint sits at depth sd + H. "Ranking degrades with horizon" and "ranking degrades with depth" are
the same measurement here; separating them needs the probe re-run from several starting depths.

fidelity(H) = Spearman over held-out shallow prefixes between the WM's predicted endpoint log-error
and the real post-VQE endpoint log-error, at the 1/4-budget checkpoint. The H sweep walks ONE
actor-sampled path per prefix (torch seed fixed per prefix, not per H), so H=1/5/10/15 are NESTED
points on the same path — that removes sampling noise, but not the depth confound. All statistics
live in `oracle_free_horizon_slope.txt` (analysis/of_horizon_slope.py); this only draws them.

⚠ Comparability: both axes are RANK statistics, and the two training targets differ by a strictly
monotone reparameterisation, so the curves ARE comparable. Absolute calibration MAE is not — never
put the two signals' MAE on one axis.

Design notes (dataviz method):
  form   — change over a continuous variable, 2 categorical series, 3 facets -> small multiples.
  color  — categorical/identity, 2 slots from the validated reference palette (#2a78d6, #eb6834);
           checked with the ported validator: CVD dE 24.7 (target 8), normal-vision 33.6 (floor 15),
           lightness band + chroma floor + contrast all pass in light mode.
  a11y   — identity is never colour-alone: solid/filled circle = oracle-free, dashed/open square =
           canonical; every panel is also direct-labelled with its slope and p.

Usage: python analysis/plot_horizon_fidelity_signals.py
Outputs: horizon_fidelity_signals.{pdf,png}
"""
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/outputs/main_results"
OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CV = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas"
CKPT = "quarter"
HS = [1, 5, 10, 15]
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH (4q)", "LiH6q": "LiH (6q)", "BeH2_8q": "BeH$_2$ (8q)"}
# prefix start depth sd = num_layers // 4 -> endpoint depth = sd + H. Drawn as a second x-axis so the
# collinearity between horizon and circuit depth is visible in the figure, not buried in the caption.
SD = {"LiH4q": 10, "LiH6q": 12, "BeH2_8q": 12}
# (display, hex, marker, facecolor, linestyle, dx)  — colour + shape + line style, never colour alone
SIG = {
    "oracle-free": ("oracle-free (paper's method)", "#2a78d6", "o", "#2a78d6", "-", -0.22),
    "canonical":   ("canonical $E_0$ reward",       "#eb6834", "s", "none",    "--", 0.22),
}
SRC = {"oracle-free": f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of/t1a_probe.json",
       "canonical":   f"{CV}/gru_energy_surrogate_{{m}}_s*_q/t1a_probe.json"}


def per_seed(pattern):
    """-> [seed, H] fidelity matrix; runs whose probe lacks a complete horizon block are skipped."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        hz = json.load(open(f)).get(CKPT, {}).get("horizon", {})
        row = [hz.get(str(H), {}).get("fidelity") for H in HS]
        if all(v is not None and np.isfinite(v) for v in row):
            rows.append([float(v) for v in row])
    return np.asarray(rows, float)


def main():
    os.makedirs(OUT, exist_ok=True)
    Hf = np.asarray(HS, float)
    fig, AX = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True)
    handles = {}
    pooled = {k: [] for k in SIG}
    for ax, mol in zip(AX, MOLS):
        for key, (disp, col, mk, fc, ls, dx) in SIG.items():
            M = per_seed(SRC[key].format(m=mol))
            if len(M) < 2:
                continue
            mu, sd = M.mean(0), M.std(0, ddof=1)
            slopes = np.array([np.polyfit(Hf, r, 1)[0] for r in M])
            pooled[key].extend(slopes)
            p = float(stats.ttest_1samp(slopes, 0.0).pvalue)
            ln = ax.errorbar(Hf + dx, mu, yerr=sd, color=col, marker=mk, mfc=fc, mec=col,
                             ms=7, mew=1.6, lw=0, capsize=3, elinewidth=1.4, zorder=4)
            b, a0 = np.polyfit(Hf, mu, 1)
            ax.plot(Hf, a0 + b * Hf, color=col, ls=ls, lw=2.0, alpha=0.75, zorder=3)
            handles[key] = ln
            # direct label: the statistic, not a number on every point. Lives in the reserved band
            # below the data (ylim is extended for it), so it can never sit on top of a marker.
            y = 0.085 if key == "oracle-free" else 0.015
            star = "*" if p < 0.05 else ""
            ax.text(0.97, y, f"slope {np.mean(slopes):+.3f}/$H$   $p$={p:.3f}{star}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6, color=col)
        ax.axhline(0.0, color="#999", ls=":", lw=0.9, zorder=1)
        # ONE x-axis. Depth is the real variable: dynamics are exact and the WM re-encodes the whole
        # sequence, so the "H-step endpoint" is just a circuit of depth sd+H. H is shown underneath as
        # the knob that produced it — a second axis would imply two variables, and there is only one.
        ax.set_xticks(HS)
        ax.set_xticklabels([f"{SD[mol] + H}\n($H$={H})" for H in HS], fontsize=8.5)
        ax.set_xlabel("endpoint circuit depth", fontsize=9.5)
        ax.set_title(DISP[mol], fontsize=10.5)
        ax.grid(True, ls="-", lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)
    AX[0].set_ylabel("endpoint ranking fidelity\n(Spearman)", fontsize=9.5)
    AX[0].set_ylim(-0.46, 1.05)                    # lower band reserved for the slope labels
    AX[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.legend([handles[k] for k in SIG if k in handles],
               [SIG[k][0] for k in SIG if k in handles],
               loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.995))
    pl = {k: np.asarray(v) for k, v in pooled.items() if v}
    note = "   ·   ".join(
        f"{k}: pooled {v.mean():+.4f}$\\pm${v.std(ddof=1):.4f}/$H$, "
        f"$p$={stats.ttest_1samp(v, 0.0).pvalue:.4f}" for k, v in pl.items())
    fig.text(0.5, 0.072, note, ha="center", va="bottom", fontsize=8, color="#333")
    # The headline caveat goes ON the figure, not only in the caption: within-arm significance is not
    # a between-arm difference, and the two arms are NOT separable at this sample size.
    if len(pl) == 2:
        a, b = pl["oracle-free"], pl["canonical"]
        pb = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        fig.text(0.5, 0.034, f"between the two signals: Welch $t$, $p$={pb:.3f} — "
                             f"NOT distinguishable at 5 seeds/task", ha="center", va="bottom",
                 fontsize=8, color="#a01d1d")
    fig.text(0.5, 0.002, "1/4-budget checkpoint  ·  mean$\\pm$std over 5 seeds  ·  * = $p<0.05$ vs 0  ·  "
                         "exact dynamics + full-sequence re-encoding $\\Rightarrow$ $H$ IS depth, "
                         "not a horizon",
             ha="center", va="bottom", fontsize=6.8, color="#666")
    fig.tight_layout(rect=(0, 0.115, 1, 0.925))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/horizon_fidelity_signals.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[written] horizon_fidelity_signals.{{pdf,png}} in {OUT}")
    for k, v in pl.items():
        print(f"  {k:12s} pooled slope {v.mean():+.4f} ± {v.std(ddof=1):.4f} "
              f"p={stats.ttest_1samp(v, 0.0).pvalue:.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
