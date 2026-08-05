"""Cheap DAgger rigor check (read-only, existing trajectory.jsonl): does DAgger help VQE-EFFICIENCY
even though it is ~neutral on final best-of-1? Full vs noDAG, seed-level VQE-to-common-target.

Per molecule: common target Q = max(median final-window best-of-1 of Full, of noDAG) so BOTH can
reach it. Per seed, first cumulative training VQE at which the window-smoothed episode-best stays
<= Q for 3 reported windows (sustained). Compare seed-level median VQE(Full) vs VQE(noDAG).
If Full reaches Q with materially fewer VQE -> DAgger buys efficiency; else DAgger is neutral here too.
"""
import glob
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_training_window import episodes_from_trajectory, run_curve  # reuse exact parsing

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}


def curves(stage, mol):
    if stage == "Full":
        ds = glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q")
    else:  # noDAG
        ds = glob.glob(f"{C}/ablations/gru_energy_surrogate_{mol}_s*_ab_noDAG")
    out = []
    for d in ds:
        eps = episodes_from_trajectory(d)
        if eps and len(eps) > 400:
            out.append(run_curve(eps))       # (cum_vqe[], window-best-of-1[])
    return out


def final_q(cvs):
    """median over seeds of each seed's final-window (last 10%) mean quality."""
    fq = [float(np.mean(y[int(len(y) * 0.9):])) for _, y in cvs]
    return np.median(fq) if fq else np.nan


def vqe_to(cv, Q, sustain=3):
    x, y = cv
    for i in range(len(y) - sustain + 1):
        if all(y[i + r] <= Q * 1.0001 for r in range(sustain)):
            return float(x[i])
    return np.nan          # never reached (right-censored)


lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 92)
P("DAgger VQE-EFFICIENCY CHECK — Full vs noDAG, seed-level VQE to a common target Q (mHa)")
P("  Q = max(median final-window best-of-1 of the two) so both can reach it; sustained crossing.")
P("  ratio = median VQE(noDAG) / median VQE(Full);  >1 = DAgger reaches Q with FEWER VQE (helps).")
P("=" * 92)
P(f"{'molecule':<11}{'target Q':>10}{'VQE Full(med)':>16}{'VQE noDAG(med)':>16}{'noDAG/Full':>12}{'reached F/nD':>14}")
P("-" * 92)
for mol in MOLS:
    cf, cn = curves("Full", mol), curves("noDAG", mol)
    if not cf or not cn:
        P(f"{DISP[mol]:<11}  (missing curves)"); continue
    Q = max(final_q(cf), final_q(cn))
    vf = np.array([vqe_to(c, Q) for c in cf]); vn = np.array([vqe_to(c, Q) for c in cn])
    rf, rn = np.isfinite(vf), np.isfinite(vn)
    mf = np.median(vf[rf]) if rf.any() else np.nan
    mn = np.median(vn[rn]) if rn.any() else np.nan
    ratio = mn / mf if (np.isfinite(mf) and np.isfinite(mn) and mf > 0) else np.nan
    P(f"{DISP[mol]:<11}{Q:>10.3f}{mf:>16.0f}{mn:>16.0f}{ratio:>12.2f}{f'{rf.sum()}/{rn.sum()}':>14}")
P("-" * 92)
P("READ: if noDAG/Full ~1 across molecules, DAgger is neutral on VQE-efficiency TOO -> lock the")
P("  'verification-mechanism-not-performance-driver' framing. If >1.3 somewhere, DAgger buys")
P("  efficiency there and the framing must credit that. (right-censored seeds excluded from median.)")

OUT = f"{HERE}/outputs/main_results/dagger_efficiency.txt"
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
