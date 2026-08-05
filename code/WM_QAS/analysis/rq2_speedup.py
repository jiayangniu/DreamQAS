"""RQ2 seed-level VQE speedup (item 4): per task-baseline, seed-level crossing to a COMMON target
Q_b, with reached/right-censored counts and a bootstrap CI on the speedup ratio. Replaces reading
the multiplier off the aggregate curve.

Q_b = median over the BASELINE's seeds of each seed's final-window (last 10%) training-window best-of-1.
Per seed of each method, first cumulative training VQE at which the window-smoothed episode-best stays
<= Q_b for `sustain` consecutive reported windows (sustained crossing). Seeds that never reach Q_b are
RIGHT-CENSORED (excluded from the median, counted separately -- NOT imputed to the final budget).
speedup = median VQE(baseline, reached) / median VQE(Full, reached); bootstrap CI resamples seeds.
"""
import glob
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_training_window import episodes_from_trajectory, episodes_from_traces, run_curve, PMK

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
TASKS = ["LiH4q", "LiH6q", "BeH2_8q"]                     # main-text Figure A tasks
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
BASES = ["No-imag", "RLQAS", "CRLQAS", "HyRLQAS"]
OUT = f"{HERE}/outputs/main_results/rq2_speedup.txt"


def curves(method, mol):
    """-> list of (cum_vqe[], window-best-of-1[]) per seed."""
    if method == "Full":
        ds, parse = glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q"), episodes_from_trajectory
    elif method == "No-imag":
        ds, parse = glob.glob(f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag"), episodes_from_trajectory
    elif method == "RLQAS":
        # RQ2 needs the training-window curve -> the trajectory-capturing re-run (dreamqas_rlqas_traj),
        # NOT the old dreamqas_rlqas (which only has frozen eval_traces, no trajectory.jsonl).
        ds, parse = glob.glob(f"{C}/dreamqas_rlqas_traj/baseline_analysis/G0_v2_{mol}/seed_*"), episodes_from_trajectory
    else:
        pdir = method.lower()
        root = f"{C}/psqas_hyrlqas_std" if method == "HyRLQAS" else f"{C}/psqas"
        ds, parse = glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*"), episodes_from_traces
    out = []
    for d in ds:
        eps = parse(d)
        if eps and len(eps) > 400:
            out.append(run_curve(eps))
    return out


def final_window_q(cvs):
    return [float(np.mean(y[int(len(y) * 0.9):])) for _, y in cvs]


def vqe_to(cv, Q, sustain=3):
    x, y = cv
    for i in range(len(y) - sustain + 1):
        if all(y[i + r] <= Q * 1.0001 for r in range(sustain)):
            return float(x[i])
    return np.nan


lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 108)
P("RQ2  SEED-LEVEL VQE SPEEDUP (Full vs each baseline) — common target Q_b = baseline final-window median")
P("  per-seed sustained crossing; right-censored seeds excluded from median (counted, NOT imputed).")
P("  speedup = median VQE(baseline) / median VQE(Full);  [a,b] = 95% bootstrap CI (resample seeds).")
P("=" * 108)
P(f"{'task':<10}{'baseline':<9}{'Q_b(mHa)':>10}{'Full VQE med':>14}{'F reach':>9}{'base VQE med':>14}{'b reach':>9}{'speedup':>10}{'95% CI':>16}")
P("-" * 108)
rng = np.random.default_rng(7)
for mol in TASKS:
    cf = curves("Full", mol)
    for b in BASES:
        cb = curves(b, mol)
        if len(cf) < 2 or len(cb) < 2:
            P(f"{DISP[mol]:<10}{b:<9}{'  (missing / incomplete)':>10}"); continue
        Q = float(np.median(final_window_q(cb)))                       # common target = baseline's floor
        vf = np.array([vqe_to(c, Q) for c in cf]); vb = np.array([vqe_to(c, Q) for c in cb])
        rf, rb = vf[np.isfinite(vf)], vb[np.isfinite(vb)]
        if len(rf) == 0 or len(rb) == 0:
            P(f"{DISP[mol]:<10}{b:<9}{Q:>10.3f}{'  - (Full never reaches Q_b)':>14}"); continue
        mf, mb = np.median(rf), np.median(rb)
        sp = mb / mf
        # bootstrap CI on ratio-of-medians (resample reached seeds independently)
        bs = []
        for _ in range(3000):
            a = np.median(rng.choice(rf, len(rf), replace=True)); c = np.median(rng.choice(rb, len(rb), replace=True))
            bs.append(c / a)
        ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        P(f"{DISP[mol]:<10}{b:<9}{Q:>10.3f}{mf:>14.0f}{f'{len(rf)}/{len(cf)}':>9}{mb:>14.0f}"
          f"{f'{len(rb)}/{len(cb)}':>9}{sp:>9.1f}x{f'[{ci[0]:.1f},{ci[1]:.1f}]':>16}")
    P("-" * 108)
P("NOTE: HyRLQAS on LiH6q may be incomplete (fixed-refill still running) -> shown only when >=2 seeds.")
P("Use the seed-level speedup (median + bootstrap CI) + reach counts for the Figure A arrows and text,")
P("NOT a multiplier read off the aggregated mean curve.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
