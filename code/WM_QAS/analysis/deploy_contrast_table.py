"""Same WM, different deployment — aggregate table (reviewer-Q1, the "why not just rank with it?" control).

Per run, `wm_deploy_contrast.py` froze ONE checkpoint (a NoDAG run, i.e. a WM trained WITH imagination)
and evaluated three deployments of that single model — the learned actor, a one-step pessimistic
WM-greedy selector (eps=0 and eps>0) — under an identical rollout harness; `wm_beam_eval.py` added a
width-10 beam over the same frozen WM. This aggregates them and pairs by seed.

Why this control exists: the end-to-end contrast (`wmgreedy_threeway.txt`) compares whole PIPELINES, so a
reviewer can answer "your greedy arm just had a worse world model". Here the model is IDENTICAL across
arms and only the decision rule changes, which is the question actually being asked.

Usage: python analysis/deploy_contrast_table.py > outputs/main_results/deploy_contrast.txt
"""
import glob
import json
import os
import re

import numpy as np

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1/nodag"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}


def load(mol):
    out = {}
    for d in sorted(glob.glob(f"{C}/*_{mol}_s*_of_nodag")):
        p = f"{d}/wm_deploy_contrast.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        s = int(re.search(r"_s(\d+)_", os.path.basename(d)).group(1))
        b = f"{d}/beam_eval.json"
        r["beam"] = json.load(open(b))["bo1_mHa"] if os.path.exists(b) else None
        out[s] = r
    return out


def main():
    rng = np.random.default_rng(7)
    print("=" * 104)
    print("SAME FROZEN WM, DIFFERENT DEPLOYMENT  (oracle-free NoDAG checkpoints — a WM trained WITH imagination)")
    print("  best-of-1 (mHa, lower better).  actor / greedy-eps = mean over 100 frozen rollouts;")
    print("  greedy0 = the single deterministic circuit; beam = width-10 search, zero real VQE inside the search.")
    print("  The MODEL is identical across all four columns; only the rule that turns it into an action differs.")
    print("=" * 104)
    for mol in MOLS:
        runs = load(mol)
        if not runs:
            print(f"\n--- {DISP[mol]}: not evaluated yet"); continue
        ss = sorted(runs)
        A = np.array([runs[s]["actor"]["bo1"] for s in ss])
        G0 = np.array([runs[s]["greedy0"]["bo1"] for s in ss])
        GE = np.array([runs[s]["greedyE"]["bo1"] for s in ss])
        BMl = [runs[s]["beam"] for s in ss]
        BM = np.array([b for b in BMl if b is not None])
        ep = {runs[s]["episode"] for s in ss}
        bad = [s for s in ss if not runs[s]["greedy0"]["deterministic"] or runs[s]["greedy_calls"] == 0]
        print(f"\n--- {DISP[mol]}   seeds={len(ss)}  checkpoint ep{sorted(ep)}  "
              f"(self-checks {'ALL PASS' if not bad else 'FAILED on seeds ' + str(bad)})")
        if mol == "LiH6q":
            print("      note: LiH6q NoDAG runs to its 2x budget, so this is ep30000, not the ep15000 the")
            print("      main table reports. Harmless here — all four arms share the SAME checkpoint.")
        print(f"{'seed':>5}{'actor':>11}{'greedy0':>11}{'greedy eps':>12}{'beam B=10':>12}")
        for i, s in enumerate(ss):
            bm = f"{BMl[i]:.3f}" if BMl[i] is not None else "—"
            print(f"{s:>5}{A[i]:>11.3f}{G0[i]:>11.3f}{GE[i]:>12.3f}{bm:>12}")

        def ms(x):
            return f"{x.mean():.3f}±{x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
        print(f"{'mean':>5}{ms(A):>13}{ms(G0):>15}{ms(GE):>14}{(ms(BM) if len(BM) else '—'):>14}")
        print(f"\n{'contrast':22s}{'n':>4}{'paired Δ':>11}{'95% CI':>21}{'actor better':>14}   verdict")
        pairs = [("actor − greedy0", G0), ("actor − greedy eps", GE)]
        if len(BM) == len(A):
            pairs.append(("actor − beam B=10", BM))
        for lab, X in pairs:
            d = A - X
            bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI crosses 0 → n.s."
            ci = f"[{lo:+.3f}, {hi:+.3f}]"
            print(f"{lab:22s}{len(d):>4}{d.mean():>+11.3f}{ci:>21}{f'{int((d < 0).sum())}/{len(d)}':>14}   {sig}")
        if len(BM) > 1 and np.std(G0, ddof=1) < 1e-9 and np.std(BM, ddof=1) < 1e-9:
            print("  NOTE: greedy0 and beam are identical across seeds -> the deterministic search collapses")
            print("        to the SAME circuit regardless of the WM weights (a structural attractor).")
    print("\n" + "=" * 104)
    print("READ: Δ<0 means the learned actor has the lower error. This is the control that answers")
    print("'why not just use the WM as a surrogate ranker' — the WM here is the GOOD one (imagination-trained),")
    print("and it is handed to greedy/beam unchanged. Saturated molecules (all arms pinned at the ansatz")
    print("floor) carry no information and must not be cited as support either way.")


if __name__ == "__main__":
    main()
