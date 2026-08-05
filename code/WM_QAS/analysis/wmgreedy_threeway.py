"""RQ3 capability ladder: NoImag -> WM-Greedy (pessimistic AND optimistic) -> DreamQAS-NoDAG.

Same frozen world model throughout; only its USE differs. Answers two reviewer objections at once:

  (a) "is a plain surrogate ranker enough?"  -> WM-Greedy scores every legal next gate with the WM and
      commits the best one. If that recovered imagined policy learning, the imagination machinery would
      be unnecessary.
  (b) "you crippled the baseline with pessimism."  -> the OPTIMISTIC arms (beta < 0, i.e. commit the
      candidate the WM rates best rather than the one that survives a pessimistic penalty) are the
      steelman. They were trained and are reported here; optimism does not rescue greedy.

⚠ TWO PROVENANCE FIXES MADE 2026-07-28, both of which changed published numbers:
  1. This script was previously `scratchpad/wmg3_threeway.py` — a SESSION-SCOPED temp directory, which
     the lineage audit cited as the generator of a paper artifact. It is now a tracked analysis script.
  2. It hard-coded FINAL_EP = {"LiH6q": 30000}, i.e. LiH-6q was reported at its 2x FULL budget while the
     main table reports every task at the locked 15,000-episode budget (storyline §1). Worse, only some
     arms had a 15,000-episode evaluation, so an unguarded "latest checkpoint" read MIXED budgets across
     arms of the same comparison. All five arms were re-evaluated at ep15000 and the script now REFUSES
     to compare arms whose checkpoint episodes differ.

Metric: frozen best-of-1 = mean episode-best over the checkpoint's 100 eval episodes, per seed; then
mean +- sample std (ddof=1) over 5 seeds. Paired-Δ over shared seeds with a percentile bootstrap.

Usage: python analysis/wmgreedy_threeway.py > outputs/main_results/wmgreedy_threeway.txt
"""
import glob
import json
import os

import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
COMMON_EP = 15000                       # locked reporting budget for EVERY task (storyline §1)
ARMS = [
    ("NoImag",                 f"{C}/ablations/gru_energy_none_{{m}}_s*_of_noimag"),
    ("WM-Greedy  beta=+1",     f"{C}/wmgreedy/gru_energy_none_{{m}}_s*_of_wmgreedy"),
    ("WM-Greedy  beta=-1 opt", f"{C}/wmgreedy_opt/gru_energy_none_{{m}}_s*_of_wmgopt1"),
    ("WM-Greedy  beta=-2 opt", f"{C}/wmgreedy_opt/gru_energy_none_{{m}}_s*_of_wmgopt2"),
    ("DreamQAS-NoDAG",         f"{C}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag"),
]
CONTRASTS = [("NoImag", "WM-Greedy  beta=+1"), ("WM-Greedy  beta=+1", "WM-Greedy  beta=-1 opt"),
             ("WM-Greedy  beta=+1", "DreamQAS-NoDAG"), ("NoImag", "DreamQAS-NoDAG")]


def seed_bo1(pattern, mol):
    """-> {seed: best-of-1} strictly at COMMON_EP. A run without that evaluation is DROPPED, never
    silently replaced by whatever checkpoint it does have — that is how the budget mixing happened."""
    out = {}
    for d in sorted(glob.glob(pattern.format(m=mol))):
        f = f"{d}/eval_traces_ep{COMMON_EP}.jsonl"
        rows = []
        if os.path.exists(f):
            rows = [json.loads(l) for l in open(f) if l.strip()]
        else:                                    # fall back to the row INSIDE the full grid, matched by
            g = f"{d}/eval_traces.jsonl"         # checkpoint episode — never "the deepest one"
            if os.path.exists(g):
                rows = [r for r in (json.loads(l) for l in open(g) if l.strip())
                        if int(r.get("_ck_ep", r.get("episode", -1))) == COMMON_EP]
        rows = [r for r in rows if r.get("ep_best")]
        if not rows:
            continue
        s = os.path.basename(d).split("_s")[-1].split("_")[0]
        e = np.array([x for x in rows[-1]["ep_best"] if np.isfinite(x)], float)
        if e.size:
            out[s] = float(e.mean())
    return out


def paired(a, b, rng, n=10000):
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    d = np.array([b[k] - a[k] for k in ks])           # b - a: negative = the SECOND arm is better
    bs = [rng.choice(d, len(d), True).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(ks), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                wins=int((d < 0).sum()))


def main():
    rng = np.random.default_rng(7)
    print("=" * 104)
    print("RQ3 CAPABILITY LADDER — the SAME frozen WM, three ways of using it")
    print(f"  frozen best-of-1 (mHa, lower better), 100 eval episodes/seed, ALL arms at ep{COMMON_EP:,}")
    print("  (the locked common budget — LiH-6q included; arms lacking that evaluation are dropped,")
    print("   never substituted, so a comparison can never mix budgets).")
    print("=" * 104)
    data = {m: {n: seed_bo1(p, m) for n, p in ARMS} for m in MOLS}
    print(f"{'arm':26s}" + "".join(f"{DISP[m]:>18}" for m in MOLS))
    print("-" * 104)
    for n, _ in ARMS:
        cells = []
        for m in MOLS:
            v = list(data[m][n].values())
            cells.append("—" if not v else
                         (f"{np.mean(v):.3g}±{np.std(v, ddof=1):.2g} ({len(v)})" if len(v) > 1
                          else f"{v[0]:.3g} (1)"))
        print(f"{n:26s}" + "".join(f"{c:>18}" for c in cells))
    print("-" * 104)
    print("(n) = seeds with an ep15000 evaluation. beta=-2 was only launched on LiH-6q.")

    print(f"\n{'contrast (b − a)':46s}{'task':10s}{'n':>3}{'paired Δ':>11}{'95% CI':>20}{'b better':>10}")
    print("-" * 104)
    for a, b in CONTRASTS:
        for m in MOLS:
            r = paired(data[m][a], data[m][b], rng)
            if r is None:
                continue
            ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            sig = "" if (r["lo"] <= 0 <= r["hi"]) else " *"
            print(f"{a + '  ->  ' + b:46s}{DISP[m]:10s}{r['n']:>3}{r['mean']:>+11.3f}{ci:>20}"
                  f"{f'{r[chr(119) + chr(105) + chr(110) + chr(115)]}/{r[chr(110)]}':>10}{sig}")
    print("-" * 104)
    print("  Δ = b − a, so Δ<0 means the SECOND (more capable) arm is better. * = CI excludes 0.")
    print("\nREADING")
    print("  * If WM-Greedy recovered imagined policy learning, NoImag -> WM-Greedy would be strongly")
    print("    negative and WM-Greedy -> NoDAG would be ~0. The data show the opposite ordering.")
    print("  * The optimistic arms exist to answer 'you crippled the baseline with pessimism'. If")
    print("    beta=+1 -> beta=-1 is ~0, pessimism was not what made greedy fail.")
    print("  * BeH2-8q sits at its ansatz floor (2.17) for every arm — a saturated cell that carries no")
    print("    method information. See plateau_diagnostic.txt; do not read differences there.")


if __name__ == "__main__":
    main()
