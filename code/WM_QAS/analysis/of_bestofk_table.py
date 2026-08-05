"""Best-of-1 AND best-of-10 at the COMMON 15,000-episode budget — oracle-free arms + externals.

Two things this table fixes relative to `of_main_table.txt` (which is best-of-1 only):
  1. it reports the K=10 order statistic alongside K=1, for every method;
  2. it reads the EXTERNAL RL baselines on LiH-6q at their **episode-15,000** checkpoint instead of
     their 30,000 final, so the one task where CRLQAS/HyRLQAS were given 2x our budget is put on the
     same footing as everything else.

ESTIMATOR (locked, imported — not reimplemented): `policy_quality_table.bestofK`,
    E[min of K deploys] = sum_i e_(i) * [ ((n-i+1)/n)^K - ((n-i)/n)^K ]     over ascending e_(i)
K=1 reduces to the arithmetic mean by construction. Per-seed statistic first, then mean +- sample std
(ddof=1) over the 5 seeds. Episodes are NEVER pooled across seeds.

⚠ THE ASYMMETRY THIS TABLE CANNOT REMOVE, and why every cell prints its n.
`bestofK` is estimated from the n evaluation episodes stored at that checkpoint, and n is NOT constant:

    DreamQAS / No-imag / DreamQAS-RL   final + LiH-6q ep15000   n = 100
    CRLQAS / HyRLQAS                   final (ep15000/30000)    n = 100
    CRLQAS / HyRLQAS                   LiH-6q AT ep15000        n =  20   <-- intermediate ckpt
    GQE                                native final             n =  50
    TF-QAS                             native final             n = 100
    QuantumDARTS                       one delivered circuit    n =   1   (BO1 == BO10 by construction)

K=10 from n=20 puts ~40% of its weight on the single smallest observation (vs ~10% at n=100), so it is
far noisier there. The `--subsample` block quantifies that on OUR OWN runs by re-reducing them at
n=20, which is a sensitivity check on the estimator, not a substitute for missing data.

⚠ THE 15,000-EPISODE CORRECTION APPLIES ONLY TO CRLQAS AND HyRLQAS. GQE (transformer epochs), TF-QAS
(one-shot sample budget) and QuantumDARTS (DARTS epochs) have no environment-episode axis at all;
their LiH-6q runs already use the same native budget as their runs on every other task, so there is no
2x advantage to remove and their cells are unchanged.

Usage: python analysis/of_bestofk_table.py > outputs/main_results/of_bestofk_table.txt
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_quality_table import bestofK, PMK, qdarts_delivered   # locked estimator + name map

OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CV = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
COMMON_EP = 15000
KS = (1, 10)

INTERNAL = [("DreamQAS (Full)", f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of"),
            ("No-imag", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag"),
            ("DreamQAS-RL", f"{OF}/dreamqas_rlqas/baseline_analysis/G0_v2_{{m}}/seed_*")]
# (display, tree root, subdir, does this method have an env-episode axis?)
EXTERNAL = [("CRLQAS", f"{CV}/psqas", "crlqas", True),
            ("HyRLQAS", f"{CV}/psqas_hyrlqas_std", "hyrlqas", True),
            ("GQE", f"{CV}/psqas", "gqeqas", False),
            ("TFQAS", f"{CV}/psqas", "tfqas", False),
            ("qdarts", f"{CV}/psqas", "qdarts", False)]


def internal_series(pattern, mol):
    """-> list over seeds of (episode, [per-episode best errors]) at the common budget."""
    out = []
    for d in sorted(glob.glob(pattern.format(m=mol))):
        # LiH6q is the only task configured for 2x; its ep15000 RE-EVALUATION (100 fresh episodes)
        # is the common-budget read, exactly as in of_main_table.py.
        f = f"{d}/eval_traces_ep{COMMON_EP}.jsonl" if mol == "LiH6q" else f"{d}/eval_traces.jsonl"
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("ep_best")]
        if rows:
            r = max(rows, key=lambda r: r.get("_ck_ep", r["episode"]))
            out.append((int(r.get("_ck_ep", r["episode"])), list(r["ep_best"])))
    return out


def external_series(root, pdir, mol, episodic):
    """-> list over seeds of (episode, [per-episode best errors]).

    For an EPISODIC baseline on LiH-6q, take the checkpoint at exactly COMMON_EP rather than the
    30,000 final. For every other cell, and for the non-episodic methods, take the last checkpoint.
    """
    out = []
    for d in sorted(glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*")):
        if not re.search(r"/seed\d+$", d):          # skip backups like seed1_collapsed_bak
            continue
        p = f"{d}/eval.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        s = sorted([(r.get("episode", 0), list(r["episode_bests"]))
                    for r in rows if r.get("episode_bests")])
        if not s:
            continue
        if episodic and mol == "LiH6q":
            hit = [x for x in s if x[0] == COMMON_EP]
            if not hit:                              # never silently fall back to a longer budget
                continue
            out.append(hit[0])
        else:
            out.append(s[-1])
    return out


def reduce_seeds(series, K, cap_n=None):
    """per-seed bestofK -> (mean, std, n_seeds, n_eval_per_seed)."""
    vals, ns = [], []
    for _, e in series:
        ee = e[:cap_n] if cap_n else e
        vals.append(bestofK(ee, K)); ns.append(len(ee))
    if not vals:
        return None
    v = np.asarray(vals, float)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, v.size,
            int(np.min(ns)) if ns else 0)


def fmt(t):
    return "—" if t is None else f"{t[0]:.3g}±{t[1]:.2g}"


def main():
    print("=" * 126)
    print("BEST-OF-1 vs BEST-OF-10 AT THE COMMON 15,000-EPISODE BUDGET  (mHa, lower better)")
    print("  E[min of K deploys], order-statistic estimator; per-seed first, then mean ± sample std")
    print("  (ddof=1) over 5 seeds. K=1 is the paper's headline; K=10 is a SEARCH-BUDGET statistic.")
    print(f"  ⚠ LiH(6q): internal arms read their ep{COMMON_EP} re-evaluation; CRLQAS/HyRLQAS read")
    print(f"    their ep{COMMON_EP} CHECKPOINT instead of their 30,000 final. GQE/TFQAS/QuantumDARTS")
    print("    have no env-episode axis — their cells are the native final and are unchanged.")
    print("=" * 126)

    rows = []
    for label, pat in INTERNAL:
        rows.append((label, "oracle-free", {m: internal_series(pat, m) for m in MOLS}))
    for label, root, pdir, episodic in EXTERNAL:
        rows.append((label, "canonical", {m: external_series(root, pdir, m, episodic) for m in MOLS}))

    for K in KS:
        print(f"\n--- best-of-{K}")
        print(f"{'method':18}{'source':14}" + "".join(f"{DISP[m]:>19}" for m in MOLS))
        print("-" * 126)
        for label, src, per in rows:
            if label == "qdarts":                     # deterministic NAS: ONE circuit per run
                cells = []
                for m in MOLS:
                    v = qdarts_delivered(m)
                    cells.append(fmt((float(np.mean(v)), float(np.std(v, ddof=1)), len(v), 1))
                                 if v else "—")
                print(f"{label:18}{src:14}" + "".join(f"{c:>19}" for c in cells))
                continue
            cells = [fmt(reduce_seeds(per[m], K)) for m in MOLS]
            print(f"{label:18}{src:14}" + "".join(f"{c:>19}" for c in cells))

    # ---- provenance of every cell: which checkpoint, how many seeds, how many eval episodes ----
    print("\n" + "=" * 126)
    print("CELL PROVENANCE — checkpoint episode / seeds / evaluation episodes per seed")
    print("  n_eval is the sample the K=10 order statistic is estimated FROM. It is not constant;")
    print("  see the sensitivity block below before comparing K=10 across rows with different n.")
    print("=" * 126)
    print(f"{'method':18}" + "".join(f"{DISP[m]:>21}" for m in MOLS))
    for label, src, per in rows:
        if label == "qdarts":
            print(f"{label:18}" + "".join(f"{'1 circuit/run, 5 sd':>21}" for _ in MOLS))
            continue
        cs = []
        for m in MOLS:
            s = per[m]
            if not s:
                cs.append("—"); continue
            ep = s[0][0]; n = min(len(e) for _, e in s)
            cs.append(f"ep{ep} {len(s)}sd n={n}")
        print(f"{label:18}" + "".join(f"{c:>21}" for c in cs))

    # ---- how much does n=20 vs n=100 move a K=10 estimate? measured on our own runs ----
    print("\n" + "=" * 126)
    print("SENSITIVITY: how much does the K=10 estimator move when n drops 100 -> 20?")
    print("  Only the LiH-6q CRLQAS/HyRLQAS cells are estimated at n=20. The size of that effect")
    print("  depends on the ARM's OWN deploy dispersion, so it is measured on every arm that has")
    print("  n=100 available — not just ours — by truncating to the first 20 evaluation episodes.")
    print("  ratio > 1 = the n=20 estimate is PESSIMISTIC (too high) relative to the n=100 estimate.")
    print("=" * 126)
    print(f"{'method':18}" + "".join(f"{DISP[m]:>15}" for m in MOLS))
    for label, src, per in rows:
        if label == "qdarts":
            continue
        cs = []
        for m in MOLS:
            s = per[m]
            if not s or min(len(e) for _, e in s) < 100:
                cs.append("n<100")                    # cannot run the check on this cell
                continue
            a = reduce_seeds(s, 10)
            b = reduce_seeds(s, 10, cap_n=20)
            cs.append(f"{b[0] / a[0]:.2f}x" if (a and b and a[0]) else "—")
        print(f"{label:18}" + "".join(f"{c:>15}" for c in cs))
    print("  'n<100' = that cell already has fewer than 100 evaluation episodes, so the check cannot")
    print("  be run there; those are exactly the GQE (n=50) and LiH-6q CRLQAS/HyRLQAS (n=20) cells.")
    print("  ⚠ Where the ratio is far from 1.00, an n=20 cell is NOT comparable to an n=100 cell at")
    print("  K=10. Read the LiH-6q column of THIS block for the arms that do have n=100 there — it is")
    print("  the closest available proxy for the bias in the two n=20 cells.")

    print("\n" + "=" * 126)
    print("HOW TO READ")
    print("=" * 126)
    print("  * K=1 is the paper's headline metric. K=10 is a SEARCH-BUDGET statistic: it answers 'what")
    print("    if the practitioner ran the frozen policy 10 times and kept the best circuit?'. It must")
    print("    never be mixed into a K=1 comparison and never used in a VQE-efficiency claim (the 10")
    print("    deploys cost 10x the evaluation VQE, which this number does not charge).")
    print("  * A method whose K=1 -> K=10 gap is LARGE has high deploy variance: repeated sampling")
    print("    helps it. A method already at its ansatz floor gains nothing, because every deploy")
    print("    returns the same circuit — that is why the saturated cells barely move.")
    print("  * qdarts delivers ONE architecture per run, so its K=1 and K=10 are identical by")
    print("    construction and its ± is search-to-search variance, not deploy variance.")
    print("  * Internal arms are oracle-free; externals ran under their own native rewards and")
    print("    protocols. Two provenances, labelled per row — do not merge the vocabularies.")


if __name__ == "__main__":
    main()
