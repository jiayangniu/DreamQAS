import glob
import json
import os

import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_quality_table import bestofK

from analysis_paths import campaign_root, require_campaign, OUTPUTS

C = campaign_root("oracle_free_v1")
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
COMMON_EP = 15000
CHEM = 1.6
ARMS = [("Full",           f"{C}/dreamqas/gru_energy_surrogate_{{m}}_s*_of"),
        ("−DAgger",        f"{C}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag"),
        ("−DIR reweight",  f"{C}/ablations/gru_energy_surrogate_{{m}}_s*_of_noDIR"),
        ("−uncertainty",   f"{C}/ablations/gru_energy_surrogate_{{m}}_s*_of_noUNC"),
        ("−imagination †", f"{C}/ablations/gru_energy_none_{{m}}_s*_of_noimag")]


def seed_bo1(pattern, mol, K=1):
    out = {}
    for d in sorted(glob.glob(pattern.format(m=mol))):
        rows = []
        f = f"{d}/eval_traces_ep{COMMON_EP}.jsonl"
        if os.path.exists(f):
            rows = [json.loads(l) for l in open(f) if l.strip()]
        else:
            g = f"{d}/eval_traces.jsonl"
            if os.path.exists(g):
                rows = [r for r in (json.loads(l) for l in open(g) if l.strip())
                        if int(r.get("_ck_ep", r.get("episode", -1))) == COMMON_EP]
        rows = [r for r in rows if r.get("ep_best") and int(r.get("_ck_ep", r.get("episode", -1))) == COMMON_EP]
        if not rows:
            continue
        s = os.path.basename(d).split("_s")[-1].split("_")[0]
        e = [x for x in rows[-1]["ep_best"] if np.isfinite(x)]
        if e:
            out[s] = float(bestofK(e, K))
    return out


def paired(full, arm, rng, n=10000):
    ks = sorted(set(full) & set(arm))
    if len(ks) < 3:
        return None
    d = np.array([arm[k] - full[k] for k in ks])
    bs = [rng.choice(d, len(d), True).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(ks), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                hurt=int((d > 0).sum()))


def main():
    require_campaign(C)
    rng = np.random.default_rng(7)
    print("=" * 118)
    print("ORACLE-FREE WM COMPONENT ABLATION — frozen best-of-1 (mHa) at the common 15,000-episode budget")
    print("  5 molecules x 5 arms x 5 seeds, all one training signal and one code era.")
    print("  Δ = arm − Full (paired by seed). POSITIVE Δ = removing the component HURT (the component helps).")
    print("=" * 118)
    data = {m: {n: seed_bo1(p, m) for n, p in ARMS} for m in MOLS}
    print(f"{'arm':18s}" + "".join(f"{DISP[m]:>19}" for m in MOLS))
    print("-" * 118)
    for n, _ in ARMS:
        cells = []
        for m in MOLS:
            v = list(data[m][n].values())
            cells.append("—" if not v else
                         (f"{np.mean(v):.3g}±{np.std(v, ddof=1):.2g}" if len(v) > 1 else f"{v[0]:.3g}"))
        print(f"{n:18s}" + "".join(f"{c:>19}" for c in cells))
    print("-" * 118)
    print(f"\n{'component removed':18s}{'task':11s}{'n':>3}{'paired Δ':>11}{'95% CI':>22}"
          f"{'hurt':>7}   verdict")
    print("-" * 118)
    for n, _ in ARMS[1:]:
        for m in MOLS:
            r = paired(data[m]["Full"], data[m][n], rng)
            if r is None:
                print(f"{n:18s}{DISP[m]:11s}  (fewer than 3 shared seeds)")
                continue
            clean = r["lo"] > 0 or r["hi"] < 0
            v = ("component HELPS" if (clean and r["mean"] > 0) else
                 "component HURTS" if clean else "n.s.")
            ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            print(f"{n:18s}{DISP[m]:11s}{r['n']:>3}{r['mean']:>+11.3f}{ci:>22}"
                  f"{f'{r[chr(104)+chr(117)+chr(114)+chr(116)]}/{r[chr(110)]}':>7}   {v}")
        print()
    print("-" * 118)
    print("† −imagination disables imagination AND DAgger together (calibrate() is gated on")
    print("  imagination==surrogate), so its Δ is a TWO-mechanism effect. To isolate imagination use")
    print("  (−imagination) − (−DAgger); the Full − (−DAgger) row is what isolates DAgger.")
    print("\nInterpret intervals using the actual paired-seed counts. Non-significance is not equivalence.")
    print("Check saturation separately; this release does not embed historical plateau values.")
    bestofk_block(rng)


def bestofk_block(rng):
    print("\n" + "=" * 118)
    print("BEST-OF-K SENSITIVITY — does any verdict depend on reporting the mean rather than the tail?")
    print("  Δ = arm − Full at the SAME checkpoint/episodes, re-reduced with E[min of K deploys].")
    print("  K=1 is the headline (mean); K=10/100 are deployment/search-budget statistics (appendix).")
    print("=" * 118)
    for n, pat in ARMS[1:]:
        print(f"\n{n}")
        print(f"{'task':11s}" + "".join(f"{'K=' + str(k):>28}" for k in (1, 10, 100)))
        for m in MOLS:
            cells = []
            for K in (1, 10, 100):
                r = paired(seed_bo1(ARMS[0][1], m, K), seed_bo1(pat, m, K), rng)
                if r is None:
                    cells.append("—"); continue
                star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
                cells.append(f"{r['mean']:+.4f} [{r['lo']:+.3f},{r['hi']:+.3f}]{star}")
            print(f"{DISP[m]:11s}" + "".join(f"{c:>28}" for c in cells))
    print("\n  * = CI excludes 0. ⚠ Saturated molecules read ~0.0000 at every K because BOTH arms sit at")
    print("  the ansatz floor — that is ABSENCE OF SIGNAL, not evidence the component is harmless.")
    print("  Only the non-saturated molecules carry information here.")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Read campaign data under DREAMQAS_RESULTS_ROOT; see REPRODUCE.md").parse_args()
    main()
