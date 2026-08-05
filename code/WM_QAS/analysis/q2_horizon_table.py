"""Reviewer-Q2: transition-matched imagined horizon H in {1, 5, 15} — frozen best-of-1 + paired deltas.

Arms (oracle-free, --dagger 0 throughout, so ONLY the imagined-rollout structure differs):
    H1  = h1/   with --imag_transition_budget T   (T trusted imagined transition loss-terms per update)
    H5  = h5/   with --imag_transition_budget T
    H15 = the reference: nodag/ (seeds 0-4, whose realized T *defines* T) + h15x/ (seeds 5-9)
T is measured per task from the reference arm's imag.jsonl (BeH2_8q 601, LiH4q 954, LiH6q 960); the budget
was verified on the live runs to hold on TWO axes at once — identical trusted-transition count AND nearly
identical raw WM step-queries — so an H-difference cannot be attributed to "more imagined data" or "more
WM compute" (see SUPPLEMENTARY_EXPERIMENTS.md §3-Q2).

Metric = the paper's unchanged frozen best-of-1: per seed, the mean episode-best true error over the
final-checkpoint evaluation episodes; then paired across seeds by SEED INDEX (the arms share seeds).

CLAIM GATE (do not exceed): only if H15 is better than H1 *under matched T*, with a paired CI that
excludes 0, may the paper say multi-step credit assignment CAUSES the gain. A null result licenses only
"imagined policy learning beats direct surrogate search" (the RQ3 conclusion), not a horizon claim.

Usage: python analysis/q2_horizon_table.py [--mols BeH2_8q] > outputs/main_results/q2_horizon.txt
"""
import argparse
import glob
import json
import os

import numpy as np

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
T_REF = {"LiH4q": 954, "LiH6q": 960, "BeH2_8q": 601}
CHEM_ACC = 1.6      # mHa — a CI that excludes 0 well below this is clean but chemically negligible
# arm -> list of (glob template, seed range) — H15 is stitched from the two sources that define it
ARMS = {
    "H1":  [(f"{C}/h1/gru_energy_surrogate_{{m}}_s{{s}}_h1", range(10))],
    "H5":  [(f"{C}/h5/gru_energy_surrogate_{{m}}_s{{s}}_h5", range(5))],
    "H15": [(f"{C}/nodag/gru_energy_surrogate_{{m}}_s{{s}}_of_nodag", range(5)),
            (f"{C}/h15x/gru_energy_surrogate_{{m}}_s{{s}}_h15", range(5, 10))],
}


def bo1(run_dir, only_ep=None):
    """best-of-1 at the reported checkpoint = mean episode-best over that checkpoint's eval episodes.

    With --only_ep N the dedicated `eval_traces_epN.jsonl` is preferred, but an arm that reached ep N as
    its FINAL checkpoint already carries that row (at the same 100-episode protocol) inside the full-grid
    `eval_traces.jsonl` — so fall back to it rather than spending real VQE to reproduce an existing number.
    The fallback matches on the checkpoint episode, never on 'whatever is deepest', so it cannot silently
    compare arms evaluated at different budgets.
    """
    cands = [f"{run_dir}/eval_traces_ep{only_ep}.jsonl"] if only_ep else []
    cands.append(f"{run_dir}/eval_traces.jsonl")
    for f in cands:
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("ep_best")]
        if only_ep and f.endswith("eval_traces.jsonl"):
            rows = [r for r in rows if int(r.get("_ck_ep", r["episode"])) == int(only_ep)]
        if not rows:
            continue
        r = max(rows, key=lambda r: r.get("_ck_ep", r["episode"]))     # the deepest checkpoint present
        return float(np.mean(r["ep_best"])), int(r.get("_ck_ep", r["episode"])), len(r["ep_best"])
    return None


def arm_seeds(arm, mol, only_ep=None):
    """-> {seed: bo1}."""
    out = {}
    for tmpl, rng in ARMS[arm]:
        for s in rng:
            d = tmpl.format(m=mol, s=s)
            v = bo1(d, only_ep)
            if v:
                out[s] = v[0]
    return out


def paired(a, b, rng, n_boot=10000):
    """paired mean difference a-b over shared seeds + percentile bootstrap CI."""
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    d = np.array([a[k] - b[k] for k in ks])
    bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(ks), seeds=ks, mean=float(d.mean()), lo=float(lo), hi=float(hi),
                wins=int((d < 0).sum()), per_seed=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", default="BeH2_8q")
    ap.add_argument("--only_ep", type=int, default=0,
                    help="evaluate the ep<N> checkpoint instead of the deepest one — needed when the arms "
                         "ran to DIFFERENT budgets (LiH6q: H1 was capped at 3750 iters = ep15000, while "
                         "H5/H15 ran to 7500). 0 = use each run's deepest checkpoint.")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    only = a.only_ep or None
    print("=" * 100)
    print("REVIEWER-Q2 — TRANSITION-MATCHED IMAGINED HORIZON  H in {1,5,15}   (oracle-free, dagger=0)")
    print("  best-of-1 (mHa, lower better) at " + (f"checkpoint ep{only}" if only else "each run's final checkpoint"))
    print("  H15 = the reference arm that DEFINES the transition budget T; H1/H5 run at --imag_transition_budget T")
    print("  Paired by seed index; 10,000-sample percentile bootstrap on the paired mean difference.")
    print("=" * 100)
    for mol in a.mols.split(","):
        vals = {k: arm_seeds(k, mol, only) for k in ARMS}
        if not any(vals.values()):
            print(f"\n--- {DISP.get(mol, mol)}: no evaluated runs yet"); continue
        print(f"\n--- {DISP.get(mol, mol)}   (matched budget T = {T_REF.get(mol, '?')} trusted transitions/update)")
        print(f"{'arm':6s}{'seeds':>7}{'best-of-1 mean±std':>22}{'<chem-acc':>11}   per-seed")
        for k in ("H1", "H5", "H15"):
            v = vals[k]
            if not v:
                print(f"{k:6s}{0:>7}   (not evaluated)"); continue
            x = np.array([v[s] for s in sorted(v)])
            ok = f"{int((x < CHEM_ACC).sum())}/{len(x)}"
            print(f"{k:6s}{len(x):>7}{x.mean():>13.3f}±{x.std(ddof=1):<8.3f}{ok:>11}   "
                  f"{{{', '.join(f'{s}:{v[s]:.2f}' for s in sorted(v))}}}")
        print(f"\n{'contrast':14s}{'n':>4}{'paired Δ':>11}{'95% CI':>20}{'better':>9}   "
              f"{'significance':22s}magnitude")
        for lab, x, y in (("H1 − H15", "H1", "H15"), ("H5 − H15", "H5", "H15"), ("H1 − H5", "H1", "H5")):
            r = paired(vals[x], vals[y], rng)
            if r is None:
                print(f"{lab:14s}{'—':>4}   (需要 ≥3 个共同 seed)"); continue
            clean = r["lo"] > 0 or r["hi"] < 0
            sig = "CI excludes 0" if clean else "CI crosses 0 → n.s."
            # A CI can exclude 0 on a difference that is deep inside chemical accuracy. Separate the two
            # questions mechanically: "is it real?" (the CI) vs "does it matter chemically?" (this column).
            # Material = the arms differ in how many seeds REACH chemical accuracy, or |Δ| >= chem-acc.
            ks = sorted(set(vals[x]) & set(vals[y]))
            nx = sum(vals[x][k] < CHEM_ACC for k in ks)
            ny = sum(vals[y][k] < CHEM_ACC for k in ks)
            if not clean:
                mag = "—"
            elif nx != ny or abs(r["mean"]) >= CHEM_ACC:
                mag = f"MATERIAL (chem-acc seeds {nx}/{len(ks)} vs {ny}/{len(ks)})"
            else:
                mag = f"sub-chem-acc (|Δ|={abs(r['mean']):.3f} = {CHEM_ACC / abs(r['mean']):.0f}x inside)"
            ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            wins = f"{r['wins']}/{r['n']}"
            print(f"{lab:14s}{r['n']:>4}{r['mean']:>+11.3f}{ci:>20}{wins:>9}   {sig:22s}{mag}")
        print("\n  Δ<0 means the FIRST arm has the lower (better) error. 'better' = seeds where that holds.")
        print("  'significance' answers whether the difference is real; 'magnitude' answers whether it")
        print("  matters chemically. A sub-chem-acc row is a genuine but chemically negligible ordering —")
        print("  report it as such, never as a policy-quality improvement.")
        # SATURATION GUARD: if an arm's seeds are all pinned at the same value, that arm is at the ansatz
        # floor and the task carries no information about the horizon — say so instead of letting the
        # reader treat a floor-vs-floor tie (or a few non-converged seeds in the other arm) as a result.
        floored = [k for k, v in vals.items() if len(v) >= 3 and np.std(list(v.values()), ddof=1) < 1e-2]
        if floored:
            lo = min(min(v.values()) for v in vals.values() if v)
            print(f"\n  ⚠ SATURATED: arm(s) {', '.join(floored)} sit at the ansatz floor (~{lo:.2f} mHa, std<0.01)")
            print("    -> this task CANNOT discriminate between horizons. Any apparent direction comes from")
            print("       the non-floored arm's failed-to-converge seeds (variance/collapse), NOT from a")
            print("       horizon effect on quality. Do not cite this molecule for the Q2 claim gate; report")
            print("       it as an uninformative (saturated) control and rely on the non-saturated tasks.")
    print("\n" + "=" * 100)
    print("CLAIM GATE: a horizon claim requires H15 better than H1 UNDER MATCHED T with a paired CI that")
    print("excludes 0. If every CI crosses 0, the honest statement is: at a matched trusted-transition")
    print("budget the imagined horizon does NOT measurably change final policy quality on this task —")
    print("which still leaves RQ3 intact (imagined policy learning beats direct surrogate search) but")
    print("removes 'multi-step credit assignment' as a demonstrated cause. Do not write 'myopia'.")


if __name__ == "__main__":
    main()
