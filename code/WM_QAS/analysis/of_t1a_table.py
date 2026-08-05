"""RQ3(a) action-ranking probe on the ORACLE-FREE checkpoints, side by side with the canonical table.

Aggregates the t1a_probe.json written by `t1a_action_ranking.py` (launcher
`experiments/launchers/of_t1a_probe.sh`) over 5 seeds x 3 representative tasks, at the SAME two
checkpoints as the canonical convention: `wmstart` (imagination warm-up boundary) and `quarter`
(1/4-budget operating point). Canonical column = the E0-trained runs under horizon_rerun/dreamqas.

Why these metrics survive the change of training signal: the oracle-free WM regresses the frontier
score S(E) instead of log10(true error), but S is a strictly monotone function of the energy E and so
is the true error (variational bound). Every metric here is RANK-based on the candidate actions
(Spearman, recall@k, normalized regret computed on the measured error of the SELECTED action,
WM-top-1 vs matched-random), so it is invariant to that monotone reparametrisation and IS comparable
across the two arms. Absolute calibration (MAE) is NOT — it lives in each WM's own target space and is
reported separately by `of_rq3_tables.py`.

Ground truth = training-identical VQE on the candidate continuations; E0 enters only as the eval-side
readout that turns an energy into an error (same licence as best-of-1). No E0 selects actions, prunes,
or picks checkpoints in either arm.

Usage: python analysis/of_t1a_table.py > outputs/main_results/oracle_free_t1a.txt
"""
import glob
import json
import os

import numpy as np

OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1/dreamqas"
CANON = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
CKPTS = [("wmstart", "imag-start"), ("quarter", "1/4-budget")]
FIELDS = [("rho_action", "rho_action"), ("regret", "regret(norm)"),
          ("recall1", "recall@1"), ("recall3", "recall@3"), ("wm_beats_rand", "wm>rand")]
HZ = ["1", "5", "10", "15"]


def load(base, mol, suffix):
    out = []
    for d in sorted(glob.glob(f"{base}/gru_energy_surrogate_{mol}_s*_{suffix}")):
        p = f"{d}/t1a_probe.json"
        if os.path.exists(p):
            try:
                out.append(json.load(open(p)))
            except Exception:
                pass
    return out


def ms(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return (np.nan, np.nan, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def cell(rows, sel, key):
    return ms([r[sel]["agg"].get(key) for r in rows if sel in r])


def fmt(t):
    m, s, _n = t
    return "     n/a  " if np.isnan(m) else f"{m:.3f}±{s:.3f}"


def main():
    of = {m: load(OF, m, "of") for m in MOLS}
    cn = {m: load(CANON, m, "q") for m in MOLS}
    print("=" * 112)
    print("T1.a COUNTERFACTUAL ACTION-RANKING PROBE — ORACLE-FREE checkpoints vs the canonical (E0-trained)")
    print("  Same protocol, same two checkpoints, 3 tasks x 5 seeds, no retraining (frozen ckpt reload).")
    print("  All metrics are RANK-based over the candidate actions -> comparable across training signals.")
    print("  Ground truth = training-identical VQE; E0 only converts energy->error at readout.")
    print("=" * 112)
    for mol in MOLS:
        print(f"\n--- {DISP[mol]}   [oracle-free seeds={len(of[mol])} | canonical seeds={len(cn[mol])}]")
        print(f"{'ckpt':<13}{'arm':<14}" + "".join(f"{lbl:>15}" for _k, lbl in FIELDS))
        for sel, lbl in CKPTS:
            for arm, rows in (("oracle-free", of[mol]), ("canonical", cn[mol])):
                if not rows:
                    continue
                print(f"{lbl:<13}{arm:<14}" + "".join(f"{fmt(cell(rows, sel, k)):>15}" for k, _l in FIELDS))
            print("-" * 112)
    print("\nHORIZON ENDPOINT FIDELITY (1/4-budget ckpt; ranking of imagined endpoints, mean±std / 5 seeds)")
    print(f"{'molecule':<11}{'arm':<14}" + "".join(f"{'H=' + h:>12}" for h in HZ))
    for mol in MOLS:
        for arm, rows in (("oracle-free", of[mol]), ("canonical", cn[mol])):
            if not rows:
                continue
            vals = []
            for h in HZ:
                vals.append(ms([r["quarter"].get("horizon", {}).get(h, {}).get("fidelity")
                                for r in rows if "quarter" in r]))
            print(f"{DISP[mol]:<11}{arm:<14}" + "".join(
                f"{('  n/a' if np.isnan(v[0]) else f'{v[0]:.2f}±{v[1]:.2f}'):>12}" for v in vals))
        print("-" * 60)
    print("\nREAD:")
    print("  rho_action / recall / wm>rand HIGHER = better;  regret LOWER = better.")
    print("  The claim this table can support is about the WM remaining a DECISION-USEFUL RANKER when the")
    print("  exact ground-state energy is removed from training — compare the two arms row by row at the")
    print("  1/4-budget operating point, and check that the imag-start -> 1/4-budget GROWTH still holds.")
    print("  Do NOT read a small oracle-free-vs-canonical difference as significant: n=5 seeds, and the")
    print("  per-seed spread here is the same order as the gap on several cells.")


if __name__ == "__main__":
    main()
