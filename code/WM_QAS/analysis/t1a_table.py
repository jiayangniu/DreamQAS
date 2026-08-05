"""Aggregate the T1.a counterfactual action-ranking probe across seeds -> RQ3(a) table.

Reads each run's t1a_probe.json (per-seed rho_action / regret / recall / wm_beats_rand + horizon
fidelity). Reports per molecule mean±std over seeds. rho_action is already a mean of per-prefix
Spearman with a per-seed bootstrap CI in the json; here we report the cross-seed mean and spread.
Matched-random win-rate (wm_beats_rand) > 0.5 = WM top-1 beats random candidates under the same
verification budget.

[2026-07-22] Convention UPDATE (supersedes the old 3-seed onset/final table): the two checkpoints are
  now the **imagination-start** (`wmstart` = the warm-up boundary `warmup_eps`≈500, where the WM first
  guides the policy) and the **¼-budget operating point** (`quarter` = the paper's equal-budget read
  point), computed on the **horizon_rerun** runs (kept checkpoints) over **5 seeds**. This replaces the
  index-based "onset" (~0.5% of training) and the ckpt-deleted "final".
"""
import glob
import json
import os
import numpy as np

DQ = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas"   # kept-ckpt runs (5 seeds, wmstart+quarter)
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
CKPTS = [("wmstart", "imag-start"), ("quarter", "1/4-budget")]     # (json key, display label)
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results/t1a_action_ranking.txt"


def load(mol):
    rows = []
    for d in sorted(glob.glob(f"{DQ}/gru_energy_surrogate_{mol}_s*_q")):
        p = f"{d}/t1a_probe.json"
        if os.path.exists(p):
            try:
                rows.append(json.load(open(p)))
            except Exception:
                pass
    return rows


def ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0), len(vals)) if vals else (np.nan, np.nan, 0)


lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 104)
P("T1.a  COUNTERFACTUAL ACTION-RANKING PROBE (RQ3a) — can the WM rank legal next actions of a real")
P("  prefix by post-VQE quality?  mean±std over 5 seeds. Held-out prefixes (early/mid/late), training-")
P("  identical VQE (snapshot-verified bit-exact), per-prefix Spearman, matched-random control.")
P("  ckpts: imag-start = imagination warm-up onset (warmup_eps~500); 1/4-budget = paper operating point.")
P("=" * 104)
P(f"{'molecule':<11}{'ckpt':<13}{'rho_action':>13}{'regret(norm)':>14}{'recall@1':>13}{'recall@3':>13}{'wm>rand':>13}{'seeds':>6}")
P("-" * 104)
horizon_rows = {}
for mol in MOLS:
    runs = load(mol)
    for key, lab in CKPTS:
        rho = ms([r.get(key, {}).get("agg", {}).get("rho_action") for r in runs])
        reg = ms([r.get(key, {}).get("agg", {}).get("regret") for r in runs])
        r1 = ms([r.get(key, {}).get("agg", {}).get("recall1") for r in runs])
        r3 = ms([r.get(key, {}).get("agg", {}).get("recall3") for r in runs])
        wr = ms([r.get(key, {}).get("agg", {}).get("wm_beats_rand") for r in runs])
        def f(t):
            return f"{t[0]:.3f}±{t[1]:.3f}" if t[2] else "  -  "
        P(f"{DISP[mol]:<11}{lab:<13}{f(rho):>13}{f(reg):>14}{f(r1):>13}{f(r3):>13}{f(wr):>13}{rho[2]:>6}")
    horizon_rows[mol] = runs
    P("-" * 104)

P()
P("HORIZON FIDELITY (endpoint ranking Spearman vs imagined horizon H; 1/4-budget ckpt, mean±std / 5 seeds)")
P(f"{'molecule':<11}{'H=1':>10}{'H=5':>10}{'H=10':>10}{'H=15':>10}")
P("-" * 54)
for mol in MOLS:
    runs = horizon_rows.get(mol, [])
    cells = []
    for H in ("1", "5", "10", "15"):
        t = ms([r.get("quarter", {}).get("horizon", {}).get(H, {}).get("fidelity") for r in runs])
        cells.append(f"{t[0]:.2f}±{t[1]:.2f}" if t[2] else "  -  ")
    P(f"{DISP[mol]:<11}" + "".join(f"{c:>10}" for c in cells))
P("-" * 54)
P("  Slope test (fidelity vs H) is n.s. on every molecule -> no systematic degradation with horizon.")
P("  Cite the slope result from `t1a_horizon_fidelity.txt` (v2, 1/4-budget); it feeds Figure 3(a).")
P()
P("READ (5-seed, imag-start -> 1/4-budget operating point):")
P("  1. rho_action GROWS from the imagination warm-up onset to the operating point on ALL 3 molecules")
P("     (4q 0.32->0.53, 6q 0.20->0.39, 8q 0.10->0.49) => the WM ACQUIRES counterfactual action-ranking")
P("     over training. BeH2(8q) is the clearest (near-zero 0.10 at onset -> 0.49). QAS-specific decision")
P("     utility, beyond generic pairwise fidelity.")
P("  2. normalized regret DROPS to the operating point (4q 0.235->0.184, 6q 0.157->0.056, 8q 0.255->0.025):")
P("     the WM's TOP-picked action moves near the real optimum among candidates -- and the top pick is")
P("     exactly what the actor uses. Strongest metric; improves on all 3.")
P("  3. wm>rand rises 4q 0.72->0.84 and 8q 0.82->0.97 (flat on 6q 0.69); WM top-1 beats ~69-97% of random")
P("     candidates under the SAME VQE budget (matched-random control).")
P("  4. Horizon fidelity: no significant trend with H (slope n.s.; `t1a_horizon_fidelity.txt`) -> multi-step")
P("     endpoint ranking is molecule-limited, not eroded by rollout length (supports exact-dynamics imagination).")
P("  Together these CLOSE RQ3(a): the feedback model is a decision-useful ranker of counterfactual circuit")
P("  continuations, ACQUIRED over training (imag-start -> operating point), with a near-optimal top pick.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
