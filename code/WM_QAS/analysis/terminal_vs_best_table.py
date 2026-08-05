"""FINAL-CIRCUIT vs EPISODE-BEST reduction, at the common 15,000-episode budget. READ-ONLY.

The paper's metric reduces an evaluation episode with `e_j = min_t err(prefix_{j,t})` — the best
VQE-optimised PREFIX. A reviewer will reasonably ask what happens under the other reduction:
`e_j = err(prefix_{j,T})`, the circuit the policy actually COMMITS to at the depth cap. This table
puts the two side by side on the same episodes, same checkpoint, same seeds.

    EPISODE-BEST    mean_j  min_t err(prefix_{j,t})     <- the paper's metric (best-of-1)
    FINAL-CIRCUIT   mean_j  err(prefix_{j,T})           <- the committed full-depth circuit
    ratio           FINAL / BEST                        <- how much the prefix rule is worth

Both are read from the SAME row of the SAME eval file (`ep_best` and `ep_final` are recorded together
by `eval_harness.evaluate_checkpoint`), so nothing is re-derived and the two cannot drift apart.

⚠ COVERAGE IS INCOMPLETE AND CANNOT BE COMPLETED READ-ONLY. `ep_final` exists only where a frozen
evaluation stored the full per-prefix trace:
    DreamQAS Full / No-imag / DreamQAS-RL   all 5 tasks, n=100          available
    CRLQAS                                  4 tasks at ep15000          available
    CRLQAS  LiH-6q                          only ckpts >= 22500          MISSING at 15k
    HyRLQAS (the REPORTED psqas_hyrlqas_std tree)                        MISSING, 0/25 runs
    GQE / TF-QAS / QuantumDARTS                                          MISSING, 0/25 runs each
The native PSQAS `eval.jsonl` stores `episode_bests` only; its `mean`/`median`/`best` keys are
statistics OVER those episode-bests, not the terminal circuit. Missing cells print `MISSING`; none is
imputed.

⚠ THE POLICY-BASED MISSING CELLS ARE PERMANENT (verified 2026-07-29). Backfilling a policy method
means replaying 100 frozen rollouts, which needs the actor weights — and **every PSQAS training
checkpoint has been deleted**: `ckpt/` is empty and `*.pt` count is 0 for crlqas, hyrlqas (BOTH
trees), tfqas and qdarts, on disk AND in the 16 GB `dreamqas_data_full.tar.gz` archive (its only
psqas `.pt` entries are GQE's 50 best_model/final_model files; its 128 `/ckpt/` paths all belong to
DreamQAS's own campaign_v1 runs). `PSQASBench/offline_final_eval.py` reloads `ckpt/ep*.pt`, so it
cannot be run for CRLQAS LiH-6q@15000 or for HyRLQAS-std at all. Do not plan around it.

What CAN still be produced, if wanted, is a DIFFERENT quantity: GQE, TF-QAS and QuantumDARTS each
kept 25/25 `best_circuit.txt`, and all three deliver ONE circuit rather than a policy, so their
"terminal vs best-prefix" question is answerable by re-running VQE over the prefixes of that single
delivered circuit (<=50 VQE calls per run). That is a single-circuit statistic, NOT a 100-episode
mean, so it would occupy its own column and must never be placed in the same column as the rows here.

Usage: python analysis/terminal_vs_best_table.py > outputs/main_results/terminal_vs_best.txt
"""
import glob
import json
import os
import re

import numpy as np

OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CV = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
COMMON_EP = 15000

INTERNAL = [("DreamQAS (Full)", f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of"),
            ("No-imag", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag"),
            ("DreamQAS-RL", f"{OF}/dreamqas_rlqas/baseline_analysis/G0_v2_{{m}}/seed_*")]
# externals that have any full-trace file at all
EXTERNAL = [("CRLQAS", f"{CV}/psqas/crlqas")]
# externals with NO terminal reduction anywhere (documented, never imputed)
NO_TRACE = ["HyRLQAS", "GQE", "TFQAS", "qdarts"]


def internal_rows(pattern, mol):
    """-> list over seeds of (ep_best[], ep_final[]) at the common budget."""
    out = []
    for d in sorted(glob.glob(pattern.format(m=mol))):
        f = f"{d}/eval_traces_ep{COMMON_EP}.jsonl" if mol == "LiH6q" else f"{d}/eval_traces.jsonl"
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("ep_best") and r.get("ep_final")]
        if rows:
            r = max(rows, key=lambda r: r.get("_ck_ep", r["episode"]))
            out.append((list(r["ep_best"]), list(r["ep_final"])))
    return out


def external_rows(root, mol):
    """-> list over seeds of (ep_best[], ep_final[]) at EXACTLY the common budget, or [] if absent.

    Never falls back to a later checkpoint: on LiH-6q the only traced checkpoints are >= 22500, and
    silently substituting one of those would compare a 15k arm against a 22.5k+ arm.
    """
    out = []
    for d in sorted(glob.glob(f"{root}/{PMK[mol]}/*/*/seed*")):
        if not re.search(r"/seed\d+$", d):
            continue
        f = f"{d}/eval_traces_final.jsonl"
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        hit = [r for r in rows if r.get("_ck_ep", r.get("episode")) == COMMON_EP
               and r.get("ep_best") and r.get("ep_final")]
        if hit:
            out.append((list(hit[0]["ep_best"]), list(hit[0]["ep_final"])))
    return out


N_REQUIRED = 100          # the standard final-checkpoint evaluation size


def cell(rows):
    """-> (best_mean, best_sd, fin_mean, fin_sd, n_kept, n_dropped) or None.

    Seeds whose trace holds fewer than N_REQUIRED episodes are DROPPED, not averaged in. One CRLQAS
    BeH2-8q seed has only 6 traced episodes (the offline re-eval did not finish it); folding a
    6-episode seed mean into a 5-seed average would silently weight it equally with 100-episode
    seeds and quietly shrink the reported std.
    """
    if not rows:
        return None
    keep = [(x, y) for x, y in rows if len(x) >= N_REQUIRED and len(y) >= N_REQUIRED]
    dropped = len(rows) - len(keep)
    if not keep:
        return None
    b = np.array([np.mean(x) for x, _ in keep], float)
    f = np.array([np.mean(y) for _, y in keep], float)
    sd = lambda v: float(v.std(ddof=1)) if v.size > 1 else 0.0
    return float(b.mean()), sd(b), float(f.mean()), sd(f), b.size, dropped


def main():
    print("=" * 122)
    print("FINAL-CIRCUIT vs EPISODE-BEST — same episodes, same checkpoint, common 15,000-episode budget")
    print("  EPISODE-BEST  = mean_j min_t err(prefix_j,t)   <- the paper's best-of-1 metric")
    print("  FINAL-CIRCUIT = mean_j err(prefix_j,T)         <- the committed full-depth circuit")
    print("  mHa, lower better; per-seed mean first, then mean ± sample std (ddof=1) over 5 seeds.")
    print(f"  LiH(6q): internal arms read their ep{COMMON_EP} re-evaluation.")
    print("=" * 122)

    data = {}
    for label, pat in INTERNAL:
        data[label] = {m: internal_rows(pat, m) for m in MOLS}
    for label, root in EXTERNAL:
        data[label] = {m: external_rows(root, m) for m in MOLS}

    for m in MOLS:
        print(f"\n--- {DISP[m]}")
        print(f"{'method':18}{'EPISODE-BEST':>22}{'FINAL-CIRCUIT':>22}{'ratio F/B':>12}"
              f"{'seeds':>7}{'dropped':>9}")
        for label in list(data):
            c = cell(data[label][m])
            if c is None:
                print(f"{label:18}{'MISSING':>22}{'MISSING':>22}{'—':>12}{'—':>7}{'—':>9}")
                continue
            bm, bs, fm, fs, ns, nd = c
            r = fm / bm if bm else float("nan")
            print(f"{label:18}{f'{bm:.4g}±{bs:.2g}':>22}{f'{fm:.4g}±{fs:.2g}':>22}"
                  f"{r:>11.1f}x{ns:>7}{(str(nd) if nd else '-'):>9}")
        for label in NO_TRACE:
            print(f"{label:18}{'MISSING':>22}{'MISSING':>22}{'—':>12}{'—':>7}{'—':>9}")
    print(f"\n  'dropped' = seeds excluded because their trace held < {N_REQUIRED} episodes.")

    # ---- consistency: is this file's EPISODE-BEST the same number the main table reports? ----
    print("\n" + "=" * 122)
    print("⚠ CONSISTENCY CHECK — the EPISODE-BEST column above is NOT the main table's best-of-1 for")
    print("  the EXTERNAL rows, and the gap is systematic, not sampling noise.")
    print("  Internal arms: both columns come from the SAME eval_traces.jsonl that feeds of_main_table,")
    print("  so their EPISODE-BEST reproduces it exactly. CRLQAS's two columns come from")
    print("  eval_traces_final.jsonl, written by a SEPARATE offline re-evaluation pass")
    print("  (PSQASBench/offline_final_eval.py), while of_main_table reads the NATIVE eval.jsonl.")
    print("=" * 122)
    print(f"{'task':10}{'native eval.jsonl':>20}{'eval_traces_final':>20}{'ratio':>9}   both = episode-best @ep15000")
    for m in MOLS:
        a, b = [], []
        for d in sorted(glob.glob(f"{CV}/psqas/crlqas/{PMK[m]}/*/*/seed*")):
            if not re.search(r"/seed\d+$", d):
                continue
            p = f"{d}/eval.jsonl"
            if os.path.exists(p):
                h = [r for r in (json.loads(l) for l in open(p) if l.strip())
                     if r.get("episode") == COMMON_EP and r.get("episode_bests")]
                if h:
                    a.append(np.mean(h[0]["episode_bests"]))
            p2 = f"{d}/eval_traces_final.jsonl"
            if os.path.exists(p2):
                h = [r for r in (json.loads(l) for l in open(p2) if l.strip())
                     if r.get("_ck_ep", r.get("episode")) == COMMON_EP
                     and r.get("ep_best") and len(r["ep_best"]) >= N_REQUIRED]
                if h:
                    b.append(np.mean(h[0]["ep_best"]))
        if a and b:
            print(f"{DISP[m]:10}{np.mean(a):>20.4g}{np.mean(b):>20.4g}"
                  f"{np.mean(b) / np.mean(a):>9.2f}x")
        else:
            print(f"{DISP[m]:10}{'—':>20}{'—':>20}{'—':>9}")
    print("  The offline pass is consistently OPTIMISTIC (ratio < 1) on every task where both exist.")
    print("  It is an independent frozen evaluation with its own RNG, and its rollout path")
    print("  (`BaseRunner._std_eval_rollout_trace`) is not verified to match the native eval's")
    print("  termination/optimiser settings — that has NOT been checked and is an open item.")
    print("  CONSEQUENCE: quote the RATIO F/B from this table (numerator and denominator share one")
    print("  pass, so it is internally valid) and keep quoting of_main_table for the best-of-1 LEVEL.")
    print("  Do NOT put this table's EPISODE-BEST column next to of_main_table's numbers.")

    print("\n" + "=" * 122)
    print("WHAT IS MISSING, AND WHAT IT WOULD COST")
    print("=" * 122)
    print("  The native PSQAS `eval.jsonl` records `episode_bests` only. Its `mean`/`median`/`best`")
    print("  keys are statistics OVER those episode-bests — NOT the terminal circuit — so the")
    print("  final-circuit reduction cannot be recovered from them by any amount of re-analysis.")
    print()
    print("  ⚠ EVERY PSQAS TRAINING CHECKPOINT HAS BEEN DELETED (verified 2026-07-29): ckpt/ is empty")
    print("  and *.pt = 0 for crlqas, hyrlqas (BOTH trees), tfqas and qdarts — on disk AND inside")
    print("  dreamqas_data_full.tar.gz (16 GB, 2026-07-18). That archive's only psqas *.pt entries are")
    print("  GQE's 50 best_model/final_model files; its 128 /ckpt/ paths are DreamQAS's own runs.")
    print("  oracle_free_v1_curated_20260728.tar.gz contains oracle_free_v1 only.")
    print()
    print("  MISSING cell                              status")
    print("  " + "-" * 100)
    print("  CRLQAS  LiH(6q) @ep15000                  PERMANENTLY LOST. offline_final_eval.py needs")
    print("                                           ckpt/ep15000.pt; traced ckpts there start at 22500.")
    print("  HyRLQAS all 5 tasks (psqas_hyrlqas_std)   PERMANENTLY LOST. 0/25 traces, 0 checkpoints.")
    print("            ⚠ the EXCLUDED psqas/hyrlqas tree HAS 25/25 traces, but that is the crippled")
    print("            reward regime the paper does not report — it is NOT a substitute.")
    print("  GQE / TF-QAS / QuantumDARTS               a DIFFERENT quantity is still obtainable: all")
    print("            three kept 25/25 best_circuit.txt and each delivers ONE circuit, so VQE over the")
    print("            prefixes of that circuit answers the question (<=50 VQE/run). Single-circuit,")
    print("            NOT a 100-episode mean -> separate column, never merged with the rows above.")
    print()
    print("  CRLQAS/HyRLQAS additionally kept global_best_state_*.npz, which holds the op_history of")
    print("  the single best circuit found during training. That supports a one-circuit comparison")
    print("  only, on a different sample from the 100-episode rows — do not mix it in either.")

    print("\n" + "=" * 122)
    print("HOW TO READ")
    print("=" * 122)
    print("  * ratio F/B is how much the min-over-prefix rule buys. A large ratio does NOT mean the")
    print("    metric is inflated: every prefix in the episode was really VQE-optimised and really")
    print("    observed, and the selection uses the lowest OBSERVED energy, never E0. It means the")
    print("    policy's terminal circuit is worse than something it passed through — which is a")
    print("    property of the search, and is exactly what the reviewer is asking about.")
    print("  * The comparison is only meaningful WITHIN a row's own protocol. A method whose search")
    print("    commits to its last gate (QuantumDARTS delivers one circuit) has no meaningful")
    print("    distinction between the two reductions.")
    print("  * Both columns come from ONE row of ONE file, so they cannot drift; if a cell is")
    print("    MISSING it is missing in BOTH columns, never half-populated.")


if __name__ == "__main__":
    main()
