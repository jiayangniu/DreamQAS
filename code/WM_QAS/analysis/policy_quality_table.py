"""Main policy-quality TABLE (episode-best, min-prefix reduction).

Per method x molecule, two blocks x two order-statistics, aggregated at the SEED level:
  Block-1 (converged): the FINAL saved checkpoint.
  Block-2 (timing-robust): mean over the LAST K_CKPT saved checkpoints (deploy-faithful,
           frozen, insensitive to which exact checkpoint was saved last).
  best-of-1  = E[min of 1 deploy]  == the mean deploy quality   (AVERAGE level)
  best-of-10 = E[min of 10 deploys] via the order-statistic estimator (OPTIMAL level)
Both use ONE estimator (bestofK) with K in {1,10}; K=1 reduces to the mean by construction.
Seed is the experimental unit: per-seed statistic first, then mean +/- std (+95% CI, t4) over
the 5 seeds -- episodes are NEVER pooled across seeds. qdarts is near-degenerate (deterministic
NAS): reported the same way but flagged; caller may collapse it to a single value.

Sources: DreamQAS-family -> analysis eval_traces.jsonl (ep_best); PSQAS -> native eval.jsonl
(episode_bests), which is the same 20/100 frozen protocol.
"""
import re
import json, glob, os
import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH4q", "BeH2": "BeH2_6q", "LiH6q": "LiH6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
METH = ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS", "GQE", "TFQAS", "qdarts"]
K_CKPT = 3          # Block-2 window: last-3 converged checkpoints
T4 = 2.776         # 95% CI multiplier, dof=4 (5 seeds)


def bestofK(errs, K):
    """E[min of K i.i.d. deploys] estimated from n samples (order-statistic weights).
    K=1 -> uniform weights -> arithmetic mean. n-consistent across different n."""
    e = np.sort(np.asarray([x for x in errs if np.isfinite(x)], float))
    n = e.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    w = ((n - i + 1) / n) ** K - ((n - i) / n) ** K
    return float((e * w).sum())


def run_series(method, mol):
    """per-run list of (ckpt_key, episode_best_list) sorted ascending by checkpoint."""
    out = []
    if method in ("Full", "No-imag", "RLQAS"):
        pat = {"Full": f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q/eval_traces.jsonl",
               "No-imag": f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag/eval_traces.jsonl",
               "RLQAS": f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{mol}/seed_*/eval_traces.jsonl"}[method]
        for f in glob.glob(pat):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            s = [(r.get("_ck_ep", r.get("episode", 0)), r["ep_best"]) for r in rows if r.get("ep_best")]
            if s:
                out.append(sorted(s))
    else:
        pdir = {"GQE": "gqeqas", "qdarts": "qdarts"}.get(method, method.lower())
        # HyRLQAS: read the fixed good-regime runs (masking-fix + accept_err=2.45) in the
        # separate psqas_hyrlqas_std tree; psqas/hyrlqas is the old crippled reward regime.
        root = f"{C}/psqas_hyrlqas_std" if method == "HyRLQAS" else f"{C}/psqas"
        for d in glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*"):
            if not re.search(r"/seed\d+$", d):   # exclude backups like seed1_collapsed_bak
                continue
            p = f"{d}/eval.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p) if l.strip()]
            s = [(r.get("episode", 0), r["episode_bests"]) for r in rows if r.get("episode_bests")]
            if s:
                out.append(sorted(s))
    return out


def qdarts_delivered(mol):
    """QuantumDARTS is deterministic NAS: its deliverable is a SINGLE circuit (best_circuit /
    result_seed.npz['energy_error_ha']), NOT a policy-sample distribution. One value per seed;
    avg==opt by construction (there is no re-sampling at deploy)."""
    import numpy as _np
    vals = []
    for d in glob.glob(f"{C}/psqas/qdarts/{PMK[mol]}/*/*/seed*"):
        f = glob.glob(f"{d}/result_seed*.npz")
        if not f:
            continue
        z = _np.load(f[0], allow_pickle=True)
        vals.append(float(z["energy_error_ha"]) * 1000.0)
    return vals


def agg(method, mol):
    """returns dict with (mean,std,ci) for each of B1/B2 x bok1/bok10 over seeds."""
    if method == "qdarts":
        v = qdarts_delivered(mol)
        if not v:
            return None
        x = np.asarray(v, float)
        sd = x.std(ddof=1) if x.size > 1 else 0.0
        t = (x.mean(), sd, T4 * sd / np.sqrt(x.size))
        return {"n": len(v), "B1_1": t, "B1_10": t, "B2_1": t, "B2_10": t, "single": True}
    b1_1, b1_10, b2_1, b2_10 = [], [], [], []
    for series in run_series(method, mol):
        b1_1.append(bestofK(series[-1][1], 1))
        b1_10.append(bestofK(series[-1][1], 10))
        tail = series[-K_CKPT:]
        b2_1.append(np.mean([bestofK(e, 1) for _, e in tail]))
        b2_10.append(np.mean([bestofK(e, 10) for _, e in tail]))
    if not b1_1:
        return None

    def ms(x):
        x = np.asarray(x, float)
        sd = x.std(ddof=1) if x.size > 1 else 0.0
        return x.mean(), sd, T4 * sd / np.sqrt(x.size)
    return {"n": len(b1_1), "B1_1": ms(b1_1), "B1_10": ms(b1_10),
            "B2_1": ms(b2_1), "B2_10": ms(b2_10)}


def cell(t):
    return f"{t[0]:.3g}±{t[1]:.2g}"


DOC = r"""================================================================================
DreamQAS — MAIN POLICY-QUALITY TABLE  (methodology + experiments, for writing)
================================================================================

0. TASK
   Quantum architecture search (QAS) for VQE ground-state energy. An agent builds
   a parameterized quantum circuit gate-by-gate; at each added gate the circuit's
   ground-state energy is optimized (VQE) and the true error vs the FCI energy is
   the signal. Goal: reach chemical accuracy (1.6 mHa) with few VQE calls, and,
   for a *reusable* agent, produce good circuits reliably at deployment.

1. METHODS (7)
   Ours / ablations (share the DreamQAS codebase; on-policy REINFORCE actor:
   hidden 512x3, lr 3e-5, entropy 1e-3, gamma 0.99, grad-clip 10):
     - DreamQAS-Full : model-based RL. Energy-surrogate GRU world model (WM),
                       ensemble K=3 (independent members + Random-Prior Functions,
                       beta=3), replay buffer (prioritized/elite/stratified/random),
                       multi-step imagination (horizon 15, 64 seeds, lambda 0.95,
                       gamma 0.99), actor conditioned on WM ensemble features,
                       DAgger, DIR error-histogram reweighting, pessimism beta=1,
                       fidelity gate tau=0.70.  (PopArt off, potential-head off.)
     - No-imag       : DreamQAS with imagination DISABLED (keeps WM representation
                       + buffer + DAgger). Isolates the imagination contribution.
                       NOTE: not run for BeH2_6q (that molecule was main-only).
     - RLQAS (v2)    : pure on-policy REINFORCE on raw observations; NO world model,
                       NO buffer, NO imagination, NO DAgger. Tight ablation =
                       DreamQAS minus the entire WM pipeline.
   External baselines (PSQASBench):
     - CRLQAS        : DQN + N-step returns + curriculum.
     - HyRLQAS       : hybrid REINFORCE (discrete gate + continuous angle).
     - GQE           : generative model over circuits.
     - QuantumDARTS  : differentiable NAS -> a single deterministic architecture.

2. MOLECULES (5) and reference (FCI) energies; chemical accuracy = 1.6 mHa:
     LiH4q   :  4 qubits, max depth 40, FCI = -7.789089 Ha
     BeH2_6q :  6 qubits, max depth 50, FCI = -14.861589 Ha
     LiH6q   :  6 qubits, max depth 50, FCI = -7.844879 Ha   (2x training budget)
     BeH2_8q :  8 qubits, max depth 50, FCI = -15.761504 Ha  (GPU rotosolve VQE)
     BeH2_10q: 10 qubits, max depth 50, FCI = -15.765124 Ha  (GPU rotosolve VQE)

3. TRAINING BUDGET
   15,000 episodes (30,000 for LiH6q) = 3,750 iterations (7,500 for LiH6q) at
   4 real episodes/iteration. 5 random seeds per (method, molecule).
   Total training VQE (Full): ~0.56M (LiH4q), 0.62M (6q), 1.63M (LiH6q),
   0.75M (8q), 0.77M (10q). VQE COMPARABILITY: Full/No-imag/RLQAS/CRLQAS/HyRLQAS
   use comparable per-step VQE (~0.6-0.8M); GQE (~7k) and QuantumDARTS (~200
   circuits) use far fewer, non-comparable -- this affects the speed/efficiency
   axis ONLY, never the quality numbers below.

4. FROZEN OFFLINE EVALUATION (how every number here is measured)
   Checkpoints are saved on a fixed schedule identical across ALL methods: 15
   log-spaced points within the first 1/4 of the budget + one every 1,500 episodes
   to the full budget (~23 checkpoints; 30 for LiH6q). At each checkpoint the policy
   is FROZEN and run for N fresh, side-effect-free episodes to the fixed max depth
   (accuracy early-stop OFF; only structural termination). N = 20 (intermediate),
   100 (final checkpoint). Training RNG is isolated so eval never perturbs training.
   Per eval episode j the reduction is EPISODE-BEST (min-over-prefix):
       e_j = min_t | E(prefix_t) - E_FCI | * 1000   [mHa]
   i.e. the best circuit the policy yields within that episode (each VQE-evaluated
   prefix is a candidate circuit). This is the reduction used by the whole table.

5. METRICS (the two reported columns)
   best-of-K = expected error of the best of K i.i.d. deployments, estimated from
   the N per-episode e_j by an order-statistic estimator:
       sort e ascending; w_i = ((n-i+1)/n)^K - ((n-i)/n)^K ; best-of-K = sum_i e_i w_i
   ONE estimator, two K:
       best-of-1  (K=1) -> uniform weights -> arithmetic mean  = AVERAGE single-
                    deploy quality  (headline metric: reliability on the FIRST try)
       best-of-10 (K=10)-> OPTIMAL error over 10 deployments   (secondary column)

6. AGGREGATION OVER 5 SEEDS (principled, defensible)
   The per-checkpoint best-of-K is computed PER SEED; we then report mean +/- std
   over the 5 seed values. The SEED is the experimental unit -- episodes are NEVER
   pooled across seeds (that would fake n and shrink the variance). 95% CI (shown on
   request) = t_4 * std / sqrt(5), t_4 = 2.776.
   Block-1 (main): the converged (final) checkpoint.
   Block-2 (robustness): per seed, mean over the last <<K>> checkpoints of best-of-K,
     then aggregated over seeds. Block-2 ~= Block-1 shows the converged number is
     NOT an artifact of which checkpoint happened to be saved last (timing-robust).

7. qdarts (QuantumDARTS) SPECIAL-CASE
   Deterministic differentiable NAS: it delivers a SINGLE circuit, not a redeployable
   stochastic policy, so best-of-1/best-of-10 collapse to one value. It is INCLUDED in
   Table A on a separate '†' row, reporting its delivered circuit =
   result_seed.npz['energy_error_ha'] (= best_circuit.txt), mean +/- std over seeds
   (best-of-1 == best-of-10). The '†' footnote states it is the delivered circuit (best
   over its search), NOT a sampled-policy value, hence not directly comparable to the
   sampled best-of-K columns -- a reader must not read qdarts vs the policy rows as
   like-for-like (qdarts' number is a best-over-search, the others' is a single deploy).

8. WHAT IS *NOT* USED AS A QUALITY METRIC (appendix, transparency only)
   "best-found" = the single best circuit ever found during training. It is
   confounded by exploration breadth: a higher-variance / more random policy samples
   a wider circuit distribution and can attain equal or lower best-found (in the
   limit, random search maximizes it given enough budget). It saturates near the
   accuracy floor across all methods here and does NOT distinguish policy quality;
   reported in the appendix for transparency, not used to rank methods.

9. DATA SOURCES
   DreamQAS-family: analysis/eval_traces.jsonl (ep_best per checkpoint), written by
     analysis/eval_policy_traces.py (read-only reuse of the training rollout).
   PSQAS (crlqas/hyrlqas/gqe/qdarts): native eval.jsonl (episode_bests), written by
     PSQASBench's inline frozen eval (same N=20/100 protocol + same checkpoint grid).
   Final-circuit reduction (appendix): PSQASBench/offline_final_eval.py.

--------------------------------------------------------------------------------
TABLES BELOW: all values mHa (lower is better), mean +/- std over 5 seeds.
Table A = MAIN (policy quality, 6 policy methods, no qdarts).
Detail  = both blocks + all methods (incl. qdarts single-value + Block-2 robustness).
--------------------------------------------------------------------------------""".replace("<<K>>", str(K_CKPT))

POLICY_METHODS = ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS", "GQE", "TFQAS"]


def _tableA(out, key, title):
    out.append(f"\n{'='*100}\n{title}   [mean +/- std over 5 seeds, mHa]\n{'='*100}")
    out.append(f"{'method':10s} | " + " | ".join(f"{DISP[m]:^15s}" for m in MOLS))
    out.append("-" * 100)
    for m in POLICY_METHODS:                        # sampled-policy methods (best-of-K well-defined)
        cells = []
        for mol in MOLS:
            a = agg(m, mol)
            cells.append(cell(a[key]) if a else "-")
        out.append(f"{m:10s} | " + " | ".join(f"{c:^15s}" for c in cells))
    # QuantumDARTS: deterministic NAS -> single delivered circuit (best-of-1 == best-of-10), marked '†'
    cells = []
    for mol in MOLS:
        a = agg("qdarts", mol)
        cells.append(cell(a[key]) if a else "-")
    out.append(f"{'qdarts †':10s} | " + " | ".join(f"{c:^15s}" for c in cells))
    out.append("  † QuantumDARTS = deterministic differentiable NAS: value is its SINGLE DELIVERED circuit")
    out.append("    (best over its search, result_seed.npz); best-of-1 == best-of-10 by construction. Not a")
    out.append("    sampled-policy value -> not directly comparable to the sampled best-of-K columns above.")


def main():
    out = [DOC]
    _tableA(out, "B1_1", "TABLE A1 -- best-of-1  (AVERAGE single-deploy policy quality)  [HEADLINE]")
    _tableA(out, "B1_10", "TABLE A2 -- best-of-10 (OPTIMAL over 10 deploys)              [secondary]")
    out.append(f"\n{'='*100}\nDETAIL -- all methods, Block-1 (converged) vs Block-2 (last-{K_CKPT}, timing-robust)\n{'='*100}")
    for mol in MOLS:
        out.append(f"\n### {DISP[mol]}")
        hdr = (f"  {'method':9s} | {'B1 avg(bo1)':>14s} {'B1 opt(bo10)':>14s} | "
               f"{'B2 avg(bo1)':>14s} {'B2 opt(bo10)':>14s}")
        out.append(hdr); out.append("  " + "-" * (len(hdr) - 2))
        for m in METH:
            a = agg(m, mol)
            if not a:
                out.append(f"  {m:9s} | pending (No-imag not run for BeH2_6q)"); continue
            note = "  <qdarts: single delivered circuit; APPENDIX only>" if m == "qdarts" else ""
            out.append(f"  {m:9s} | {cell(a['B1_1']):>14s} {cell(a['B1_10']):>14s} | "
                       f"{cell(a['B2_1']):>14s} {cell(a['B2_10']):>14s}{note}")
    rep = "\n".join(out)
    print(rep)
    d = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/policy_quality_table.txt", "w").write(rep)
    print(f"\n[written] {d}/policy_quality_table.txt")


if __name__ == "__main__":
    main()
