"""Unified FULL-cost accounting for the supplementary contrast (reviewer-Q3).

Recovers per (method, molecule) cost fields from EXISTING artifacts and tags each field's provenance:
  measured      = read directly from a run log/metric;
  reconstructed = deterministically derived from config + code + logged counts;
  unavailable   = not reliably recoverable (must come from the controlled timing pass).

Quantum/VQE side (per training run): vqe_calls, calib_vqe_calls (measured, metrics.jsonl last row),
  vqe_nfev (measured, best_circuit.json). Eval-VQE is separate (frozen best-of-1) and reconstructed from
  the eval protocol. Classical side: WM-queries reconstructed (imag.jsonl imag_horizon_used*n_seeds*iters,
  or raw_wm_queries when transition-budgeted; beam from beam_eval.json); wall-clock reconstructed from the
  console `t=` (flagged cross-run-INCOMPARABLE -> needs controlled timing). Statevector = simulation only.

Usage: python analysis/cost_table.py   (edit C / METHODS below or import build_row)
"""
import json, glob, os, re
import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
# method -> (glob of run dirs, has_imagination)
METHODS = {
    "NoImag":    (f"{C}/ablations/gru_energy_none_{{m}}_s*_of_noimag", False),
    "WM-Greedy": (f"{C}/wmgreedy/gru_energy_none_{{m}}_s*_of_wmgreedy", False),
    "NoDAG-H15": (f"{C}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag", True),
    "Full":      (f"{C}/dreamqas/gru_energy_surrogate_{{m}}_s*_of", True),  # 5-mol oracle-free Full
}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]   # BeH2 == BeH2-6q (repo naming)
# Locked reporting budget (storyline §1): EVERY task is reported at 15,000 episodes = 3750 iters.
# LiH6q is the only task CONFIGURED for 2x, so without this cap its cost row would be a 7500-iter
# total sitting next to 3750-iter totals — which is exactly the doc/artifact drift this table is
# supposed to prevent. `vqe_nfev` cannot be capped (best_circuit.json is written once at run end and
# COBYLA's nfev grows with depth, so halving it would overestimate) -> reported as n/a when capped.
CAP_ITERS = {"LiH6q": 3750}
_TPAT = re.compile(r"t=(\d+)s")


def _cap_for(d):
    for m, c in CAP_ITERS.items():
        if f"_{m}_" in os.path.basename(d) or f"_{m}s" in os.path.basename(d):
            return c
    return None


def _last_metric(d):
    """Last metrics row AT OR BELOW the reporting cap (whole run when the task has no cap)."""
    f = f"{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    cap = _cap_for(d)
    if cap:
        rows = [r for r in rows if int(r.get("iter", 0)) < cap]
    return rows[-1] if rows else None


def _wm_queries(d, has_imag):
    """reconstructed classical WM-query count over training."""
    if not has_imag:
        return 0, "reconstructed(=0; NoImag/greedy select-only counted at deploy)"
    f = f"{d}/imag.jsonl"
    if not os.path.exists(f): return None, "unavailable"
    rows = [json.loads(l) for l in open(f) if l.strip()]
    cap = _cap_for(d)
    if cap:
        rows = [r for r in rows if int(r.get("iter", 0)) < cap]   # same reporting budget as vqe_calls
    # transition-budgeted runs log raw_wm_queries directly; else reconstruct horizon_used*n_seeds
    if rows and "raw_wm_queries" in rows[0]:
        q = sum(r.get("raw_wm_queries", 0) for r in rows)
    else:
        q = sum(r.get("imag_horizon_used", 0) * r.get("n_seeds", 0) for r in rows)
    # imag.jsonl logs once per wm_refresh_every iters -> scale to per-iter updates is already per-log-row
    return int(q), "reconstructed(imag.jsonl horizon_used*n_seeds)"


def _verify_vqe(d, has_imag):
    """DAgger/calibration verification VQE for ONE run -> (calls, provenance).

    dagger=1 (Full): the probe's VQE is charged to `vqe_calls` (the verified circuit trains the WM, so it
      belongs to the acceleration budget) and no separate counter exists -> reconstruct from the candidate
      lengths in calibration.jsonl. That over-counts slightly, because a candidate can terminate early;
      measured on the NoDAG arm the reconstruction runs ~10-17% above the true counter. RECONSTRUCTED.
    dagger=0 (NoDAG): the identical probe is charged to `calib_vqe_calls` and is NOT in the budget. MEASURED.
    NoImag / WM-Greedy: `calibrate()` is gated on imagination == "surrogate", so there is none.
    """
    if not has_imag:
        return 0, "none (calibrate() requires imagination=surrogate)"
    mt = _last_metric(d)
    if mt and mt.get("calib_vqe_calls"):
        return int(mt["calib_vqe_calls"]), "measured (calib_vqe_calls; NOT in budget)"
    f = f"{d}/calibration.jsonl"
    if not os.path.exists(f):
        return None, "unavailable"
    n = 0
    cap = _cap_for(d)          # same reporting budget as vqe_calls, else Full's verify would be a
    for l in open(f):          # full-budget total sitting inside a capped row
        if not l.strip():
            continue
        r = json.loads(l)
        if cap and int(r.get("iter", 0)) >= cap:
            continue
        if r.get("selection") != "SUMMARY" and r.get("len"):
            n += int(r["len"])
    return n, "reconstructed (candidate lengths; INSIDE vqe_calls; ~10-17% high)"


def _incomplete(d):
    """True if this run has not reached its configured budget. Cost rows are per-RUN totals, so a
    still-training run would silently contribute a partial cost under a final-result label."""
    try:
        c = json.load(open(f"{d}/config.json"))["config"]
        target = int(c["n_iterations"])
    except Exception:
        return False
    m = f"{d}/metrics.jsonl"
    if not os.path.exists(m):
        return True
    return sum(1 for l in open(m) if l.strip()) < target


def build_row(method, mol, run_glob, has_imag):
    all_dirs = sorted(glob.glob(run_glob.format(m=mol)))
    dirs = [d for d in all_dirs if not _incomplete(d)]
    n_partial = len(all_dirs) - len(dirs)
    vqe, nfev, wmq, bo1, ver = [], [], [], [], []
    vsrc = ""
    for d in dirs:
        mt = _last_metric(d)
        if mt: vqe.append(mt.get("vqe_calls")); bo1.append(mt.get("best_err_mHa"))
        bc = f"{d}/best_circuit.json"
        # nfev is an END-OF-RUN artifact; on a capped task it would describe a longer run than the
        # rest of the row, so it is withheld rather than silently mixed across budgets.
        if os.path.exists(bc) and not _cap_for(d): nfev.append(json.load(open(bc)).get("vqe_nfev"))
        q, _ = _wm_queries(d, has_imag)
        if q is not None: wmq.append(q)
        v, vsrc = _verify_vqe(d, has_imag)
        if v is not None: ver.append(v)
    def ms(x): x=[v for v in x if v is not None]; return (float(np.mean(x)) if x else None)
    return {
        "method": method, "molecule": mol, "n_seeds": len(dirs), "n_partial": n_partial,
        "vqe_calls": ms(vqe), "vqe_nfev": ms(nfev), "wm_queries": ms(wmq), "verify_vqe": ms(ver),
        "src_verify": vsrc,
        "train_best_mHa": ms(bo1),
        "src_vqe": "measured", "src_nfev": "measured",
        "src_wmq": "reconstructed" if has_imag else "reconstructed(=0)",
        "src_wall": "unavailable(controlled-timing)",
    }


if __name__ == "__main__":
    print("UNIFIED COST TABLE (existing runs; new H1/H5/Beam rows appended once available)")
    print(f"{'method':11s}{'mol':10s}{'seeds':>6s}{'vqe_calls':>12s}{'vqe_nfev':>12s}{'wm_queries':>12s}"
          f"{'verify_vqe':>12s}{'train_best':>11s}  verify-VQE provenance")
    for method, (g, hi) in METHODS.items():
        for mol in MOLS:
            r = build_row(method, mol, g, hi)
            if not r["n_seeds"]:
                if r["n_partial"]:
                    print(f"{method:11s}{mol:10s}{'—':>6s}   (all {r['n_partial']} run(s) still training — no row)")
                continue
            def f(x, k=0): return "  n/a" if x is None else (f"{x:.0f}" if k == 0 else f"{x:.3g}")
            print(f"{r['method']:11s}{r['molecule']:10s}{r['n_seeds']:>6d}{f(r['vqe_calls']):>12s}"
                  f"{f(r['vqe_nfev']):>12s}{f(r['wm_queries']):>12s}{f(r['verify_vqe']):>12s}"
                  f"{f(r['train_best_mHa'],1):>11s}  {r['src_verify']}"
                  + (f"   [+{r['n_partial']} run(s) STILL TRAINING, excluded]" if r["n_partial"] else ""))
    print("\nWM-queries=0 for NoImag/WM-Greedy is TRAINING WM-queries — those arms do no imagination.")
    print("Their DEPLOY-time candidate scoring is deliberately NOT tabulated here (decision 2026-07-28):")
    print("the storyline's cost spec covers Full/NoImag/NoDAG, and the Q1 conclusion does not rest on it.")
    print("If ever needed, beam's per-search WM-query count is already recorded in each beam_eval.json.")
    print("Wall-clock is UNAVAILABLE from historical logs (load-contended) -> controlled timing pass required.")
