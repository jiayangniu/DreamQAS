"""VQE-to-target tables for the RL methods (comparable per-step VQE), two modes:

  --mode policy (default) : cumulative training VQE to reach each K=1 POLICY-QUALITY node.
      K=1(t) = sliding-100-iter mean of the policy's per-episode full-depth error; running-best
      K1 vs cum-VQE is monotone. Nodes auto-inferred per molecule. Speedup rows =
      baseline_VQE / Full_VQE at the same policy level.
      -> outputs/main_results/vqe_to_policy_level.txt

  --mode error : cumulative training VQE to reach each ERROR node (training running-min
      best-error trajectory). DreamQAS-Full is CAPPED at 1/4 of its training budget (the
      intended claim is 1/4-budget DreamQAS vs full-budget baselines). Per (method, node) ->
      (#reaching seeds / N, median VQE among reachers).
      -> outputs/main_results/vqe_to_error_nodes.txt

(merged 2026-07-18 from the former vqe_to_nodes.py = the error mode; policy mode unchanged.)
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"; W = 100
PMK = {"LiH4q": "DQ_LiH_4q", "LiH6q": "DQ_LiH_6q", "BeH2": "DQ_BeH2_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH4q", "BeH2": "BeH2_6q", "LiH6q": "LiH6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
RL = ["Full", "No-imag", "RLQAS", "crlqas", "hyrlqas"]
NAME = {"Full": "DreamQAS-Full", "No-imag": "No-imag", "RLQAS": "RLQAS", "crlqas": "CRLQAS", "hyrlqas": "HyRLQAS"}


# ---------------------------------------------------------------- policy mode

def series(method, mol):
    """list of (cum_vqe[], mean_ep_err[]) per seed, ascending by vqe."""
    out = []
    if method == "Full":
        G = glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q/metrics.jsonl")
    elif method == "No-imag":
        G = glob.glob(f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag/metrics.jsonl")
    elif method == "RLQAS":
        G = glob.glob(f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{mol}/seed_*/metrics.jsonl")
    else:
        G = []
    for f in G:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        fld, sc = ("mean_ep_error", 1000.0) if method == "RLQAS" else ("mean_ep_err_mHa", 1.0)
        v = np.array([r["vqe_calls"] for r in rows if r.get(fld) is not None], float)
        e = np.array([r[fld] * sc for r in rows if r.get(fld) is not None], float)
        if len(e) >= W:
            out.append((v, e))
    if method in ("crlqas", "hyrlqas"):
        for d in glob.glob(f"{C}/psqas/{method}/{PMK[mol]}/*/*/seed*"):
            try:
                rows = list(csv.DictReader(open(f"{d}/episode_summary.tsv"), delimiter="\t"))
                err = np.array([float(r["final_energy_error_ha"]) * 1000 for r in rows])
                nst = np.array([float(r["n_steps"]) for r in rows])
                k = len(err) // 4 * 4
                e = err[:k].reshape(-1, 4).mean(1)                    # mean_ep_err per iter (4 eps)
                v = np.cumsum(nst[:k].reshape(-1, 4).sum(1))          # cum per-step VQE per iter
                if len(e) >= W:
                    out.append((v, e))
            except Exception:
                pass
    return out


def running_best_k1(v, e):
    """monotone (cum_vqe, best-K1-so-far): K1(i)=mean(e[i-W+1:i+1]); running cummin."""
    c = np.cumsum(np.insert(e, 0, 0))
    win = (c[W:] - c[:-W]) / W                                       # window means, ends at iter W..len
    vv = v[W - 1:]
    return vv, np.minimum.accumulate(win)


def vqe_to_policy(method, mol, node):
    hits = []
    for v, e in series(method, mol):
        vv, rb = running_best_k1(v, e)
        idx = np.where(rb <= node)[0]
        if idx.size:
            hits.append(float(vv[idx[0]]))
    return hits, len(series(method, mol))


def infer_policy_nodes(mol):
    finals = []
    for m in RL:
        s = series(m, mol)
        if s:
            finals.append(np.median([running_best_k1(v, e)[1][-1] for v, e in s]))
    if not finals:
        return []
    fine, coarse = min(finals), max(finals) * 1.5
    if coarse <= fine:
        coarse = fine * 50
    return [float(f"{x:.2g}") for x in np.geomspace(coarse, fine, 5)]


def run_policy_mode():
    out = ["VQE calls (k) to reach each K=1 POLICY-QUALITY node — median(#reachers/N).  RL methods (comparable per-step VQE)."]
    for mol in MOLS:
        nodes = infer_policy_nodes(mol)
        if not nodes:
            continue
        out.append(f"\n### {DISP[mol]}   policy-quality nodes K=1 (mHa): " + ", ".join(f"{n:g}" for n in nodes))
        hdr = f"{'method':14s} " + " ".join(f"{n:>13g}" for n in nodes); out.append(hdr); out.append("-" * len(hdr))
        vqe_full = {}
        for m in RL:
            cells = []
            for n in nodes:
                h, N = vqe_to_policy(m, mol, n)
                if m == "Full":
                    vqe_full[n] = np.median(h) if h else None
                cells.append(f"{np.median(h)/1000:.0f}k({len(h)}/{N})" if h else f"-(0/{N})")
            out.append(f"{NAME[m]:14s} " + " ".join(f"{c:>13s}" for c in cells))
        # speedup rows: baseline_VQE / Full_VQE at matched policy level (only where both reach)
        for m in ["RLQAS", "crlqas", "hyrlqas"]:
            cells = []
            for n in nodes:
                h, _ = vqe_to_policy(m, mol, n)
                vf = vqe_full.get(n)
                cells.append(f"{(np.median(h)/vf):.1f}x" if (h and vf) else "—")
            out.append(f"  speed {NAME[m][:8]:8s} " + " ".join(f"{c:>13s}" for c in cells))
    rep = "\n".join(out)
    print(rep)
    os.makedirs(f"{HERE}/outputs/main_results", exist_ok=True)
    open(f"{HERE}/outputs/main_results/vqe_to_policy_level.txt", "w").write(rep)


# ----------------------------------------------------------------- error mode
# (former vqe_to_nodes.py; needs main_results helpers + the training config)

ERR_RL = ["Full", "RLQAS", "crlqas", "hyrlqas"]


def run_error_mode():
    sys.path.insert(0, HERE)
    from main_results import find_runs, training_best_traj, MOL_DISP, METHOD_DISP
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase2_surrogate"))
    from config import N_ITERS

    def quarter_vqe(run_dir, mol):
        """cum_vqe at iter = N_ITERS[mol]/4 for a DreamQAS-Full run (the 1/4-budget point)."""
        rows = [json.loads(l) for l in open(f"{run_dir}/metrics.jsonl") if l.strip()]
        q_iter = N_ITERS[mol] / 4.0
        row = min(rows, key=lambda r: abs(r["iter"] - q_iter))
        return float(row["vqe_calls"])

    def traj_capped(run_dir, method, mol):
        """(cum_vqe, running-min best) — Full capped at its 1/4-budget VQE."""
        t = training_best_traj(run_dir, method)
        if t is None:
            return None
        v, b = t
        if method == "Full":
            cap = quarter_vqe(run_dir, mol)
            m = v <= cap
            v, b = v[m], b[m]
        return (v, b) if v.size else None

    def infer_nodes(per_method_finals):
        """5 log-spaced error nodes spanning a molecule's achieved range:
        from 2x the WORST method's final (easy, ~all reach) down to the BEST final (hardest)."""
        vals = [x for x in per_method_finals.values() if x is not None and np.isfinite(x)]
        if not vals:
            return []
        fine = min(vals)
        coarse = max(vals) * 2.0
        if coarse <= fine:
            coarse = fine * 100
        nodes = np.geomspace(coarse, fine, 5)
        # round to 2 significant figures for readability
        return [float(f"{x:.2g}") for x in nodes]

    def vqe_to(vb, thr):
        v, b = vb
        idx = np.where(b <= thr)[0]
        return float(v[idx[0]]) if idx.size else None

    g = find_runs()
    out = []
    for mol in MOLS:
        # per-method final running-min best (median across seeds), for node inference
        finals = {}
        per_run_traj = {}
        for m in ERR_RL:
            trajs = []
            for r in sorted(g.get((m, mol), [])):
                vb = traj_capped(r, m, mol)
                if vb:
                    trajs.append(vb)
            per_run_traj[m] = trajs
            if trajs:
                finals[m] = float(np.median([b[-1] for _, b in trajs]))
        nodes = infer_nodes(finals)
        if not nodes:
            continue
        out.append(f"\n### {MOL_DISP[mol]}   (Full = 1/4 budget; baselines = full)   error nodes (mHa): "
                   + ", ".join(f"{n:g}" for n in nodes))
        hdr = f"{'method':14s} " + " ".join(f"{n:>13g}" for n in nodes)
        out.append(hdr); out.append("-" * len(hdr))
        for m in ERR_RL:
            trajs = per_run_traj[m]
            cells = []
            for n in nodes:
                hits = [vqe_to(vb, n) for vb in trajs]
                hits = [h for h in hits if h is not None]
                if hits:
                    cells.append(f"{np.median(hits)/1000:.1f}k({len(hits)}/{len(trajs)})")
                else:
                    cells.append(f"-({0}/{len(trajs)})")
            tag = METHOD_DISP[m] + ("*" if m == "Full" else "")
            out.append(f"{tag:14s} " + " ".join(f"{c:>13s}" for c in cells))
    rep = "\n".join(out)
    print(rep)
    os.makedirs(f"{HERE}/outputs/main_results", exist_ok=True)
    open(f"{HERE}/outputs/main_results/vqe_to_error_nodes.txt", "w").write(
        "VQE calls (thousands) to reach each error node — median(#reachers/N).\n"
        "* DreamQAS-Full capped at 1/4 training budget; baselines at full budget.\n" + rep)
    print(f"\n[written] outputs/main_results/vqe_to_error_nodes.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["policy", "error"], default="policy")
    args = ap.parse_args()
    run_policy_mode() if args.mode == "policy" else run_error_mode()
