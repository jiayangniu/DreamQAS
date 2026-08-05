"""Standardized-eval-protocol driver + aggregation (C1).

Two stages:
  1. eval-pass: for a training run dir, reload each episode-checkpoint, run the frozen offline
     eval (20 fresh episodes for intermediate checkpoints, 100 for the final), and write one
     JSON line per checkpoint to `<run_dir>/eval.jsonl` — schema = episode_best_stats +
     {episode, cum_train_vqe_calls, cum_train_vqe_nfev}. Dispatches DreamQAS-method vs
     RLQAS-baseline checkpoints automatically.
  2. aggregate: read many runs' eval.jsonl, group by (method,molecule) across seeds, and emit
     the per-cum-VQE learning curve (cross-seed median + stratified-bootstrap 95% CI), the main
     table (mean±std / median / SR@chem at the ¼-budget point and the final point), and
     VQE-to-target (first checkpoint whose cross-seed median crosses a threshold).

CLI:
  python analysis/eval_protocol.py pass  <run_dir> [--device cuda:0] [--n_final 100]
  python analysis/eval_protocol.py agg   <glob...>  [--chem 1.6]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase2_surrogate"))
import numpy as np
import torch


# ---------------- stage 1: eval pass ----------------
def eval_one(ckpt_path, n_episodes, device):
    """Dispatch a single checkpoint to the right frozen-eval function -> stats dict."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "wm" in ck:                                    # DreamQAS method checkpoint
        from phase2_surrogate import eval_harness
        return eval_harness.evaluate_checkpoint(ckpt_path, n_episodes, device=device)
    else:                                             # RLQAS baseline checkpoint
        from runner_baseline import evaluate_baseline_checkpoint
        return evaluate_baseline_checkpoint(ckpt_path, n_episodes, device=device)


def run_eval_pass(run_dir, device="cuda:0", n_inter=20, n_final=100):
    cks = sorted(glob.glob(f"{run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int(os.path.basename(p)[2:-3]))
    if not cks:
        print(f"[eval-pass] no checkpoints under {run_dir}/ckpt/"); return
    final_ep = max(int(os.path.basename(p)[2:-3]) for p in cks)
    out = open(f"{run_dir}/eval.jsonl", "w")
    for p in cks:
        ep = int(os.path.basename(p)[2:-3])
        n = n_final if ep == final_ep else n_inter
        stats = eval_one(p, n, device)
        out.write(json.dumps(stats, default=str) + "\n"); out.flush()
        print(f"[eval-pass] ep{ep:>6} n={n:>3} median={stats['median']} SR={stats['success_rate']} "
              f"cum_vqe={stats['cum_train_vqe_calls']}", flush=True)
    out.close()
    print(f"[eval-pass] wrote {run_dir}/eval.jsonl ({len(cks)} checkpoints)")


# ---------------- stage 2: aggregation ----------------
def _boot_ci(vals, reps=2000, lo=2.5, hi=97.5, seed=0):
    """Bootstrap CI of the median across seeds (few seeds -> percentile bootstrap)."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, v.size, replace=True)) for _ in range(reps)]
    return (float(np.median(v)), float(np.percentile(meds, lo)), float(np.percentile(meds, hi)))


def aggregate(run_dirs, chem=1.6, targets=(1.6,)):
    """Group runs by (method, molecule); cross-seed learning curve + VQE-to-target."""
    from collections import defaultdict
    groups = defaultdict(list)   # (method,mol) -> list of per-run [ {episode,median,SR,cum_vqe,...} ]
    for rd in run_dirs:
        ejl = f"{rd}/eval.jsonl"
        if not os.path.exists(ejl):
            continue
        rows = [json.loads(l) for l in open(ejl) if l.strip()]
        name = os.path.basename(rd.rstrip("/"))       # e.g. gru_energy_surrogate_BeH2_8q_s0
        # crude parse: method-ish prefix + molecule token + seed
        mol = next((m for m in ("BeH2_10q", "BeH2_8q", "BeH2", "LiH6q", "LiH4q") if m in name), "?")
        method = name.split(f"_{mol}_")[0] if f"_{mol}_" in name else name
        groups[(method, mol)].append(rows)

    report = {}
    for (method, mol), runs in sorted(groups.items()):
        # align by episode across seeds
        by_ep = defaultdict(lambda: {"median": [], "cum_vqe": [], "SR": []})
        for rows in runs:
            for r in rows:
                by_ep[r["episode"]]["median"].append(r["median"])
                by_ep[r["episode"]]["cum_vqe"].append(r["cum_train_vqe_calls"])
                by_ep[r["episode"]]["SR"].append(r["success_rate"])
        curve = []
        for ep in sorted(by_ep):
            med, lo, hi = _boot_ci(by_ep[ep]["median"])
            curve.append({"episode": ep, "cum_vqe": float(np.mean(by_ep[ep]["cum_vqe"])),
                          "median": med, "ci_lo": lo, "ci_hi": hi,
                          "SR_mean": float(np.mean([s for s in by_ep[ep]["SR"] if s is not None]))
                          if by_ep[ep]["SR"] else None, "n_seeds": len(by_ep[ep]["median"])})
        vqe_to = {}
        for t in targets:
            hit = next((c for c in curve if c["median"] is not None and c["median"] <= t), None)
            vqe_to[t] = hit["cum_vqe"] if hit else None
        report[f"{method} | {mol}"] = {"curve": curve, "vqe_to_target": vqe_to,
                                       "n_seeds": len(runs)}
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pass"); p1.add_argument("run_dir"); p1.add_argument("--device", default="cuda:0")
    p1.add_argument("--n_inter", type=int, default=20); p1.add_argument("--n_final", type=int, default=100)
    p2 = sub.add_parser("agg"); p2.add_argument("globs", nargs="+"); p2.add_argument("--chem", type=float, default=1.6)
    a = ap.parse_args()
    if a.cmd == "pass":
        run_eval_pass(a.run_dir, a.device, a.n_inter, a.n_final)
    else:
        run_dirs = [d for g in a.globs for d in glob.glob(g) if os.path.isdir(d)]
        rep = aggregate(run_dirs, a.chem)
        for k, v in rep.items():
            last = v["curve"][-1] if v["curve"] else {}
            print(f"\n=== {k}  (n_seeds={v['n_seeds']}) ===")
            print(f"  final: cum_vqe={last.get('cum_vqe')}, median={last.get('median')}, "
                  f"SR={last.get('SR_mean')}")
            print(f"  VQE-to-target: {v['vqe_to_target']}")


if __name__ == "__main__":
    main()
