import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase2_surrogate"))
import numpy as np
import torch


def eval_one(ckpt_path, n_episodes, device, legacy_noise_config=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "wm" in ck:
        from phase2_surrogate import eval_harness
        return eval_harness.evaluate_checkpoint(ckpt_path, n_episodes, device=device)
    else:
        from runner_baseline import evaluate_baseline_checkpoint
        return evaluate_baseline_checkpoint(ckpt_path, n_episodes, device=device, legacy_noise_config=legacy_noise_config)


def run_eval_pass(run_dir, device=None, n_inter=20, n_final=100, legacy_noise_config=None):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if min(n_inter, n_final) < 1:
        raise ValueError("Evaluation episode counts must be positive")
    cks = sorted(glob.glob(f"{run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int(os.path.basename(p)[2:-3]))
    if not cks:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/ckpt/")
    final_ep = max(int(os.path.basename(p)[2:-3]) for p in cks)
    out = open(f"{run_dir}/eval.jsonl", "w")
    for p in cks:
        ep = int(os.path.basename(p)[2:-3])
        n = n_final if ep == final_ep else n_inter
        stats = eval_one(p, n, device, legacy_noise_config)
        out.write(json.dumps(stats, default=str) + "\n"); out.flush()
        print(f"[eval-pass] ep{ep:>6} n={n:>3} median={stats['median']} SR={stats['success_rate']} "
              f"cum_vqe={stats['cum_train_vqe_calls']}", flush=True)
    out.close()
    print(f"[eval-pass] wrote {run_dir}/eval.jsonl ({len(cks)} checkpoints)")


def _boot_ci(vals, reps=2000, lo=2.5, hi=97.5, seed=0):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, v.size, replace=True)) for _ in range(reps)]
    return (float(np.median(v)), float(np.percentile(meds, lo)), float(np.percentile(meds, hi)))


def aggregate(run_dirs, chem=1.6, targets=(1.6,)):
    from collections import defaultdict
    groups = defaultdict(list)
    for rd in run_dirs:
        ejl = f"{rd}/eval.jsonl"
        if not os.path.exists(ejl):
            continue
        rows = [json.loads(l) for l in open(ejl) if l.strip()]
        name = os.path.basename(rd.rstrip("/"))
        baseline = name.startswith("seed_")
        if baseline:
            name = os.path.basename(os.path.dirname(rd.rstrip("/")))
        mol = next((m for m in ("BeH2_12q", "BeH2_10q", "BeH2_8q", "H2O_8q", "BeH2", "LiH6q", "LiH4q") if m in name), "?")
        method = name.split(f"_{mol}_")[0] if f"_{mol}_" in name else name
        if baseline:
            method = "RLQAS"
        groups[(method, mol)].append(rows)

    report = {}
    for (method, mol), runs in sorted(groups.items()):
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
    p1 = sub.add_parser("pass"); p1.add_argument("run_dir"); p1.add_argument("--device", default=None)
    p1.add_argument("--legacy_noise_config", type=json.loads)
    p1.add_argument("--n_inter", type=int, default=20); p1.add_argument("--n_final", type=int, default=100)
    p2 = sub.add_parser("agg"); p2.add_argument("globs", nargs="+"); p2.add_argument("--chem", type=float, default=1.6)
    a = ap.parse_args()
    if a.cmd == "pass":
        run_eval_pass(a.run_dir, a.device, a.n_inter, a.n_final, a.legacy_noise_config)
    else:
        run_dirs = [d for g in a.globs for d in glob.glob(g) if os.path.isdir(d)]
        rep = aggregate(run_dirs, a.chem)
        if not rep:
            raise FileNotFoundError("No evaluation records matched the supplied run directories")
        for k, v in rep.items():
            last = v["curve"][-1] if v["curve"] else {}
            print(f"\n=== {k}  (n_seeds={v['n_seeds']}) ===")
            print(f"  final: cum_vqe={last.get('cum_vqe')}, median={last.get('median')}, "
                  f"SR={last.get('SR_mean')}")
            print(f"  VQE-to-target: {v['vqe_to_target']}")


if __name__ == "__main__":
    main()
