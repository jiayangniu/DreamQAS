"""WM-Greedy 'search' deployment eval: N stochastic epsilon-greedy rollouts on the FINAL checkpoint,
so best-of-1 / best-of-10 reflect deploy-as-search (fair upper bound) rather than the single
deterministic greedy circuit (eps=0). Writes <run_dir>/eval_explore.json = {"ep_best":[...], ...}.

Usage: python analysis/wmg_explore_eval.py <run_dir> [--device cpu] [--eps 0.1] [--n 100]
No params/buffers/training-VQE touched (side-effect-free like eval_policy_traces).
"""
import sys, os, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_surrogate"))
import numpy as np
import torch
from config import Config
from runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    cks = sorted(glob.glob(f"{a.run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid})
    cfg.device = a.device
    cfg.log_raw = False
    cfg.wm_greedy_eps = a.eps
    r = Runner(cfg, _eval_mode=True)
    r.actor.load_state_dict(ck["actor"])
    r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and getattr(r, "scale", None) is not None:
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval(); r.wm.eval()
    r.greedy_on = True                       # activate WM-greedy selection (eval_mode=False => uses cfg.eps)

    seed = 100003 + 1009 * int(cfg.seed) + int(ck["episode"])
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 31 - 1))
    ep_best = []
    for _ in range(a.n):
        out = r._rollout(count_vqe=False, update_buffer=False, update_best=False,
                         run_to_max_depth=True, eval_mode=False)   # eval_mode=False -> eps-greedy
        errs = out["errs"]
        ep_best.append(float(np.min(errs)) if len(errs) else float("inf"))
    rec = {"episode": int(ck["episode"]), "eps": a.eps, "n": a.n,
           "ep_best": ep_best, "mean": float(np.mean(ep_best)), "min": float(np.min(ep_best))}
    json.dump(rec, open(f"{a.run_dir}/eval_explore.json", "w"))
    print(f"[explore] {os.path.basename(a.run_dir)} eps={a.eps} n={a.n} "
          f"bo1(mean)={rec['mean']:.3f} min={rec['min']:.3f}")


if __name__ == "__main__":
    main()
