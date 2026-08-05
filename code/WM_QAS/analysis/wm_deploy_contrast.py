"""Same WM, different DEPLOYMENT: does the learned actor beat direct surrogate exploitation?

The existing three-way contrast (`wmgreedy_threeway.txt`) compares whole PIPELINES — a run trained with
WM-greedy selection versus a run trained with imagined policy learning. A reviewer can object that the
greedy arm was handed a *worse* world model (one trained under its own greedy data distribution). This
script removes that objection by holding the MODEL fixed: it takes a run that was trained WITH imagination
(NoDAG / Full), freezes its WM **and** its actor, and evaluates three deployments of that one checkpoint:

    actor    the learned policy (the method's own deployment)
    greedy0  WM-Greedy, eps=0: enumerate all legal next gates, score each one step ahead under the frozen
             WM with the deployed pessimistic value phi = -(mean + beta*std), take argmax. No VQE.
    greedyE  the same selector with eps>0, N rollouts — a search-budget upper bound for the surrogate.

Every arm uses the identical rollout harness, the identical `run_to_max_depth` semantics and the identical
per-prefix real-VQE readout, so the ONLY thing that varies is how the WM is turned into an action.

⚠ SILENT-FALLBACK GUARD. `Runner._rollout` only takes the greedy branch when `cfg.select_mode ==
"wm_greedy"`. A run trained with the actor has `select_mode == "actor"`, so merely setting `greedy_on`
(as `wmg_explore_eval.py` does — correct there, because those runs were *trained* greedy) would silently
produce ACTOR numbers under a "greedy" label. Here we force `select_mode`, count the actual calls into
`_wm_greedy_action`, and abort if the count is zero.

Usage: python analysis/wm_deploy_contrast.py <run_dir> [--device cpu] [--n 100] [--eps 0.1]
Writes <run_dir>/wm_deploy_contrast.json. Side-effect-free: no params, buffers or training VQE touched.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_surrogate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from config import Config
from runner import Runner


def _load(run_dir, device, allow_partial=False):
    cks = sorted(glob.glob(f"{run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    if not cks:
        raise SystemExit(f"[deploy] no checkpoints in {run_dir}")
    # COMPLETENESS GUARD: this script reports the LAST checkpoint as if it were the trained model. A run
    # that is still training would silently contribute an early checkpoint under a final-result label.
    last_ep = int("".join(filter(str.isdigit, os.path.basename(cks[-1]))))
    try:
        c0 = json.load(open(f"{run_dir}/config.json"))["config"]
        target = int(c0["real_eps_per_iter"]) * int(c0["n_iterations"])
    except Exception:
        target = 0
    if target and last_ep < target and not allow_partial:
        raise SystemExit(f"[deploy] SKIP {os.path.basename(run_dir)}: still training "
                         f"(ep{last_ep}/{target}). Re-run when complete, or pass --allow_partial.")
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid})
    cfg.device = device
    cfg.log_raw = False
    r = Runner(cfg, _eval_mode=True)
    r.actor.load_state_dict(ck["actor"])
    r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and getattr(r, "scale", None) is not None:
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval(); r.wm.eval()
    return r, ck, cfg


def rollout_best(r, n, eval_mode):
    """n frozen rollouts -> per-episode best true error (mHa). eval_mode=True => deterministic greedy."""
    out = []
    for _ in range(n):
        o = r._rollout(count_vqe=False, update_buffer=False, update_best=False,
                       run_to_max_depth=True, eval_mode=eval_mode)
        e = o["errs"]
        out.append(float(np.min(e)) if len(e) else float("inf"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--allow_partial", action="store_true")
    a = ap.parse_args()

    r, ck, cfg = _load(a.run_dir, a.device, a.allow_partial)
    name = os.path.basename(a.run_dir)
    seed0 = 100003 + 1009 * int(cfg.seed) + int(ck["episode"])
    rec = {"run": name, "episode": int(ck["episode"]), "n": a.n, "eps": a.eps,
           "trained_select_mode": getattr(cfg, "select_mode", "actor"),
           "beta": float(getattr(cfg, "pessimism_beta", 0.0))}

    # ---- 1) the learned actor (this run's own deployment) ----
    cfg.select_mode = "actor"; r.greedy_on = False
    torch.manual_seed(seed0); np.random.seed(seed0 % (2 ** 31 - 1))
    act = rollout_best(r, a.n, eval_mode=False)
    rec["actor"] = {"bo1": float(np.mean(act)), "bo10": float(np.min(act[:10])), "min": float(np.min(act))}

    # ---- 2/3) the SAME frozen WM as a selector; guard against silently falling back to the actor ----
    calls = {"n": 0}
    orig = r._wm_greedy_action

    def counted(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)
    r._wm_greedy_action = counted
    cfg.select_mode = "wm_greedy"; r.cfg.select_mode = "wm_greedy"; r.greedy_on = True

    torch.manual_seed(seed0); np.random.seed(seed0 % (2 ** 31 - 1))
    g0 = rollout_best(r, 2, eval_mode=True)          # eps=0 is deterministic -> two draws must agree
    if calls["n"] == 0:
        raise SystemExit(f"[deploy] FATAL {name}: greedy selector never fired — the run silently used "
                         f"the actor. Do NOT report these numbers.")
    det = abs(g0[0] - g0[1]) < 1e-9
    rec["greedy0"] = {"bo1": float(g0[0]), "deterministic": bool(det)}

    cfg.wm_greedy_eps = a.eps; r.cfg.wm_greedy_eps = a.eps
    torch.manual_seed(seed0 + 7); np.random.seed((seed0 + 7) % (2 ** 31 - 1))
    ge = rollout_best(r, a.n, eval_mode=False)       # eval_mode=False -> eps-greedy over the SAME selector
    rec["greedyE"] = {"bo1": float(np.mean(ge)), "bo10": float(np.min(ge[:10])), "min": float(np.min(ge))}
    rec["greedy_calls"] = calls["n"]

    json.dump(rec, open(f"{a.run_dir}/wm_deploy_contrast.json", "w"))
    print(f"[deploy] {name:44s} actor={rec['actor']['bo1']:8.3f}  greedy0={rec['greedy0']['bo1']:8.3f}"
          f"  greedy_eps={rec['greedyE']['bo1']:8.3f}   det={det}  calls={calls['n']}")


if __name__ == "__main__":
    main()
