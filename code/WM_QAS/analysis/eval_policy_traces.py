import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase2_surrogate"))
import numpy as np
import torch


def _eval_seed(seed, episode):
    return 100003 + 1009 * int(seed) + int(episode)


def _reload_dreamqas(ck, device):
    from config import Config
    from runner import Runner
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid})
    # Enable reference-energy reporting for frozen evaluation.
    cfg.enable_ground_truth_diagnostics = True
    if device is not None:
        cfg.device = device
    r = Runner(cfg, _eval_mode=True)
    r.actor.load_state_dict(ck["actor"]); r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and getattr(r, "scale", None) is not None:
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval(); r.wm.eval()
    return r, int(cfg.seed)


def _trace_dreamqas(r):
    out = r._rollout(count_vqe=False, update_buffer=False, update_best=False,
                     run_to_max_depth=True, eval_mode=True)
    errs = np.asarray(out["errs"], np.float64)
    nl = out.get("errs_noiseless")
    errs_nl = errs if nl is None else np.asarray(nl, np.float64)
    return errs, errs_nl


def _reload_rlqas(ck, device, legacy_noise_config=None):
    from runner_baseline import load_baseline_checkpoint
    return load_baseline_checkpoint(ck, device, legacy_noise_config), int(ck.get("seed", 0))


def _trace_rlqas(r):
    env = r.env; env.reset()
    obs = torch.tensor(r._get_obs_structure(), device=r.device)
    errs, errs_nl = [], []
    with torch.no_grad():
        for step in range(env.num_layers + 1):
            ill = env.illegal_action_new()
            if ill and len(ill) >= r.action_size:
                break
            env_action, _ = r.policy.act(obs, ill_actions=ill if ill else None)
            env.step(env_action, torch.tensor(0.0, device=r.device), train_flag=True)
            errs.append(float(abs(env.min_energy - env.energy)) * 1000.0)
            errs_nl.append(float(abs(
                env.min_energy - getattr(env, "energy_noiseless", env.energy))) * 1000.0)
            obs = torch.tensor(r._get_obs_structure(), device=r.device)
            if step >= env.num_layers - 1:
                break
    return np.asarray(errs, np.float64), np.asarray(errs_nl, np.float64)


def _episode_reductions(errs, errs_nl=None):
    if errs_nl is None:
        errs_nl = errs
    if errs.size == 0:
        return dict(best=float("inf"), mean=float("inf"), median=float("inf"),
                    var=0.0, final=float("inf"), n_prefix=0,
                    best_noiseless=float("inf"), final_noiseless=float("inf"))
    # Select using observed errors; report the same prefix without noise.
    i = int(np.argmin(errs))
    return dict(best=float(errs[i]), mean=float(errs.mean()), median=float(np.median(errs)),
                var=float(errs.var(ddof=1)) if errs.size > 1 else 0.0,
                final=float(errs[-1]), n_prefix=int(errs.size),
                best_noiseless=float(errs_nl[i]), final_noiseless=float(errs_nl[-1]))


def run(run_dir, device=None, n_inter=20, n_final=100, overwrite=False, last_k=0, only_ep=0, legacy_noise_config=None):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if min(n_inter, n_final) < 1:
        raise ValueError("Evaluation episode counts must be positive")
    cks = sorted(glob.glob(f"{run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int(os.path.basename(p)[2:-3]))
    if not cks:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/ckpt/")
    if only_ep:
        pick = min(cks, key=lambda p: abs(int(os.path.basename(p)[2:-3]) - only_ep))
        picked_ep = int(os.path.basename(pick)[2:-3])
        out_path = f"{run_dir}/eval_traces_ep{picked_ep}.jsonl"
        targets = [pick]
        n_inter = n_final
        print(f"[traces] only_ep~{only_ep} -> {os.path.basename(pick)} -> {os.path.basename(out_path)}")
        final_ep = picked_ep
    else:
        out_path = f"{run_dir}/eval_traces.jsonl"
        final_ep = max(int(os.path.basename(p)[2:-3]) for p in cks)
        targets = cks[-last_k:] if last_k and last_k < len(cks) else cks
        if final_ep not in [int(os.path.basename(p)[2:-3]) for p in targets]:
            targets = targets + [p for p in cks if int(os.path.basename(p)[2:-3]) == final_ep]
    done_eps, kept = set(), []
    if os.path.exists(out_path) and not overwrite:
        for l in open(out_path):
            if not l.strip():
                continue
            try:
                rec = json.loads(l); done_eps.add(rec.get("_ck_ep", rec["episode"])); kept.append(l.rstrip("\n"))
            except Exception:
                pass
    todo = [p for p in targets if int(os.path.basename(p)[2:-3]) not in done_eps]
    if not todo:
        print(f"[traces] skip (targets present: {len(done_eps)} recs) {run_dir}"); return
    fout = open(out_path, "w")
    for l in kept:
        fout.write(l + "\n")
    fout.flush()
    for p in todo:
        ep = int(os.path.basename(p)[2:-3])
        n = n_final if ep == final_ep else n_inter
        ck = torch.load(p, map_location="cpu", weights_only=False)
        is_dq = "wm" in ck
        r, seed = _reload_dreamqas(ck, device) if is_dq else _reload_rlqas(ck, device, legacy_noise_config)
        torch.manual_seed(_eval_seed(seed, ep)); np.random.seed(_eval_seed(seed, ep) % (2 ** 31 - 1))
        trace = _trace_dreamqas if is_dq else _trace_rlqas
        eps = [_episode_reductions(*trace(r)) for _ in range(n)]
        rec = {"_ck_ep": ep, "episode": int(ck["episode"]), "cum_train_vqe_calls": int(ck["vqe_calls"]),
               "cum_train_vqe_nfev": int(ck.get("vqe_nfev", 0)), "n_eval_episodes": n,
               "method": "Full" if is_dq else "RLQAS",
               "ep_best": [e["best"] for e in eps], "ep_mean": [e["mean"] for e in eps],
               "ep_median": [e["median"] for e in eps], "ep_var": [e["var"] for e in eps],
               "ep_final": [e["final"] for e in eps]}
        rec["ep_best_noiseless"] = [e["best_noiseless"] for e in eps]
        rec["ep_final_noiseless"] = [e["final_noiseless"] for e in eps]
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        bb = np.array(rec["ep_best"])
        print(f"[traces] ep{ep:>6} n={n:>3} best-of-N={bb.min():.4f} mean-ebest={bb.mean():.4f} "
              f"cum_vqe={rec['cum_train_vqe_calls']}", flush=True)
        del r
    fout.close()
    print(f"[traces] wrote {out_path} ({len(cks)} checkpoints)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default=None)
    ap.add_argument("--legacy_noise_config", type=json.loads, help="Explicit noise JSON for legacy baseline checkpoints")
    ap.add_argument("--n_inter", type=int, default=20)
    ap.add_argument("--n_final", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--last_k", type=int, default=0, help="only eval the last K checkpoints (+final); resumable")
    ap.add_argument("--only_ep", type=int, default=0, help="eval ONLY the ckpt nearest this episode -> eval_traces_ep<N>.jsonl")
    a = ap.parse_args()
    run(a.run_dir, a.device, a.n_inter, a.n_final, a.overwrite, a.last_k, a.only_ep, a.legacy_noise_config)
