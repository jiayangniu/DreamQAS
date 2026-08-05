"""Standardized frozen offline evaluation + checkpointing (energy-surrogate method).

The shared reducer `episode_best_stats` is mirrored on the PSQASBench side so ALL methods
report identically. `save_checkpoint` / `evaluate_checkpoint` reload a frozen actor + world
model and run fresh, side-effect-free episodes to the FIXED max depth (accuracy-based early
stop disabled; only structural termination), taking e_j = min_t eps_{j,t} over the episode's
VQE-evaluated prefixes. For the rotosolve (8q/10q) path the recorded prefix energies come from
a complex128 reference backend, not the GPU complex64 search (handled in Runner._energy).
"""
import numpy as np
import torch


def episode_best_stats(episode_bests, chem_acc_mHa: float = 1.6) -> dict:
    """Reduce a list of per-episode best errors e_j (mHa) to the reported stats.
    Identical formula used by DreamQAS, the RLQAS baseline, and the PSQASBench baselines."""
    a = np.asarray([e for e in episode_bests if np.isfinite(e)], dtype=float)
    n = int(a.size)
    return {
        "n": n,
        "mean": float(a.mean()) if n else None,
        "median": float(np.median(a)) if n else None,
        "std": float(a.std(ddof=1)) if n > 1 else 0.0,
        "success_rate": float((a <= chem_acc_mHa).mean()) if n else None,   # SR@chem-acc
        "best": float(a.min()) if n else None,                              # best-of-N
    }


def save_checkpoint(runner, episode: int, path: str) -> None:
    """Frozen-EVAL checkpoint: actor + world model + config + cumulative TRAINING VQE
    counters + best circuit/depth + RNG. Reloadable by `evaluate_checkpoint`.
    NOTE: optimizer states are deliberately NOT saved -- evaluate_checkpoint only loads
    actor+wm for a frozen rollout and there is no resume-training path, so persisting the
    Adam moments (~2x the model each, per WM-ensemble member) would ~3x the checkpoint size
    and the on-disk footprint for nothing."""
    torch.save({
        "episode": episode,
        "iter": getattr(runner, "_cur_iter", -1),
        "actor": runner.actor.state_dict(),
        "wm": runner.wm.state_dict(),
        "cfg": dict(runner.cfg.__dict__),
        "vqe_calls": runner.vqe_calls,                       # real rollout + DAgger (budget)
        "vqe_nfev": getattr(runner, "vqe_nfev", 0),          # supplementary inner feval count
        "calib_vqe_calls": runner.calib_vqe_calls,           # diagnostic, off-budget
        "best_err": runner.best_err,
        "best_seq": list(runner.best_seq) if runner.best_seq is not None else None,
        "best_depth": runner.best_depth,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        # oracle-free scale state (None on the legacy path); additive key — old ckpts unaffected
        "escale": (runner.scale.state_dict() if getattr(runner, "scale", None) is not None else None),
    }, path)


def evaluate_checkpoint(ckpt_path: str, n_episodes: int, device: str = None,
                        chem_acc_mHa: float = 1.6, eval_seed: int = None) -> dict:
    """Reload a checkpoint, FREEZE actor + WM, run `n_episodes` fresh side-effect-free episodes
    to fixed max depth (no accuracy early stop; structural termination only), and return the
    per-checkpoint stats plus the cumulative TRAINING vqe from the checkpoint."""
    from config import Config
    from runner import Runner

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid})
    if device is not None:
        cfg.device = device

    r = Runner(cfg, _eval_mode=True)                         # build env+wm+actor, no train logs/dump
    r.actor.load_state_dict(ck["actor"])
    r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and r.scale is not None:  # oracle-free: restore the frozen scale
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval()
    r.wm.eval()

    if eval_seed is None:                                    # distinct, deterministic per (seed, episode)
        eval_seed = 100003 + 1009 * int(cfg.seed) + int(ck["episode"])
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed % (2 ** 31 - 1))

    ebests = []
    for _ in range(n_episodes):
        out = r._rollout(count_vqe=False, update_buffer=False, update_best=False,
                         run_to_max_depth=True, eval_mode=True)
        errs = out["errs"]
        ebests.append(float(np.min(errs)) if len(errs) else float("inf"))

    stats = episode_best_stats(ebests, chem_acc_mHa)
    stats.update({
        "episode": int(ck["episode"]),
        "cum_train_vqe_calls": int(ck["vqe_calls"]),
        "cum_train_vqe_nfev": int(ck.get("vqe_nfev", 0)),
        "n_eval_episodes": n_episodes,
        "eval_seed": eval_seed,
        "episode_bests": [float(x) for x in ebests],
    })
    return stats
