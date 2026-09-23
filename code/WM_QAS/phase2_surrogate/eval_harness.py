import numpy as np
import torch


def episode_best_stats(episode_bests, chem_acc_mHa: float = 1.6) -> dict:
    a = np.asarray([e for e in episode_bests if np.isfinite(e)], dtype=float)
    n = int(a.size)
    return {
        "n": n,
        "mean": float(a.mean()) if n else None,
        "median": float(np.median(a)) if n else None,
        "std": float(a.std(ddof=1)) if n > 1 else 0.0,
        "success_rate": float((a <= chem_acc_mHa).mean()) if n else None,
        "best": float(a.min()) if n else None,
    }


def save_checkpoint(runner, episode: int, path: str) -> None:
    torch.save({
        "episode": episode,
        "iter": getattr(runner, "_cur_iter", -1),
        "actor": runner.actor.state_dict(),
        "wm": runner.wm.state_dict(),
        "cfg": dict(runner.cfg.__dict__),
        "vqe_calls": runner.vqe_calls,
        "vqe_nfev": getattr(runner, "vqe_nfev", 0),
        "calib_vqe_calls": runner.calib_vqe_calls,
        "best_err": runner.best_err,
        "best_seq": list(runner.best_seq) if runner.best_seq is not None else None,
        "best_depth": runner.best_depth,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        "escale": (runner.scale.state_dict() if getattr(runner, "scale", None) is not None else None),
    }, path)


def evaluate_checkpoint(ckpt_path: str, n_episodes: int, device: str = None,
                        chem_acc_mHa: float = 1.6, eval_seed: int = None) -> dict:
    from config import Config
    from runner import Runner

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid})
    # Enable reference-energy reporting for frozen evaluation.
    cfg.enable_ground_truth_diagnostics = True
    if device is not None:
        cfg.device = device

    r = Runner(cfg, _eval_mode=True)
    r.actor.load_state_dict(ck["actor"])
    r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and r.scale is not None:
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval()
    r.wm.eval()

    if eval_seed is None:
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
