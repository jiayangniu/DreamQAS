"""Cost benchmark — run this BEFORE launching any noisy fleet.

Measures real DreamQAS episodes (the actual CircuitEnv + COBYLA loop, not a microbench)
with the noise backend on and off, and projects to the full 15,000-episode budget.

The number that matters is the **ratio** noisy/clean: the clean LiH-4q runs are already
done, so their recorded wall_clock times the ratio is the fleet cost.

Usage:
    python code/noise_ptm/tests/bench_cost.py --episodes 3
    python code/noise_ptm/tests/bench_cost.py --episodes 3 --device gpu

Notes
-----
* Defaults to JAX-on-CPU. The box normally has a training fleet on its GPUs, and a 4q
  PTM kernel is round-trip-latency-bound rather than FLOP-bound, so CPU is not obviously
  worse — that is exactly what this script is for.
* ``--device gpu`` sets XLA_PYTHON_CLIENT_PREALLOCATE=false so JAX does not grab the
  whole card out from under other jobs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_CODE, os.path.join(_CODE, "WM_QAS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="episodes discarded (JIT compile)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--molecule-cfg", default="G0_v2_LiH4q.cfg")
    ap.add_argument("--max-step", type=int, default=0, help="0 = env's num_layers")
    ap.add_argument("--total-episodes", type=int, default=15000, help="budget to project to")
    a = ap.parse_args()

    if a.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("DREAMQAS_NO_MPS", "1")

    import torch
    from environment import CircuitEnv
    from utils import get_config
    from noise_ptm.integration import build_evaluator
    from noise_ptm.spec import COMPOSITE

    os.chdir(os.path.join(_CODE, "WM_QAS"))
    conf = get_config("analysis/", a.molecule_cfg)

    def build(noisy):
        torch.manual_seed(0)
        np.random.seed(0)
        env = CircuitEnv(conf, device=torch.device("cpu"))
        if noisy:
            env.attach_noisy_evaluator(build_evaluator(
                env, mode="ptm", p1q=COMPOSITE.p1q, p2q=COMPOSITE.p2q,
                p_ro=COMPOSITE.p_ro, basis_change=True, verbose=False))
        return env

    def run_episode(env, rng, max_step):
        """Random-action episode of the same shape the RL loop produces."""
        env.reset()
        n = env.num_qubits
        nfev = 0
        for _ in range(max_step):
            if rng.random() < 0.4:
                c = int(rng.integers(n))
                off = 1 + int(rng.integers(n - 1))
                action = [c, off, n, 0]
            else:
                action = [n, 0, int(rng.integers(n)), int(rng.integers(1, 4))]
            env.step(action, torch.tensor(0.0), train_flag=False)
            nfev += int(getattr(env, "last_nfev", 0) or 0)
        return nfev

    results = {}
    for tag, noisy in (("clean", False), ("noisy", True)):
        env = build(noisy)
        max_step = a.max_step or env.num_layers
        rng = np.random.default_rng(7)
        for _ in range(a.warmup):                     # JIT compile / cache warm
            run_episode(env, np.random.default_rng(999), max_step)
        t0 = time.perf_counter()
        total_nfev = 0
        for _ in range(a.episodes):
            total_nfev += run_episode(env, rng, max_step)
        dt = time.perf_counter() - t0
        per_ep = dt / a.episodes
        results[tag] = (per_ep, total_nfev / a.episodes)
        print(f"[{tag:>5}] {per_ep:8.3f} s/episode   {total_nfev / a.episodes:8.0f} nfev/episode "
              f"  ({per_ep / max(total_nfev / a.episodes, 1) * 1e3:.3f} ms/eval)")

    c, nvals = results["clean"][0], results["clean"][1]
    nz, nzval = results["noisy"][0], results["noisy"][1]
    print()
    print(f"  ratio noisy/clean       : {nz / c:6.2f}x")
    print(f"  projected {a.total_episodes} episodes:")
    print(f"      clean  {c * a.total_episodes / 3600:8.2f} h  ({c * a.total_episodes / 86400:.2f} d)")
    print(f"      noisy  {nz * a.total_episodes / 3600:8.2f} h  ({nz * a.total_episodes / 86400:.2f} d)")
    print(f"  15 runs (noisy), serial : {nz * a.total_episodes * 15 / 86400:.1f} d")
    print(f"  backend: JAX on {a.device.upper()}   max_step={a.max_step or 'num_layers'}")
    print()
    print("  NOTE: nfev/episode differs between the two — COBYLA follows a different path on the")
    print("  noisy landscape. Compare s/episode (what the fleet costs), and ms/eval for the kernel.")


if __name__ == "__main__":
    main()
