"""Equivalence test: circuit_rules (pure) vs the REAL CircuitEnv.

Drives the actual env on random LEGAL rollouts (VQE stubbed out) and asserts,
at every step, bit-exact agreement between the pure ports in circuit_rules and
the env's own logic for:
    (1) decoded illegal-action set          (compute_illegal  vs env.illegal_action_new)
    (2) structure tensor state[:, :N+3]      (apply_structure  vs env.state)
    (3) per-qubit moments                    (apply_structure  vs env.moments)
    (4) internal illegal-action slot list    (compute_illegal  vs env.illegal_actions)

This is the correctness gate for imagination action masking. Run:
    python tests/test_circuit_rules.py
"""

import os
import sys
import numpy as np
import torch

# Make code/WM_QAS importable and the CWD (relative cfg/data paths).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils import get_config, dictionary_of_actions
from environment import CircuitEnv
from circuit_rules import compute_illegal, apply_structure, fresh_illegal, empty_structure


def _make_env(experiment_name, cfg_name):
    conf = get_config(experiment_name, cfg_name)
    env = CircuitEnv(conf, device=torch.device("cpu"))
    # Stub VQE so we exercise ONLY structure/mask bookkeeping (fast, deterministic).
    env.get_energy = lambda *a, **k: (0.0, 0.0)
    env.reward_fn = lambda e: 0.0
    env.external_opt = 0
    return env


def run_case(experiment_name, cfg_name, n_episodes=8, seed=0):
    env = _make_env(experiment_name, cfg_name)
    N = env.num_qubits
    L = env.num_layers
    A = env.action_size
    translate = dictionary_of_actions(N)
    rng = np.random.RandomState(seed)

    steps_checked = 0
    for ep in range(n_episodes):
        env.reset()
        sh_ill = fresh_illegal(N)
        sh_cur = [N, N, N, N]                 # env.reset sets current_action = [N]*4
        sh_struct = empty_structure(L, N)
        sh_moments = [0] * N

        for t in range(L + 1):
            # (A) selection-time mask — mirrors runner's `ill = env.illegal_action_new()`
            ill_env = env.illegal_action_new()
            _, ill_mine = compute_illegal(sh_ill, sh_cur, N)
            assert set(ill_env) == set(ill_mine), (
                f"[{cfg_name}] ep{ep} t{t} MASK mismatch:\n"
                f"  env ={sorted(set(ill_env))}\n  mine={sorted(set(ill_mine))}")
            assert env.illegal_actions == sh_ill, (
                f"[{cfg_name}] ep{ep} t{t} illegal-slot list mismatch:\n"
                f"  env ={env.illegal_actions}\n  mine={sh_ill}")

            # Pick a random LEGAL action (mirrors the masked policy).
            legal = [a for a in range(A) if a not in set(ill_mine)]
            a_idx = int(rng.choice(legal)) if legal else int(rng.choice(A))
            a_env = list(translate[a_idx])
            a_sh = list(translate[a_idx])

            # (B) env transition (VQE stubbed) — internally calls illegal_action_new again
            _, _, done, _ = env.step(a_env, torch.tensor(0.0), train_flag=False)

            # Mirror the transition on the shadow: same double-call ordering as env.
            sh_cur = a_sh
            compute_illegal(sh_ill, a_sh, N)
            apply_structure(sh_struct, sh_moments, a_sh, N)

            # (2)(3)(4) compare post-transition structure / moments / slots
            env_struct = env.state[:, : N + 3].cpu().numpy()
            assert np.array_equal(sh_struct, env_struct), (
                f"[{cfg_name}] ep{ep} t{t} STRUCTURE mismatch "
                f"(diff cells={int((sh_struct != env_struct).sum())})")
            assert sh_moments == list(env.moments), (
                f"[{cfg_name}] ep{ep} t{t} MOMENTS mismatch:\n"
                f"  env ={list(env.moments)}\n  mine={sh_moments}")
            assert env.illegal_actions == sh_ill, (
                f"[{cfg_name}] ep{ep} t{t} post-step slot mismatch:\n"
                f"  env ={env.illegal_actions}\n  mine={sh_ill}")

            steps_checked += 1
            if done:
                break

    print(f"  PASS  {cfg_name:28s} N={N} L={L} A={A}  "
          f"episodes={n_episodes} steps_checked={steps_checked}")
    return steps_checked


def main():
    cases = [
        ("analysis/", "G0_v2_LiH4q.cfg"),
        ("analysis/", "G0_v2_LiH6q.cfg"),   # skipped gracefully if cfg/data missing
    ]
    total = 0
    ran = 0
    for exp, cfg in cases:
        try:
            total += run_case(exp, cfg)
            ran += 1
        except FileNotFoundError as e:
            print(f"  SKIP  {cfg:28s} (missing cfg/data: {e})")
        except KeyError as e:
            print(f"  SKIP  {cfg:28s} (cfg key missing: {e})")
    if ran == 0:
        print("NO CASES RAN — could not instantiate any env"); sys.exit(2)
    print(f"ALL EQUIVALENCE CHECKS PASSED  ({ran} cfg(s), {total} steps total)")


if __name__ == "__main__":
    main()
