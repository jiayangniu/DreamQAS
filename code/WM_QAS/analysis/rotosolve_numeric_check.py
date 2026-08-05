"""Rotosolve numeric-safety check (verification item 1, load-bearing).

Confirms that on the 8q/10q GPU-rotosolve path the REPORTED energy is complex128 (Qulacs),
not the GPU complex64 search energy — so the 1.6 mHa success label never flips on fp32 error.

For a random circuit stepped through the rotosolve env, at each step compares:
  (a) env.energy (what step() reports)  vs  env.reference_energy()  -> must be ~0 (both Qulacs)
  (b) env.energy (Qulacs complex128)    vs  env._energy_via_bvqe() (GPU complex64) -> fp32 gap
Reports the max gaps in mHa. (a) must be ~0; (b) is the raw fp32 error (informational; it does
NOT affect the label because the reported energy is (a)'s complex128 value).

Run:  CUDA_VISIBLE_DEVICES=0 DREAMQAS_NO_MPS=1 python analysis/rotosolve_numeric_check.py --molecule BeH2_8q
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from utils import get_config
from environment import CircuitEnv
from circuit_rules import _dict_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="BeH2_8q")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conf = get_config("analysis/", f"G0_v2_{a.molecule}.cfg")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CircuitEnv(conf, device=dev)
    trans = _dict_actions(env.num_qubits)
    print(f"molecule={a.molecule} nq={env.num_qubits} optim_alg={getattr(env,'optim_alg',None)} "
          f"has_bvqe={hasattr(env,'_batched_vqe')} min_energy(true)={env.min_energy:.6f}")

    env.reset()
    gap_ref, gap_fp32, energies = [], [], []
    for step in range(a.steps):
        ill = env.illegal_action_new()
        legal = [i for i in range(env.action_size) if not (ill and i in ill)]
        if not legal:
            break
        act = int(np.random.choice(legal))
        env.step(list(trans[act]), torch.tensor(0.0), train_flag=True)
        e_step = float(env.energy)                       # Qulacs complex128 (what step reports)
        e_ref = float(env.reference_energy())            # Qulacs complex128 (recompute)
        e_bvqe = float(env._energy_via_bvqe())           # GPU complex64
        gap_ref.append(abs(e_step - e_ref) * 1000.0)     # mHa
        gap_fp32.append(abs(e_step - e_bvqe) * 1000.0)   # mHa
        energies.append(e_step)

    print(f"steps evaluated: {len(gap_ref)}")
    print(f"(a) |env.energy - reference_energy| : max={max(gap_ref):.2e} mHa  (must be ~0)")
    print(f"(b) |Qulacs(c128) - BatchedVQE(c64)|: max={max(gap_fp32):.4f} mHa  mean={np.mean(gap_fp32):.4f}")
    print(f"    -> reported energy/label uses (a)'s complex128 value; fp32 gap (b) does not flip labels.")
    ok = max(gap_ref) < 1e-6
    print("RESULT:", "OK — reported energy is complex128" if ok else "WARN — reference mismatch!")


if __name__ == "__main__":
    main()
