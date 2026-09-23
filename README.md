# DreamQAS

Research code for quantum architecture search with an energy-surrogate world model,
imagined policy updates, and VQE verification. This repository contains the reference
DreamQAS implementation, its model-free REINFORCE control, a four-qubit PTM noise
backend, and selected frozen-policy evaluation and table scripts.

## Scope

- Training and small-scale reproduction use the bundled 4–8 qubit Hamiltonians.
- Canonical and oracle-free training are separate protocols. The default canonical
  method uses ground-state error in its training signal; `--oracle_free 1` uses an
  energy-frontier score instead. Ground-state energies may still be used for diagnostics.
- Raw experiment campaigns, trained checkpoints, and external comparison methods are
  **not bundled**. Complete paper tables require those run outputs. This is a source
  release, not a precomputed-results archive or a claim that every paper figure can
  be reproduced from the files alone.
- Cluster schedulers, migration utilities, temporary exploratory analyses, and large
  output folders are excluded. Historical configuration files remain for checkpoint
  compatibility; use `G0_v2_*` for the documented protocol.

## Install

The reference environment uses Python 3.10 and Linux. A CPU is sufficient for the
LiH4q example; larger runs benefit from a compatible NVIDIA GPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional: PTM noise experiments and the full release regression suite
python -m pip install -r requirements-noise.txt
```

The dependency files pin the reference versions. If using CUDA, install a PyTorch
build compatible with your driver. NVIDIA MPS is disabled by default; starting a
shared MPS service requires the explicit `DREAMQAS_ENABLE_MPS=1` environment setting.

## Small CPU example

Run from the repository root. This performs **one iteration / four real episodes**;
it checks training and checkpoint evaluation, not convergence or the imagination
phase (which normally starts after warm-up).

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/WM_QAS/phase2_surrogate/runner.py \
  --molecule LiH4q --seed 0 --device cpu --n_iterations 1 \
  --out_dir runs/smoke --tag _smoke
python code/WM_QAS/analysis/eval_policy_traces.py \
  runs/smoke/gru_energy_surrogate_LiH4q_s0_smoke \
  --device cpu --n_inter 1 --n_final 1
```

Outputs include `config.json`, training logs, `ckpt/ep4.pt`, and
`eval_traces.jsonl`. Reuse a run directory only when deliberately replacing its
training outputs. See [REPRODUCE.md](REPRODUCE.md) for matched baseline commands,
campaign naming, noise evaluation, tests, and protocol limits.

## Layout

| Path | Purpose |
|---|---|
| `code/WM_QAS/phase2_surrogate/` | World model, training loop, imagination, checkpoint evaluation |
| `code/WM_QAS/main_baseline.py` | Model-free REINFORCE control |
| `code/WM_QAS/analysis/` | Selected evaluation and policy-quality table scripts |
| `code/WM_QAS/configuration_files/` | Environment and historical configuration files |
| `code/noise_ptm/` | Four-qubit noisy energy evaluator and physics checks |
| `data/` | Small Hamiltonians, file hashes, and data scope |
| `docs/method/ARCHITECTURE.md` | Implementation map and method conventions |

See [data/README.md](data/README.md) for Hamiltonian availability, and
[LICENSE](LICENSE) for the MIT license. Please identify the repository commit and
experimental configuration when reporting results based on this code.
