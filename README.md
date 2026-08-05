# DreamQAS

**A known-dynamics (energy-feedback) world model for VQE-efficient Quantum Architecture Search.**

Reference implementation for the paper. This repository contains the method, the ablation
arms, the analysis pipeline that produces every reported table and figure, and the noise
backend used in the noise section.

---

## What the method does

RL-based Quantum Architecture Search (RLQAS) runs a VQE optimization at *every* step to score
each candidate circuit. VQE dominates the cost and becomes the bottleneck as molecules grow.

DreamQAS learns an **energy surrogate** that predicts a circuit prefix's post-VQE error — and
its epistemic uncertainty — *without* running VQE, and plans with it. Because the
circuit-construction dynamics are **known** (they are given by the circuit rules), the agent
generates multi-step **imagined rollouts** entirely in symbolic space to train the policy with
far fewer real VQE calls.

It is model-based RL on a *known-dynamics* world model: circuit prefixes evolve exactly; we do
**not** learn transitions or variational-parameter dynamics. Only the *energy feedback* is
learned. In lineage this sits between AlphaZero (known dynamics + learned value) and Dreamer
(imagination-based policy gradient); unlike MuZero, latent dynamics are not learned.

Full method specification — tensor shapes, losses, hyperparameters:
**[`docs/method/ARCHITECTURE.md`](docs/method/ARCHITECTURE.md)**.

---

## Layout

```
code/WM_QAS/
  phase2_surrogate/      the method
    runner.py              main loop: real + imagination REINFORCE, WM refresh,
                           fidelity gate, DAgger
    surrogate_wm.py        energy-surrogate world model
                           (GRU encoder + K-ensemble + Randomized Prior Functions)
    imagine.py             known-dynamics imagination
                           (`Shadow` legal masking + surrogate scoring + λ-returns)
    buffer.py              replay buffer (mixed elite / prioritized / stratified sampling)
    escale.py              oracle-free empirical-frontier scoring (signed-log)
    config.py              config dataclass + CLI contract
    eval_harness.py        frozen offline evaluation
  environment.py         circuit-construction MDP + Hamiltonian loading
  VQE.py                 VQE backends (COBYLA statevector; rotosolve for 8q+)
  circuit_rules.py       legal-action rules
  main_baseline.py       model-free RLQAS baseline (no world model)
  runner_baseline.py
  agent/, world_model/   policy networks and the shared MLP
  analysis/              every table and figure in the paper is produced here
  configuration_files/   .cfg files for the baseline arm
  tests/

code/noise_ptm/          PTM / Pauli–Liouville noisy-expectation backend (noise section)

data/mol_data/           Hamiltonians, 4–8 qubits (see "Molecules" below)
experiments/launchers/   the launch scripts used for the reported campaigns
```

---

## Quick start

```bash
pip install -r requirements.txt
cd code/WM_QAS

# DreamQAS (Full)
python phase2_surrogate/runner.py --molecule LiH4q --seed 0 --out_dir ./runs \
    --encoder gru --reward energy --imagination surrogate \
    --independent_ensemble 1 --dagger 1 --dir_reweight 1 --imag_adv_normalize 1

# ablation: no imagination
python phase2_surrogate/runner.py --molecule LiH4q --seed 0 --out_dir ./runs \
    --encoder gru --reward energy --imagination none \
    --independent_ensemble 1 --dagger 1 --dir_reweight 1 --imag_adv_normalize 1

# model-free RLQAS baseline
python main_baseline.py --config G0_baseline_LiH4q --experiment_name analysis/ --seed 0
```

`--out_dir` is required in practice — the default points at the authors' cluster. See
[`REPRODUCE.md`](REPRODUCE.md) for the full protocol, the exact arms, and the analysis pipeline.

---

## Molecules

Hamiltonians up to 8 qubits ship with this repository (≈3 MB total):

| Molecule | Qubits | Mapping | Role |
|---|---|---|---|
| LiH | 4 | parity | main task |
| BeH₂ | 6 | Jordan–Wigner | easy — search is close to trivial |
| LiH | 6 | Jordan–Wigner | hard — energy plateau, luck-dominated |
| BeH₂ | 8 | Jordan–Wigner | scale-up (6-31G, 2e/4o) |
| H₂O | 8 | Jordan–Wigner | scale-up (STO-3G, 4e/4o) |

The 10- and 12-qubit Hamiltonians are 8–270 MB each and exceed GitHub's file-size limit, so
they are **not** in this repository. Regenerate them with OpenFermion/PySCF from the basis and
active space recorded in `docs/method/ARCHITECTURE.md`, or request the exact `.npz` files.

Two H₂O-8q geometries are shipped. The paper uses the equilibrium one
(`O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586`).

---

## Noise experiments

`code/noise_ptm/` implements the noisy expectation used in the noise section: the density
matrix is propagated in the Pauli–Liouville (PTM) representation, so every ⟨H⟩ is an **exact**
noisy expectation — no measurement sampling, no shots, hence deterministic given (circuit, θ).

Three channels: depolarizing after each single-qubit rotation (`p1q`), depolarizing after each
CNOT (`p2q`), and symmetric readout assignment error (`p_ro`). Readout error and
measurement-basis-change depolarizing are folded into the Hamiltonian's Pauli vector at compile
time, so they cost nothing at run time. Noise levels are taken from IBM device calibration
medians; see `code/noise_ptm/spec.py`.

Everything is off by default — with `noise_mode="off"` the code path is byte-identical to the
noiseless one.

```bash
python code/noise_ptm/tests/test_equivalence.py     # PTM vs qulacs, noiseless and noisy
```

---

## Baselines

The comparison baselines (CRLQAS, HyRLQAS, GQE, TF-QAS, QuantumDARTS) are **not** in this
repository — they come from their own published implementations, collected in a separate
benchmark harness. `REPRODUCE.md` records which implementation each number came from and
every modification that was made to them.

---

## Licence

See [`LICENSE`](LICENSE).
