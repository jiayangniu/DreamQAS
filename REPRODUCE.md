# Reproducing the paper

This document says exactly what runs, what it produces, and what this repository does *not*
contain. Read the "Known limitations" section first — it will save you time.

---

## 0. Known limitations of this release

**(a) Output and analysis paths are hard-coded to the authors' cluster.**
The code was written for one machine and has not been made path-portable. Concretely:

| What | Status |
|---|---|
| Hamiltonian loading (`environment.py`) | ✅ **relative** — works out of the box |
| Baseline output root (`main_baseline.py`) | ✅ defaults to `./results` |
| DreamQAS output root (`phase2_surrogate/config.py`, field `out_dir`) | ⚠️ absolute — **pass `--out_dir` on every run** |
| Analysis scripts (`code/WM_QAS/analysis/*.py`) | ⚠️ absolute — edit one constant per file |

`phase2_surrogate/runner.py` has **no `--device` flag**: 4q and 6q select CPU automatically, 8q
and larger select CUDA. `main_baseline.py` calls `torch.cuda.set_device()` unconditionally and
therefore **requires a visible GPU even for 4-qubit runs**; on a CPU-only machine that arm will
fail at startup.

Every analysis script declares its input root as a module-level constant in its first ~50
lines — `C`, `OF`, `CV`, `CANON`, `ABL`, or `HR`. Repoint those to wherever you wrote your runs.
The two campaign roots referenced are:

```
.../dreamqas_campaign/campaign_v1     # canonical campaign (E0-based training signal)
.../dreamqas_campaign/oracle_free_v1  # oracle-free (E0-free) campaign
```

Two additional constants need attention if you use those scripts:
`analysis/main_results.py` (`OUT`, the artifact output directory) and
`analysis/import_psqas.py` (`PB`, the baseline harness's molecule directory).

**(b) No result artifacts are shipped.** Run logs, checkpoints, `eval_traces.jsonl`, and the
generated tables/figures are not in this repository (the raw run tree is several GB). The
analysis scripts are included so the derivation of every reported number is inspectable; to
regenerate the numbers you must first reproduce the runs.

**(c) No baseline implementations.** See §5.

**(d) Exact numerical reruns do not reproduce.** Re-running an identical configuration and
seed does **not** reproduce a previous run bit-for-bit — the VQE optimizer path is sensitive to
floating-point non-determinism and to library versions, and a single seed's final error can
move by several mHa. Seed-level *aggregates* (5-seed mean ± std) are the reproducible unit;
individual seed values are not. Do not compare a rerun seed against a previously reported one.

---

## 1. Environment

```bash
pip install -r requirements.txt
```

Python 3.10.20. `qulacs` version affects VQE numerics — keep the pin. Torch is only needed on
GPU for the 8-qubit-and-larger rotosolve path; 4q and 6q run on CPU.

Every `.cfg` sets `[runtime] num_threads = 1`, read by `runtime_init.py` *before* numpy/torch
import. Do not remove it: without it each run spawns a thread pool per process and a batch of
concurrent runs will oversubscribe the machine.

---

## 2. The arms

Five arms appear in the paper. All are launched from `code/WM_QAS`.

| Arm | Command |
|---|---|
| **DreamQAS (Full)** | `python phase2_surrogate/runner.py --molecule $MOL --seed $S --out_dir $OUT` |
| **No-imag** | same `+ --imagination none` |
| **noDAG** | same `+ --dagger 0` |
| **DreamQAS-RL** (model-free control) | `python main_baseline.py --config G0_v2_$MOL --experiment_name analysis/ --seed $S --out_dir $OUT` |
| **Oracle-free variant** | any of the above `+ --oracle_free 1` |

Canonical Full flags are the config defaults; `--assert_full 1` hard-errors at startup if any
mechanism flag deviates from the canonical Full configuration. Use it when launching the
headline runs.

Episode budgets are per-molecule constants in `phase2_surrogate/config.py` (`N_ITERS`);
LiH-6q uses 2× the others. All reported comparisons are read at the **locked common budget of
15,000 episodes**, which for LiH-6q means the ep-15000 re-evaluation rather than its final
checkpoint.

Reference launchers, as used for the reported campaigns, are in `experiments/launchers/`
(`oracle_free_main.sh`, `oracle_free_ablations.sh`, `run_campaign.sh` + `campaign_manifest.py`,
`scaleup_v1.sh`, `noise_v1.sh`, `of_post_batch_eval.sh`). They contain the authors' absolute
paths and cluster-specific GPU/MPS/core-pinning logic; read them as a specification of what was
run, not as scripts to execute unmodified.

---

## 3. The reported metric

**Frozen best-of-1.** Not the training-time best. The protocol is:

1. Load a saved checkpoint and freeze the policy.
2. Roll out 100 fresh evaluation episodes (20 for intermediate checkpoints).
3. Within each episode, run VQE on every circuit prefix and keep the prefix with the lowest
   **observed** energy — this is a deployment-faithful selection rule and never consults the
   exact ground-state energy `E0`.
4. That episode's error is `ep_best`. best-of-1 = the mean of `ep_best` over the 100 episodes.
5. Report mean ± sample std (`ddof=1`) over the 5 seeds. **The seed is the experimental unit**;
   episodes are never pooled across seeds.

```bash
python analysis/eval_policy_traces.py $RUN_DIR --device cpu --only_ep 15000 --n_final 100
```

writes `eval_traces_ep15000.jsonl` in the run directory. The table scripts read that file.

⚠ Do **not** read the headline number from `metrics.jsonl:best_err`. That is a training-time
quantity computed under a different selection rule and is not the reported metric.

`analysis/README.md` maps each script to the artifact it produces.

---

## 4. Oracle-freedom

Two training signals appear in the paper and must not be conflated.

- **Canonical runs** use `E0` in the training reward.
- **Oracle-free runs** (`--oracle_free 1`) never use `E0` during training: the reward is a
  signed-log score against an empirical energy frontier maintained from observed energies
  only, with a fixed margin of 0.1 mHa (`phase2_surrogate/escale.py`).

The oracle-free variant is a *viable variant, not an equivalence* — it costs measurably more on
the hardest task. Results obtained under the canonical signal must not be redescribed as
oracle-free outputs.

In both cases `E0` is used **only** for post-hoc error reporting: no search method uses the
true error or `E0` to select actions, prune, or choose a trajectory or checkpoint.

---

## 5. Baselines

CRLQAS, HyRLQAS, GQE, TF-QAS and QuantumDARTS are **not** in this repository. They were run
from their own implementations inside a separate benchmark harness, each under its own native
reward and stopping protocol. Two consequences for anyone comparing:

- **VQE-count comparability.** DreamQAS arms and CRLQAS/HyRLQAS share a VQE-counting
  convention; GQE, TF-QAS and QuantumDARTS use native accounting, so those three support
  *quality* comparison only — never a matched-VQE speedup claim.
- **Modifications made.** Two baselines required fixes before they produced valid results:
  an action-masking correction plus an acceptance-threshold setting for HyRLQAS, and a
  supernet-size correction for QuantumDARTS (its `num_layers` parameter means slots-per-qubit,
  not circuit depth; misreading it made the supernet deliver empty circuits). After the fix,
  QuantumDARTS still exceeds the 50-gate budget on some tasks — the budget is a recorded field
  for that method, not an enforced constraint. That is a concession in its favour and is
  reported as such.

All baselines shared **bit-identical Hamiltonians** with the DreamQAS arms; this was asserted
rather than assumed.

---

## 6. Noise experiments

`code/noise_ptm/` computes ⟨H⟩ on the noisy density matrix in the Pauli–Liouville (PTM)
representation. Protocol:

- Every objective evaluation inside COBYLA is an **exact** noisy expectation. There is no
  measurement sampling and no shot budget, so the energy is deterministic given (circuit, θ)
  and COBYLA sees no sampling variance. Noise still changes the energy landscape, the optimal
  parameters, and the ordering of architectures.
- One-qubit depolarizing after every single-qubit gate; two-qubit depolarizing after every
  CNOT; measurement basis-change gates are treated as single-qubit gates; readout error applies
  only at measurement. Circuit transitions remain exact.
- Readout and basis-change noise reduce to a compile-time diagonal rescale of the Hamiltonian's
  Pauli vector, so they cost nothing at run time.
- The oracle-free frontier is updated with the **noisy** expected energy. Ensemble, ranking
  gate, pessimism, truncation and verification are unchanged from the noiseless algorithm.

Noise levels come from IBM device calibration medians (`code/noise_ptm/spec.py`). Those tiers
carry gate and readout error but **no T1/T2**, so they are a *lower bound* on real device noise.

Validation:

```bash
python code/noise_ptm/tests/test_equivalence.py   # PTM vs qulacs: noiseless and noisy
python code/noise_ptm/tests/bench_cost.py         # per-step cost vs the clean twin
```

Enable noise with `--noise_mode ptm --noise_p1q ... --noise_p2q ... --noise_p_ro ...`. All noise
switches default to off, and with them off the code path is byte-identical to the noiseless one
(`code/WM_QAS/tests/test_e0_invariance.py` guards this).

**Operational note.** Each noise run spawns ~150 OS threads from XLA's CPU thread pool.
`OMP_NUM_THREADS` and `XLA_FLAGS` do *not* constrain it; only `taskset -c <core>` does, at a
~10% cost. Launching a batch without core pinning will overload the machine.

---

## 7. Tests

```bash
cd code/WM_QAS
python tests/test_circuit_rules.py     # legal-action rules (~1 min)
python tests/test_e0_invariance.py     # E0 never enters the training path (~10 min)
```

`test_e0_invariance.py` is the load-bearing one: it re-runs training with `E0` shifted by
0.5 Ha and asserts the training state is **bit-identical**, then re-runs with `min_energy=None`
and asserts the run completes with finite scores. Only an explicit whitelist of post-hoc
diagnostic fields is allowed to move.

Verified on this release: `test_circuit_rules` 720 steps PASS; `test_e0_invariance` all cases
PASS; `noise_ptm/tests/test_equivalence.py` all four checks PASS (worst deviation vs qulacs
3.7 × 10⁻⁷ Ha over 200 random circuits, noiseless and noisy).
