# Reproduction guide

Commands below start at the repository root, with the installed environment active.
Use a fresh output directory for each protocol. Training may overwrite existing logs;
evaluation traces resume by checkpoint unless `--overwrite` is specified.

## 1. Inputs and output paths

Six Hamiltonian files are included in `data/mol_data`, with hashes in
`data/manifest.json`. The documented self-contained tasks are `LiH4q`, `BeH2`
(BeH2-6q), `LiH6q`, `BeH2_8q`, and `H2O_8q`. The 10q/12q configurations require
external Hamiltonians; see [data/README.md](data/README.md).

- `--out_dir PATH`: training output root, resolved from the invocation directory.
- `DREAMQAS_MOL_DATA`: optional replacement Hamiltonian directory. It must contain
  the exact filenames requested by the selected configuration.
- `DREAMQAS_RESULTS_ROOT`: parent of `campaign_v1` and `oracle_free_v1` for table readers;
  defaults to the repository's `runs` directory.
- `DREAMQAS_ANALYSIS_OUT`: table output directory; defaults to repository `outputs`.

Configuration loading is independent of the current directory. Quote paths containing
spaces. No server home directory or mounted research disk is required.

## 2. Small training and evaluation checks

The README runs a four-episode DreamQAS example. The matched model-free control uses
**the same `G0_v2_LiH4q` environment** (40 layers), not the older 30-layer
`G0_baseline_LiH4q` configuration:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/WM_QAS/main_baseline.py \
  --config G0_v2_LiH4q --experiment_name analysis/ --molecule LiH4q \
  --seed 0 --device cpu --max_episodes 4 --out_dir runs/baseline-smoke
python code/WM_QAS/analysis/eval_policy_traces.py \
  runs/baseline-smoke/baseline_analysis/G0_v2_LiH4q/seed_0 \
  --device cpu --n_inter 1 --n_final 1
```

These short runs do not estimate paper-level performance. Training and evaluation
budgets must be stated separately. A full DreamQAS run normally uses 3,750 iterations
(15,000 real episodes); LiH6q uses 7,500 iterations (30,000 episodes). DAgger's
verified VQE calls are counted in addition to real rollout calls. Checkpoint schedules
and default budgets live in `phase2_surrogate/config.py`.

## 3. Canonical campaign and table input names

This example shows **one seed** at the full LiH4q budget. For a five-seed campaign,
repeat with seeds 0–4. Use the same molecule and environment for every compared arm.
For LiH6q, change the baseline budget to 30,000 episodes.

```bash
python code/WM_QAS/phase2_surrogate/runner.py \
  --molecule LiH4q --seed 0 --assert_full 1 \
  --out_dir runs/campaign_v1/dreamqas --tag _q
python code/WM_QAS/phase2_surrogate/runner.py \
  --molecule LiH4q --seed 0 --imagination none \
  --out_dir runs/campaign_v1/ablations --tag _ab_noimag
python code/WM_QAS/main_baseline.py \
  --config G0_v2_LiH4q --experiment_name analysis/ --molecule LiH4q \
  --seed 0 --max_episodes 15000 --out_dir runs/campaign_v1/dreamqas_rlqas
```

Run `eval_policy_traces.py RUN_DIRECTORY` for each resulting run, normally with
20 fresh episodes at intermediate checkpoints and 100 at the final checkpoint.
The table loader expects these paths beneath `DREAMQAS_RESULTS_ROOT`:

```text
campaign_v1/
  dreamqas/gru_energy_surrogate_<molecule>_s<seed>_q/eval_traces.jsonl
  ablations/gru_energy_none_<molecule>_s<seed>_ab_noimag/eval_traces.jsonl
  dreamqas_rlqas/baseline_analysis/G0_v2_<molecule>/seed_<seed>/eval_traces.jsonl
```

```bash
python code/WM_QAS/analysis/policy_quality_table.py
```

The table reports final-checkpoint quality and the last-three-checkpoint average.
Best-of-K is the expected minimum from K independent draws from the empirical
per-episode distribution; K=1 is its arithmetic mean. Aggregate across seeds,
not pooled episodes. Missing arms/tasks remain missing, and sample counts are shown.
The historical table task list includes BeH2-10q; that cell requires external inputs.

External CRLQAS, HyRLQAS, GQE, TFQAS, and QuantumDARTS campaign outputs can also be
read by the table loader, but their implementations and DreamQAS-specific protocol
adapters are not included here. Their path patterns are in `run_series()` and
`qdarts_delivered()` in `policy_quality_table.py`. Do not substitute unrelated
benchmark runs without checking Hamiltonians, budgets, optimizer, and evaluation
conventions. QuantumDARTS contributes a single delivered circuit rather than a
sampled-policy distribution. Native VQE counts across all methods are not necessarily
comparable.

## 4. Oracle-free training and ablations

Add `--oracle_free 1` to the DreamQAS command. The energy-frontier score is used for
training; canonical results must not be mixed into the oracle-free campaign.
`--ground_truth_diagnostics 0` additionally removes access to the exact ground-state
energy from training diagnostics. Frozen evaluators explicitly restore reference-energy diagnostics for reporting;
this does not change the saved training configuration or update learned parameters.

```bash
python code/WM_QAS/phase2_surrogate/runner.py \
  --molecule LiH4q --seed 0 --oracle_free 1 --assert_full 1 \
  --out_dir runs/oracle_free_v1/dreamqas --tag _of
```

The table readers use a common **15,000-episode checkpoint** for all tasks, including
LiH6q. For each run evaluate it with:

```bash
python code/WM_QAS/analysis/eval_policy_traces.py RUN_DIRECTORY \
  --only_ep 15000 --n_final 100
```

`--only_ep` picks the nearest saved checkpoint; the table readers accept only a
record whose actual episode is 15,000. They do not treat a short smoke run as a
full-budget result.

| Arm | Extra flags | Directory under `oracle_free_v1` | Tag |
|---|---|---|---|
| Full | `--oracle_free 1` | `dreamqas` | `_of` |
| No-imag | `--oracle_free 1 --imagination none` | `ablations` | `_of_noimag` |
| No DAgger | `--oracle_free 1 --dagger 0` | `nodag` | `_of_nodag` |
| No DIR | `--oracle_free 1 --dir_reweight 0` | `ablations` | `_of_noDIR` |
| No pessimism/confidence truncation | `--oracle_free 1 --pessimism_beta 0 --imag_conf_tau 1e9` | `ablations` | `_of_noUNC` |

These match the historical table names. The no-uncertainty arm disables both the
pessimism penalty and confidence truncation; retain the ensemble and other settings.
No-imag also
bypasses DAgger calibration in this implementation, so its effect combines two
mechanisms. The REINFORCE control retains its environment reward; it is not an
oracle-free DreamQAS world-model run simply because it is placed in this campaign.

```bash
python code/WM_QAS/analysis/of_main_table.py
python code/WM_QAS/analysis/of_ablation_table.py
```

The combined main table labels the campaign/control provenance. It can include
external results from `campaign_v1`, whose native rewards and budgets differ.
Ablation intervals use paired seeds; fewer than three paired seeds are reported
as insufficient. A non-significant difference is not evidence of equivalence.

## 5. Noise and checkpoint compatibility

Install `requirements-noise.txt`. The PTM backend supports **four qubits only**.
For example, append these to either the DreamQAS or baseline training command:

```text
--noise_mode ptm --noise_p1q 0.0005 --noise_p2q 0.005 --noise_p_ro 0.001 --noise_basis_change 1
```

New baseline checkpoints save and restore the actual noise specification. Old
baseline checkpoints contain no noise metadata; evaluation therefore fails until
the caller explicitly supplies the known training conditions:

```bash
# Only if the old run is known to be noiseless:
python code/WM_QAS/analysis/eval_policy_traces.py RUN_DIRECTORY \
  --legacy_noise_config '{"mode":"off"}' --overwrite
# For an old noisy run, replace values with its documented training settings:
python code/WM_QAS/analysis/eval_policy_traces.py RUN_DIRECTORY \
  --legacy_noise_config '{"mode":"ptm","p1q":0.0005,"p2q":0.005,"p_ro":0.001,"basis_change":true}' \
  --overwrite
```

Do not infer those settings from the checkpoint weights or filename. Previously
generated noisy-baseline evaluations should be regenerated with verified settings.
The override is used only when checkpoint metadata is missing.

`eval_policy_traces.py` saves observed prefix errors and the noiseless errors of
**the same observed-selected prefixes** (`ep_best_noiseless`). Selecting the minimum
of the noiseless trace instead would use information unavailable in the noisy
experiment. The table scripts above consume `ep_best` and are intended for clean
campaigns; they are not noise-paper table generators. The simpler
`eval_protocol.py pass` / `--eval_only` interfaces report observed errors only.
Only load checkpoints from a trusted source: the PyTorch loading path allows
Python objects for configuration compatibility.

## 6. Validation

After installing both dependency files:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
python code/WM_QAS/tests/test_release.py
python code/WM_QAS/tests/test_circuit_rules.py
python code/noise_ptm/tests/test_equivalence.py
python code/WM_QAS/tests/test_e0_invariance.py
```

The checks cover bundled data hashes, working-directory independence, noisy
baseline save/load and seeded traces, prefix selection, legality masks, PTM versus
independent quantum simulators, and oracle-free E0 invariance through WM refresh,
imagination, and DAgger. `analysis/rotosolve_numeric_check.py` is an additional
reference-energy check for the larger-system optimizer.

This maintenance release does not rerun full multi-seed paper campaigns. Small
validation runs establish execution and invariants, not reproduction of published
aggregate performance. Record the commit, full configuration, seed, dependencies,
Hamiltonian hash, and training/evaluation budgets with each result.
