# Implementation map

The maintained reference method lives in `code/WM_QAS/phase2_surrogate/`.

| Component | Implementation | Role |
|---|---|---|
| Environment / VQE | `environment.py`, `VQE.py` | Apply gates, optimize angles, return energies |
| Circuit legality | `circuit_rules.py` | Shared structural rules used by imagination |
| Training loop | `phase2_surrogate/runner.py` | Real rollouts, actor updates, WM refresh, verification, logging |
| Surrogate ensemble | `phase2_surrogate/surrogate_wm.py` | Predict energy-derived scores from gate sequences |
| Replay | `phase2_surrogate/buffer.py` | Store real prefixes and sample training/imagination seeds |
| Imagination | `phase2_surrogate/imagine.py` | Masked virtual gate rollouts and policy-gradient loss |
| Oracle-free score | `phase2_surrogate/escale.py` | Energy-frontier scale without exact ground-state error as the training target |
| Model-free control | `runner_baseline.py`, `agent/baseline_policy.py` | REINFORCE on structural observations |
| Frozen evaluation | `phase2_surrogate/eval_harness.py`, `analysis/eval_policy_traces.py` | Reconstruct checkpoints and sample without updating training state |
| PTM noise | `code/noise_ptm/` | Four-qubit gate and measurement noise energy evaluation |

The actor receives detached world-model features; surrogate parameters are trained
by their own supervised objective. Imagination uses the same circuit constraints as
the real environment. Its activation depends on warm-up and fidelity checks, so a
short entry-point smoke run is not an imagination-performance experiment.

DAgger verifies selected imagined circuits using real VQE and, when enabled, feeds
those observations back into replay. No-imag also bypasses this calibration path;
it is not a pure single-mechanism DAgger-preserving ablation in this implementation.

Canonical training uses ground-state-error-derived targets. Oracle-free training
uses the energy-frontier score; exact energies can remain available for diagnostics
unless explicitly disabled. The model-free control uses the environment's existing
reward and should be labelled separately from oracle-free world-model arms.

`Config` and the CLI in `runner.py` define the surrogate method's active hyperparameters.
Environment `.cfg` files also contain sections from older experiments; their
`[world_model]` values do not override the `Config` dataclass in this runner.
The baseline reads its `[baseline]` section, with explicit CLI arguments taking precedence.

For budgets, concrete commands, data scope, and interpretation of evaluation outputs,
see [REPRODUCE.md](../../REPRODUCE.md). No new research-method changes are introduced
by the publication-cleanup release.
