# Selected analysis entry points

Run these with Python from the repository root; examples are in
[REPRODUCE.md](../../../REPRODUCE.md).

| Script | Input and purpose |
|---|---|
| `eval_policy_traces.py` | Run directory with `ckpt/ep*.pt`; frozen evaluation, per-prefix summaries, noise-aware selection |
| `eval_protocol.py` | `pass RUN_DIR` writes simpler `eval.jsonl`; `agg GLOB...` aggregates those records |
| `policy_quality_table.py` | Canonical campaign `eval_traces.jsonl`; best-of-K statistics over seeds |
| `of_main_table.py` | Common-budget oracle-free campaign plus separately labelled control/external results |
| `of_ablation_table.py` | Oracle-free component ablations with paired-seed intervals |
| `rotosolve_numeric_check.py` | Optimizer energy check against a higher-precision reference |

Table roots are configured through `DREAMQAS_RESULTS_ROOT`. Generated reports belong
under `outputs/`, not source code. Missing data is not filled with historical numbers.
Table scripts do not rerun training or download external baseline outputs.

Operational scripts, exploratory plots, migration tools, and campaign-specific
postprocessing are excluded from the maintained release. Their removal does not
change the training algorithms or delete historical experimental data.
