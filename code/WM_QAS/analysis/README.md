# Analysis pipeline

Every table and figure in the paper is produced by a script in this directory. Each script
declares its input run-tree root as a module-level constant in its first ~50 lines (`C`, `OF`,
`CV`, `CANON`, `ABL`, `HR`); those are absolute paths to the authors' cluster and must be
repointed. See [`../../../REPRODUCE.md`](../../../REPRODUCE.md) §0.

Two run sets are read, and they must never be merged:

- **canonical** — training reward uses the exact ground-state energy `E0`
- **oracle-free** — training never uses `E0` (`--oracle_free 1`); scripts prefixed `of_` and
  `plot_rq2_w100_of` read this set

Most scripts are standalone (stdlib + numpy/matplotlib, no method imports). The exceptions are
`wm_beam_eval.py` and `wmg_explore_eval.py`, which reload a frozen world-model checkpoint.

## Locked conventions

- Headline metric is **frozen best-of-1** — never `metrics.jsonl:best_err`.
- Dispersion is the **sample** standard deviation (`ddof=1`) over the 5 seeds. The seed is the
  experimental unit; episodes are never pooled across seeds.
- All methods are reported as mean ± std. The median is a diagnostic only.
- Curve figures use median + IQR over seeds.
- VQE-speedup and DAgger-efficiency tables use the **W = 100** accounting basis (`*_w100`).

## Core pipeline

| Script | Artifact | What it is |
|---|---|---|
| `eval_protocol.py` | (driver) | standardized checkpoint grid + frozen offline eval protocol |
| `eval_policy_traces.py` | per-run `eval_traces*.jsonl` | frozen rollout of saved checkpoints (20 / 100 episodes, side-effect-free) |
| `policy_quality_table.py` | policy-quality table | best-of-1 / best-of-10 across tasks and methods |
| `of_main_table.py` | unified main table | oracle-free internal arms + external baselines, both provenances labelled |
| `ladder_table.py`, `ladder_full_export.py` | causal-ladder tables | four-stage ladder RLQAS → No-imag → noDAG → Full |
| `rq2_speedup.py`, `plot_rq2_w100.py`, `plot_rq2_w100_of.py` | speedup tables + Figure 2 | seed-level VQE speedup with right-censoring and bootstrap CIs |
| `of_gap_table.py` | matched-error VQE gap ladder | the source of the speed figures' annotations |
| `t1a_action_ranking.py` + `t1a_table.py` | action-ranking probe | counterfactual per-prefix ρ, top-pick regret, matched-random comparison |
| `wm_diagnostics.py` | WM fidelity table | pairwise ranking fidelity, calibration MAE, disagreement correlation |
| `dagger_diagnostic.py`, `dagger_efficiency*.py` | DAgger tables | selection bias diagnostics and Full-vs-noDAG VQE-to-target |
| `main_results.py` | per-episode / per-checkpoint CSVs, coverage | bulk exports |
| `cost_table.py` | cost accounting | VQE / WM-query / wall-clock, each field tagged `measured \| reconstructed \| unavailable` |

## Figures

`plot_training_window.py`, `plot_speed_curves.py`, `plot_figure3_reliability.py`,
`plot_fidelity_curves.py`, `plot_t1a_horizon.py`, `plot_imag_strength.py`,
`plot_risk_coverage.py`, `plot_horizon_fidelity_signals.py`, `plot_wm_accuracy_by_depth.py`.

## Supplementary

`wm_beam_eval.py` (WM-guided beam search, B = 10, asserts zero real VQE inside the search),
`wmg_explore_eval.py` and `wmgreedy_threeway.py` (WM-greedy deployment variants),
`of_horizon_slope.py`, `of_bestofk_table.py`, `of_ablation_table.py`, `of_rq3_tables.py`,
`of_t1a_table.py`, `of_action_utility_table.py`, `of_vqe_to_target_table.py`,
`terminal_vs_best_table.py`, `frozen_final_table.py`, `deploy_contrast_table.py`,
`vqe_to_policy_level.py`, `verification_correction.py`, `risk_coverage.py`,
`plateau_diagnostic.py`, `reward_fidelity_audit.py`, `speed_ladder.py`, `timing_table.py`,
`q2_horizon_table.py`, `lineage_audit.py`, `imag_dose_speed.py`.

`import_psqas.py` imports baseline runs and molecules from the external benchmark harness (not
included in this repository); `rotosolve_numeric_check.py` verifies the rotosolve numerics that
the 8-qubit-and-larger results depend on.
