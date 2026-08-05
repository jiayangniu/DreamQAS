# Launchers

The scripts used to launch the reported campaigns, included as a **specification of what was
run** — not as scripts to execute unmodified. They contain the authors' absolute paths and
cluster-specific logic (GPU assignment, NVIDIA MPS, core pinning, free-memory vetoes).

| Script | Campaign |
|---|---|
| `run_campaign.sh` + `campaign_manifest.py` | canonical campaign — five tasks × arms × 5 seeds |
| `oracle_free_main.sh` | oracle-free DreamQAS-Full, 5 tasks × 5 seeds |
| `oracle_free_ablations.sh` | oracle-free No-imag + model-free RLQAS control |
| `scaleup_v1.sh` | scale-up tasks (BeH₂-12q, H₂O-8q), 8 arms × 2 tasks × 5 seeds |
| `noise_v1.sh` | noise campaign — 5 arms × 5 seeds × 2 device tiers |
| `of_post_batch_eval.sh` | frozen best-of-1 evaluation over a finished batch |

Two operational details are load-bearing rather than incidental:

- **Core pinning** (`noise_v1.sh`). Each noise run spawns ~150 OS threads from XLA's CPU thread
  pool. `OMP_NUM_THREADS` and `XLA_FLAGS` do not constrain it; only `taskset -c` does, at a ~10%
  cost per run. Two concurrent batches must use disjoint core offsets.
- **`--assert_full 1`** hard-errors at startup unless every mechanism flag matches the canonical
  Full configuration. It is the guard against a silently mis-flagged headline run.
