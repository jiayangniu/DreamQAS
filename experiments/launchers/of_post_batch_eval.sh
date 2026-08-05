#!/bin/bash
# Frozen best-of-1 evaluation at ep15000 for the batches that finished on 2026-07-28.
# Run this AFTER all training batches have reached their cap, and BEFORE the curated backup —
# the backup ships eval_traces_ep15000.jsonl, and once it exists the checkpoints stop being critical.
#
# Two groups, different reasons:
#   h1 LiH6q  (10 runs)  — REQUIRED. The reviewer-Q2 horizon gate needs the ep15000 checkpoint of the
#                          H=1 arm on the only other non-saturated task. Without it §3-Q2 stays a
#                          LiH-4q-only statement. LiH4q h1 and BeH2_8q h1 are already evaluated.
#   ablations noDIR/noUNC (50 runs) — WANTED. Gives the oracle-free component-ablation policy-quality
#                          table. NOT needed for the Acc_pair curves: those read fidelity.jsonl, which
#                          every ablation run already has (25/25 each).
# Timing runs are deliberately NOT evaluated: timing_table.py reads metrics/timing counters only.
#
# ep15000 = 3750 iters = the locked common reporting budget (§1 of the storyline). 100 eval episodes.
# Writes <run_dir>/eval_traces_ep15000.jsonl — a separate file; nothing existing is overwritten.
# Usage: ./of_post_batch_eval.sh [--dry] [--max N] [--only h1|abl]
set -eo pipefail
source /home/USER/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env
DREAMQAS=/home/USER/DreamQAS/code/WM_QAS
C=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1
LOGS=$C/logs/post_batch_eval; mkdir -p "$LOGS"
MAX=16; DRY=0; ONLY=all
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --dry) DRY=1; shift;; --only) ONLY="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;; esac; done

CMDS=()
add() {                                   # skip runs that already have the artifact -> resumable
    [ -d "$2" ] || { echo "[skip] missing $2"; return; }
    [ -f "$2/eval_traces_ep15000.jsonl" ] && { echo "[have] $1"; return; }
    CMDS+=("$1"$'\t'"$2")
}
if [ "$ONLY" = all ] || [ "$ONLY" = h1 ]; then
  for s in 0 1 2 3 4 5 6 7 8 9; do
    add "h1_LiH6q_s${s}" "$C/h1/gru_energy_surrogate_LiH6q_s${s}_h1"
  done
fi
if [ "$ONLY" = all ] || [ "$ONLY" = abl ]; then
  for v in noDIR noUNC; do
    for d in "$C"/ablations/*_of_"$v"; do
      add "$(basename "$d")" "$d"
    done
  done
fi

echo "[post-batch-eval] to run=${#CMDS[@]} MAX=$MAX ONLY=$ONLY"
if [ "$DRY" = 1 ]; then printf '%s\n' "${CMDS[@]}" | sed 's/\t/  ::  /'; exit 0; fi
[ "${#CMDS[@]}" -eq 0 ] && { echo "[post-batch-eval] nothing to do."; exit 0; }

for line in "${CMDS[@]}"; do
  label="${line%%$'\t'*}"; d="${line#*$'\t'}"
  while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 10; done
  # 8q/10q trained under ROTOSOLVE on GPU; everything else is COBYLA on CPU.
  dev=cpu; case "$d" in *BeH2_8q*|*BeH2_10q*) dev=cuda;; esac
  echo "[post-batch-eval] launch $label (dev=$dev)"
  ( cd "$DREAMQAS" && DREAMQAS_NO_MPS=1 OMP_NUM_THREADS=1 python -u analysis/eval_policy_traces.py "$d" \
      --device "$dev" --only_ep 15000 --n_final 100 > "$LOGS/${label}.log" 2>&1 ) &
  sleep 2
done
wait; touch "$C/_OF_POST_BATCH_EVAL_DONE"; echo "[post-batch-eval] ALL DONE."
