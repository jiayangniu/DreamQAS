#!/bin/bash
# Oracle-free main-experiment contrast (2026-07-24): DreamQAS-Full with oracle_free=1 (empirical-
# frontier signed-log score, fixed margin 0.1 mHa) x 5 tasks x 5 seeds = 25 full-budget runs.
# Purpose: measure deviation of the oracle-free practical variant vs the canonical E0-based main
# table (policy_quality_table.txt Full row). Canonical flags otherwise; budgets auto per molecule
# (config.py N_ITERS; LiH6q 2x). Frozen-eval via the standard ckpt grid + eval_policy_traces later.
#
# Usage:  ./oracle_free_main.sh [--smoke] [--dry] [--max N]
# Detached:  setsid bash ./oracle_free_main.sh < /dev/null > <log> 2>&1 &
set -eo pipefail
source /home/USER/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env
DREAMQAS=/home/USER/DreamQAS/code/WM_QAS
C=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1
MAX=$(( $(nproc) - 4 )); SMOKE=0; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --smoke) SMOKE=1; shift;; --dry) DRY=1; shift;;
  *) echo "unknown arg: $1"; exit 2;; esac; done

MOLS=(LiH4q BeH2 LiH6q BeH2_8q BeH2_10q)
EXTRA=""; [ "$SMOKE" = 1 ] && EXTRA=" --n_iterations 4 --warmup_eps 8 --wm_refresh_every 3"
mkdir -p "$C/dreamqas" "$C/logs"

CMDS=()
for mol in "${MOLS[@]}"; do
  for s in 0 1 2 3 4; do
    CMDS+=("${mol}_s${s}"$'\t'"python phase2_surrogate/runner.py --molecule ${mol} --seed ${s} --oracle_free 1 --out_dir ${C}/dreamqas --tag _of${EXTRA}")
  done
done
echo "[oracle-free-main] runs=${#CMDS[@]} MAX=$MAX smoke=$SMOKE margin=0.1mHa(fixed default)"
if [ "$DRY" = 1 ]; then printf '%s\n' "${CMDS[@]}"; exit 0; fi

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  pgrep -f "nvidia-cuda-mps-control -d" >/dev/null 2>&1 || nvidia-cuda-mps-control -d
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
idx=0
for line in "${CMDS[@]}"; do
  label="${line%%$'\t'*}"; cmd="${line#*$'\t'}"
  while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 10; done
  g=$(( idx % NGPU ))
  echo "[oracle-free-main] launch $label gpu=$g"
  ( cd "$DREAMQAS" && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 bash -lc "$cmd" > "$C/logs/${label}.log" 2>&1 ) &
  idx=$(( idx + 1 ))
  sleep 3
done
echo "[oracle-free-main] all ${#CMDS[@]} launched; waiting..."
wait
touch "$C/_ORACLE_FREE_MAIN_DONE"
echo "[oracle-free-main] ALL DONE."
