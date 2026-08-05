#!/usr/bin/env bash
# Canonical-AC campaign runner. Consumes campaign_manifest.py, owns MPS + GPU
# round-robin + concurrency throttle + disconnect-safe launch. Correctness of the
# run commands (canonical flags, budgets, mol keys) lives in campaign_manifest.py.
#
# Usage:
#   ./run_campaign.sh --queue q1_main_4q6q --max 88 [--seeds 0,1,2,3,4]
#                     [--methods Full,crlqas] [--smoke] [--dry] [--stagger 3] [--ngpu 4]
#
# (no `set -u`: conda activate references unbound vars)
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE=""; MAX=88; SEEDS="0,1,2,3,4"; METHODS=""; SMOKE=0; DRY=0; STAGGER=3; NGPU=4
while [ $# -gt 0 ]; do
  case "$1" in
    --queue) QUEUE="$2"; shift 2;;
    --max) MAX="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --methods) METHODS="$2"; shift 2;;
    --stagger) STAGGER="$2"; shift 2;;
    --ngpu) NGPU="$2"; shift 2;;
    --smoke) SMOKE=1; shift;;
    --dry) DRY=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$QUEUE" ] || { echo "ERROR: --queue required"; exit 2; }

source /home/sh0/S4068570/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env

GENARGS=(--queue "$QUEUE" --seeds "$SEEDS")
[ -n "$METHODS" ] && GENARGS+=(--methods "$METHODS")
[ "$SMOKE" = 1 ] && GENARGS+=(--smoke)
mapfile -t RUNS < <(python "$HERE/campaign_manifest.py" "${GENARGS[@]}")
echo "[campaign] queue=$QUEUE  runs=${#RUNS[@]}  MAX=$MAX  NGPU=$NGPU  smoke=$SMOKE"

if [ "$DRY" = 1 ]; then
  printf '%s\n' "${RUNS[@]}"; echo "[campaign] --dry: ${#RUNS[@]} runs printed, nothing launched."; exit 0
fi

# ---- single shared MPS daemon at the default pipe (so each client's ensure_mps
#      sees it running and no-ops instead of starting a conflicting daemon) ----
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if ! pgrep -f "nvidia-cuda-mps-control -d" >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d && echo "[campaign] single MPS daemon up (all $NGPU GPUs)"
  else
    echo "[campaign] MPS daemon already running"
  fi
fi

LOGROOT="/home/sh0/S4068570/DreamQAS/experiments/campaign_v1/logs/$QUEUE"
mkdir -p "$LOGROOT"

idx=0
for line in "${RUNS[@]}"; do
  label="${line%%$'\t'*}"; rest="${line#*$'\t'}"
  workdir="${rest%%$'\t'*}"; cmd="${rest#*$'\t'}"
  g=$(( idx % NGPU ))
  log="$LOGROOT/${label//[:\/]/_}.log"
  # throttle to MAX concurrent (plain `&` so jobs -r / wait can track them;
  # disconnect safety comes from running THIS launcher detached -- see footer).
  while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 5; done
  echo "[campaign] launch idx=$idx gpu=$g  $label"
  ( cd "$workdir" && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 bash -lc "$cmd" >"$log" 2>&1 ) &
  idx=$(( idx + 1 ))
  sleep "$STAGGER"
done

echo "[campaign] all ${#RUNS[@]} launched; waiting..."
wait
echo "[campaign] queue $QUEUE COMPLETE."
# leave the MPS daemon up for subsequent queues; stop with: echo quit | nvidia-cuda-mps-control
#
# DISCONNECT SAFETY: run this launcher itself detached so it (and its child runs)
# survive an SSH disconnect, e.g.:
#   setsid nohup ./run_campaign.sh --queue q1_main_4q6q --max 88 >campaign.q1.log 2>&1 &
