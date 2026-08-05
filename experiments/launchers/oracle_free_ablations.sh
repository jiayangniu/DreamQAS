#!/bin/bash
# Oracle-free contrast, wave 2 (2026-07-24): NoImag (oracle_free=1) + DreamQAS-RL, 5 tasks x 5 seeds
# each = 50 runs. Completes the oracle-free ladder contrast alongside oracle_free_main.sh (Full).
#   NoImag      : phase2 runner --imagination none --oracle_free 1 (WM+replay trained on the
#                 frontier score; no imagination, no DAgger).
#   DreamQAS-RL : main_baseline.py, UNMODIFIED — its REINFORCE reward is env.reward_fn on the
#                 fake_min_energy reference (cfg constant), so its TRAINING is already E0-free;
#                 fresh seeds here give a same-code-state control. E0 appears in its logs only.
# Output -> oracle_free_v1/{ablations,dreamqas_rlqas}. Safe to run concurrently with wave 1
# (96-thread node; MAX guard shares the box).
#
# Usage:  ./oracle_free_ablations.sh [--dry] [--max N]
set -eo pipefail
source /home/USER/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env
DREAMQAS=/home/USER/DreamQAS/code/WM_QAS
C=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1
MAX=$(( $(nproc) - 4 )); DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --dry) DRY=1; shift;;
  *) echo "unknown arg: $1"; exit 2;; esac; done

# molecule -> (dq key, cfg, episodes) — mirrors campaign_manifest.py MOLS
MOLS="LiH4q:G0_v2_LiH4q:15000 BeH2:G0_v2_BeH2:15000 LiH6q:G0_v2_LiH6q:30000 BeH2_8q:G0_v2_BeH2_8q:15000 BeH2_10q:G0_v2_BeH2_10q:15000"
mkdir -p "$C/ablations" "$C/dreamqas_rlqas" "$C/logs"

CMDS=()
for spec in $MOLS; do
  dq="${spec%%:*}"; rest="${spec#*:}"; cfgname="${rest%%:*}"; eps="${rest##*:}"
  for s in 0 1 2 3 4; do
    CMDS+=("noimag_${dq}_s${s}"$'\t'"python phase2_surrogate/runner.py --molecule ${dq} --seed ${s} --imagination none --oracle_free 1 --out_dir ${C}/ablations --tag _of_noimag")
    CMDS+=("rlqas_${dq}_s${s}"$'\t'"python main_baseline.py --config ${cfgname} --experiment_name analysis/ --molecule ${dq} --seed ${s} --gpu_id 0 --max_episodes ${eps} --out_dir ${C}/dreamqas_rlqas")
  done
done
echo "[of-abl] runs=${#CMDS[@]} MAX=$MAX"
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
  echo "[of-abl] launch $label gpu=$g"
  ( cd "$DREAMQAS" && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 bash -lc "$cmd" > "$C/logs/${label}.log" 2>&1 ) &
  idx=$(( idx + 1 ))
  sleep 3
done
echo "[of-abl] all ${#CMDS[@]} launched; waiting..."
wait
touch "$C/_ORACLE_FREE_ABL_DONE"
echo "[of-abl] ALL DONE."
