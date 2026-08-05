#!/bin/bash
# End-to-end NOISE experiment — LiH-4q, 2 device tiers x 5 arms x 5 seeds = 50 runs.
#
# WHAT THIS RUNS
#   arms   DreamQAS (Full) / No-imag / DreamQAS-RL  + the two RL baselines on the same
#          matched-VQE axis, CRLQAS and HyRLQAS. CRLQAS especially: its entire
#          contribution is QAS under hardware errors, so a noise section that omits it
#          invites the obvious reviewer question. GQE / TF-QAS / QuantumDARTS are left
#          out on purpose — quality-only, off the shared VQE axis (of_main_table.py
#          caption 3), and would only add caveats.
#          All five arms have finished CLEAN LiH-4q counterparts, so this is a
#          like-for-like overlay on the existing table.
#
#   ONE NOISE MODEL FOR ALL FIVE. CRLQAS/HyRLQAS live in PSQASBench, whose own qulacs
#   noise path supports only 1q/2q depolarizing with NO readout — using it would mean
#   comparing two different noise models. Instead PSQASBench embeds the same noise_ptm
#   package (validated against its own qulacs density matrix to 2.3e-07 Ha).
#   DreamQAS arms take the spec via CLI flags; PSQASBench arms via DREAMQAS_NOISE_* env
#   vars, because their CircuitEnv is built in 8 places and an env var cannot be missed
#   at one of them.
#
#   ⚠ TWO EMBEDDED COPIES. noise_ptm lives in BOTH trees. If they drift, the five arms
#   stop sharing a noise model and the comparison is invalid — while everything still
#   imports and still produces plausible energies, so nothing else would catch it. This
#   script therefore runs verify_sync.py first and REFUSES to launch on any drift.
#   tiers  low  = boston (IBM Heron r3, best current hardware)
#          high = miami  (IBM Nighthawk r1, harshest current hardware)
#          Both are device-wide medians from real IBM calibration CSVs (2026-05-20),
#          taken via code/noise_ptm/spec.py — this script never hardcodes the numbers,
#          it asks Python for them, so there is exactly one source of truth.
#
# ⚠ THE TIERS ARE A LOWER BOUND ON DEVICE NOISE. The calibration anchors also carry
# T1/T2 (amp_damp + dephasing); this project's noise spec has no time-like noise, so
# only the gate + readout part is used. A real device is worse. State this in the paper.
#
# WHY CPU. The PTM backend is a 256-dim kernel: measured 2.01 ms/eval on CPU vs 3.11 ms
# on GPU, because at 4 qubits the cost is CPU<->GPU dispatch latency, not FLOPs (the
# same conclusion NAHyRLQAS reached, docs/noise_acceleration.md §6). So arms A/B run
# JAX-on-CPU at one thread each and never touch the GPU. Arm C still needs a CUDA device
# for its torch actor, but its JAX stays on CPU.
#
# COST. Measured against a same-seed/same-load clean twin: ~3.2x per RL step, ~1-2 days
# per run at 15,000 episodes. 5 arms x 5 seeds x 2 tiers = 50 runs; stage one tier at a
# time with --tiers low, then --tiers high.
#
# Usage: ./noise_v1.sh [--dry] [--max N] [--tiers "low high"] [--arms "Full No-imag DreamQAS-RL"] [--seeds 0,1,2,3,4]
set -eo pipefail
source /home/sh0/S4068570/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env
REPO=/home/sh0/S4068570/DreamQAS
DREAMQAS=$REPO/code/WM_QAS
PSQAS=/home/sh0/S4068570/NeurIPS2026/PSQASBench
NZ=/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/noise_v1
LOGS=$NZ/logs

MAX=25; DRY=0; SEEDS="0,1,2,3,4"; TIERS="low high"
# Free VRAM a GPU arm needs before it may launch. Only the small torch policy
# network lives on the GPU (JAX/PTM is on CPU), but this batch puts 15 GPU
# processes onto 3 cards that the scale-up fleet is already using, so launches
# WAIT for room instead of OOM-ing. Same veto pattern as scaleup_v1.sh.
MIN_FREE_MB=4000
# CPU core pinning. MEASURED 2026-08-05: without it each run spawns ~150 OS threads
# (XLA's CPU thread pool — OMP_NUM_THREADS does NOT control it, and neither does
# XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1, both
# tested and both still gave 150). 25 runs x 150 spinning threads drove load to 104 on a
# 56-core box and was starving the existing scale-up fleet.
# `taskset -c <core>` cuts it to 7 threads for a 10% per-eval cost (1.19 -> 1.31 ms),
# i.e. the pool was spinning, not working — the 256-dim kernel gets nothing from it.
# CORE_OFFSET keeps the noise batch off the low cores the other fleet tends to land on.
PIN=1
CORE_OFFSET=8
ARMS="Full No-imag DreamQAS-RL crlqas hyrlqas"
MOL=LiH4q; EPISODES=15000
# PSQASBench names the same task differently from DreamQAS: DQ_LiH_4q vs LiH4q. Keep the
# two keys explicit — deriving one from the other ("DQ_$MOL") silently produces
# DQ_LiH4q, a config that does not exist, and the run dies at launch.
PSMOL=DQ_LiH_4q
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --dry) DRY=1; shift;;
  --tiers) TIERS="$2"; shift 2;; --arms) ARMS="$2"; shift 2;;
  --seeds) SEEDS="$2"; shift 2;; --episodes) EPISODES="$2"; shift 2;;
  --min_free) MIN_FREE_MB="$2"; shift 2;;
  --no_pin) PIN=0; shift;; --core_offset) CORE_OFFSET="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;; esac; done
mkdir -p "$LOGS"

# Hard gate: the two embedded copies of noise_ptm must be byte-identical, or the five
# arms are not comparable. Cheap (9 hashes) and worth it before a 25-run batch.
if ! python "$REPO/code/noise_ptm/verify_sync.py"; then
  echo "!! noise_ptm copies have DRIFTED — refusing to launch."
  echo "   Fix, or re-sync with: python $REPO/code/noise_ptm/verify_sync.py --update"
  exit 4
fi

# Single source of truth for the noise strengths: read them out of spec.py rather than
# duplicating them here, so the launcher can never drift from what the paper cites.
tier_flags() {
  PYTHONPATH="$REPO/code" JAX_PLATFORMS=cpu python - "$1" <<'PY'
import sys
from noise_ptm.spec import DEVICE_TIERS, TIER_NAMES
t = DEVICE_TIERS[TIER_NAMES[sys.argv[1]]]
print(f"--noise_mode ptm --noise_p1q {t.p1q:.6g} --noise_p2q {t.p2q:.6g} --noise_p_ro {t.p_ro:.6g}")
PY
}
tier_env() {
  PYTHONPATH="$REPO/code" JAX_PLATFORMS=cpu python - "$1" <<'PY'
import sys
from noise_ptm.spec import DEVICE_TIERS, TIER_NAMES
t = DEVICE_TIERS[TIER_NAMES[sys.argv[1]]]
print(f"DREAMQAS_NOISE_MODE=ptm DREAMQAS_NOISE_P1Q={t.p1q:.6g} "
      f"DREAMQAS_NOISE_P2Q={t.p2q:.6g} DREAMQAS_NOISE_PRO={t.p_ro:.6g} "
      f"DREAMQAS_NOISE_BASIS_CHANGE=1")
PY
}
want() { case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }

CMDS=()
for tier in $TIERS; do
  FLAGS=$(tier_flags "$tier") || { echo "!! unknown tier: $tier"; exit 3; }
  TAG="_nz_${tier}"
  for s in $(echo "$SEEDS" | tr ',' ' '); do
    # ---- DreamQAS (Full) ------------------------------------------------------------
    # NOTE: no --assert_full. noise_mode is in CANONICAL_FULL, so a noisy run is by
    # construction NOT canonical Full and --assert_full would (correctly) abort it.
    want Full && CMDS+=("Full_${tier}_s${s}"$'\t'"cpu"$'\t'"$DREAMQAS"$'\t'\
"python -u phase2_surrogate/runner.py --molecule $MOL --seed $s --oracle_free 1 \
--n_iterations $((EPISODES/4)) $FLAGS --out_dir $NZ/$tier/dreamqas --tag $TAG"$'\t'\
"$NZ/$tier/dreamqas/gru_energy_surrogate_${MOL}_s${s}${TAG}"$'\t'\
"runner.py --molecule $MOL --seed $s --oracle_free 1 --n_iterations")

    # ---- No-imag --------------------------------------------------------------------
    want No-imag && CMDS+=("Noimag_${tier}_s${s}"$'\t'"cpu"$'\t'"$DREAMQAS"$'\t'\
"python -u phase2_surrogate/runner.py --molecule $MOL --seed $s --oracle_free 1 \
--imagination none --n_iterations $((EPISODES/4)) $FLAGS --out_dir $NZ/$tier/ablations --tag ${TAG}_noimag"$'\t'\
"$NZ/$tier/ablations/gru_energy_none_${MOL}_s${s}${TAG}_noimag"$'\t'\
"runner.py --molecule $MOL --seed $s --oracle_free 1 --imagination none")

    # ---- DreamQAS-RL ----------------------------------------------------------------
    want DreamQAS-RL && CMDS+=("RL_${tier}_s${s}"$'\t'"gpu"$'\t'"$DREAMQAS"$'\t'\
"python -u main_baseline.py --config G0_v2_$MOL --experiment_name analysis/ --molecule $MOL \
--seed $s --gpu_id 0 --max_episodes $EPISODES $FLAGS --out_dir $NZ/$tier/dreamqas_rlqas"$'\t'\
"$NZ/$tier/dreamqas_rlqas/baseline_analysis/G0_v2_${MOL}/seed_${s}"$'\t'\
"main_baseline.py --config G0_v2_$MOL --experiment_name analysis/ --molecule $MOL --seed $s")

    # ---- CRLQAS / HyRLQAS (PSQASBench; noise arrives via DREAMQAS_NOISE_* env vars) --
    want crlqas && CMDS+=("crlqas_${tier}_s${s}"$'\t'"gpu"$'\t'"$PSQAS"$'\t'\
"python -u main.py --method crlqas --mol $PSMOL --config DreamQAS_EXP/${PSMOL}.cfg --seed $s --device cuda:0 --results-root $NZ/$tier/psqas --use-wandb 0"$'\t'\
"$NZ/$tier/psqas/crlqas/$PSMOL/DreamQAS_EXP/$PSMOL/seed$s"$'\t'\
"main.py --method crlqas --mol $PSMOL --config DreamQAS_EXP/${PSMOL}.cfg --seed $s ")

    want hyrlqas && CMDS+=("hyrlqas_${tier}_s${s}"$'\t'"gpu"$'\t'"$PSQAS"$'\t'\
"python -u main.py --method hyrlqas --mol $PSMOL --config DreamQAS_STD/${PSMOL}.cfg --seed $s --device cuda:0 --results-root $NZ/$tier/psqas_hyrlqas_std --use-wandb 0"$'\t'\
"$NZ/$tier/psqas_hyrlqas_std/hyrlqas/$PSMOL/DreamQAS_STD/$PSMOL/seed$s"$'\t'\
"main.py --method hyrlqas --mol $PSMOL --config DreamQAS_STD/${PSMOL}.cfg --seed $s ")
  done
done

# A run is DONE when its metrics.jsonl has as many rows as iterations requested. The
# DreamQAS arms write one row per iteration; the RL baseline one per iteration too.
finished() {
  local d="$1" want_rows="$2"
  # PSQASBench stamps wall_clock into run_meta.txt only at the very end.
  if [ -f "$d/run_meta.txt" ]; then
    grep -q wall_clock "$d/run_meta.txt" 2>/dev/null && return 0 || return 1
  fi
  [ -f "$d/metrics.jsonl" ] || return 1
  [ "$(wc -l < "$d/metrics.jsonl")" -ge "$want_rows" ]
}

QUEUE=()
for line in "${CMDS[@]}"; do
  IFS=$'\t' read -r label dev wd cmd outdir guard <<< "$line"
  if finished "$outdir" $((EPISODES/4)); then echo "[done]      $label"; continue; fi
  # In-flight guard. Uses the system pgrep -- do NOT let conda's lib dir onto the global
  # LD_LIBRARY_PATH or this silently stops matching and duplicate runs get launched into
  # the same directory (that exact failure cost 69 contaminated run dirs on 2026-07-31).
  if pgrep -f -- "$guard" > /dev/null 2>&1; then echo "[in-flight] $label"; continue; fi
  QUEUE+=("$line")
done
echo "[noise] queued=${#QUEUE[@]} of ${#CMDS[@]}  MAX=$MAX  tiers='$TIERS'  arms='$ARMS'  episodes=$EPISODES"
for t in $TIERS; do echo "         $t: $(tier_flags "$t")"; done
if [ "$DRY" = 1 ]; then
  printf '%s\n' "${QUEUE[@]}" | awk -F'\t' '{print "  " $1 "  [" $2 "]  -> " $5}'
  exit 0
fi
echo "[noise] free disk: $(df -h "$NZ" --output=avail | tail -1)"

# Count globally: after a restart the previous launcher's children are orphaned and
# invisible to `jobs`, so a fresh --max would stack on top of them.
running() { pgrep -fa -- "python -u (phase2_surrogate/runner.py|main_baseline.py)" 2>/dev/null \
              | grep -cF -- "$NZ" || true; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
NCORE=$(nproc)
if [ "$PIN" = 1 ] && [ "${#QUEUE[@]}" -gt "$((NCORE - CORE_OFFSET))" ]; then
  echo "!! ${#QUEUE[@]} runs but only $((NCORE - CORE_OFFSET)) cores from offset $CORE_OFFSET."
  echo "   Two runs would share a core and halve each other. Lower --max or --core_offset."
  exit 5
fi
i=0; core_i=0
for line in "${QUEUE[@]}"; do
  IFS=$'\t' read -r label dev wd cmd outdir guard <<< "$line"
  while [ "$(running)" -ge "$MAX" ]; do sleep 20; done
  if [ "$dev" = gpu ]; then
    # Pick the first card with room; if none has it, wait rather than OOM. NGPU comes
    # from the driver -- a hardcoded card count is how 4 runs died on 2026-07-31.
    g=""; tries=0
    while [ -z "$g" ]; do
      for _ in $(seq 1 "$NGPU"); do
        c=$(( i % NGPU )); i=$((i+1))
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null || echo 0)
        [ "$free" -ge "$MIN_FREE_MB" ] && { g=$c; break; }
      done
      if [ -z "$g" ]; then
        tries=$((tries+1))
        echo "[noise] all cards below ${MIN_FREE_MB}MiB free; wait 60s ($tries)"
        sleep 60
      fi
    done
    VIS="$g"
  else
    VIS=""                     # arms A/B: JAX on CPU, no GPU at all
  fi
  # PSQASBench arms get the noise through the process environment (see header).
  case "$label" in
    crlqas_*|hyrlqas_*) NOISE_ENV=$(tier_env "$(echo "$label" | cut -d_ -f2)");;
    *)                  NOISE_ENV="";;
  esac
  if [ "$PIN" = 1 ]; then
    PIN_CMD="taskset -c $(( (CORE_OFFSET + core_i) % NCORE ))"; core_i=$((core_i+1))
  else
    PIN_CMD=""
  fi
  echo "[noise] launch $label  (jax=cpu, torch=${dev}${VIS:+ gpu$VIS}${PIN_CMD:+, core ${PIN_CMD#taskset -c }})"
  # bash -c, NOT -lc: a login shell re-reads /etc/profile, resets PATH and drops the
  # conda env -> every child dies with ModuleNotFoundError: numpy.
  # LD_LIBRARY_PATH is CHILD-SCOPED: python needs conda's libstdc++ (the host's lacks
  # GLIBCXX_3.4.29), but exporting it globally breaks the system pgrep above.
  ( cd "$wd" && \
      JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="$VIS" DREAMQAS_NO_MPS=1 OMP_NUM_THREADS=1 \
      LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      $PIN_CMD env $NOISE_ENV bash -c "$cmd" > "$LOGS/${label}.log" 2>&1 ) &
  sleep 3
done
wait
touch "$LOGS/_NOISE_V1_DONE"
echo "[noise] ALL DONE -> $NZ"
