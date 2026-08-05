#!/bin/bash
# SCALE-UP campaign: the two larger tasks, run under the EXISTING settings, nothing invented.
#
#   BeH2 12q  cc-pVDZ (2e,6o)  E0 = -15.76934270 Ha   -- next rung of the basis ladder
#                                                        STO-3G 6q -> 6-31G 8q -> 6-311G 10q -> cc-pVDZ 12q
#   H2O  8q   STO-3G (4e,4o)   E0 = -74.97036284 Ha   -- a DIFFERENT molecule at a size already
#                                                        in the suite (tests transfer, not just scale)
#
# Both Hamiltonians are installed bit-identically into the DreamQAS and PSQASBench trees by
# experiments/scaleup/prepare_scaleup_mols.py (|dE0| = 0 verified); the baseline configs come from
# experiments/scaleup/gen_scaleup_baseline_configs.py, which DERIVES its protocol overlay from the
# shipped DQ_BeH2_8q/10q configs so the new tasks cannot drift from the paper's five.
#
# ARMS (5 seeds each; 8 arms x 2 tasks x 5 = 80 runs):
#   Full        DreamQAS canonical AC, ORACLE-FREE            -> oracle_free_v1/dreamqas
#   No-imag     --imagination none, oracle-free               -> oracle_free_v1/ablations
#   DreamQAS-RL WM-free REINFORCE (main_baseline.py)           -> oracle_free_v1/dreamqas_rlqas
#   crlqas / qdarts / gqeqas / tfqas   DreamQAS_EXP configs   -> campaign_v1/psqas
#   hyrlqas                            DreamQAS_STD config    -> campaign_v1/psqas_hyrlqas_std
#                                      (STD = HyRLQAS's own published protocol; it is the row
#                                       policy_quality_table.py:59 reads, so it is what we run)
#
# Output paths and run_name suffixes replicate the existing trees EXACTLY, so of_main_table.py /
# policy_quality_table.py pick the new columns up as soon as their MOLS lists are extended. New
# molecules mean new subdirectories: no existing run directory is written to or deleted.
#
# BUDGET: 15,000 episodes for every arm and both tasks -- the locked common budget. DreamQAS
# inherits it from config.EPISODES (3750 iters x 4 real eps); the baseline configs carry
# episodes = 15000. No task-specific budget is introduced.
#
# MEASURED COST (probe, 26 iters, 3 probes + 16 concurrent baseline reruns on the box, so only the
# RATIOS are meaningful): H2O_8q 24.4 s/iter, BeH2_12q 27.2 s/iter, BeH2_10q 33.2 s/iter -- i.e.
# 0.73x and 0.82x of the 10q task. 12q is NOT a cost blow-up: its dense-H energy einsum is only
# ~1.6x the 10q one (4096^2 complex64 = 134 MB resident) and the WM/actor cost dominates. Against
# the production 10q run (19743 s at 25-way concurrency) expect roughly 4-10 h per DreamQAS run;
# the spread is real, because the 8q task cost MORE than the 10q one in production (8.9 h vs 5.5 h)
# -- steady-state VQE volume, not qubit count, sets the price.
#
# Resumable: a run whose expected output already exists is skipped.
#
# Usage: ./scaleup_v1.sh [--dry] [--smoke] [--max N] [--mols "BeH2_12q H2O_8q"]
#                        [--only "Full No-imag ..."] [--seeds 0,1,2,3,4]
set -eo pipefail
source /home/USER/miniconda3/etc/profile.d/conda.sh && conda activate crlqas_env
DREAMQAS=/home/USER/DreamQAS/code/WM_QAS
PSQAS=/home/USER/NeurIPS2026/PSQASBench
OF=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1
CV=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1
LOGS=/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/scaleup_v1_logs
# MAX: sized from the measured box, not guessed (96 cores / 4x L4 23 GB, 2026-07-30):
#   CPU      1 core per run (OMP_NUM_THREADS=1) -> 96 - ~8 reserved = 88 runs.  NOT binding.
#   RAM      ~1.6 GB RSS per run of 363 GB.                                     NOT binding.
#   GPU mem  mean 1.12 GB per process, 4x23034 MiB total; filling to 80% -> ~65 processes.
#            WORST case is CRLQAS on 12q at 4.9 GB ([5000x5] DQN + complex128 dense H), so a
#            card must never host many of those -> MIN_FREE_MB veto below.       BINDING.
#   GPU SM   at 8 processes/card two of four cards already sit at SM 100%, so past ~12/card the
#            returns fall off. 32 here + 12 (hyrlqas rerun) + 5 (crlqas LiH-6q) = 49 concurrent,
#            which matches the "50 concurrent, no pressure" measurement recorded for this node.
MAX=32; DRY=0; SMOKE=0; MOLS="BeH2_12q H2O_8q"; ONLY=""; SEEDS="0,1,2,3,4"
# Headroom to leave on a card AFTER the run being launched has taken its share.
# A flat "skip any card under N MiB" threshold must be sized for the worst tenant (CRLQAS-12q,
# 6.3 GB on the cluster) -- and then a 480 MiB DreamQAS-RL run is refused a card with 7 GB free and the
# launcher stalls with GPUs half empty. Require instead: free >= need(this run) + MARGIN.
MARGIN_MB=4096
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --dry) DRY=1; shift;; --smoke) SMOKE=1; shift;;
  --mols) MOLS="$2"; shift 2;; --only) ONLY="$2"; shift 2;; --seeds) SEEDS="$2"; shift 2;;
  --margin) MARGIN_MB="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;; esac; done
mkdir -p "$LOGS" "$OF/dreamqas" "$OF/ablations" "$OF/dreamqas_rlqas" "$CV/psqas" "$CV/psqas_hyrlqas_std"
SEEDLIST=$(echo "$SEEDS" | tr ',' ' ')

# DreamQAS molecule key -> PSQASBench --mol key.  12q uses the NATIVE benchmark molecule (as
# 8q/10q do); H2O came in through the DQ_* conversion route (as LiH 4q/6q and BeH2 6q did).
psmol() { case "$1" in
    BeH2_12q) echo T5_BeH2_CCPVDZ_12q;;
    H2O_8q)   echo DQ_H2O_8q;;
    *) echo "!! no PSQASBench mol key for $1" >&2; exit 3;; esac; }
# baseline config stem (same for every method; the family differs, see below)
csmol() { case "$1" in
    BeH2_12q) echo DQ_BeH2_12q;;
    H2O_8q)   echo DQ_H2O_8q;;
    *) exit 3;; esac; }

# Measured GPU footprint per arm on the cluster (nvidia-smi --query-compute-apps, 2026-08-02).
# Sizes the pre-launch headroom check; update if an arm's model size changes.
need_mb() { case "$1" in
    crlqas_BeH2_12q*)  echo 6300;;   # [5000x5] DQN + complex128 4096^2 dense H -- by far the largest
    crlqas_*)          echo 3100;;
    gqeqas_*)          echo 1500;;
    qdarts_*)          echo 1300;;
    tfqas_*)           echo 1200;;
    hyrlqas_BeH2_12q*) echo 1200;;
    hyrlqas_*)         echo  800;;
    Full_*)            echo 1250;;
    Noimag_*)          echo 1100;;
    RL_*)              echo  600;;
    *)                 echo 2000;;   # unknown label -> conservative
  esac; }

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }

DQ_SMOKE=""; EP_SMOKE=""
[ "$SMOKE" = 1 ] && DQ_SMOKE=" --n_iterations 4 --warmup_eps 8 --wm_refresh_every 3" && EP_SMOKE=" --max_episodes 16"

CMDS=()
# label, workdir, cmd, done-marker path, in-flight guard pattern.
# The GUARD is what makes this launcher safe to restart at a different --max while runs are in
# flight: a queued run whose process is already alive is skipped. Without it a restart would
# double-launch every in-flight run -- and DreamQAS's runner.py has NO result-dir lock (only
# PSQASBench does, RLQAS/result_logger.py:120-133), so two processes would write one output dir.
add() { CMDS+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"); }

# SEED-major, MOLECULE-inner: the two molecules interleave instead of running back to back.
# Molecule-major (the original order) put all 40 BeH2-12q runs ahead of every H2O-8q run, so at
# MAX=30 the H2O column would not start for 3-5 days. Interleaving also mixes the cheap task in
# with the expensive one: H2O-8q needs a 256-dim state vector against 12q's 4096, so it both
# finishes sooner and holds far less GPU memory.
for s in $SEEDLIST; do
  for mol in $MOLS; do
    pm=$(psmol "$mol"); cm=$(csmol "$mol")
    # ---- internal DreamQAS arms (oracle-free) --------------------------------------------
    want Full && add "Full_${mol}_s${s}" "$DREAMQAS" \
      "python -u phase2_surrogate/runner.py --molecule $mol --seed $s --oracle_free 1 --assert_full 1 --out_dir $OF/dreamqas --tag _of$DQ_SMOKE" \
      "$OF/dreamqas/gru_energy_surrogate_${mol}_s${s}_of/eval_traces.jsonl" \
      "runner.py --molecule $mol --seed $s --oracle_free 1 --assert_full"
    want No-imag && add "Noimag_${mol}_s${s}" "$DREAMQAS" \
      "python -u phase2_surrogate/runner.py --molecule $mol --seed $s --oracle_free 1 --imagination none --out_dir $OF/ablations --tag _of_noimag$DQ_SMOKE" \
      "$OF/ablations/gru_energy_none_${mol}_s${s}_of_noimag/eval_traces.jsonl" \
      "runner.py --molecule $mol --seed $s --oracle_free 1 --imagination none"
    want DreamQAS-RL && add "RL_${mol}_s${s}" "$DREAMQAS" \
      "python -u main_baseline.py --config G0_v2_${mol} --experiment_name analysis/ --molecule $mol --seed $s --gpu_id 0 --max_episodes 15000 --out_dir $OF/dreamqas_rlqas$EP_SMOKE" \
      "$OF/dreamqas_rlqas/baseline_analysis/G0_v2_${mol}/seed_${s}/eval_traces.jsonl" \
      "main_baseline.py --config G0_v2_${mol} --experiment_name analysis/ --molecule $mol --seed $s "
    # ---- external baselines ---------------------------------------------------------------
    for meth in crlqas qdarts gqeqas tfqas; do
      want "$meth" && add "${meth}_${mol}_s${s}" "$PSQAS" \
        "python -u main.py --method $meth --mol $pm --config DreamQAS_EXP/${cm}.cfg --seed $s --device cuda:0 --results-root $CV/psqas --use-wandb 0" \
        "$CV/psqas/$meth/$pm/DreamQAS_EXP/${cm}/seed${s}/run_meta.txt" \
        "main.py --method $meth --mol $pm --config DreamQAS_EXP/${cm}.cfg --seed $s "
    done
    want hyrlqas && add "hyrlqas_${mol}_s${s}" "$PSQAS" \
      "python -u main.py --method hyrlqas --mol $pm --config DreamQAS_STD/${cm}.cfg --seed $s --device cuda:0 --results-root $CV/psqas_hyrlqas_std --use-wandb 0" \
      "$CV/psqas_hyrlqas_std/hyrlqas/$pm/DreamQAS_STD/${cm}/seed${s}/run_meta.txt" \
      "main.py --method hyrlqas --mol $pm --config DreamQAS_STD/${cm}.cfg --seed $s "
  done
done

# Is this run FINISHED?  The two families need different tests, and getting this wrong is not
# cosmetic -- a marker that appears while the run is still going makes a crashed run look complete
# and it would never be resumed:
#   DreamQAS  eval_traces.jsonl is written only at the end (an in-flight run dir holds just
#             metrics/trajectory/fidelity/imag/calibration + ckpt), so mere existence is valid.
#   PSQASBench  eval.jsonl is appended every eval_every episodes from the start, so it is NOT a
#             completion signal. run_meta.txt is rewritten with wall_clock_sec only on completion
#             (verified: in-flight 12q run has 2 eval lines and no wall_clock; finished 8q run has it).
finished() {
  case "$1" in
    */run_meta.txt) [ -f "$1" ] && grep -q wall_clock "$1" ;;
    *)              [ -s "$1" ] ;;
  esac
}

# drop runs that are already finished OR already in flight
QUEUE=()
for line in "${CMDS[@]}"; do
  IFS=$'\t' read -r label wd cmd marker guard <<< "$line"
  if [ "$SMOKE" = 0 ] && finished "$marker"; then echo "[done]      $label"; continue; fi
  if pgrep -f -- "$guard" > /dev/null 2>&1; then echo "[in-flight] $label"; continue; fi
  QUEUE+=("$line")
done

echo "[scaleup] queued=${#QUEUE[@]} of ${#CMDS[@]}  MAX=$MAX  mols='$MOLS'  only='${ONLY:-all 8 arms}'  seeds=$SEEDS  smoke=$SMOKE"
if [ "$DRY" = 1 ]; then
  for line in "${QUEUE[@]}"; do IFS=$'\t' read -r l w c m g <<< "$line"; printf '%-26s %s\n' "$l" "$c"; done
  exit 0
fi
echo "[scaleup] free disk: $(df -h "$OF" --output=avail | tail -1)"

# One MPS daemon for ALL cards (measured ~4x on this workload when many processes share a GPU).
#
# ⚠ THE DAEMON INHERITS CUDA_VISIBLE_DEVICES AND SERVES ONLY WHAT IT CAN SEE.
# Observed on the cluster 2026-07-31: a run launched with CUDA_VISIBLE_DEVICES=0 won the
# runtime_init.ensure_mps() lock and auto-started the daemon, which therefore managed GPU 0 alone.
# Every subsequent client on gpu1/gpu2 then died with "RuntimeError: No CUDA GPUs are available"
# -- 2 of 3 runs, silently, with an error that looks like a driver problem and is not. So:
#   1. start the daemon HERE, with CVD explicitly listing every card, before any run exists;
#   2. verify it actually serves all of them;
#   3. export DREAMQAS_NO_MPS=1 into the children so no run can ever auto-start a narrowed daemon
#      (they still USE this daemon -- /tmp/nvidia-mps is the CUDA default pipe dir, found without
#      any opt-in; DREAMQAS_NO_MPS only disables ensure_mps()'s *daemon-starting* path).
# Stop it with: echo quit | nvidia-cuda-mps-control
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  ALLGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
  # Try to start it; "An instance of this daemon is already running" is a normal, expected outcome
  # and must not abort the launcher (set -e would). Do NOT gate this on pgrep -- whether a daemon
  # exists is not the question. The question is whether it serves every card, and only the
  # per-card probe below can answer that, so that probe is the sole gate.
  ( unset CUDA_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES="$ALLGPU" nvidia-cuda-mps-control -d ) 2>&1 | \
    grep -v "already running" || true
  sleep 2
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    CUDA_VISIBLE_DEVICES=$g python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
      || { echo "[scaleup] FATAL: MPS daemon does not serve gpu$g. Run:  echo quit | nvidia-cuda-mps-control"
           echo "[scaleup]        then re-launch (the daemon must start with every card visible)."; exit 3; }
  done
  echo "[scaleup] MPS ok on all cards: $ALLGPU"
fi
export DREAMQAS_NO_MPS=1
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
gpu_nproc()   { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
                | grep -cF "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1")"; }
# Start the round-robin on the currently LEAST loaded card. Round-robin is balanced within one
# launcher, but three launchers each starting at card 0 is what produced the 2-saturated/2-idle
# split observed at 01:24 -- this offset removes that.
best=0; bestn=999999
for g in $(seq 0 $((NGPU-1))); do n=$(gpu_nproc "$g"); [ "$n" -lt "$bestn" ] && { bestn=$n; best=$g; }; done
echo "[scaleup] gpus=$NGPU  starting round-robin at gpu$best (has $bestn procs)  headroom=need+${MARGIN_MB}MiB"

# Count scale-up runs GLOBALLY, not via `jobs -r`. When this launcher is restarted at a different
# --max, the previous launcher's children are already orphaned to init and would be invisible to
# `jobs`, so a fresh MAX=32 on top of 16 live orphans would give 48. Matching on the two scale-up
# molecules counts orphans and own children alike, and deliberately excludes the baseline_rerun /
# crlqas-LiH-6q processes, which have their own cap.
scaleup_running() {
  pgrep -fa -- "python -u " 2>/dev/null \
    | grep -cE -- "--molecule (BeH2_12q|H2O_8q)|G0_v2_(BeH2_12q|H2O_8q)|--mol (T5_BeH2_CCPVDZ_12q|DQ_H2O_8q)" \
    || true
}

i=$best
for line in "${QUEUE[@]}"; do
  IFS=$'\t' read -r label wd cmd marker guard <<< "$line"
  while [ "$(scaleup_running)" -ge "$MAX" ]; do sleep 20; done
  # round-robin, but veto any card that lacks room for the biggest process we have measured
  # (CRLQAS on 12q peaks at 4.9 GB). Without the veto a clustered placement OOMs the card.
  want_mb=$(( $(need_mb "$label") + MARGIN_MB ))
  g=""; tries=0
  while [ -z "$g" ]; do
    for _ in $(seq 1 "$NGPU"); do
      c=$(( i % NGPU )); i=$((i+1))
      if [ "$(gpu_free_mb "$c")" -ge "$want_mb" ]; then g=$c; break; fi
    done
    if [ -z "$g" ]; then
      tries=$((tries+1))
      echo "[scaleup] no card has ${want_mb}MiB free for $label; waiting 60s (try $tries)"
      sleep 60
    fi
  done
  # bash -c, NOT bash -lc: a LOGIN shell re-reads /etc/profile + ~/.bash_profile and resets PATH,
  # which drops the conda env this launcher activated and lands the child in `base` -- every run
  # then dies with "ModuleNotFoundError: No module named 'numpy'". Observed on the cluster 2026-07-31;
  # the AWS box only survived it because its ~/.bashrc happened to re-activate the env.
  echo "[scaleup] launch $label -> gpu$g (free $(gpu_free_mb "$g")MiB)"
  # LD_LIBRARY_PATH is set HERE, on the child only -- never globally (e.g. via activate.d).
  # Why python needs it: this RHEL host's /lib64/libstdc++.so.6 lacks GLIBCXX_3.4.29, which
  # numpy's _pocketfft_umath requires; conda's libstdc++.so.6.0.34 has it.
  # Why it must NOT be global: it also makes SYSTEM binaries load conda's libsystemd.so.0,
  # which needs a newer glibc than the host has, so `pgrep` dies with
  #   "pgrep: /lib64/libc.so.6: version `GLIBC_2.30' not found".
  # A dead pgrep makes the in-flight guard above silently never match -> every live run gets
  # launched a SECOND time into the same output dir. That happened on the cluster 2026-07-31 and
  # corrupted a batch of runs. Confine the variable to the child; keep the launcher's own
  # pgrep/nvidia-smi on the system libraries.
  ( cd "$wd" && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 \
      LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      bash -c "$cmd" > "$LOGS/${label}.log" 2>&1 ) &
  # 6 s, not 2: CUDA init + the 268 MB 12q Hamiltonian load take ~10-20 s, so a shorter gap makes
  # the free-memory reading stale and lets several heavy runs pile onto one card.
  sleep 6
done
wait
touch "$LOGS/_SCALEUP_V1_DONE"
echo "[scaleup] ALL DONE -> $LOGS"
