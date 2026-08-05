"""Pre-import BLAS thread limiter.

Must be imported (and `setup_blas_threads()` invoked) at the very top of
`main_*.py`, BEFORE any `import numpy / torch / scipy`. The BLAS thread
pool is sized when those libraries first initialize, so setting
`OMP_NUM_THREADS` later has no effect.

Reads `[runtime] num_threads` from the experiment's .cfg file. If the
section or key is missing, leaves the environment unchanged.

Uses only stdlib (`configparser`, `argparse`, `pathlib`) to avoid
pulling in numpy as a transitive import.
"""

import argparse
import configparser
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_MPS_PIPE_DIR = "/tmp/nvidia-mps"
_MPS_LOG_DIR = "/tmp/nvidia-mps-log"
_MPS_LOCK = "/tmp/dreamqas_mps.lock"


def ensure_mps() -> bool:
    """Route this process through NVIDIA MPS, starting the control daemon if
    none is running (once, flock-guarded so many concurrent runs don't race).

    Why: without MPS, GPU kernels from different processes are *time-sliced*.
    With MPS they run concurrently. Measured on this project: at 24 concurrent
    6q runs, mean iter_time dropped 12.8s -> 3.2s (~4x). Each run's WM training
    + imagination is small on a big GPU, so many runs share one card cheaply.

    Must run BEFORE the first CUDA call — CUDA reads CUDA_MPS_PIPE_DIRECTORY at
    context creation. Safe to call from runtime_init (precedes `import torch`).

    Disable with env DREAMQAS_NO_MPS=1. No-ops (returns False) when the MPS
    tools are absent, so it is safe on machines without MPS. Returns True if
    this process is wired to use MPS.

    Note: the daemon persists after runs exit (a shared, reusable service).
    Stop it with `echo quit | nvidia-cuda-mps-control`; the campaign launcher
    does this automatically at the end.
    """
    if os.environ.get("DREAMQAS_NO_MPS"):
        return False
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False  # MPS tools not installed -> run normally, no MPS

    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", _MPS_PIPE_DIR)
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", _MPS_LOG_DIR)
    pipe_dir = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    control_pipe = os.path.join(pipe_dir, "control")
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(os.environ["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    except OSError:
        return False

    # Serialize the check/start across all runs: the FIRST to grab the lock
    # starts the daemon and waits for it to be ready; the rest block on the
    # lock, then find it already running. Blocking (not LOCK_NB) so no run
    # proceeds to CUDA init before the daemon's control pipe exists.
    started = False
    try:
        lock = open(_MPS_LOCK, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(control_pipe):
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(50):  # up to ~5s for the daemon to come up
                if os.path.exists(control_pipe):
                    started = True
                    break
                time.sleep(0.1)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    except OSError:
        return False

    if started:
        print(f"[runtime_init] NVIDIA MPS daemon started (pipe={pipe_dir})")
    return os.path.exists(control_pipe)


def setup_blas_threads(
    cfg_root: str = "configuration_files",
    default_config: str | None = None,
    default_experiment_name: str | None = None,
) -> int | None:
    """Read [runtime] num_threads from the active cfg and set BLAS env vars.

    Args:
        cfg_root: directory holding {experiment_name}/{config}.cfg
        default_config: fallback if --config not on CLI (so the limiter
            works even when main_*.py is invoked with no args and the
            argparse defaults take over)
        default_experiment_name: fallback if --experiment_name not on CLI

    Returns the integer num_threads that was applied, or None if no cfg
    was located / no [runtime] section was present. Caller can pass the
    return value to `torch.set_num_threads` to keep PyTorch intra-op
    threading in sync with BLAS.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    pre.add_argument("--experiment_name", default=default_experiment_name)
    args, _ = pre.parse_known_args(sys.argv[1:])
    if not args.config or not args.experiment_name:
        return None

    cfg_path = (
        Path(__file__).parent / cfg_root / args.experiment_name / f"{args.config}.cfg"
    )
    if not cfg_path.exists():
        return None

    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    n = parser.get("runtime", "num_threads", fallback=None)
    if n is None:
        return None

    n_int = int(n)
    n_str = str(n_int)
    for var in _BLAS_ENV_VARS:
        os.environ[var] = n_str
    print(f"[runtime_init] BLAS threads = {n_str} (from {cfg_path.name})")
    return n_int
