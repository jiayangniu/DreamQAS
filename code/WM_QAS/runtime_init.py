import argparse
import configparser
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
    """Start or reuse NVIDIA MPS only when explicitly enabled."""
    if os.environ.get("DREAMQAS_ENABLE_MPS") != "1" or os.environ.get("DREAMQAS_NO_MPS"):
        return False
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False

    import fcntl

    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", _MPS_PIPE_DIR)
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", _MPS_LOG_DIR)
    pipe_dir = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    control_pipe = os.path.join(pipe_dir, "control")
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(os.environ["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    except OSError:
        return False

    started = False
    try:
        lock = open(_MPS_LOCK, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(control_pipe):
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(50):
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
