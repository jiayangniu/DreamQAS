"""Entry point for WM-QAS v2 baseline (no world model / imagination).

Usage:
    python main_baseline.py --config LiH4q_baseline --experiment_name baseline/ --gpu_id 0 --seed 0

Hyperparameters are read from [baseline] section of the .cfg file.
Priority: CLI args > [baseline] in cfg > argparse defaults.
"""

# IMPORTANT: runtime_init must run before any scientific-library import so
# BLAS thread pools see the env vars before initialization. Defaults match
# main_baseline's argparse defaults so the limiter works even when invoked
# with no CLI args.
import runtime_init
_RUNTIME_NUM_THREADS = runtime_init.setup_blas_threads(
    default_config="LiH4q_baseline",
    default_experiment_name="baseline/",
)
# Enable NVIDIA MPS by default (concurrent GPU sharing, ~4x under many runs).
# Must precede `import torch`; no-op without MPS tools or with DREAMQAS_NO_MPS=1.
runtime_init.ensure_mps()

import sys
import argparse
import pathlib
import torch

import utils
from utils import get_config
from environment import CircuitEnv
from runner_baseline import BaselineRunner


_BASELINE_INT_KEYS = {
    "max_episodes", "real_episodes_per_iter", "actor_hidden_dim",
}
_BASELINE_FLOAT_KEYS = {
    "actor_lr", "entropy_coef", "reinforce_gamma",
}


def _parse_baseline_conf(raw: dict) -> dict:
    result = {}
    for k, v in raw.items():
        if k in _BASELINE_INT_KEYS:
            result[k] = int(v)
        elif k in _BASELINE_FLOAT_KEYS:
            result[k] = float(v)
    return result


def get_args(argv, baseline_defaults=None):
    parser = argparse.ArgumentParser(description="WM-QAS v2 Baseline (no WM)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="LiH4q_baseline")
    parser.add_argument("--experiment_name", type=str, default="baseline/")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--molecule", type=str, default="")          # for the standardized ckpt schedule
    parser.add_argument("--out_dir", type=str, default=None)         # override output root
    parser.add_argument("--eval_only", type=str, default=None)       # checkpoint .pt -> frozen offline eval
    parser.add_argument("--n_eval_episodes", type=int, default=20)

    parser.add_argument("--max_episodes",          type=int,   default=20000)
    parser.add_argument("--real_episodes_per_iter", type=int,   default=4)
    parser.add_argument("--actor_hidden_dim",       type=int,   default=512)
    parser.add_argument("--actor_lr",               type=float, default=3e-5)
    parser.add_argument("--entropy_coef",           type=float, default=1e-3)
    parser.add_argument("--reinforce_gamma",        type=float, default=0.99)

    # noisy environment (PTM backend, code/noise_ptm/). Default off -> legacy path.
    # This arm has no Config dataclass, so the flags live directly on args and are
    # handed to noise_ptm.integration.build_evaluator after the env is constructed.
    parser.add_argument("--noise_mode",   type=str,   default="off")   # off | ptm
    parser.add_argument("--noise_p1q",    type=float, default=0.0)
    parser.add_argument("--noise_p2q",    type=float, default=0.0)
    parser.add_argument("--noise_p_ro",   type=float, default=0.0)
    parser.add_argument("--noise_basis_change", type=int, default=1)

    if baseline_defaults:
        parser.set_defaults(**baseline_defaults)

    return parser.parse_args(argv)


if __name__ == "__main__":
    # Step 1: pre-parse to locate cfg file
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="LiH4q_baseline")
    pre.add_argument("--experiment_name", default="baseline/")
    pre_args, _ = pre.parse_known_args(sys.argv[1:])

    # Step 2: load cfg
    conf = get_config(pre_args.experiment_name, f"{pre_args.config}.cfg")

    # Step 3: extract [baseline] section
    baseline_defaults = _parse_baseline_conf(conf.get("baseline", {}))

    # Step 4: full arg parse (cfg fills defaults, CLI overrides)
    args = get_args(sys.argv[1:], baseline_defaults)

    torch.cuda.set_device(args.gpu_id)
    # Keep PyTorch intra-op threading in sync with the BLAS cap set by runtime_init.
    torch.set_num_threads(_RUNTIME_NUM_THREADS if _RUNTIME_NUM_THREADS is not None else 1)
    utils.set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")

    print("=" * 62)
    for section, values in conf.items():
        if section == "baseline":
            continue
        print(f"[CONFIG] {section}")
        for k, v in values.items():
            print(f"         {k}: {v}")
    print("=" * 62)
    print("[BASELINE CONFIG]")
    for k in sorted(_BASELINE_INT_KEYS | _BASELINE_FLOAT_KEYS):
        if hasattr(args, k):
            print(f"  {k}: {getattr(args, k)}")
    print("=" * 62)

    # frozen offline eval of ONE baseline checkpoint -> print stats (no training)
    if args.eval_only:
        import json
        from runner_baseline import evaluate_baseline_checkpoint
        stats = evaluate_baseline_checkpoint(args.eval_only, args.n_eval_episodes,
                                             device=f"cuda:{args.gpu_id}")
        print(json.dumps(stats, default=str), flush=True)
        sys.exit(0)

    environment = CircuitEnv(conf, device=device)

    # ---- noisy environment (default OFF; no-op unless --noise_mode ptm) ----
    if str(args.noise_mode).lower() != "off":
        _code_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        if _code_dir not in sys.path:
            sys.path.insert(0, _code_dir)        # .../DreamQAS/code -> `import noise_ptm`
        from noise_ptm.integration import build_evaluator
        _ev = build_evaluator(
            environment, mode=args.noise_mode, p1q=args.noise_p1q, p2q=args.noise_p2q,
            p_ro=args.noise_p_ro, basis_change=bool(args.noise_basis_change),
        )
        environment.attach_noisy_evaluator(_ev)

    root = args.out_dir or "results"
    output_path = f"{root}/baseline_{args.experiment_name}{args.config}/seed_{args.seed}"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    runner = BaselineRunner(
        env=environment,
        conf=conf,
        device=device,
        output_path=output_path,
        seed=args.seed,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_lr=args.actor_lr,
        entropy_coef=args.entropy_coef,
        reinforce_gamma=args.reinforce_gamma,
        max_episodes=args.max_episodes,
        real_episodes_per_iter=args.real_episodes_per_iter,
        molecule=args.molecule,
        env_cfg=args.config,
        experiment_name=args.experiment_name,
    )

    runner.run()
