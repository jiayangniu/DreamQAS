import runtime_init
_RUNTIME_NUM_THREADS = runtime_init.setup_blas_threads(
    default_config="G0_v2_LiH4q",
    default_experiment_name="analysis/",
)
runtime_init.ensure_mps()

import sys
import json
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
    parser.add_argument("--config", type=str, default="G0_v2_LiH4q")
    parser.add_argument("--experiment_name", type=str, default="analysis/")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--legacy_noise_config", type=json.loads, help='Explicit JSON noise settings for old checkpoints, e.g. {"mode":"off"}')
    parser.add_argument("--molecule", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--eval_only", type=str, default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=20)

    parser.add_argument("--max_episodes",          type=int,   default=20000)
    parser.add_argument("--real_episodes_per_iter", type=int,   default=4)
    parser.add_argument("--actor_hidden_dim",       type=int,   default=512)
    parser.add_argument("--actor_lr",               type=float, default=3e-5)
    parser.add_argument("--entropy_coef",           type=float, default=1e-3)
    parser.add_argument("--reinforce_gamma",        type=float, default=0.99)

    parser.add_argument("--noise_mode",   type=str,   default="off")
    parser.add_argument("--noise_p1q",    type=float, default=0.0)
    parser.add_argument("--noise_p2q",    type=float, default=0.0)
    parser.add_argument("--noise_p_ro",   type=float, default=0.0)
    parser.add_argument("--noise_basis_change", type=int, default=1)

    if baseline_defaults:
        parser.set_defaults(**baseline_defaults)

    return parser.parse_args(argv)


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="G0_v2_LiH4q")
    pre.add_argument("--experiment_name", default="analysis/")
    pre_args, _ = pre.parse_known_args(sys.argv[1:])

    conf = get_config(pre_args.experiment_name, f"{pre_args.config}.cfg")

    baseline_defaults = _parse_baseline_conf(conf.get("baseline", {}))

    args = get_args(sys.argv[1:], baseline_defaults)

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
    torch.set_num_threads(_RUNTIME_NUM_THREADS if _RUNTIME_NUM_THREADS is not None else 1)
    utils.set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")

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

    if args.eval_only:
        import json
        from runner_baseline import evaluate_baseline_checkpoint
        stats = evaluate_baseline_checkpoint(args.eval_only, args.n_eval_episodes,
                                             device=str(device), legacy_noise_config=args.legacy_noise_config)
        print(json.dumps(stats, default=str), flush=True)
        sys.exit(0)

    environment = CircuitEnv(conf, device=device)

    if str(args.noise_mode).lower() != "off":
        _code_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        if _code_dir not in sys.path:
            sys.path.insert(0, _code_dir)
        from noise_ptm.integration import build_evaluator
        _ev = build_evaluator(
            environment, mode=args.noise_mode, p1q=args.noise_p1q, p2q=args.noise_p2q,
            p_ro=args.noise_p_ro, basis_change=bool(args.noise_basis_change),
        )
        environment.attach_noisy_evaluator(_ev)

    root = str(pathlib.Path(args.out_dir or "runs/baseline").expanduser().resolve())
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
