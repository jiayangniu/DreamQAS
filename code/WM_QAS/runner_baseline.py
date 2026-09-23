from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

import utils
from agent.baseline_policy import BaselineDiscreteActor, BaselinePolicy
from phase2_surrogate import eval_harness
from phase2_surrogate.config import checkpoint_episodes as _ckpt_sched_for


class BaselineRunner:

    def __init__(
        self,
        env,
        conf: dict,
        device: torch.device,
        output_path: str,
        seed: int,
        actor_hidden_dim: int = 512,
        actor_lr: float = 3e-5,
        entropy_coef: float = 1e-3,
        reinforce_gamma: float = 0.99,
        max_episodes: int = 20000,
        real_episodes_per_iter: int = 4,
        molecule: str = "",
        env_cfg: str = "",
        experiment_name: str = "",
        ckpt_episodes=None,
        chem_acc_mHa: float = 1.6,
    ):
        self.env = env
        self.conf = conf
        self.device = device
        self.output_path = output_path
        self.seed = seed
        self.max_episodes = max_episodes
        self.real_episodes_per_iter = real_episodes_per_iter
        self.reinforce_gamma = reinforce_gamma
        self.molecule = molecule
        self.env_cfg = env_cfg
        self.experiment_name = experiment_name
        self.chem_acc_mHa = chem_acc_mHa
        if ckpt_episodes:
            self.ckpt_sched = sorted({int(x) for x in ckpt_episodes})
        elif molecule:
            try:
                self.ckpt_sched = _ckpt_sched_for(molecule)
            except KeyError:
                self.ckpt_sched = []
        else:
            self.ckpt_sched = []
        self.ckpt_ptr = 0
        self.best_seq = None
        self.best_depth = 0

        self.obs_dim = env.num_layers * (env.num_qubits + 3) * env.num_qubits
        self.action_size = env.action_size
        self.actor_hidden_dim = actor_hidden_dim

        self.actor = BaselineDiscreteActor(
            obs_dim=self.obs_dim,
            action_size=self.action_size,
            hidden_dim=actor_hidden_dim,
            entropy_coef=entropy_coef,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.policy = BaselinePolicy(
            actor=self.actor,
            translate=utils.dictionary_of_actions(env.num_qubits),
            device=device,
        )

        self.best_error = float("inf")
        self.total_eps = 0
        self.total_vqe_calls = 0


    def _get_obs_structure(self) -> np.ndarray:
        state = self.env.state
        structure = state[:, : self.env.num_qubits + 3]
        return structure.reshape(-1).cpu().numpy().astype(np.float32)

    def _collect_episode(self) -> dict:
        env = self.env
        env.reset()

        first_obs = self._get_obs_structure()
        obs_list = [first_obs]
        action_list, reward_list = [], []
        ill_mask_list: list[torch.Tensor] = []
        ep_return = 0.0

        obs_tensor = torch.tensor(first_obs, device=self.device)
        step_errs = []
        step_energies = []
        step_dones = []

        for _ in range(env.num_layers + 1):
            ill = env.illegal_action_new()

            ill_mask = torch.zeros(self.action_size, dtype=torch.bool)
            if ill:
                ill_mask[ill] = True
            ill_mask_list.append(ill_mask)

            env_action, action_idx = self.policy.act(
                obs_tensor,
                ill_actions=ill if ill else None,
            )

            act_param = torch.tensor(0.0, device=self.device)
            _, reward, done, _ = env.step(env_action, act_param, train_flag=True)

            step_errs.append(float(abs(env.min_energy - env.energy)))
            step_energies.append(float(env.energy))
            step_dones.append(bool(done))

            obs_next = self._get_obs_structure()
            obs_list.append(obs_next)
            obs_tensor = torch.tensor(obs_next, device=self.device)

            action_list.append(action_idx)
            reward_list.append(float(reward))
            ep_return += float(reward)

            if done:
                break

        true_error = float(abs(env.min_energy - env.energy))
        best_step = int(np.argmin(step_errs)) if step_errs else 0
        return {
            "obs":         np.array(obs_list, dtype=np.float32),
            "actions":     np.array(action_list, dtype=np.int64),
            "rewards":     np.array(reward_list, dtype=np.float32),
            "ill_masks":   torch.stack(ill_mask_list),
            "ep_return":   ep_return,
            "final_error": true_error,
            "min_error":   min(step_errs) if step_errs else true_error,
            "best_prefix": action_list[: best_step + 1],
            "n_steps":     len(action_list),
            "step_errs":   step_errs,
            "step_energies": step_energies,
            "step_dones":  step_dones,
        }

    def _eval_rollout(self) -> float:
        env = self.env
        env.reset()
        obs = torch.tensor(self._get_obs_structure(), device=self.device)
        best = float("inf")
        with torch.no_grad():
            for step in range(env.num_layers + 1):
                ill = env.illegal_action_new()
                if ill and len(ill) >= self.action_size:
                    break
                env_action, _ = self.policy.act(obs, ill_actions=ill if ill else None)
                env.step(env_action, torch.tensor(0.0, device=self.device), train_flag=True)
                best = min(best, float(abs(env.min_energy - env.energy)) * 1000.0)
                obs = torch.tensor(self._get_obs_structure(), device=self.device)
                if step >= env.num_layers - 1:
                    break
        return best

    def _save_ckpt(self, episode: int) -> None:
        import os
        cdir = f"{self.output_path}/ckpt"
        os.makedirs(cdir, exist_ok=True)
        torch.save({
            "episode": episode,
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vqe_calls": self.total_vqe_calls,
            "vqe_nfev": 0,
            "best_err_mHa": self.best_error * 1000.0,
            "best_seq": list(self.best_seq) if self.best_seq is not None else None,
            "best_depth": self.best_depth,
            "molecule": self.molecule, "env_cfg": self.env_cfg,
            "experiment_name": self.experiment_name, "seed": self.seed,
            "actor_hidden_dim": self.actor_hidden_dim,
            "noise_config": baseline_noise_config(self.env),
        }, f"{cdir}/ep{episode}.pt")

    def _reinforce_update(self, episodes: list[dict]) -> dict[str, float]:
        all_obs, all_actions, all_returns, all_masks = [], [], [], []

        for ep in episodes:
            T = len(ep["actions"])
            rewards = ep["rewards"]
            returns = np.zeros(T, dtype=np.float32)
            g = 0.0
            for t in reversed(range(T)):
                g = rewards[t] + self.reinforce_gamma * g
                returns[t] = g

            all_obs.append(
                torch.tensor(ep["obs"][:T], dtype=torch.float32, device=self.device)
            )
            all_actions.append(
                torch.tensor(ep["actions"], dtype=torch.long, device=self.device)
            )
            all_returns.append(
                torch.tensor(returns, dtype=torch.float32, device=self.device)
            )
            all_masks.append(ep["ill_masks"].to(self.device))

        obs_cat     = torch.cat(all_obs,     dim=0)
        actions_cat = torch.cat(all_actions, dim=0)
        returns_cat = torch.cat(all_returns, dim=0)
        masks_cat   = torch.cat(all_masks,   dim=0)

        returns_norm = (
            (returns_cat - returns_cat.mean())
            / (returns_cat.std(unbiased=False) + 1e-8)
        )

        loss = self.actor.reinforce_loss(obs_cat, actions_cat, returns_norm, masks_cat)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer.step()

        return {"actor_loss": float(loss.detach())}


    def run(self) -> None:
        t_start = time.time()

        n_iters = self.max_episodes // self.real_episodes_per_iter

        print("#" * 62)
        print("WM-QAS v2 Baseline (no world model / imagination)")
        print(f"  Obs dim: {self.obs_dim}, Action size: {self.action_size}")
        n_params = sum(p.numel() for p in self.actor.parameters())
        print(f"  Actor: MLP({self.obs_dim}→{self.action_size}), params={n_params:,}")
        print(f"  {n_iters} iters × {self.real_episodes_per_iter} eps/iter"
              f"  = {self.max_episodes} total episodes")
        print("#" * 62)

        metrics_path = pathlib.Path(self.output_path) / "metrics.jsonl"
        metrics_file = metrics_path.open("w")
        traj_path = pathlib.Path(self.output_path) / "trajectory.jsonl"
        traj_file = traj_path.open("w")

        for iteration in range(1, n_iters + 1):
            t0 = time.time()

            episodes = []
            for ep_in_iter in range(self.real_episodes_per_iter):
                traj = self._collect_episode()
                episodes.append(traj)
                self.total_eps += 1
                self.total_vqe_calls += traj["n_steps"]
                for st in range(traj["n_steps"]):
                    traj_file.write(json.dumps({
                        "iter": iteration, "ep": ep_in_iter, "step": st,
                        "action": int(traj["actions"][st]),
                        "energy": float(traj["step_energies"][st]),
                        "true_error": float(traj["step_errs"][st]) * 1000.0,
                        "reward": float(traj["rewards"][st]),
                        "done": int(traj["step_dones"][st]),
                    }) + "\n")
                if traj["min_error"] < self.best_error:
                    self.best_error = traj["min_error"]
                    self.best_seq = list(traj["best_prefix"])
                    self.best_depth = len(traj["best_prefix"])

            mean_ep_return = float(np.mean([e["ep_return"]   for e in episodes]))
            mean_ep_error  = float(np.mean([e["final_error"] for e in episodes]))
            mean_ep_steps  = float(np.mean([e["n_steps"]     for e in episodes]))

            metrics = self._reinforce_update(episodes)
            iter_time = time.time() - t0

            if iteration % 10 == 0 or iteration <= 20:
                print(
                    f"[iter {iteration:>5d}/{n_iters}]"
                    f"  ep_ret={mean_ep_return:+.4f}"
                    f"  ep_steps={mean_ep_steps:.1f}"
                    f"  error={mean_ep_error:.6f}"
                    f"  actor_loss={metrics['actor_loss']:.4f}"
                    f"  best_err={self.best_error:.6f}"
                    f"  real_eps={self.total_eps}"
                    f"  vqe={self.total_vqe_calls}"
                    f"  t={iter_time:.1f}s"
                )

            row: dict = {
                "iter":           iteration,
                "iter_time":      iter_time,
                "mean_ep_return": mean_ep_return,
                "mean_ep_error":  mean_ep_error,
                "mean_ep_steps":  mean_ep_steps,
                "best_err":       self.best_error,
                "real_eps":       self.total_eps,
                "vqe_calls":      self.total_vqe_calls,
                **metrics,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            traj_file.flush()

            while self.ckpt_ptr < len(self.ckpt_sched) and self.total_eps >= self.ckpt_sched[self.ckpt_ptr]:
                self._save_ckpt(self.ckpt_sched[self.ckpt_ptr])
                self.ckpt_ptr += 1

        metrics_file.close()
        traj_file.close()
        self._save_ckpt(self.total_eps)

        total_time = time.time() - t_start
        print(f"\nBaseline training complete: {self.max_episodes} episodes ({n_iters} iters)")
        print(f"Best energy error: {self.best_error:.6f} Ha  ({self.best_error*1000:.4f} mHa)")
        print(f"Total time: {total_time:.1f}s  |  VQE calls: {self.total_vqe_calls}  |  ckpts: {self.ckpt_ptr+1}")


def baseline_noise_config(env):
    """Return the attached evaluator's noise specification."""
    ev = getattr(env, "_noisy_eval", None)
    if ev is None:
        return {"mode": "off"}
    spec = ev.spec
    return {"mode": "ptm", "p1q": spec.p1q, "p2q": spec.p2q,
            "p_ro": spec.p_ro, "basis_change": spec.model_basis_change}


def load_baseline_checkpoint(ck, device=None, legacy_noise_config=None):
    from utils import get_config
    from environment import CircuitEnv
    # Legacy checkpoints require explicit noise settings.
    noise = ck.get("noise_config")
    if noise is None:
        if legacy_noise_config is None:
            raise ValueError('Legacy baseline checkpoint has no noise metadata. Supply '
                             '--legacy_noise_config JSON after checking its training conditions; '
                             'use {"mode":"off"} only for a known noiseless run.')
        noise = legacy_noise_config
    if not isinstance(noise, dict) or noise.get("mode") not in ("off", "ptm"):
        raise ValueError("noise_config must specify mode off or ptm")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    conf = get_config(ck.get("experiment_name", "analysis/"), ck["env_cfg"] + ".cfg")
    env = CircuitEnv(conf, device=torch.device("cpu"))
    if noise["mode"] == "ptm":
        import sys
        code_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from noise_ptm.integration import build_evaluator
        required = {"p1q", "p2q", "p_ro", "basis_change"}
        if not required.issubset(noise):
            raise ValueError(f"PTM noise_config requires {sorted(required)}")
        env.attach_noisy_evaluator(build_evaluator(env, **noise))
    r = BaselineRunner(env=env, conf=conf, device=dev, output_path="",
                       seed=ck.get("seed", 0), actor_hidden_dim=ck.get("actor_hidden_dim", 512),
                       molecule=ck.get("molecule", ""), env_cfg=ck.get("env_cfg", ""),
                       experiment_name=ck.get("experiment_name", "analysis/"))
    r.actor.load_state_dict(ck["actor"])
    r.actor.eval()
    return r


def evaluate_baseline_checkpoint(ckpt_path, n_episodes, device=None, chem_acc_mHa=1.6,
                                 eval_seed=None, legacy_noise_config=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    r = load_baseline_checkpoint(ck, device, legacy_noise_config)
    if eval_seed is None:
        eval_seed = 100003 + 1009 * int(ck.get("seed", 0)) + int(ck["episode"])
    torch.manual_seed(eval_seed); np.random.seed(eval_seed % (2 ** 31 - 1))
    ebests = [r._eval_rollout() for _ in range(n_episodes)]
    stats = eval_harness.episode_best_stats(ebests, chem_acc_mHa)
    stats.update({"episode": int(ck["episode"]), "cum_train_vqe_calls": int(ck["vqe_calls"]),
                  "cum_train_vqe_nfev": int(ck.get("vqe_nfev", 0)), "n_eval_episodes": n_episodes,
                  "eval_seed": eval_seed, "episode_bests": [float(x) for x in ebests]})
    return stats
