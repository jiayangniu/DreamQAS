"""Phase 2 runner v1 — energy-surrogate WM + surrogate-imagination.

Reuses v2 env (CircuitEnv/VQE) + WMDiscreteActor + Shadow. Core loop:
  collect real ep (VQE) -> buffer -> actor: real REINFORCE (+ imagination if gated)
  every 20 iter: fidelity self-check -> imagination gate; WM refresh (tail-weighted).
VQE-calibration + RSSM-encoder variant are added after this core is validated.

Flags: reward∈{curriculum,energy}, imagination∈{off,surrogate}, encoder=gru (rssm TODO).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from utils import get_config
from environment import CircuitEnv
from agent.wm_policy import WMDiscreteActor
from circuit_rules import _dict_actions

from config import (Config, N_ITERS, WM_HIDDEN, ROTOSOLVE_MOLS,
                    checkpoint_episodes, config_hash, mechanism_diff)
from surrogate_wm import build_wm
from buffer import ReplayBuffer
from escale import EnergyScale
from imagine import imagine_and_loss, sample_imagined_circuits
import eval_harness


class PotentialActor(nn.Module):
    """M2: wraps the base actor with a value/potential head. Presents the SAME
    (feat[B,H], mask) interface — internally augments feat with pot(feat) before the base
    actor — so collect_episode / real_reinforce / imagine.py need NO changes. The pot head
    is regressed to the episode return (a value signal as an extra actor feature)."""
    def __init__(self, base_actor, feature_dim, hidden=128):
        super().__init__()
        self.actor = base_actor                        # built with feature_dim+1
        self.pot = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.action_size = base_actor.action_size

    def pot_value(self, feat):
        return self.pot(feat).squeeze(-1)

    def forward(self, feat, mask):
        return self.actor(torch.cat([feat, self.pot(feat)], -1), mask)


class Runner:
    def __init__(self, cfg: Config, _eval_mode: bool = False):
        self.cfg = cfg
        self._eval_mode = _eval_mode
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # WM_QAS: env uses relative paths
        self.dev = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
        conf = get_config(cfg.env_experiment, cfg.env_cfg + ".cfg")
        # 8q/10q run the GPU rotosolve path -> env (BatchedVQE.H) on CUDA; 4q/6q stay CPU + COBYLA.
        self._needs_reference = cfg.molecule in ROTOSOLVE_MOLS
        env_dev = torch.device(self.dev) if self._needs_reference else torch.device("cpu")
        self.env = CircuitEnv(conf, device=env_dev)
        self.N = self.env.num_qubits
        self.A = self.env.action_size
        self.trans = _dict_actions(self.N)

        # ---- WM-Greedy guard: greedy candidate-selection REPLACES imagined policy-gradient ----
        if getattr(cfg, "select_mode", "actor") == "wm_greedy" and cfg.imagination == "surrogate":
            raise SystemExit("[wm_greedy] select_mode=wm_greedy requires imagination in {none,off} "
                             "(greedy candidate-selection replaces imagined policy-gradient; running "
                             "both is contradictory). Pass --imagination none.")

        # ---- oracle-free mode guards (no behavior change when oracle_free is off) ----
        if not getattr(cfg, "enable_ground_truth_diagnostics", True):
            if not getattr(cfg, "oracle_free", False):
                raise SystemExit("[e0] enable_ground_truth_diagnostics=False requires oracle_free=True "
                                 "(on the legacy path the true error IS the training signal)")
            if self.env.fake_min_energy is None:
                raise SystemExit("[e0] diagnostics-off needs fake_min_energy in the env cfg (else the "
                                 "done/curriculum reference falls back to the true E0, environment.py:130)")
            if self.env.fn_type in ("staircase", "incremental_with_ladder", "logprog_ladder"):
                raise SystemExit(f"[e0] fn_type={self.env.fn_type} consumes the true E0 in reward_fn; "
                                 "incompatible with diagnostics-off")
            self.env.min_energy = None      # sever E0; reset() never reassigns it (env only re-derives min_eig)

        # ---- noisy environment (default OFF; no-op unless cfg.noise_mode == "ptm") ----
        # Single construction gate, mirroring self.scale for oracle_free below. When it
        # returns None nothing is attached and every energy path stays byte-identical.
        if str(getattr(cfg, "noise_mode", "off")).lower() != "off":
            _code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _code_dir not in sys.path:
                sys.path.insert(0, _code_dir)     # .../DreamQAS/code -> `import noise_ptm`
            from noise_ptm.integration import attach as _attach_noise
            _attach_noise(self.env, cfg, verbose=not _eval_mode)

        self.wm = build_wm(self.A, cfg).to(self.dev)
        pot = getattr(cfg, "pot_head", False)
        base = WMDiscreteActor(feature_dim=cfg.wm_hidden + (1 if pot else 0), action_size=self.A,
                               hidden_dim=cfg.actor_hidden, num_layers=cfg.actor_layers)
        self.actor = (PotentialActor(base, cfg.wm_hidden) if pot else base).to(self.dev)
        self.scale = EnergyScale(cfg) if getattr(cfg, "oracle_free", False) else None
        self.buf = ReplayBuffer(cfg, self.scale)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=cfg.wm_lr, weight_decay=cfg.wm_weight_decay)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.outdir = f"{cfg.out_dir}/{cfg.run_name}"
        # ---- A0: provenance — dump the FULL resolved config + hash + canonical-diff per run ----
        self._chash = config_hash(cfg)
        self._mech_diff = mechanism_diff(cfg)
        if getattr(cfg, "assert_full", False) and self._mech_diff:
            raise SystemExit(f"[assert_full] resolved config differs from canonical Full: {self._mech_diff}")
        if not _eval_mode:
            os.makedirs(self.outdir, exist_ok=True)
            self._logs = {n: open(f"{self.outdir}/{n}.jsonl", "w")
                          for n in ("metrics", "imag", "fidelity", "trajectory", "calibration")}
            json.dump({"config": cfg.__dict__, "config_hash": self._chash,
                       "mechanism_diff": self._mech_diff, "n_qubits": self.N, "action_size": self.A},
                      open(f"{self.outdir}/config.json", "w"), indent=2, default=str)
        else:
            self._logs = {}

        # ---- phase wall-clock accounting (cfg.timing; OFF by default = zero instrumentation) ----
        # Answers "what fraction of end-to-end training time is VQE vs WM training vs imagination?",
        # which VQE-call counts alone cannot. ON adds a cuda.synchronize() at each phase boundary so a
        # GPU phase is not mis-attributed to whatever runs next (async kernel launches) — that sync is
        # the ONLY behavioural difference, and it is why this is opt-in rather than always-on.
        self._timing = bool(getattr(cfg, "timing", False))
        self._t_cuda = self._timing and str(self.dev).startswith("cuda")   # self.dev is a str, not torch.device
        self._tacc = {k: 0.0 for k in ("vqe_real", "vqe_calib", "vqe_eval", "rollout_other",
                                       "actor", "imagine", "wm_train", "other")}
        self._t_wall0 = None

        self.vqe_calls = 0
        self.vqe_nfev = 0               # supplementary inner-feval count (per-optimizer, not the unit)
        self.eval_vqe_calls = 0         # frozen-eval VQE — NOT counted anywhere in training budget
        self.calib_vqe_calls = 0        # diagnostic VQE — NOT counted in the acceleration budget
        self.best_err = float("inf")
        # Noiseless companion of best_err. Equals best_err when noise is off; under
        # noise it is the SAME delivered prefix re-scored without noise, which is the
        # number the paper reports. Monitoring only — the headline metric is still the
        # frozen best-of-1 from eval_traces.jsonl, never best_err.
        self.best_err_nl = float("inf")
        self.best_E = float("inf")      # raw-energy twin of best_err (well-defined without E0)
        self.best_seq = None            # best circuit's gate sequence + its depth (A2)
        self.best_depth = 0
        self.n_eps = 0
        self._cur_iter = -1
        self.imag_on = False
        self.greedy_on = False          # WM-Greedy activation (gated like imag_on; eval forces greedy)
        self.fidelity = 0.0
        self._calib_pts = []            # accumulated (abs_pred_err, disagree) for running corr
        # standardized checkpoint schedule (¼-dense 15 + every 1500 after); overridable via cfg
        if getattr(cfg, "ckpt_episodes", ""):
            self.ckpt_sched = sorted({int(x) for x in str(cfg.ckpt_episodes).split(",") if x.strip()})
        elif cfg.molecule in N_ITERS:
            self.ckpt_sched = checkpoint_episodes(cfg.molecule)
        else:
            self.ckpt_sched = []
        self.ckpt_ptr = 0

    def _log(self, name, row):
        self._logs[name].write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                            for k, v in row.items()}) + "\n")
        self._logs[name].flush()

    # ---- phase timing (no-op unless cfg.timing) -------------------------------------------------
    def _tick(self):
        """Start/lap the phase clock. Returns None when timing is off (callers then skip _tock)."""
        if not self._timing:
            return None
        if self._t_cuda:
            torch.cuda.synchronize()              # flush async kernels so they are billed to THIS phase
        return time.perf_counter()

    def _tock(self, t, bucket):
        if t is None:
            return
        if self._t_cuda:
            torch.cuda.synchronize()
        self._tacc[bucket] += time.perf_counter() - t

    def _timing_row(self):
        """Cumulative seconds per phase + the unattributed remainder, for metrics.jsonl."""
        if not self._timing or self._t_wall0 is None:
            return {}
        wall = time.perf_counter() - self._t_wall0
        acc = {f"t_{k}": v for k, v in self._tacc.items() if k != "other"}
        acc["t_wall"] = wall
        acc["t_other"] = wall - sum(v for k, v in self._tacc.items() if k != "other")
        return acc

    def _te(self, E):
        """E0-derived true error in mHa — DIAGNOSTIC ONLY. NaN when diagnostics are severed."""
        return abs(self.env.min_energy - E) * 1000.0 if self.env.min_energy is not None else float("nan")

    @staticmethod
    def _fin(x):
        """JSON-safe: None for non-finite diagnostics (never emit bare NaN)."""
        return x if (x is not None and math.isfinite(x)) else None

    # ---------------- energy (reference for frozen eval) ----------------
    def _energy(self, env, eval_mode):
        """Training uses the env optimizer energy (Qulacs complex128 for 4q/6q COBYLA).
        Frozen eval on the rotosolve (8q/10q) path recomputes on a complex128 reference so the
        1.6 mHa success label never flips on GPU complex64 numerical error."""
        if eval_mode and self._needs_reference:
            ref = getattr(env, "reference_energy", None)     # wired for rotosolve in A3
            if callable(ref):
                return ref()
        return env.energy

    # ---------------- WM-Greedy one-step candidate selection ----------------
    def _wm_greedy_action(self, ill, hstate, eps):
        """WM-Greedy: enumerate ALL legal one-step next-actions, roll each ONE step forward under the
        FROZEN WM (no VQE), score each with the SAME pessimistic value used in imagination
        phi = -(mean_pred + pessimism_beta*ensemble_std), return argmax phi (= argmin conservative
        predicted score; lower predicted log-error/frontier-score = better circuit in both modes) as a
        [1] long tensor (shape-matches dist.sample() so downstream :182/:210 are unchanged). ε-greedy:
        with prob eps return a uniform-random legal action. `hstate` (current real prefix state, batch=1)
        is NOT mutated -- wm.step reads h_0 and returns a fresh state; hexp is a copy."""
        illset = set(ill) if ill else set()
        legal = [a for a in range(self.A) if a not in illset]
        if not legal:                                    # degenerate (mirror actor path over full space)
            legal = list(range(self.A))
        if eps > 0.0 and np.random.rand() < eps:         # ε-greedy on the already-seeded np.random stream
            return torch.tensor([legal[np.random.randint(len(legal))]], dtype=torch.long, device=self.dev)
        acts = torch.tensor(legal, dtype=torch.long, device=self.dev)     # [B]
        exp = [-1] * hstate.dim(); exp[-2] = len(legal)  # batch axis is -2 (shared [L,B,H] / ens [K,L,B,H])
        hexp = hstate.expand(*exp).contiguous()          # fresh copy; the real prefix hstate is untouched
        beta = getattr(self.cfg, "pessimism_beta", 0.0)
        with torch.no_grad():
            _, _, ph = self.wm.step(acts, hexp)          # ph [K,B] per-member (popart-normalized)
            std = ph.std(0) if ph.shape[0] > 1 else torch.zeros_like(ph[0])
            phi = -(ph.mean(0) + beta * std)             # [B]; argmax = pessimistic-best candidate
            j = int(torch.argmax(phi))
        return torch.tensor([legal[j]], dtype=torch.long, device=self.dev)

    # ---------------- rollout (real training + frozen eval share this) ----------------
    def _rollout(self, it=-1, ep=-1, *, count_vqe=True, update_buffer=True, update_best=True,
                 run_to_max_depth=False, eval_mode=False):
        env = self.env
        env.reset()
        hstate = self.wm.init_state(1, self.dev)
        feat = torch.zeros(1, self.cfg.wm_hidden, device=self.dev)   # empty-circuit feature
        acts, errs, energies, feats, masks, env_rewards, disags = [], [], [], [], [], [], []
        errs_nl = []      # noiseless-companion error per prefix; == errs when noise is off
        self.wm.eval()
        for step in range(env.num_layers + 1):
            ill = env.illegal_action_new()
            if run_to_max_depth and ill and len(ill) >= self.A:   # eval: structural termination only
                break
            mask = torch.zeros(1, self.A, dtype=torch.bool, device=self.dev)
            if ill:
                mask[0, ill] = True
            if self.cfg.select_mode == "wm_greedy" and (eval_mode or self.greedy_on):
                # WM-Greedy baseline: frozen-WM one-step pessimistic candidate selection.
                # eval deploys pure greedy (eps=0); training uses ε-greedy while the WM is frozen.
                a = self._wm_greedy_action(ill, hstate, 0.0 if eval_mode else self.cfg.wm_greedy_eps)
            else:
                with torch.no_grad():
                    dist = self.actor(feat, mask)
                    a = dist.sample()
            _t = self._tick()
            _, r_env, done, _ = env.step(list(self.trans[int(a)]), torch.tensor(0.0), train_flag=True)
            self._tock(_t, "vqe_real" if count_vqe else "vqe_eval")   # the VQE optimisation itself
            if count_vqe:
                self.vqe_calls += 1
                self.vqe_nfev += int(getattr(env, "last_nfev", 0) or 0)
            else:
                self.eval_vqe_calls += 1                       # frozen-eval VQE: off training budget
            E = float(self._energy(env, eval_mode))
            te = self._te(E)                                   # E0-derived: DIAGNOSTIC ONLY when oracle_free
            acts.append(int(a)); errs.append(te); energies.append(E)
            # Noiseless companion of the SAME prefix. Identical to `te` when noise is off
            # (env.energy_noiseless == env.energy then), so the legacy path is unchanged.
            # Under noise this is what the paper reports the delivered circuit against.
            errs_nl.append(self._te(float(getattr(env, "energy_noiseless", E))))
            feats.append(feat.squeeze(0).detach()); masks.append(mask.squeeze(0))
            env_rewards.append(float(r_env))
            if update_best:
                if self.scale is not None:                     # oracle-free: rank on raw energy (same order)
                    if E < self.best_E:
                        self.best_E = E; self.best_err = te
                        self.best_err_nl = errs_nl[-1]
                        self.best_seq = list(acts); self.best_depth = len(acts)
                elif te < self.best_err:                       # legacy line: byte-identical behavior
                    self.best_err = te; self.best_E = E
                    self.best_err_nl = errs_nl[-1]
                    self.best_seq = list(acts); self.best_depth = len(acts)   # persist best circuit + depth
            if self.cfg.log_raw and not eval_mode:
                row = {"iter": it, "ep": ep, "step": step, "action": int(a),
                       "energy": float(env.energy), "true_error": self._fin(te) if self.scale is not None else te,
                       "error": float(env.error), "reward": float(r_env), "done": int(done)}
                if self.scale is not None:                     # oracle-free: training score (F_adopted is
                    row["frontier_score"] = (float(self.scale.score_np([E], count=False)[0])
                                             if self.scale.initialized() else None)   # fixed within an episode
                self._log("trajectory", row)
            with torch.no_grad():
                feat, hstate, ph = self.wm.step(a, hstate)
                disags.append(float(ph.std(0)) if ph.shape[0] > 1 else 0.0)  # P3: disagreement at reached state
            if run_to_max_depth:
                if step >= env.num_layers - 1:                 # eval: stop at FIXED max depth (structural)
                    break                                      # ignore accuracy early-stop, but not depth limit
            elif done:                                         # training: original break-on-done
                break
        scores = None
        if update_buffer:
            if self.scale is not None:
                if not self.scale.initialized():               # first real episode seeds pending+adopted
                    self.scale.init_from(energies)
                for E_ in energies:                            # pending frontier only (adopt at refresh)
                    self.scale.observe(E_)
                scores = self.scale.score_np(energies)         # one consistent (current) scale per episode
            self.buf.add(np.array(acts), np.array(errs, np.float32), np.array(energies, np.float32))
        if not eval_mode:
            self.n_eps += 1
        return {"acts": acts, "errs": np.array(errs, np.float32),
                "errs_noiseless": np.array(errs_nl, np.float32),
                "energies": np.array(energies, np.float32), "scores": scores,
                "feats": torch.stack(feats),
                "masks": torch.stack(masks), "env_rewards": np.array(env_rewards, np.float32),
                "disagree": np.array(disags, np.float32)}

    # ---------------- real episode (with VQE) ----------------
    def collect_episode(self, it, ep):
        return self._rollout(it, ep, count_vqe=True, update_buffer=True, update_best=True,
                             run_to_max_depth=False, eval_mode=False)

    # ---------------- rewards ----------------
    def _returns(self, ep):
        if getattr(self.cfg, "reward_kind", "energy") == "staircase":  # Q1: event-driven ladder
            # oracle-free: rungs run on the positive frontier-scale s_mHa (non-canonical path;
            # rung semantics shift by the margin — not used in main experiments)
            errs = (np.power(10.0, ep["scores"]) if self.scale is not None else ep["errs"])
            ladder = sorted(self.cfg.stair_ladder_mHa, reverse=True)   # high→low rungs
            r = np.zeros(len(errs), np.float32); best = np.inf; crossed = set()
            for t in range(len(errs)):
                best = min(best, float(errs[t]))
                for rung in ladder:                                    # +1 per newly-crossed rung
                    if best <= rung and rung not in crossed:
                        crossed.add(rung); r[t] += 1.0
        elif self.cfg.reward == "curriculum":
            r = ep["env_rewards"]
        elif self.scale is not None:
            # oracle-free energy reward: Δ(-S) per step — same telescoping shape, S = frontier score
            le = ep["scores"]
            r = np.empty_like(le); p = le[0]           # r[0]=0; r[t]=S[t-1]-S[t]
            for t in range(len(le)):
                r[t] = p - le[t]; p = le[t]
        else:  # energy: Δ(-log10 err) per step = log-error reduction (potential shaping, Φ=-logerr)
            le = np.log10(np.clip(ep["errs"], self.cfg.err_floor_mHa, None))
            r = np.empty_like(le); p = le[0]           # r[0]=0; r[t]=le[t-1]-le[t]
            for t in range(len(le)):
                r[t] = p - le[t]; p = le[t]
        # P3: curiosity — intrinsic bonus for reaching high-disagreement (uncertain) states. On the
        # 6q energy-plateau the extrinsic reward is flat (86% of circuits share one energy level), so
        # this disagreement bonus supplies gradient in that flat region to hunt rare breakthroughs.
        cb = getattr(self.cfg, "curiosity_beta", 0.0)
        if cb > 0 and "disagree" in ep and len(ep["disagree"]):
            r = r + cb * ep["disagree"][:len(r)]
        # discounted returns
        G = np.zeros_like(r); acc = 0.0
        for t in reversed(range(len(r))):
            acc = r[t] + self.cfg.reinforce_gamma * acc; G[t] = acc
        return G

    def real_reinforce(self, eps):
        feats = torch.cat([e["feats"] for e in eps])
        masks = torch.cat([e["masks"] for e in eps])
        acts = torch.cat([torch.tensor(e["acts"], device=self.dev) for e in eps])
        rets_raw = torch.tensor(np.concatenate([self._returns(e) for e in eps]),
                                dtype=torch.float32, device=self.dev)
        if getattr(self.cfg, "maxk_baseline", False):   # M1: max-seeking high-quantile baseline
            adv = (rets_raw - torch.quantile(rets_raw, self.cfg.maxk_q)) / (rets_raw.std() + 1e-6)
        else:
            adv = (rets_raw - rets_raw.mean()) / (rets_raw.std() + 1e-6)
        feats = feats.to(self.dev)
        dist = self.actor(feats, masks.to(self.dev))
        logp = dist.log_prob(acts)
        loss = -(logp * adv).mean() - self.cfg.entropy_coef * dist.entropy().mean()
        if getattr(self.cfg, "pot_head", False):        # M2: value head regressed to the return
            loss = loss + F.mse_loss(self.actor.pot_value(feats), rets_raw)
        return loss

    def _grad_share(self, real_loss, imag_loss):
        """P1 headline metric: fraction of the actor gradient that comes from imagination,
        ‖∇imag‖ / (‖∇real‖+‖∇imag‖). Both losses share the actor params but live on separate
        graphs; retain_graph keeps them alive for the subsequent combined .backward()."""
        params = [p for p in self.actor.parameters() if p.requires_grad]
        gr = torch.autograd.grad(real_loss, params, retain_graph=True, allow_unused=True)
        gi = torch.autograd.grad(imag_loss, params, retain_graph=True, allow_unused=True)
        rn = torch.sqrt(sum((g ** 2).sum() for g in gr if g is not None))
        inorm = torch.sqrt(sum((g ** 2).sum() for g in gi if g is not None))
        return float(inorm / (rn + inorm).clamp_min(1e-12))

    # ---------------- WM supervised refresh ----------------
    def _dir_bin_weights(self):
        """DIR / label-distribution smoothing (Yang'21): weight ∝ 1/sqrt(gaussian-smoothed
        log-error histogram) so the RARE low-error breakthroughs aren't drowned by the abundant
        energy-plateau mass. Recomputed once per refresh from the whole buffer."""
        from scipy.ndimage import gaussian_filter1d
        if self.scale is not None:      # oracle-free: histogram over the frontier score S
            le = np.concatenate([self.scale.score_np(e["E"], count=False) for e in self.buf.eps])
        else:
            le = np.concatenate([np.log10(np.clip(e["err"], self.cfg.err_floor_mHa, None))
                                 for e in self.buf.eps])
        hist, edges = np.histogram(le, bins=self.cfg.dir_bins)
        dens = gaussian_filter1d(hist.astype(np.float64), self.cfg.dir_smooth) + 1.0
        binw = np.clip((1.0 / np.sqrt(dens)) / (1.0 / np.sqrt(dens)).mean(), None, self.cfg.dir_w_max)
        return (torch.tensor(edges[1:-1], device=self.dev, dtype=torch.float32),
                torch.tensor(binw, device=self.dev, dtype=torch.float32))

    def wm_refresh(self, n_steps):
        self.wm.train()
        if self.scale is not None:
            # refresh boundary: adopt the pending frontier atomically with retraining; freeze the
            # margin at the first boundary after warm-up. Labels below are recomputed from raw E
            # with the just-adopted scale -> never stale.
            self.scale.adopt()
            if not self.scale.frozen and self.n_eps >= self.cfg.warmup_eps:
                self.scale.freeze_margin_from_buffer(self.buf)
        dir_on = getattr(self.cfg, "dir_reweight", False)
        if dir_on:
            dir_edges, dir_binw = self._dir_bin_weights()   # imbalanced-tail weights (once/refresh)
        losses = []
        for _ in range(n_steps):
            (acts, err, E, mask) = self.buf.sample_batch()[0]
            acts, err, E, mask = acts.to(self.dev), err.to(self.dev), E.to(self.dev), mask.to(self.dev)
            if self.scale is not None:                       # oracle-free label: frontier score S(E)
                y = self.scale.score_t(E)
            else:
                y = torch.log10(err.clamp_min(self.cfg.err_floor_mHa))
            if dir_on:                                       # DIR: inverse label-density weight
                w = dir_binw[torch.bucketize(y, dir_edges).clamp(0, dir_binw.numel() - 1)]
            elif self.scale is not None:                     # legacy tail weight on the positive s-scale
                s_mHa = torch.pow(10.0, y)
                w = (self.cfg.tail_w_A / (s_mHa + self.cfg.tail_w_eps)).clamp(1.0, self.cfg.tail_w_max)
            else:                                            # legacy hand-tuned tail weight
                w = (self.cfg.tail_w_A / (err + self.cfg.tail_w_eps)).clamp(1.0, self.cfg.tail_w_max)
            self.wm.update_popart(y[mask.bool()])           # B: track current target distribution
            y_t = self.wm.normalize(y)                       # normalized target (identity w/o popart)
            preds = self.wm.seq_preds(acts)                  # [K,B,T] per-member (normalized)
            # Bug1: train EACH head on its OWN bootstrap of the batch -> ensemble diversity
            loss = 0.0
            for k in range(self.wm.K):
                bw = (torch.rand_like(mask) < 0.8).float() * mask   # per-head 80% bootstrap
                e = preds[k] - y_t
                hub = torch.where(e.abs() <= 1.0, 0.5 * e ** 2, e.abs() - 0.5) * w * bw
                loss = loss + hub.sum() / bw.sum().clamp_min(1)
            loss = loss / self.wm.K + self.cfg.kl_scale * self.wm.kl_loss()  # KL=0 for gru
            self.wm_opt.zero_grad(); loss.backward(); self.wm_opt.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def fidelity_check(self):
        v = self.buf.recent_val(min(200, len(self.buf)))
        if v is None:
            return 0.0
        acts, err, E, mask = [t.to(self.dev) for t in v]
        self.wm.eval()
        with torch.no_grad():
            pred, _, _ = self.wm.predict(acts)
        m = mask.bool()
        # rank reference: legacy = true error; oracle-free = raw energy (bijectively the same order)
        ref = err if self.scale is None else E
        sc = pred[m].cpu().numpy(); tr = ref[m].cpu().numpy()
        # pairwise accuracy (sampled)
        rng = np.random.default_rng(0); i, j = rng.integers(0, len(sc), 50000), rng.integers(0, len(sc), 50000)
        ok = tr[i] != tr[j]
        return float((np.sign(sc[i] - sc[j]) == np.sign(tr[i] - tr[j]))[ok].mean())

    # ---------------- VQE calibration (+ DAgger feedback) ----------------
    def _vqe_circuit(self, seq, count_budget=False):
        """Replay an imagined action-sequence through the real env (VQE) -> per-step RAW optimized
        energies (Ha) + the actions actually applied. (Signature change: was per-step true errors;
        callers derive te via self._te when they need the E0 diagnostic.) Guards illegal actions
        (Shadow==env, shouldn't fire). count_budget=True counts these VQE calls into the
        acceleration budget (DAgger: the circuit trains the WM); else into calib_vqe_calls."""
        env = self.env
        env.reset()
        energies, applied = [], []
        for a in seq:
            ill = env.illegal_action_new()
            if ill and a in ill:
                break
            _t = self._tick()
            _, _, done, _ = env.step(list(self.trans[int(a)]), torch.tensor(0.0), train_flag=True)
            # DAgger verification (count_budget=True) is billed to vqe_real, the diagnostic-only
            # calibration probe (count_budget=False) to vqe_calib — mirrors the VQE-call accounting.
            self._tock(_t, "vqe_real" if count_budget else "vqe_calib")
            E = float(env.energy)
            te = self._te(E)
            if count_budget:
                self.vqe_calls += 1
                self.vqe_nfev += int(getattr(env, "last_nfev", 0) or 0)
                if self.scale is not None:
                    self.scale.observe(E)   # DAgger VQE is a REAL observation -> pending frontier
                    if E < self.best_E:     # rank on raw energy (same order as te)
                        self.best_E = E; self.best_err = te
                        self.best_seq = applied + [int(a)]; self.best_depth = len(applied) + 1
                elif te < self.best_err:    # legacy line: DAgger VQE is a REAL discovery -> credit best_err
                    self.best_err = te      # (symmetric: it charges the budget AND can improve it)
                    self.best_E = E
                    self.best_seq = applied + [int(a)]; self.best_depth = len(applied) + 1
            else:
                self.calib_vqe_calls += 1
            energies.append(E)
            applied.append(int(a))
            if done:
                break
        return energies, applied

    def calibrate(self, it):
        """VQE a few imagined circuits, compare WM prediction vs real true_error, and
        (A: DAgger) feed the VQE'd circuits back into the buffer so the WM is corrected on
        its own imagination distribution — the fix for the systematic over-optimism.
        Still logs calib_MAE + disagree_vs_err corr. DAgger VQE counts into the budget.
        """
        dagger = getattr(self.cfg, "dagger", False)
        seeds = self.buf.imag_seeds(self.cfg.imag_n_seeds, self.cfg.imag_seed_window,
                                    getattr(self.cfg, "imag_seed_strategy", "frontier"))
        cands = sample_imagined_circuits(self.wm, self.actor, seeds, self.N, self.cfg, self.dev,
                                         max_depth=self.env.num_layers)
        # dedup by sequence; prefer circuits that were actually extended past the seed
        seen, uniq = set(), []
        for c in cands:
            key = tuple(c["seq"])
            if key in seen or c["n_imagined"] == 0:
                continue
            seen.add(key); uniq.append(c)
        if not uniq:
            return
        # select which imagined circuits to VQE-verify (P4: dagger_select). Budget held constant at
        # calib_n_top+calib_n_disagree across modes so the VQE cost is identical -> fair comparison.
        mode = getattr(self.cfg, "dagger_select", "mix")
        nt, nd = self.cfg.calib_n_top, self.cfg.calib_n_disagree
        sel = {}
        if mode == "top":               # exploit: verify the predicted-best
            for c in sorted(uniq, key=lambda c: c["pred_logerr"])[: nt + nd]:
                sel[tuple(c["seq"])] = (c, "top")
        elif mode == "disagree":        # explore: verify the most uncertain (active-learning ideal)
            for c in sorted(uniq, key=lambda c: -c["disagree"])[: nt + nd]:
                sel[tuple(c["seq"])] = (c, "disagree")
        elif mode == "random":          # control: verify random imagined circuits
            k = min(nt + nd, len(uniq))
            for i in np.random.choice(len(uniq), k, replace=False):
                sel[tuple(uniq[int(i)]["seq"])] = (uniq[int(i)], "random")
        else:                           # mix (default): top-predicted + highest-disagreement
            for c in sorted(uniq, key=lambda c: c["pred_logerr"])[: nt]:
                sel[tuple(c["seq"])] = (c, "top")
            for c in sorted(uniq, key=lambda c: -c["disagree"])[: nd]:
                k = tuple(c["seq"]); sel[k] = (c, "both" if k in sel else "disagree")

        errs = []
        for c, tag in sel.values():
            Es, applied = self._vqe_circuit(c["seq"], count_budget=dagger)
            if not Es:
                continue
            tes = [self._te(E_) for E_ in Es]            # E0 diagnostics (NaN when severed)
            real_te = tes[-1]
            if self.scale is not None:
                # oracle-free: calibration error lives in the WM's own target space (frontier score);
                # E0 fields are retained as diagnostics only. abs_logerr_err is NOT emitted (its old
                # meaning |pred - log10(te)| would be a unit mismatch against a score-trained WM).
                real_score = float(self.scale.score_np([Es[-1]], count=False)[0])
                abs_err = abs(c["pred_logerr"] - real_score)
                real_logerr = (float(np.log10(max(real_te, self.cfg.err_floor_mHa)))
                               if math.isfinite(real_te) else None)
                row = {"iter": it, "selection": tag, "len": len(c["seq"]),
                       "n_imagined": c["n_imagined"], "pred_logerr": c["pred_logerr"],
                       "pred_score": c["pred_logerr"], "real_score": real_score,
                       "score_abs_err": abs_err,
                       "real_logerr": real_logerr, "real_err_mHa": self._fin(real_te),
                       "disagree": c["disagree"]}
            else:
                real_logerr = float(np.log10(max(real_te, self.cfg.err_floor_mHa)))
                abs_err = abs(c["pred_logerr"] - real_logerr)
                row = {"iter": it, "selection": tag, "len": len(c["seq"]),
                       "n_imagined": c["n_imagined"], "pred_logerr": c["pred_logerr"],
                       "real_logerr": real_logerr, "real_err_mHa": real_te,
                       "abs_logerr_err": abs_err, "disagree": c["disagree"]}
            errs.append(abs_err)
            self._calib_pts.append((abs_err, c["disagree"]))
            if dagger and applied:                       # A: DAgger — correct the WM on its own imagination
                self.buf.add(np.array(applied, np.int64), np.array(tes, np.float32),
                             np.array(Es, np.float32))
            self._log("calibration", row)
        # running summary: MAE on log-error + corr(disagree, |pred error|)
        corr = None
        if len(self._calib_pts) >= 5:
            ae = np.array([p[0] for p in self._calib_pts])
            dg = np.array([p[1] for p in self._calib_pts])
            if dg.std() > 1e-9 and ae.std() > 1e-9:
                corr = float(spearmanr(dg, ae).correlation)
        if self.scale is not None:      # oracle-free: MAE lives in frontier-score space (new names)
            self._log("calibration", {"iter": it, "selection": "SUMMARY", "n_vqe": len(errs),
                                      "calib_score_mae_this": float(np.mean(errs)) if errs else None,
                                      "calib_score_mae": float(np.mean([p[0] for p in self._calib_pts])),
                                      "disagree_vs_err_corr": corr, "calib_vqe_calls": self.calib_vqe_calls})
        else:                           # legacy row: key order preserved byte-for-byte
            self._log("calibration", {"iter": it, "selection": "SUMMARY", "n_vqe": len(errs),
                                      "calib_MAE_this": float(np.mean(errs)) if errs else None,
                                      "calib_MAE_all": float(np.mean([p[0] for p in self._calib_pts])),
                                      "disagree_vs_err_corr": corr, "calib_vqe_calls": self.calib_vqe_calls})
        return corr

    # ---------------- main loop ----------------
    def _save_ckpt(self, episode):
        cdir = f"{self.outdir}/ckpt"
        os.makedirs(cdir, exist_ok=True)
        eval_harness.save_checkpoint(self, episode, f"{cdir}/ep{episode}.pt")

    def run(self):
        cfg = self.cfg
        t0 = time.time()
        self._t_wall0 = time.perf_counter()
        for it in range(cfg.n_iterations):
            self._cur_iter = it
            _vqe_before = self._tacc["vqe_real"]
            _t = self._tick()
            eps = [self.collect_episode(it, e) for e in range(cfg.real_eps_per_iter)]
            if _t is not None:   # rollout minus the VQE already billed inside -> policy/WM forward cost
                self._tock(_t, "rollout_other")
                self._tacc["rollout_other"] -= (self._tacc["vqe_real"] - _vqe_before)
            # actor: real REINFORCE (+ imagination if gated)
            _t = self._tick()
            self.actor_opt.zero_grad()
            loss = self.real_reinforce(eps)
            if getattr(cfg, "real_dose_alpha", 0.0) > 0 and len(eps) > 0:   # NoImag-RealDose compute control
                bidx = np.random.randint(0, len(eps), len(eps))             # on-policy bootstrap of current eps
                loss = loss + cfg.real_dose_alpha * self.real_reinforce([eps[i] for i in bidx])
            self._tock(_t, "actor")
            imag_diag = {}
            _t = self._tick()
            if cfg.imagination == "surrogate" and self.imag_on:
                strat = getattr(cfg, "imag_seed_strategy", "frontier")
                seeds = self.buf.imag_seeds(cfg.imag_n_seeds, cfg.imag_seed_window, strat)
                seed_fn = ((lambda: self.buf.imag_seeds(cfg.imag_n_seeds, cfg.imag_seed_window, strat))
                           if getattr(cfg, "imag_transition_budget", 0) > 0 else None)
                iloss, imag_diag = imagine_and_loss(self.wm, self.actor, seeds, self.N, cfg, self.dev,
                                                    max_depth=self.env.num_layers, seed_fn=seed_fn)
                if iloss is not None:
                    iloss = getattr(cfg, "imag_loss_weight", 1.0) * iloss   # P1: λ_imag
                    if it % cfg.wm_refresh_every == 0:   # P1: imag vs real actor-grad share (headline metric)
                        imag_diag["imag_grad_frac"] = self._grad_share(loss, iloss)
                    loss = loss + iloss
                    self._log("imag", {"iter": it, **imag_diag})
            self._tock(_t, "imagine")
            _t = self._tick()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.actor_opt.step()
            self._tock(_t, "actor")

            wm_loss = None
            # the WM's GRU IS the actor's state encoder (all variants) -> always train it;
            # only the *use* of imagination is gated (surrogate flag + warmup + fidelity).
            if it % cfg.wm_refresh_every == 0 and len(self.buf) >= 8:
                _t = self._tick()
                self.fidelity = self.fidelity_check()
                self.imag_on = (cfg.imagination == "surrogate" and self.n_eps >= cfg.warmup_eps
                                and self.fidelity >= cfg.fidelity_tau)
                self.greedy_on = (cfg.select_mode == "wm_greedy" and self.n_eps >= cfg.warmup_eps
                                  and self.fidelity >= cfg.fidelity_tau)
                wm_loss = self.wm_refresh(cfg.wm_refresh_steps)
                self._tock(_t, "wm_train")
                self._log("fidelity", {"iter": it, "pairwise": self.fidelity, "imag_on": self.imag_on,
                                       "greedy_on": self.greedy_on, "n_eps": self.n_eps, "wm_loss": wm_loss})
            # diagnostic VQE-calibration (surrogate variants only; runs after a WM refresh)
            calib_corr = None
            if (cfg.imagination == "surrogate" and it > 0 and it % cfg.calib_every == 0
                    and len(self.buf) >= 8):
                calib_corr = self.calibrate(it)

            mean_err = float(np.mean([e["errs"][-1] for e in eps]))
            mrow = {"iter": it, **self._timing_row(), "vqe_calls": self.vqe_calls,
                    "calib_vqe_calls": self.calib_vqe_calls, "best_err_mHa": self.best_err,
                    "mean_ep_err_mHa": mean_err, "real_loss": float(loss.detach()),
                    "imag_on": self.imag_on, "greedy_on": self.greedy_on, "fidelity": self.fidelity,
                    "disagree_vs_err_corr": calib_corr,
                    "K": imag_diag.get("n_seeds", 0), "wm_loss": wm_loss}
            if self.scale is not None:                   # oracle-free: JSON-safe E0 diagnostics + scale state
                mrow["best_err_mHa"] = self._fin(self.best_err)
                mrow["mean_ep_err_mHa"] = self._fin(mean_err)
                mrow.update({"best_E": self.best_E, "frontier_E": self.scale.F_adopted,
                             "margin_mHa": self.scale.margin_mHa,
                             "scale_version": self.scale.scale_version,
                             "below_frontier_events": self.scale.below_frontier_events})
            if str(getattr(cfg, "noise_mode", "off")).lower() != "off":
                # Noisy runs only: the same delivered prefix re-scored without noise, plus
                # the mean over this iteration's episodes. Keys are ABSENT on clean runs so
                # every existing metrics.jsonl parser is unaffected.
                mrow["best_err_noiseless_mHa"] = self._fin(self.best_err_nl)
                mrow["mean_ep_err_noiseless_mHa"] = self._fin(
                    float(np.mean([e["errs_noiseless"][-1] for e in eps])))
            self._log("metrics", mrow)
            # standardized episode-checkpoint hook (episodes advance real_eps_per_iter/iter)
            while self.ckpt_ptr < len(self.ckpt_sched) and self.n_eps >= self.ckpt_sched[self.ckpt_ptr]:
                self._save_ckpt(self.ckpt_sched[self.ckpt_ptr])
                self.ckpt_ptr += 1
            if it % 25 == 0:
                print(f"[{cfg.run_name}] it {it}/{cfg.n_iterations} ep={self.n_eps} vqe={self.vqe_calls} "
                      f"best={self.best_err:.4f}mHa imag={self.imag_on} fid={self.fidelity:.2f} "
                      f"t={time.time()-t0:.0f}s", flush=True)
        # final checkpoint (at the true final episode count) + best circuit/depth
        self._save_ckpt(self.n_eps)
        bc = {"best_err_mHa": self.best_err, "best_seq": self.best_seq, "best_depth": self.best_depth,
              "vqe_calls": self.vqe_calls, "vqe_nfev": self.vqe_nfev, "config_hash": self._chash,
              "ckpt_episodes": self.ckpt_sched}
        if self.scale is not None:
            bc["best_err_mHa"] = self._fin(self.best_err)
            bc["best_E"] = self.best_E
            bc["escale"] = self.scale.state_dict()
        json.dump(bc, open(f"{self.outdir}/best_circuit.json", "w"), indent=2, default=str)
        for f in self._logs.values():
            f.close()
        print(f"[{cfg.run_name}] DONE best_err={self.best_err:.5f} mHa, vqe={self.vqe_calls}, "
              f"ckpts={self.ckpt_ptr}, hash={self._chash}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="LiH4q")
    p.add_argument("--encoder", default="gru")
    p.add_argument("--reward", default="energy")
    p.add_argument("--imagination", default="surrogate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_iterations", type=int, default=None)
    p.add_argument("--warmup_eps", type=int, default=None)
    p.add_argument("--wm_refresh_every", type=int, default=None)
    p.add_argument("--fidelity_tau", type=float, default=None)
    p.add_argument("--wm_hidden", type=int, default=None)   # override WM width (6q bigger)
    p.add_argument("--out_dir", default=None)               # improved runs -> new dir (no overwrite)
    # ablation knobs (None = use Config default)
    p.add_argument("--independent_ensemble", type=int, default=None)
    p.add_argument("--popart", type=int, default=None)
    p.add_argument("--dagger", type=int, default=None)
    p.add_argument("--imag_horizon", type=int, default=None)
    p.add_argument("--pessimism_beta", type=float, default=None)
    p.add_argument("--maxk_baseline", type=int, default=None)   # M1
    p.add_argument("--pot_head", type=int, default=None)        # M2
    p.add_argument("--reward_kind", default=None)               # energy | staircase (Q1)
    p.add_argument("--dir_reweight", type=int, default=None)    # DIR imbalanced-tail (default drop)
    p.add_argument("--dir_w_max", type=float, default=None)     # DIR inverse-density weight cap
    # P1: imagination participation / gradient share
    p.add_argument("--imag_adv_normalize", type=int, default=None)  # whiten imag advantage
    p.add_argument("--imag_loss_weight", type=float, default=None)  # lambda_imag multiplier
    p.add_argument("--imag_depth_cap", type=int, default=None)      # cap imagined depth at env num_layers
    p.add_argument("--imag_n_seeds", type=int, default=None)
    p.add_argument("--imag_conf_tau", type=float, default=None)     # rollout confidence-stop threshold
    p.add_argument("--imag_transition_budget", type=int, default=None)  # transition-matched horizon (0=off)
    p.add_argument("--real_dose_alpha", type=float, default=None)       # NoImag-RealDose control (0=off)
    # P2: uncertainty / ensemble hyperparams
    p.add_argument("--rpf_beta", type=float, default=None)
    p.add_argument("--wm_ensemble_K", type=int, default=None)
    # P3: 6q plateau attack
    p.add_argument("--curiosity_beta", type=float, default=None)
    p.add_argument("--imag_seed_strategy", default=None)           # frontier|stratified|high_disagree|elite
    # P4: DAgger active learning
    p.add_argument("--dagger_select", default=None)                # top|disagree|mix|random
    p.add_argument("--calib_n_top", type=int, default=None)
    p.add_argument("--calib_n_disagree", type=int, default=None)
    p.add_argument("--calib_every", type=int, default=None)
    # oracle-free training (empirical-frontier score; escale.py). Default OFF = legacy path.
    p.add_argument("--oracle_free", type=int, default=None)
    p.add_argument("--timing", type=int, default=None)   # phase wall-clock accounting (adds cuda.sync)
    p.add_argument("--ground_truth_diagnostics", type=int, default=None)
    p.add_argument("--score_margin_mHa", type=float, default=None)   # fixed-mode margin hyperparameter
    p.add_argument("--score_margin_mode", default=None)              # fixed | quantile
    # noisy environment (PTM backend, code/noise_ptm/). Default off -> legacy path.
    p.add_argument("--noise_mode", default=None)                     # off | ptm
    p.add_argument("--noise_p1q", type=float, default=None)          # 1q depolarizing / rotation
    p.add_argument("--noise_p2q", type=float, default=None)          # 2q depolarizing / CNOT
    p.add_argument("--noise_p_ro", type=float, default=None)         # symmetric readout error
    p.add_argument("--noise_basis_change", type=int, default=None)   # model basis-change depol
    # WM-Greedy baseline (surrogate candidate-selection; requires --imagination none)
    p.add_argument("--select_mode", default=None)                    # actor | wm_greedy
    p.add_argument("--wm_greedy_eps", type=float, default=None)      # ε-greedy exploration
    p.add_argument("--tag", default="")                     # run_name suffix (separate ablations)
    # standardized eval protocol / checkpointing + provenance
    p.add_argument("--ckpt_episodes", default=None)         # comma-list; None = auto (¼-dense 15 + every 1500)
    p.add_argument("--eval_only", default=None)             # checkpoint .pt -> frozen offline eval, print stats
    p.add_argument("--n_eval_episodes", type=int, default=20)
    p.add_argument("--assert_full", type=int, default=None)  # 1 = hard-error unless config == canonical Full
    a = p.parse_args()
    cfg = Config(molecule=a.molecule, encoder=a.encoder, reward=a.reward,
                 imagination=a.imagination, seed=a.seed)
    cfg.env_cfg = f"G0_v2_{a.molecule}"
    cfg.n_iterations = a.n_iterations or N_ITERS[a.molecule]
    if a.warmup_eps is not None: cfg.warmup_eps = a.warmup_eps
    if a.wm_refresh_every is not None: cfg.wm_refresh_every = a.wm_refresh_every
    if a.fidelity_tau is not None: cfg.fidelity_tau = a.fidelity_tau
    if a.out_dir is not None: cfg.out_dir = a.out_dir
    if a.independent_ensemble is not None: cfg.independent_ensemble = bool(a.independent_ensemble)
    if a.popart is not None: cfg.popart = bool(a.popart)
    if a.dagger is not None: cfg.dagger = bool(a.dagger)
    if a.calib_every is not None: cfg.calib_every = a.calib_every
    if a.oracle_free is not None: cfg.oracle_free = bool(a.oracle_free)
    if a.timing is not None: cfg.timing = bool(a.timing)
    if a.ground_truth_diagnostics is not None: cfg.enable_ground_truth_diagnostics = bool(a.ground_truth_diagnostics)
    if a.score_margin_mHa is not None: cfg.score_margin_mHa = a.score_margin_mHa
    if a.score_margin_mode is not None: cfg.score_margin_mode = a.score_margin_mode
    if a.noise_mode is not None: cfg.noise_mode = a.noise_mode
    if a.noise_p1q is not None: cfg.noise_p1q = a.noise_p1q
    if a.noise_p2q is not None: cfg.noise_p2q = a.noise_p2q
    if a.noise_p_ro is not None: cfg.noise_p_ro = a.noise_p_ro
    if a.noise_basis_change is not None: cfg.noise_basis_change = bool(a.noise_basis_change)
    if a.select_mode is not None: cfg.select_mode = a.select_mode
    if a.wm_greedy_eps is not None: cfg.wm_greedy_eps = a.wm_greedy_eps
    if a.imag_horizon is not None: cfg.imag_horizon = a.imag_horizon
    if a.pessimism_beta is not None: cfg.pessimism_beta = a.pessimism_beta
    if a.maxk_baseline is not None: cfg.maxk_baseline = bool(a.maxk_baseline)
    if a.pot_head is not None: cfg.pot_head = bool(a.pot_head)
    if a.reward_kind is not None: cfg.reward_kind = a.reward_kind
    if a.dir_reweight is not None: cfg.dir_reweight = bool(a.dir_reweight)
    if a.dir_w_max is not None: cfg.dir_w_max = a.dir_w_max
    if a.imag_adv_normalize is not None: cfg.imag_adv_normalize = bool(a.imag_adv_normalize)
    if a.imag_loss_weight is not None: cfg.imag_loss_weight = a.imag_loss_weight
    if a.imag_depth_cap is not None: cfg.imag_depth_cap = bool(a.imag_depth_cap)
    if a.imag_n_seeds is not None: cfg.imag_n_seeds = a.imag_n_seeds
    if a.imag_conf_tau is not None: cfg.imag_conf_tau = a.imag_conf_tau
    if a.imag_transition_budget is not None: cfg.imag_transition_budget = a.imag_transition_budget
    if a.real_dose_alpha is not None: cfg.real_dose_alpha = a.real_dose_alpha
    if a.rpf_beta is not None: cfg.rpf_beta = a.rpf_beta
    if a.wm_ensemble_K is not None: cfg.wm_ensemble_K = a.wm_ensemble_K
    if a.curiosity_beta is not None: cfg.curiosity_beta = a.curiosity_beta
    if a.imag_seed_strategy is not None: cfg.imag_seed_strategy = a.imag_seed_strategy
    if a.dagger_select is not None: cfg.dagger_select = a.dagger_select
    if a.calib_n_top is not None: cfg.calib_n_top = a.calib_n_top
    if a.calib_n_disagree is not None: cfg.calib_n_disagree = a.calib_n_disagree
    if a.ckpt_episodes is not None: cfg.ckpt_episodes = a.ckpt_episodes
    if a.assert_full is not None: cfg.assert_full = bool(a.assert_full)
    # per-molecule WM width (single source of truth in config.WM_HIDDEN)
    cfg.wm_hidden = a.wm_hidden if a.wm_hidden is not None else WM_HIDDEN.get(a.molecule, 256)
    cfg.run_name = f"{a.encoder}_{a.reward}_{a.imagination}_{a.molecule}_s{a.seed}{a.tag}"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.eval_only:                                   # frozen offline eval of ONE checkpoint -> print stats
        stats = eval_harness.evaluate_checkpoint(a.eval_only, a.n_eval_episodes, device=cfg.device)
        print(json.dumps(stats, default=str), flush=True)
    else:
        Runner(cfg).run()
