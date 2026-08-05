"""Phase 2: energy-surrogate WM + surrogate-imagination — config / flag contract.

One codebase, three orthogonal flags define the 5-variant matrix:
    encoder      ∈ {gru, rssm}          which representation
    reward       ∈ {curriculum, energy} dense energy-improvement vs v2 curriculum
    imagination  ∈ {off, surrogate}     surrogate-imagination on/off

Matrix (per molecule × seed):
    noimag            : rssm  + curriculum + off      (bar; re-run clean)
    surr_noimag_rssm  : rssm  + energy     + off      (isolate dense reward)
    surr_imag_rssm    : rssm  + energy     + surrogate(imagination on rssm)
    surr_noimag_gru   : gru   + energy     + off      (isolate encoder)
    surr_imag_gru     : gru   + energy     + surrogate(imagination on gru)

Results -> /home/USER/DreamQAS/experiments/raw_run_outputs/DreamQAS_phase2_results
           (inside the repo so it migrates with DreamQAS/; persistent /dev/root, NO nvme symlink).
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- identity ----
    molecule: str = "LiH4q"           # LiH4q | LiH6q  (drives env cfg + n_iterations)
    seed: int = 0
    run_name: str = "surr_imag_gru_LiH4q_s0"

    # ---- the 3 axis flags ----
    encoder: str = "gru"              # gru | rssm
    reward: str = "energy"            # curriculum | energy
    imagination: str = "surrogate"    # off | surrogate

    # ---- env / RL loop ----
    env_cfg: str = "G0_v2_LiH4q"      # reuse an analysis cfg for env params
    env_experiment: str = "analysis/"
    n_iterations: int = 1000          # 4q=1000 / 6q=1500 (set per molecule)
    real_eps_per_iter: int = 4
    actor_hidden: int = 512
    actor_layers: int = 3
    actor_lr: float = 3e-5
    entropy_coef: float = 1e-3
    reinforce_gamma: float = 0.99
    grad_clip: float = 10.0

    # ---- surrogate WM (energy head) ----
    wm_embed: int = 64
    wm_hidden: int = 256              # 4q=256 / 6q=384 (set per molecule in runner)
    wm_gru_layers: int = 2
    wm_ensemble_K: int = 3            # ensemble members (epistemic confidence)

    # ---- root-cause fixes (all cfg-gated so old behavior is reproducible) ----
    independent_ensemble: bool = True # C: K INDEPENDENT encoders + RPF (vs shared trunk)
    rpf_beta: float = 3.0             # C: randomized-prior scale (Osband'18); disagreement OOD
    popart: bool = False              # B: adaptive target-normalization — PROVEN HARMFUL (2026-07) -> OFF in canonical Full
    popart_momentum: float = 0.99
    dagger: bool = True               # A: feed VQE-calibrated imagined circuits back to buffer
    pessimism_beta: float = 1.0       # D: imag value = -(mean_logerr + beta*disagree) (avoid model error)

    # ---- beneficial reward-shaping techniques (150-run G2; stack on A+C to boost precision) ----
    maxk_baseline: bool = False       # M1: max-seeking high-quantile REINFORCE baseline (best 4q)
    maxk_q: float = 0.8               # baseline = quantile(returns, q): only top-(1-q) get +advantage
    pot_head: bool = False            # M2: extra actor feature = predicted sum of next-K rewards — PROVEN HARMFUL -> OFF in canonical Full
    pot_K: int = 5
    reward_kind: str = "energy"       # energy | staircase (Q1: event-driven best_err-crossing bonus)
    stair_ladder_mHa: tuple = (100.0, 30.0, 10.0, 3.0, 1.6, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    wm_lr: float = 1e-3
    wm_weight_decay: float = 1e-5
    err_floor_mHa: float = 1e-3
    tail_w_A: float = 5.0             # legacy tail-weight w = clip(A/(err_mHa+eps),1,max) (dir off)
    tail_w_max: float = 20.0
    tail_w_eps: float = 0.3
    # imbalanced-regression (Yang'21 DIR): weight ∝ 1/sqrt(smoothed label density) so the RARE
    # low-error breakthroughs (~4%) aren't drowned by the abundant energy-plateau mass.
    dir_reweight: bool = True
    dir_bins: int = 40
    dir_smooth: float = 2.0           # LDS gaussian-kernel sigma (bins)
    dir_w_max: float = 50.0           # cap on the inverse-density weight

    # ---- oracle-free training (empirical-frontier score; escale.py) ----
    # OFF (default): training primitive stays the E0-derived true error — byte-identical legacy path.
    # ON: reward/WM-label/DIR/priority switch to the symmetric signed-log frontier score
    # S(E) = sgn(d)·log10(1 + |d|/m), d = (E − F_adopted)·1e3 [mHa], m = margin (see escale.py for the
    # exact form; for d ≥ 0 this equals the legacy log-error shape up to a constant, so step rewards are
    # unchanged there), built ONLY from observed real-VQE energies; E0 then feeds diagnostics/eval only.
    oracle_free: bool = False
    # ---- phase wall-clock accounting (VQE vs WM-training vs imagination share of end-to-end time) ----
    # OFF (default): no timers, no cuda.synchronize -> byte-identical legacy path.
    # ON: per-phase cumulative seconds logged into metrics.jsonl (t_vqe_real / t_vqe_calib /
    # t_rollout_other / t_actor / t_imagine / t_wm_train / t_other / t_wall). The added
    # cuda.synchronize() at phase boundaries makes GPU work bill to the phase that issued it;
    # it slightly serialises the pipeline, so timing runs are for MEASUREMENT, not for results.
    timing: bool = False
    # Batched seed encoding for imagination. ON (default) collapses sum(len)=~2240 batch-1 GPU
    # launches per update into T=~35 batched ones; the legacy per-seed loop was measured at ~96%
    # of imagined-rollout time and grew linearly with circuit depth. Mathematically identical per
    # seed (only the GEMM batch size differs). Set False to reproduce the legacy path exactly.
    imag_batched_encode: bool = True
    enable_ground_truth_diagnostics: bool = True   # False (requires oracle_free): env.min_energy severed;
                                                   # all E0-derived log fields become null
    # margin = the score's resolution floor near the frontier (oracle-free analogue of err_floor_mHa).
    # "fixed" (default): plain hyperparameter score_margin_mHa, stationary from the start.
    #   0.1 mHa keeps ~83%/50% of the legacy reward mass in the 0.1–1 / 0.01–0.1 mHa bands
    #   (offline replay of the canonical LiH4q trajectory); 16× finer than chemical accuracy.
    #   COBYLA (complex128) molecules tolerate down to ~1e-3; the 8q/10q complex64 rotosolve path
    #   has ~1e-2 mHa numeric noise — do not go below ~0.1 there.
    # "quantile": legacy variant — provisional init, frozen at the first refresh after warm-up
    #   from the q-quantile of warm-up distances (clipped [min,max]).
    score_margin_mode: str = "fixed"               # fixed | quantile
    score_margin_mHa: float = 0.1                  # the fixed-mode hyperparameter
    score_margin_init_mHa: float = 10.0            # quantile mode: provisional pre-freeze margin
    score_margin_q: float = 0.10                   # quantile mode: warm-up quantile of (E − frontier)
    score_margin_min_mHa: float = 1.0
    score_margin_max_mHa: float = 50.0
    frontier_jump_warn_mHa: float = 50.0           # warn on single-observation frontier drops > this

    # ---- noisy quantum environment (PTM backend; code/noise_ptm/) ----
    # OFF (default, "off"): no evaluator is built, env._noisy_eval stays None, and every
    # branch in environment.py takes the byte-identical noiseless path.
    # ON  ("ptm"): <H> during training and during the optimizer's inner loop becomes the
    # EXACT noisy density-matrix expectation (no shots -> deterministic per (circuit, θ)).
    # The noiseless companion energy keeps being computed alongside it, so the delivered
    # circuit can be reported both ways.
    # These live here rather than in the .cfg on purpose: the .cfg is re-read from disk at
    # eval time (eval_harness.py / eval_policy_traces.py / runner_baseline.py), so a noise
    # setting kept only there would silently differ between training and evaluation. Config
    # is serialised into the checkpoint, so it cannot drift.
    noise_mode: str = "off"                        # off | ptm
    noise_p1q: float = 0.0                         # depolarizing after every 1q rotation
    noise_p2q: float = 0.0                         # depolarizing after every CNOT
    noise_p_ro: float = 0.0                        # symmetric readout, P(1̃|0) = P(0̃|1)
    # Model the depolarizing of the measurement basis-change gates (H / S†H) that a real
    # experiment needs for X/Y Pauli terms. Exact and free — folded into the Hamiltonian's
    # Pauli vector at compile time. See code/noise_ptm/readout.py.
    noise_basis_change: bool = True

    # ---- RSSM encoder (categorical latent bottleneck; only when encoder=rssm) ----
    wm_latent_groups: int = 8         # DreamerV3-style discrete latent: G groups
    wm_latent_classes: int = 8        # × C classes each (straight-through categorical)
    kl_scale: float = 0.1             # weight of the latent KL term in the WM loss
    kl_free_bits: float = 0.5         # nats/group below which KL isn't penalized (anti-collapse floor)

    # ---- WM update schedule + data ----
    warmup_eps: int = 500             # imagination stays OFF until this many real eps collected
    wm_refresh_every: int = 20        # check fidelity / maybe refresh every N iters
    wm_refresh_steps: int = 200       # grad steps per refresh (fine-tune)
    fidelity_tau: float = 0.70        # pairwise-acc gate: imagination on only if fidelity>tau
    # replay batch composition
    batch_size: int = 64
    elite_N: int = 1000               # elite buffer size (best circuits ever)
    frac_prioritized: float = 0.35    # p ∝ quality×recency×wm_error
    frac_elite: float = 0.25
    frac_stratified: float = 0.20     # B: even coverage across error deciles (keeps coarse ranking)
    frac_random: float = 0.20

    # ---- imagination ----
    imag_horizon: int = 15            # canonical unified default (horizon sweep {0,5,10,15,20} validates)
    imag_n_seeds: int = 64            # imagined trajectories per update
    imag_seed_window: int = 200       # seeds drawn from most-recent N episodes (frontier)
    imag_seed_strategy: str = "frontier"  # frontier | stratified | high_disagree | elite (P3: where to seed)
    imag_lambda: float = 0.95         # λ-return
    imag_conf_tau: float = 0.60       # ensemble std above which rollout STOPS (normalized units w/ popart)
    imag_gamma: float = 0.99
    # imagination gradient share (P1: raise imagination's influence on the actor from ~7%)
    imag_adv_normalize: bool = False  # whiten imag advantage by its std (match real arm's O(1) scale)
    imag_loss_weight: float = 1.0     # λ_imag: explicit multiplier on the imagination REINFORCE loss
    imag_depth_cap: bool = False      # ablation: cap imagined circuit depth at env num_layers (mirror real env)
    # transition-matched horizon control: 0 = legacy; >0 = generate imagined trajectories until >=T
    # TRUSTED transition loss-terms accrue, then randomly downsample to EXACTLY T (returns computed on
    # full trajectories first). Makes H in {1,5,15} use the same trusted-transition budget per update.
    imag_transition_budget: int = 0
    # NoImag "RealDose" compute control (default OFF): add alpha * real_reinforce(bootstrap of current eps).
    real_dose_alpha: float = 0.0

    # ---- WM-Greedy baseline (surrogate candidate-selection; NoImag + rollout-selection override) ----
    # select_mode="wm_greedy": at each real construction step, enumerate ALL legal next-actions, roll each
    # ONE step under the frozen WM (no VQE), score with the SAME pessimistic value as imagination
    # phi = -(mean_pred + pessimism_beta*ensemble_std), commit argmax -> exactly ONE VQE. No imagined PG.
    select_mode: str = "actor"        # actor | wm_greedy (rollout-time action selection)
    wm_greedy_eps: float = 0.1        # ε-greedy exploration while the WM is frozen between refreshes

    # ---- curiosity / exploration (P3: attack the 6q energy-plateau) ----
    curiosity_beta: float = 0.0       # intrinsic reward += beta * ensemble-disagreement at each real step

    # ---- VQE calibration (DAgger + active learning) ----
    calib_every: int = 20             # every N iters
    calib_n_top: int = 5              # VQE this many top-predicted-good imagined circuits
    calib_n_disagree: int = 5         # + this many highest-ensemble-disagreement ones
    dagger_select: str = "mix"        # top | disagree | mix | random (P4: which circuits to VQE-verify)

    # ---- standardized eval protocol / checkpointing (2026-07) ----
    ckpt_episodes: str = ""           # comma-sep episode checkpoints; "" = auto (¼-dense 15 + every 1500)
    eval_only: str = ""               # path to a checkpoint .pt -> run frozen offline eval instead of training
    n_eval_episodes: int = 20         # fresh eval episodes per intermediate checkpoint (final uses 100)
    eval_final_episodes: int = 100    # fresh eval episodes for the final checkpoint (best-of-N reported)
    assert_full: bool = False         # if True, hard-error at startup unless mechanism flags == canonical Full

    # ---- output / logging ----
    out_dir: str = "/home/USER/DreamQAS/experiments/raw_run_outputs/DreamQAS_phase2_results"
    log_raw: int = 1                  # write trajectory.jsonl (per-step)
    device: str = "cuda"


# ---- per-molecule constants (single source of truth) ----
# Full RLQAS-matched episode budget. DreamQAS trains the FULL budget; the headline
# "equal-budget" result is read from the ¼-budget checkpoint (no separate ¼ run).
EPISODES = {"LiH4q": 15000, "LiH6q": 30000, "BeH2": 15000, "BeH2_8q": 15000, "BeH2_10q": 15000,
            # ---- scale-up tasks (added 2026-07-30). Both take the LOCKED common budget of
            # 15,000 episodes, so no new budget convention enters the paper.
            "BeH2_12q": 15000, "H2O_8q": 15000}
N_ITERS = {m: e // 4 for m, e in EPISODES.items()}          # 4 real eps / iter
# WM width ladder by qubit count: 4q=256, 6q/8q=384, >=10q=512. The scale-up tasks REUSE
# existing rungs (H2O_8q -> 384 like BeH2_8q; BeH2_12q -> 512 like BeH2_10q) rather than
# introducing a new width, so capacity is never a confound when comparing across sizes.
WM_HIDDEN = {"LiH4q": 256, "LiH6q": 384, "BeH2": 384, "BeH2_8q": 384, "BeH2_10q": 512,
             "BeH2_12q": 512, "H2O_8q": 384}
# molecules whose VQE runs on the GPU rotosolve path (tractable large systems).
# Convention: >= 8 qubits -> ROTOSOLVE on GPU; 4q/6q stay CPU + COBYLA.
ROTOSOLVE_MOLS = ("BeH2_8q", "BeH2_10q", "BeH2_12q", "H2O_8q")


def quarter_episodes(molecule: str) -> int:
    """The ¼-budget headline episode count, snapped to a multiple of real_eps_per_iter=4."""
    return (EPISODES[molecule] // 4 // 4) * 4


def checkpoint_episodes(molecule: str, n_dense: int = 15, tail_every: int = 1500) -> list:
    """Standardized checkpoint schedule: `n_dense` log-spaced points within the ¼ budget
    (front-dense; last == ¼ point == headline equal-budget read point), then one every
    `tail_every` episodes to the full budget. All snapped to a multiple of 4."""
    total = EPISODES[molecule]
    q = quarter_episodes(molecule)
    start = max(4, (q // 200 // 4) * 4)
    dense = []
    for i in range(n_dense):
        f = i / (n_dense - 1)
        e = start * (q / start) ** f            # geometric start -> q
        dense.append(int(round(e / 4)) * 4)
    dense[-1] = q                               # force exact ¼ point
    dense = sorted({e for e in dense if e > 0})
    tail = list(range(q + tail_every, total + 1, tail_every))
    if total not in tail:
        tail.append(total)
    return sorted({*dense, *(t for t in tail if t <= total)})


# ---- canonical Full ("AC") mechanism flags — anti-drift single source of truth ----
CANONICAL_FULL = dict(
    encoder="gru", reward="energy", imagination="surrogate", reward_kind="energy",
    independent_ensemble=True, wm_ensemble_K=3, rpf_beta=3.0,
    dir_reweight=True, dagger=True, pessimism_beta=1.0,
    fidelity_tau=0.70, imag_horizon=15,
    pot_head=False, popart=False,               # the two proven-harmful modules, OFF
    # Noise is a mechanism change like any other: without it here, mechanism_diff() —
    # and therefore --assert_full 1 — would be blind to a run that accidentally trained
    # in a noisy environment. The noisy runs are a separate arm and deliberately fail
    # this check; they must be launched WITHOUT --assert_full.
    noise_mode="off",
)


def mechanism_diff(cfg) -> dict:
    """Which canonical-Full mechanism flags this config differs from (empty = exactly Full).
    Surfaces any unintended default leak (pot_head=1 / popart=1 / horizon=5) per run."""
    d = {}
    for k, v in CANONICAL_FULL.items():
        cur = getattr(cfg, k, None)
        if cur != v:
            d[k] = {"canonical": v, "actual": cur}
    return d


def config_hash(cfg) -> str:
    """Stable short hash of the fully-resolved config (sorted) — provenance guard across runs."""
    import hashlib
    import json
    from dataclasses import asdict
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
