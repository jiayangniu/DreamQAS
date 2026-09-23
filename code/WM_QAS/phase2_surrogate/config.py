from dataclasses import dataclass


@dataclass
class Config:
    molecule: str = "LiH4q"
    seed: int = 0
    run_name: str = "surr_imag_gru_LiH4q_s0"

    encoder: str = "gru"
    reward: str = "energy"
    imagination: str = "surrogate"

    env_cfg: str = "G0_v2_LiH4q"
    env_experiment: str = "analysis/"
    n_iterations: int = 1000
    real_eps_per_iter: int = 4
    actor_hidden: int = 512
    actor_layers: int = 3
    actor_lr: float = 3e-5
    entropy_coef: float = 1e-3
    reinforce_gamma: float = 0.99
    grad_clip: float = 10.0

    wm_embed: int = 64
    wm_hidden: int = 256
    wm_gru_layers: int = 2
    wm_ensemble_K: int = 3

    independent_ensemble: bool = True
    rpf_beta: float = 3.0
    popart: bool = False
    popart_momentum: float = 0.99
    dagger: bool = True
    pessimism_beta: float = 1.0

    maxk_baseline: bool = False
    maxk_q: float = 0.8
    pot_head: bool = False
    pot_K: int = 5
    reward_kind: str = "energy"
    stair_ladder_mHa: tuple = (100.0, 30.0, 10.0, 3.0, 1.6, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    wm_lr: float = 1e-3
    wm_weight_decay: float = 1e-5
    err_floor_mHa: float = 1e-3
    tail_w_A: float = 5.0
    tail_w_max: float = 20.0
    tail_w_eps: float = 0.3
    dir_reweight: bool = True
    dir_bins: int = 40
    dir_smooth: float = 2.0
    dir_w_max: float = 50.0

    oracle_free: bool = False
    timing: bool = False
    imag_batched_encode: bool = True
    enable_ground_truth_diagnostics: bool = True
    score_margin_mode: str = "fixed"
    score_margin_mHa: float = 0.1
    score_margin_init_mHa: float = 10.0
    score_margin_q: float = 0.10
    score_margin_min_mHa: float = 1.0
    score_margin_max_mHa: float = 50.0
    frontier_jump_warn_mHa: float = 50.0

    noise_mode: str = "off"
    noise_p1q: float = 0.0
    noise_p2q: float = 0.0
    noise_p_ro: float = 0.0
    noise_basis_change: bool = True

    wm_latent_groups: int = 8
    wm_latent_classes: int = 8
    kl_scale: float = 0.1
    kl_free_bits: float = 0.5

    warmup_eps: int = 500
    wm_refresh_every: int = 20
    wm_refresh_steps: int = 200
    fidelity_tau: float = 0.70
    batch_size: int = 64
    elite_N: int = 1000
    frac_prioritized: float = 0.35
    frac_elite: float = 0.25
    frac_stratified: float = 0.20
    frac_random: float = 0.20

    imag_horizon: int = 15
    imag_n_seeds: int = 64
    imag_seed_window: int = 200
    imag_seed_strategy: str = "frontier"
    imag_lambda: float = 0.95
    imag_conf_tau: float = 0.60
    imag_gamma: float = 0.99
    imag_adv_normalize: bool = False
    imag_loss_weight: float = 1.0
    imag_depth_cap: bool = False
    imag_transition_budget: int = 0
    real_dose_alpha: float = 0.0

    select_mode: str = "actor"
    wm_greedy_eps: float = 0.1

    curiosity_beta: float = 0.0

    calib_every: int = 20
    calib_n_top: int = 5
    calib_n_disagree: int = 5
    dagger_select: str = "mix"

    ckpt_episodes: str = ""
    eval_only: str = ""
    n_eval_episodes: int = 20
    eval_final_episodes: int = 100
    assert_full: bool = False

    out_dir: str = "runs/dreamqas"
    log_raw: int = 1
    device: str = "cuda"


EPISODES = {"LiH4q": 15000, "LiH6q": 30000, "BeH2": 15000, "BeH2_8q": 15000, "BeH2_10q": 15000,
            "BeH2_12q": 15000, "H2O_8q": 15000}
N_ITERS = {m: e // 4 for m, e in EPISODES.items()}
WM_HIDDEN = {"LiH4q": 256, "LiH6q": 384, "BeH2": 384, "BeH2_8q": 384, "BeH2_10q": 512,
             "BeH2_12q": 512, "H2O_8q": 384}
ROTOSOLVE_MOLS = ("BeH2_8q", "BeH2_10q", "BeH2_12q", "H2O_8q")


def quarter_episodes(molecule: str) -> int:
    return (EPISODES[molecule] // 4 // 4) * 4


def checkpoint_episodes(molecule: str, n_dense: int = 15, tail_every: int = 1500) -> list:
    total = EPISODES[molecule]
    q = quarter_episodes(molecule)
    start = max(4, (q // 200 // 4) * 4)
    dense = []
    for i in range(n_dense):
        f = i / (n_dense - 1)
        e = start * (q / start) ** f
        dense.append(int(round(e / 4)) * 4)
    dense[-1] = q
    dense = sorted({e for e in dense if e > 0})
    tail = list(range(q + tail_every, total + 1, tail_every))
    if total not in tail:
        tail.append(total)
    return sorted({*dense, *(t for t in tail if t <= total)})


CANONICAL_FULL = dict(
    encoder="gru", reward="energy", imagination="surrogate", reward_kind="energy",
    independent_ensemble=True, wm_ensemble_K=3, rpf_beta=3.0,
    dir_reweight=True, dagger=True, pessimism_beta=1.0,
    fidelity_tau=0.70, imag_horizon=15,
    pot_head=False, popart=False,
    noise_mode="off",
)


def mechanism_diff(cfg) -> dict:
    d = {}
    for k, v in CANONICAL_FULL.items():
        cur = getattr(cfg, k, None)
        if cur != v:
            d[k] = {"canonical": v, "actual": cur}
    return d


def config_hash(cfg) -> str:
    import hashlib
    import json
    from dataclasses import asdict
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
