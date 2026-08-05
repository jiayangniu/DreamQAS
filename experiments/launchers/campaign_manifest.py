#!/usr/bin/env python3
"""Canonical-AC campaign manifest generator (single source of truth for run commands).

Emits one run per line, TAB-separated:  <label>\t<workdir>\t<command>
consumed by run_campaign.sh (which owns MPS + GPU round-robin + throttling).

WHY THIS EXISTS (pre-launch review finding, CRITICAL): every legacy launcher
(run_program.sh, beh2.sh, ...) hardcodes `--pot_head 1 --imag_horizon 5` -- the two
proven-harmful / non-canonical flags -- and budgets 5000/7500. Reusing them would
silently run the WRONG config for all ~190 DreamQAS runs. This generator instead:
  * OMITS --pot_head / --imag_horizon  -> phase2 runner.py inherits config.py canonical
    Full (pot_head=0, imag_horizon=15).
  * OMITS --n_iterations                -> inherits config.N_ITERS (3750/7500/3750...).
  * adds --assert_full 1 to every canonical-Full run  -> hard-errors on any drift.
  * pins baseline mol keys (10q -> T5_BeH2_6311G_10q, NOT the L6_BeH2_10q Scalability
    molecule) so every method uses the same physical Hamiltonian.

The 6 methods:
  Full     = DreamQAS canonical AC     (phase2_surrogate/runner.py)
  RLQAS    = WM-free REINFORCE baseline (main_baseline.py)
  crlqas   = CRLQAS  (DQN)   |  hyrlqas = HyRLQAS (REINFORCE)     -- PSQASBench main.py
  qdarts   = QuantumDARTS     |  gqeqas  = GQE                    -- PSQASBench main.py

Usage:  python campaign_manifest.py --queue q1_main_4q6q [--seeds 0,1,2,3,4] [--smoke]
"""
import argparse

DREAMQAS = "/home/USER/DreamQAS/code/WM_QAS"
PSQAS = "/home/USER/NeurIPS2026/PSQASBench"
OUT_ROOT = "/home/USER/DreamQAS/experiments/campaign_v1"

# molecule -> (DreamQAS key, baseline DQ key/config stem, PSQASBench COBYLA mol key)
# depth/optimizer/budget all come from config.py (DreamQAS) and the DreamQAS_EXP cfgs
# (baselines) -- NOT hardcoded here, so the five-way consistency is preserved.
MOL = {
    "LiH4q":    dict(dq="LiH4q",    dq_cfg="G0_v2_LiH4q",  base="DQ_LiH_4q",  episodes=15000),
    "LiH6q":    dict(dq="LiH6q",    dq_cfg="G0_v2_LiH6q",  base="DQ_LiH_6q",  episodes=30000),
    "BeH2_6q":  dict(dq="BeH2",     dq_cfg="G0_v2_BeH2",   base="DQ_BeH2_6q", episodes=15000),
    "BeH2_8q":  dict(dq="BeH2_8q",  dq_cfg="G0_v2_BeH2_8q", base="DQ_BeH2_8q",
                     ps_mol="T5_BeH2_631G_8q", episodes=15000),
    "BeH2_10q": dict(dq="BeH2_10q", dq_cfg="G0_v2_BeH2_10q", base="DQ_BeH2_10q",
                     ps_mol="T5_BeH2_6311G_10q", episodes=15000),
}

QUEUES = {
    "q1_main_4q6q":  ["LiH4q", "LiH6q", "BeH2_6q"],
    "q2_main_8q10q": ["BeH2_8q", "BeH2_10q"],
}

# ---- Ablations (all DreamQAS phase2 runs; NO --assert_full, each differs from canonical).
#      Molecule set = {LiH4q,LiH6q (CPU), BeH2_8q,BeH2_10q (GPU)} -- BeH2_6q is "easy",
#      main-only. Deduped vs main + each other:
#        causal      {RLQAS,No-imag,Full}: Full+RLQAS already in main  -> adds No-imag
#        horizon     imag_horizon {0,5,10,15,20}: 15=canonical(main), 0~=No-imag(causal)
#                                                  -> adds {5,10,20}
#        reliability {Full,-DIR,-unc,-DAgger}: Full=main  -> adds {-DIR,-unc,-DAgger}
ABL_MOLS = {"abl_cpu": ["LiH4q", "LiH6q"], "abl_gpu": ["BeH2_8q", "BeH2_10q"]}
ABLATIONS = [
    ("noimag", "--imagination none"),                       # causal: No-imagination
    ("hz5",    "--imag_horizon 5"),                         # imagination-horizon sweep
    ("hz10",   "--imag_horizon 10"),
    ("hz20",   "--imag_horizon 20"),
    ("noDIR",  "--dir_reweight 0"),                         # reliability: -DIR
    ("noUNC",  "--pessimism_beta 0 --imag_conf_tau 999"),   # reliability: -uncertainty-control
    ("noDAG",  "--dagger 0"),                               # reliability: -DAgger
]


def dreamqas_ablation(m, seed, tag, flags, smoke):
    extra = " --n_iterations 4 --warmup_eps 8 --wm_refresh_every 3" if smoke else ""
    cmd = (f"python phase2_surrogate/runner.py --molecule {m['dq']} --seed {seed} {flags} "
           f"--out_dir {OUT_ROOT}/ablations --tag _ab_{tag}{extra}")
    return DREAMQAS, cmd


def dreamqas_full(m, seed, smoke):
    """Canonical-AC Full: omit harmful flags + budget -> inherit config.py; assert_full guards drift."""
    extra = " --n_iterations 4 --warmup_eps 8 --wm_refresh_every 3" if smoke else ""
    cmd = (f"python phase2_surrogate/runner.py --molecule {m['dq']} --seed {seed} "
           f"--assert_full 1 --out_dir {OUT_ROOT}/dreamqas --tag _q{extra}")
    return DREAMQAS, cmd


def dreamqas_rlqas(m, seed, smoke):
    """WM-free REINFORCE baseline. Same env cfg as Full; budget = molecule episodes."""
    ep = 16 if smoke else m["episodes"]
    cmd = (f"python main_baseline.py --config {m['dq_cfg']} --experiment_name analysis/ "
           f"--molecule {m['dq']} --seed {seed} --gpu_id 0 --max_episodes {ep} "
           f"--out_dir {OUT_ROOT}/dreamqas_rlqas")
    return DREAMQAS, cmd


def psqas(method, m, seed, smoke):
    """PSQASBench baseline (crlqas/hyrlqas/qdarts/gqeqas). Config + budget from DreamQAS_EXP cfg."""
    ps_mol = m.get("ps_mol", m["base"])          # 4q/6q use DQ_* key; 8q/10q pin T5_* key
    cfg = f"DreamQAS_EXP/{m['base']}.cfg"
    cmd = (f"python main.py --method {method} --mol {ps_mol} --config {cfg} "
           f"--seed {seed} --device cuda:0 --results-root {OUT_ROOT}/psqas")
    return PSQAS, cmd


METHODS = [
    ("Full",    lambda m, s, sm: dreamqas_full(m, s, sm)),
    ("RLQAS",   lambda m, s, sm: dreamqas_rlqas(m, s, sm)),
    ("crlqas",  lambda m, s, sm: psqas("crlqas", m, s, sm)),
    ("hyrlqas", lambda m, s, sm: psqas("hyrlqas", m, s, sm)),
    ("qdarts",  lambda m, s, sm: psqas("qdarts", m, s, sm)),
    ("gqeqas",  lambda m, s, sm: psqas("gqeqas", m, s, sm)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, choices=list(QUEUES) + list(ABL_MOLS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--methods", default="", help="comma list to restrict (default all 6)")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets for a launcher dry-run")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    only = set(a.methods.split(",")) if a.methods else None
    if a.queue in ABL_MOLS:                                   # ablation queue
        for molname in ABL_MOLS[a.queue]:
            m = MOL[molname]
            for tag, flags in ABLATIONS:
                if only and tag not in only:
                    continue
                for seed in seeds:
                    workdir, cmd = dreamqas_ablation(m, seed, tag, flags, a.smoke)
                    print(f"{a.queue}:{molname}:{tag}:s{seed}\t{workdir}\t{cmd}")
        return
    for molname in QUEUES[a.queue]:                           # main queue (6 methods)
        m = MOL[molname]
        for mname, fn in METHODS:
            if only and mname not in only:
                continue
            for seed in seeds:
                workdir, cmd = fn(m, seed, a.smoke)
                label = f"{a.queue}:{molname}:{mname}:s{seed}"
                print(f"{label}\t{workdir}\t{cmd}")


if __name__ == "__main__":
    main()
