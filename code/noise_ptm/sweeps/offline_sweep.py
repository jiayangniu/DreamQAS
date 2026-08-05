"""Offline noise sweeps — 1q / 2q / readout, four levels each.

WHAT THIS ANSWERS
-----------------
The end-to-end runs answer "how well does each method do under noise". These sweeps
answer the mechanism question: **which part of the learning signal does noise break, and
at what strength?** They are counterfactual replays over the ALREADY-FINISHED clean
LiH-4q runs, so they need no new training — only re-evaluation of the same circuits at
each noise level.

Say so in the paper: these are replays on clean-trained trajectories. The agent that
produced them never saw noise, so this isolates "noise perturbs the signal" from "the
agent adapted to noise".

METRICS (definitions reused from the existing analysis code, not reinvented)
---------------------------------------------------------------------------
rho_act    Spearman(WM score, real post-VQE log-error) over a prefix's legal SIBLING
           actions. Chance = 0. Definition: analysis/of_action_utility_table.py:9.
NReg       (y_selected - y_best) / (y_worst - y_best) for the WM's top-1; 0 = it picked
           the best available gate. Same file, line 13.
rank_rho   Spearman(noisy error, noiseless error) over the same siblings. This is the
           direct "does noise scramble the ordering the agent ranks on" number — 1.0
           means noise shifts every energy but preserves the ranking, so the search is
           unaffected; lower means the ordering itself degrades.
AURC       Area under the risk-coverage curve using ensemble disagreement as the
           selective-prediction score (analysis/risk_coverage.py). Lower = disagreement
           is a better abstention signal.
frontier   Replaying EnergyScale (phase2_surrogate/escale.py) over the trajectory's
           energies, clean vs noisy:
             bias        F_adopted(noisy) - F_adopted(clean), in mHa
             false_rate  fraction of noisy frontier updates that the clean energies
                         would NOT have produced

WHY NOT RE-OPTIMIZE THE ANGLES
------------------------------
Candidates are evaluated by `env.step()`, which runs the full inner optimizer under
whatever noise is attached — so the angles ARE re-optimized under noise, exactly as the
agent would experience. Nothing here reuses clean-optimal parameters.

Usage
-----
    python code/noise_ptm/sweeps/offline_sweep.py --sweep 2q --prefixes 40 --seeds 0,1
    python code/noise_ptm/sweeps/offline_sweep.py --sweep all --out sweeps.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, "..", ".."))          # .../DreamQAS/code
_WM = os.path.join(_CODE, "WM_QAS")
for _p in (_CODE, _WM, os.path.join(_WM, "analysis"), os.path.join(_WM, "phase2_surrogate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DREAMQAS_NO_MPS", "1")

from noise_ptm import NoiseSpec4q, SWEEP_1Q, SWEEP_2Q, SWEEP_RO   # noqa: E402
from noise_ptm.integration import build_evaluator                 # noqa: E402

FLOOR = 1e-3          # mHa floor for log10, matching t1a_action_ranking.py
CLEAN_GLOB = ("/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/"
              "oracle_free_v1/dreamqas/gru_energy_surrogate_LiH4q_s{seed}_of")
SWEEPS = {"1q": SWEEP_1Q, "2q": SWEEP_2Q, "readout": SWEEP_RO}
LEVELS = ("clean", "mild", "medium", "strong")


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# one prefix -> all legal sibling actions, evaluated under the attached noise
# ---------------------------------------------------------------------------

def probe(r, prefix, max_cand, rng):
    """Evaluate every legal next action from `prefix`.

    Returns per-candidate noisy error, noiseless error, WM score and WM disagreement.
    Built on t1a_action_ranking's snapshot/restore + Shadow legality so the env
    bookkeeping is the already-validated one; the only additions are the noiseless
    companion and the disagreement, which the noise metrics need.
    """
    from circuit_rules import Shadow
    from imagine import _encode_seeds, _ens_std
    from t1a_action_ranking import _snap, _restore

    _, applied = r._vqe_circuit(list(prefix), count_budget=False)
    if len(applied) < 2:
        return None
    prefix = list(applied)
    env = r.env
    snap = _snap(env)

    sh = Shadow(r.N)
    for a in prefix:
        sh.commit(r.trans[int(a)])
    ill = set(sh.mask_indices() or [])
    legal = [a for a in range(r.A) if a not in ill]
    if len(legal) < 3:
        return None
    cands = legal if len(legal) <= max_cand else list(rng.choice(legal, max_cand, replace=False))

    noisy = np.empty(len(cands), float)
    noiseless = np.empty(len(cands), float)
    for i, a in enumerate(cands):
        _restore(env, snap)
        env.step(list(r.trans[int(a)]), torch.tensor(0.0), train_flag=True)
        # env.energy is the OBSERVED (noisy when a backend is attached) energy; the
        # noiseless companion is stored alongside it by environment.py.
        noisy[i] = abs(env.min_energy - env.energy) * 1000.0
        noiseless[i] = abs(env.min_energy - getattr(env, "energy_noiseless", env.energy)) * 1000.0
    _restore(env, snap)

    seqs = [prefix + [a] for a in cands]
    _, _, preds = _encode_seeds(r.wm, [list(s) for s in seqs], r.dev)   # [K, N]
    pred = preds.mean(0).detach().cpu().numpy()
    disagree = _ens_std(preds).detach().cpu().numpy()

    return dict(cands=cands, noisy=noisy, noiseless=noiseless, pred=pred, disagree=disagree)


def metrics_from_probe(p):
    """rho_act / NReg / rank_rho for one prefix."""
    noisy, noiseless, pred = p["noisy"], p["noiseless"], p["pred"]
    lnoisy = np.log10(np.maximum(noisy, FLOOR))
    pick = int(np.argsort(pred)[0])                       # WM's top-1
    lo, hi = noisy.min(), noisy.max()
    return dict(
        rho_act=_spearman(pred, lnoisy),                  # WM vs what the agent observes
        nreg=float((noisy[pick] - lo) / (hi - lo)) if hi > lo else 0.0,
        rank_rho=_spearman(noisy, noiseless),             # does noise scramble the order?
        n_cand=len(noisy),
    )


def aurc_from_probes(probes):
    """Risk-coverage AURC using ensemble disagreement as the abstention score.

    Pooled over all probed candidates: sort by disagreement ascending, sweep coverage,
    risk = mean noisy error of the retained set (normalised per prefix so prefixes with
    very different error scales do not dominate).
    """
    scores, risks = [], []
    for p in probes:
        e = p["noisy"]
        rng_ = e.max() - e.min()
        if rng_ <= 0:
            continue
        scores.append(p["disagree"])
        risks.append((e - e.min()) / rng_)               # per-prefix normalised regret
    if not scores:
        return float("nan")
    s = np.concatenate(scores)
    y = np.concatenate(risks)
    order = np.argsort(s)
    y = y[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(y) / np.arange(1, len(y) + 1)
    _trap = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed it
    return float(_trap(risk, cov))


# ---------------------------------------------------------------------------
# frontier replay
# ---------------------------------------------------------------------------

def frontier_replay(energies_clean, energies_noisy, cfg):
    """Replay EnergyScale over both energy streams; report bias and false-update rate."""
    from escale import EnergyScale
    sc_c, sc_n = EnergyScale(cfg), EnergyScale(cfg)
    sc_c.init_from(energies_clean[:4]); sc_n.init_from(energies_noisy[:4])
    false_updates = 0
    n_updates = 0
    for ec, en in zip(energies_clean, energies_noisy):
        before = sc_n.pending_frontier
        sc_c.observe(ec); sc_n.observe(en)
        if en < before:                                   # the noisy stream moved the frontier
            n_updates += 1
            if ec >= before:                              # ... but the clean energy would NOT have
                false_updates += 1
    sc_c.adopt(); sc_n.adopt()
    return dict(
        frontier_bias_mHa=float((sc_n.F_adopted - sc_c.F_adopted) * 1000.0),
        false_update_rate=float(false_updates / n_updates) if n_updates else 0.0,
        n_frontier_updates=int(n_updates),
    )


# ---------------------------------------------------------------------------

def run_level(run_dir, spec, prefixes, max_cand, seed):
    """Reload the clean run, attach `spec` (None = clean), probe every prefix."""
    from eval_policy_traces import _reload_dreamqas
    cks = sorted(glob.glob(f"{run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int(os.path.basename(p)[2:-3]))
    if not cks:
        raise SystemExit(f"no checkpoints in {run_dir}")
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    r, _ = _reload_dreamqas(ck, "cpu")
    if spec is not None and not spec.is_noiseless:
        r.env.attach_noisy_evaluator(build_evaluator(
            r.env, mode="ptm", p1q=spec.p1q, p2q=spec.p2q, p_ro=spec.p_ro,
            basis_change=spec.model_basis_change, verbose=False))
    rng = np.random.default_rng(20260805 + seed)
    out = []
    for pre in prefixes:
        p = probe(r, pre, max_cand, rng)
        if p is not None:
            out.append(p)
    del r
    return out


def load_prefixes(run_dir, n_prefix, rng):
    """Action prefixes sampled from the clean run's own trajectory."""
    rows = []
    with open(f"{run_dir}/trajectory.jsonl") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    eps = {}
    for x in rows:
        eps.setdefault((x["iter"], x["ep"]), []).append(int(x["action"]))
    seqs = [v for v in eps.values() if len(v) >= 4]
    if not seqs:
        return []
    idx = rng.choice(len(seqs), size=min(n_prefix, len(seqs)), replace=False)
    # a mid-episode prefix, so there are still legal siblings to rank
    return [seqs[i][: max(2, len(seqs[i]) // 2)] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="all", choices=("all", "1q", "2q", "readout"))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--prefixes", type=int, default=25)
    ap.add_argument("--max_cand", type=int, default=12,
                    help="legal sibling actions probed per prefix (each is one real VQE). "
                         "NReg saturates at 0 with too few candidates -- the WM's top-1 is "
                         "then almost always the best of a tiny set. Use >=10 for a "
                         "meaningful NReg.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    names = list(SWEEPS) if a.sweep == "all" else [a.sweep]
    results = {}

    for sname in names:
        table = SWEEPS[sname]
        for seed in seeds:
            run_dir = CLEAN_GLOB.format(seed=seed)
            if not os.path.isdir(run_dir):
                print(f"[skip] missing {run_dir}")
                continue
            rng = np.random.default_rng(1234 + seed)
            prefixes = load_prefixes(run_dir, a.prefixes, rng)
            print(f"\n=== sweep {sname}  seed {seed}  ({len(prefixes)} prefixes, "
                  f"<= {a.max_cand} siblings each) ===")
            print(f"  {'level':<8}{'rho_act':>10}{'NReg':>9}{'rank_rho':>10}"
                  f"{'AURC':>9}{'noise_pen_mHa':>15}")
            for level in LEVELS:
                spec = table[level]
                probes = run_level(run_dir, spec, prefixes, a.max_cand, seed)
                if not probes:
                    print(f"  {level:<8} (no usable prefix)")
                    continue
                per = [metrics_from_probe(p) for p in probes]
                agg = {k: float(np.nanmean([m[k] for m in per]))
                       for k in ("rho_act", "nreg", "rank_rho")}
                agg["aurc"] = aurc_from_probes(probes)
                pen = float(np.mean([np.mean(p["noisy"] - p["noiseless"]) for p in probes]))
                agg["noise_penalty_mHa"] = pen
                agg["n_prefix"] = len(probes)
                agg["spec"] = spec.to_tag()
                results[f"{sname}/seed{seed}/{level}"] = agg
                print(f"  {level:<8}{agg['rho_act']:>10.3f}{agg['nreg']:>9.3f}"
                      f"{agg['rank_rho']:>10.3f}{agg['aurc']:>9.3f}{pen:>15.2f}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[sweep] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
