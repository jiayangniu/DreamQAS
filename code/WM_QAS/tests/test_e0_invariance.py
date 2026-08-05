"""E0-invariance test for oracle-free training (cfg.oracle_free=True).

Acceptance criterion (approved plan): with oracle_free on, deleting or SHIFTING the exact
ground-state energy E0 (env.min_energy) may change ONLY the whitelisted ground-truth diagnostic
fields — never any training state, action, weight, priority, or verification decision.

Cases:
  1. Two Runners, same seed, one with env.min_energy shifted DOWN by 0.5 Ha (down, not up: E0 stays
     below every variational energy, so |E - E0| shifts by exactly +500 mHa — up-shifts can cross
     observed energies and break the constant-offset check).
     - MAIN assertions: buffer actions/energies, EnergyScale state (incl. version), replay
       priorities, WM + actor weights, best_seq/best_E — all identical.
     - WHITELIST (must CHANGE, by exactly +500 mHa where applicable): best_err, trajectory
       true_error, calibration real_err_mHa / real_logerr.
  2. EnergyScale unit checks: first-episode init gives finite scores before any refresh;
     consecutive new bests below the adopted frontier get DISTINCT, correctly ordered scores
     (C1 extension, no clamp-induced ties/zero rewards).
  3. min_energy=None + enable_ground_truth_diagnostics=False: the run completes warm-up, WM
     refresh, imagination and DAgger calibrate; true_error is null in trajectory.jsonl.

Run:  python tests/test_e0_invariance.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2_surrogate"))
os.chdir(_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from config import Config                      # noqa: E402
from runner import Runner                      # noqa: E402
from escale import EnergyScale                 # noqa: E402

TMP = tempfile.mkdtemp(prefix="e0inv_")
WHITELIST_NOTE = ("whitelisted E0 diagnostics: best_err, trajectory.true_error, "
                  "calibration real_err_mHa/real_logerr, metrics best_err_mHa/mean_ep_err_mHa")


def make_cfg(tag, **over):
    cfg = Config()
    cfg.molecule = "LiH4q"
    cfg.env_cfg = "G0_v2_LiH4q"
    cfg.device = "cpu"
    cfg.oracle_free = True
    cfg.n_iterations = 3
    cfg.warmup_eps = 4
    cfg.wm_refresh_every = 1
    cfg.calib_every = 1
    cfg.fidelity_tau = 0.0
    cfg.n_iterations = 3
    cfg.out_dir = TMP
    cfg.run_name = f"inv_{tag}"
    cfg.ckpt_episodes = "999999"     # skip mid-run ckpts (schedule fires only at final save)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def run_one(tag, e0_shift=0.0, **over):
    torch.manual_seed(0); np.random.seed(0)
    r = Runner(make_cfg(tag, **over))
    if e0_shift:
        r.env.min_energy = r.env.min_energy + e0_shift
    r.run()
    return r


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def case1_shift_invariance():
    r1 = run_one("base")
    r2 = run_one("shift", e0_shift=-0.5)      # DOWN 0.5 Ha -> te' = te + 500 mHa exactly

    # ---- MAIN: training state identical ----
    assert len(r1.buf.eps) == len(r2.buf.eps), "buffer episode count differs"
    for a, b in zip(r1.buf.eps, r2.buf.eps):
        assert a["acts"].tolist() == b["acts"].tolist(), "action sequences differ"
        assert np.allclose(a["E"], b["E"], atol=0), "buffered raw energies differ"
    assert r1.scale.state_dict() == r2.scale.state_dict(), "EnergyScale state differs"
    assert np.allclose(r1.buf._priorities(), r2.buf._priorities()), "replay priorities differ"
    for k in r1.wm.state_dict():
        assert torch.equal(r1.wm.state_dict()[k], r2.wm.state_dict()[k]), f"WM weight differs: {k}"
    for k in r1.actor.state_dict():
        assert torch.equal(r1.actor.state_dict()[k], r2.actor.state_dict()[k]), f"actor weight differs: {k}"
    assert r1.best_seq == r2.best_seq and r1.best_E == r2.best_E, "best circuit differs"
    assert r1.vqe_calls == r2.vqe_calls and r1.calib_vqe_calls == r2.calib_vqe_calls

    # trajectory: every non-diagnostic column identical; true_error differs by exactly +500
    t1 = read_jsonl(f"{r1.outdir}/trajectory.jsonl")
    t2 = read_jsonl(f"{r2.outdir}/trajectory.jsonl")
    assert len(t1) == len(t2)
    for x, y in zip(t1, t2):
        for k in x:
            if k == "true_error":
                assert abs((y[k] - x[k]) - 500.0) < 1e-6, "true_error must shift by exactly +500 mHa"
            elif k == "error":
                # env.error is vs fake_min_energy (cfg constant) -> unchanged by an E0 shift
                assert x[k] == y[k]
            else:
                assert x[k] == y[k], f"non-diagnostic trajectory field changed: {k}"

    # calibration: pred/real scores + selection identical; real_err_mHa shifted
    c1 = [r for r in read_jsonl(f"{r1.outdir}/calibration.jsonl") if r.get("selection") != "SUMMARY"]
    c2 = [r for r in read_jsonl(f"{r2.outdir}/calibration.jsonl") if r.get("selection") != "SUMMARY"]
    assert len(c1) == len(c2)
    for x, y in zip(c1, c2):
        assert x["selection"] == y["selection"] and x["pred_score"] == y["pred_score"]
        assert x["real_score"] == y["real_score"] and x["score_abs_err"] == y["score_abs_err"]
        assert abs((y["real_err_mHa"] - x["real_err_mHa"]) - 500.0) < 1e-6

    # WHITELIST diagnostics DID change
    assert abs((r2.best_err - r1.best_err) - 500.0) < 1e-6, "best_err must shift by exactly +500 mHa"
    print(f"  case1 OK: training state bit-identical under E0-0.5Ha; diagnostics shifted +500 mHa "
          f"({len(t1)} traj rows, {len(c1)} calib rows). {WHITELIST_NOTE}")


def case2_escale_units():
    cfg = make_cfg("unit")
    sc = EnergyScale(cfg)
    # first-episode init -> finite scores before any refresh
    ep0 = np.array([-7.60, -7.65, -7.70], np.float64)
    sc.init_from(ep0)
    s0 = sc.score_np(ep0)
    assert np.all(np.isfinite(s0)), "first-episode scores must be finite pre-refresh"
    # consecutive new bests BELOW the adopted frontier: distinct, ordered, no zero-reward ties
    lows = np.array([-7.72, -7.74, -7.76], np.float64)     # each below F_adopted=-7.70
    sl = sc.score_np(lows)
    assert np.all(np.isfinite(sl))
    assert sl[0] > sl[1] > sl[2], "scores of successive new bests must be strictly decreasing"
    rewards = -np.diff(np.concatenate([s0[-1:], sl]))       # r = S[t-1]-S[t] > 0 for improvements
    assert np.all(rewards > 0), "no clamp-induced zero rewards on below-frontier improvements"
    # monotone across the full range (mixed d<0 / d>=0)
    Es = np.linspace(-7.80, -7.40, 41)
    ss = sc.score_np(Es, count=False)
    assert np.all(np.diff(ss) > 0), "S(E) must be strictly increasing in E"
    # adopt: version bump + s_mHa positivity
    sc.adopt()
    assert np.all(sc.s_mHa_np(Es) > 0), "s_mHa = 10**S must be strictly positive"
    print("  case2 OK: C1 extension strictly monotone, finite, positive s_mHa, no zero rewards")


def case3_diagnostics_off():
    r = run_one("noe0", enable_ground_truth_diagnostics=False)
    assert r.env.min_energy is None
    assert r.scale.frozen, "warm-up freeze must have fired"
    t = read_jsonl(f"{r.outdir}/trajectory.jsonl")
    assert all(row["true_error"] is None for row in t), "true_error must be null with E0 severed"
    assert all(np.isfinite(row["frontier_score"]) for row in t if row["frontier_score"] is not None)
    m = read_jsonl(f"{r.outdir}/metrics.jsonl")
    assert all(row["best_err_mHa"] is None for row in m)
    assert any(row.get("imag_on") for row in m), "imagination must have switched on"
    c = read_jsonl(f"{r.outdir}/calibration.jsonl")
    assert any(row.get("selection") == "SUMMARY" for row in c), "calibrate must have run"
    for row in c:
        if row.get("selection") != "SUMMARY":
            assert row["real_err_mHa"] is None and row["real_logerr"] is None
            assert np.isfinite(row["real_score"]) and np.isfinite(row["score_abs_err"])
    print(f"  case3 OK: full run (warmup+WM+imagination+DAgger) with min_energy=None; "
          f"E0 fields null, scores finite ({len(t)} traj rows)")


if __name__ == "__main__":
    try:
        case2_escale_units()      # cheap unit checks first
        case1_shift_invariance()
        case3_diagnostics_off()
        print("ALL E0-INVARIANCE CHECKS PASSED")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
