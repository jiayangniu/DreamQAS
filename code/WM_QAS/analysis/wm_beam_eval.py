"""WM-guided Beam Search (eval-only) on an existing WM-Greedy checkpoint's FROZEN WM.

Answers reviewer-Q1: can a stronger (finite-width) surrogate SEARCH with the SAME WM replace imagined
policy learning? Beam width B keeps the top-B prefixes ranked by the DEPLOYED pessimistic value
phi = -(mean + beta*std); expands over legal next-gates (Shadow legality); NO real VQE / true energy /
mHa / E0 anywhere in search or pruning. The WM alone selects ONE final trajectory, which is then handed
to the STANDARD best-of-1 frozen eval (per-prefix VQE, lowest observed-energy prefix). Real VQE is used
ONLY for that final standard eval, never for beam pruning.

Usage: python analysis/wm_beam_eval.py <wmgreedy_run_dir> [--device cpu|cuda:0] [--B 10]
Writes <run_dir>/beam_eval.json.
"""
import sys, os, glob, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_surrogate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from config import Config
from runner import Runner
from circuit_rules import Shadow, _dict_actions


def _reload(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    valid = set(Config().__dict__.keys())
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in valid}); cfg.device = device
    r = Runner(cfg, _eval_mode=True)
    r.actor.load_state_dict(ck["actor"]); r.wm.load_state_dict(ck["wm"])
    if ck.get("escale") is not None and getattr(r, "scale", None) is not None:
        r.scale.load_state_dict(ck["escale"])
    r.actor.eval(); r.wm.eval()
    return r, ck


@torch.no_grad()
def beam_search(r, B, beta):
    """WM-only beam search from the empty circuit. Returns (best_action_seq, stats). No VQE."""
    A, N = r.A, r.N
    tr = _dict_actions(N)
    # each entry: prefix (list[int]), h (WM hstate, batch=1), sh (Shadow at that prefix), phi
    beam = [{"prefix": [], "h": r.wm.init_state(1, r.dev), "sh": Shadow(N), "phi": 0.0}]
    wm_q = 0; cand_layer = []; dedup = 0; empty_layer = None
    for depth in range(r.env.num_layers):
        cands = []
        for e in beam:
            illegal = set(e["sh"].mask_indices())            # mutates e['sh'] (single-use); mirrors _build_shadows
            legal = [a for a in range(A) if a not in illegal]
            if not legal:
                continue
            acts = torch.tensor(legal, dtype=torch.long, device=r.dev)
            exp = [-1] * e["h"].dim(); exp[-2] = len(legal)
            hexp = e["h"].expand(*exp).contiguous()
            _, newh, ph = r.wm.step(acts, hexp)              # ph [K, nlegal]
            wm_q += len(legal)
            std = ph.std(0) if ph.shape[0] > 1 else torch.zeros_like(ph[0])
            phi = -(ph.mean(0) + beta * std)                 # [nlegal]; higher = better
            for j, a in enumerate(legal):
                nsh = e["sh"].copy(); nsh.commit(tr[a])
                nh = newh[..., j:j + 1, :].contiguous()
                cands.append((float(phi[j]), e["prefix"] + [a], nh, nsh))
        if not cands:
            empty_layer = depth
            break
        cand_layer.append(len(cands))
        # dedup identical prefixes (keep best phi), then keep global top-B
        best = {}
        for phi_v, pre, nh, nsh in cands:
            k = tuple(pre)
            if k not in best or phi_v > best[k][0]:
                best[k] = (phi_v, pre, nh, nsh)
        dedup += len(cands) - len(best)
        top = sorted(best.values(), key=lambda x: -x[0])[:B]
        beam = [{"prefix": t[1], "h": t[2], "sh": t[3], "phi": t[0]} for t in top]
    best_traj = beam[0]["prefix"] if beam else []            # WM picks the top-phi final trajectory
    stats = {"B": B, "wm_queries": wm_q, "cand_per_layer": float(np.mean(cand_layer)) if cand_layer else 0.0,
             "dedup": dedup, "empty_layer": empty_layer, "traj_len": len(best_traj)}
    return best_traj, stats


def standard_bo1(r, seq):
    """Standard best-of-1 on ONE delivered trajectory: replay to FULL depth (run_to_max_depth semantics —
    IGNORE the accuracy early-stop `done`, structural termination only), VQE every prefix, select the
    lowest observed post-VQE ENERGY prefix, report its true error (E0 only for the mHa readout).
    Returns (bo1_mHa, terminal_mHa, n_vqe)."""
    tr = _dict_actions(r.N)
    env = r.env; env.reset()
    energies = []
    for a in seq:
        ill = env.illegal_action_new()
        if ill and int(a) in ill:                    # Shadow-legal => should not happen; guard anyway
            break
        env.step(list(tr[int(a)]), torch.tensor(0.0), train_flag=True)   # ignore returned `done`
        energies.append(float(env.energy))
    n_vqe = len(energies)
    i_star = int(np.argmin(energies))                # select by lowest observed energy (NOT by E0)
    errs = [abs(env.min_energy - E) * 1000.0 for E in energies]          # E0 only for mHa readout
    return float(errs[i_star]), float(errs[-1]), n_vqe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=10)
    a = ap.parse_args()
    cks = sorted(glob.glob(f"{a.run_dir}/ckpt/ep*.pt"),
                 key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    r, ck = _reload(cks[-1], a.device)
    beta = float(getattr(r.cfg, "pessimism_beta", 0.0))
    v0 = r.vqe_calls
    t0 = time.perf_counter()
    traj, stats = beam_search(r, a.B, beta)
    stats["beam_wall_s"] = time.perf_counter() - t0
    stats["search_vqe_calls"] = r.vqe_calls - v0            # MUST be 0 (no VQE in search)
    bo1, term, n_vqe = standard_bo1(r, traj)
    rec = {"episode": int(ck["episode"]), "beta": beta, "bo1_mHa": bo1, "terminal_mHa": term,
           "bo1_eval_vqe_calls": n_vqe, **stats}
    json.dump(rec, open(f"{a.run_dir}/beam_eval.json", "w"))
    print(f"[beam] {os.path.basename(a.run_dir)} B={a.B} bo1={bo1:.3f}mHa term={term:.3f} "
          f"wm_q={stats['wm_queries']} search_vqe={stats['search_vqe_calls']} eval_vqe={n_vqe} "
          f"cand/layer={stats['cand_per_layer']:.0f} empty_layer={stats['empty_layer']}")
    assert stats["search_vqe_calls"] == 0, "beam search consumed real VQE!"


if __name__ == "__main__":
    main()
