"""How far is the ORACLE-FREE training reward from the reward FCI would have given? (appendix §5)

SAME-SAMPLE, PAIRED, READ-ONLY. Both rewards are recomputed on the IDENTICAL transitions — the real
rollout steps the oracle-free runs actually trained on — so this is a paired audit of the training
signal, not a cross-campaign quality comparison. No training, no VQE, no checkpoint is touched.

    r_OF (t)  = S(E_t) - S(E_{t+1})              S = the empirical-frontier signed-log score
    r_FCI(t)  = log10(E_t - E0) - log10(E_{t+1} - E0)      the canonical E0-based log-error reward

Both are read straight out of `trajectory.jsonl`, which logs, per step, `frontier_score` = S(E_t)
(computed with the scale that was live at that moment, i.e. exactly what training consumed) and
`true_error` = |E_t - E0| in mHa (an E0 DIAGNOSTIC that was never fed to training). The two therefore
describe the same transition and differ only in the reference point: a moving, observed frontier
F_adopted versus the fixed exact ground state E0.

Sign convention: runner._returns builds r[t] = phi[t-1] - phi[t] with phi = the score, i.e. POSITIVE
reward = the energy went DOWN. Both quantities here use that same convention, so they are directly
comparable term by term.

⚠ WHAT COUNTS AS A FLAT TRANSITION — this determines whether the headline is right or nonsense.
S(E) and log10(E - E0) are BOTH strictly increasing in E, so on any transition where the energy
actually moves the two rewards CANNOT disagree in sign: that is an identity, not an empirical result,
and the audit's job is to confirm it numerically. Flatness must therefore be decided on the ENERGY,
not on whether a reward rounds to zero. `runner._log` rounds every logged float to 6 dp, and the two
fields do not survive that equally: `true_error` is in mHa (values ~1-1000, so 6 dp still resolves
1e-6 mHa) while `frontier_score` is a log-scale value ~0-3 whose 6th decimal is far coarser. On a
repeated identical energy the score difference rounds to exactly 0 while the true-error difference
keeps a meaningless ~1e-8 log wiggle. Calling those "sign disagreements" produced a spurious ~47%
disagreement rate on LiH-6q in the first draft of this script. They are excluded here and counted
separately as `res-lim` (resolution-limited).

WHAT THIS CAN AND CANNOT ANSWER
  CAN  reward-level agreement on the training distribution: Spearman, sign agreement, the signed
       distortion dr = r_OF - r_FCI, and how all three move with frontier lag / distance-to-frontier /
       refresh epoch / training phase.
  CANNOT  top-1 / top-k action agreement, or "FCI regret when the action is chosen by r_OF". Those
       need the COUNTERFACTUAL candidate set at a state (several actions from one prefix). Rollout
       logs contain exactly one taken action per state, and the one probe that does enumerate
       candidates (t1a_action_ranking.py) stores only per-prefix aggregates, not per-candidate
       energies. Producing them would require new VQE, which is out of scope. Reported as MISSING.

Stages: the budget is split into thirds by ITERATION (early/mid/late) so the reward audit can be read
against the training curve. The frontier moves fastest early, so a single pooled number would hide
the phase where the two signals disagree most.

Usage: python analysis/reward_fidelity_audit.py [--seeds 5] [--max-eps N]
       > outputs/main_results/reward_fidelity_audit.txt
"""
import argparse
import glob
import json
import math
import os

import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
CAP = {"LiH6q": 3750}          # locked reporting budget, in ITERATIONS
FLOOR = 1e-3                   # cfg.err_floor_mHa — the canonical reward's own clip


def _spearman(a, b):
    """Rank correlation with AVERAGE ranks for ties (scipy-equivalent). The project's t1a probe uses
    ordinal argsort-of-argsort ranks instead, which is wrong under ties; ties are common here because
    plateau steps repeat an energy exactly, so the correct tie handling is not optional."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _avg_rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):                       # average ranks within each tie group
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return r


def load_run(d, mol, max_eps=None):
    """-> dict of per-transition arrays over the run's real rollout steps.

    A transition is a consecutive (step t -> t+1) pair INSIDE one episode; episode boundaries are
    never crossed (the frontier reference is constant within an episode but can change between them).
    """
    f = f"{d}/trajectory.jsonl"
    if not os.path.exists(f):
        return None
    cap = CAP.get(mol)
    rof, rfci, iters, dmHa, terr, moved = [], [], [], [], [], []
    cur_ep, S, TE, EN, IT = None, [], [], [], None
    n_ep = 0

    def flush():
        if len(S) < 2:
            return
        s = np.asarray(S, float); te = np.asarray(TE, float); en = np.asarray(EN, float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(te))):
            return
        le = np.log10(np.clip(te, FLOOR, None))
        rof.extend(s[:-1] - s[1:])            # r_OF  = S(E_t) - S(E_{t+1})
        rfci.extend(le[:-1] - le[1:])         # r_FCI = log10 err_t - log10 err_{t+1}
        moved.extend(en[:-1] != en[1:])       # did the ENERGY move at logged precision?
        iters.extend([IT] * (len(s) - 1))
        terr.extend(te[:-1])
        dmHa.extend(s[:-1])                   # score of the ORIGIN state (proxy for distance to frontier)

    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        it = r.get("iter", -1)
        if cap is not None and it >= cap:
            break
        key = (it, r.get("ep"))
        if key != cur_ep:
            flush()
            n_ep += 1
            if max_eps and n_ep > max_eps:
                break
            cur_ep, S, TE, EN, IT = key, [], [], [], it
        fs, te, en = r.get("frontier_score"), r.get("true_error"), r.get("energy")
        if fs is None or te is None or en is None:
            S, TE, EN = [], [], []            # pre-scale-init episode: unusable, drop it whole
            continue
        S.append(float(fs)); TE.append(float(te)); EN.append(float(en))
    flush()
    if len(rof) < 100:
        return None
    return dict(rof=np.asarray(rof), rfci=np.asarray(rfci), it=np.asarray(iters),
                s0=np.asarray(dmHa), te0=np.asarray(terr), moved=np.asarray(moved, bool))


def frontier_lag(d, mol):
    """-> {iter: (F_adopted - E0) in mHa} from metrics.jsonl, joined to the audit by iteration.

    E0 is recovered from the run's own logs (best_E and best_err_mHa are the same circuit), so no
    Hamiltonian file is read and the value is guaranteed to be the one this run was scored against.
    """
    f = f"{d}/metrics.jsonl"
    if not os.path.exists(f):
        return {}, float("nan")
    cap = CAP.get(mol)
    rows = []
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if cap is not None and r.get("iter", -1) >= cap:
            break
        rows.append(r)
    E0 = float("nan")
    for r in rows:                            # E0 = best_E + best_err/1000 (variational: E >= E0)
        be, bm = r.get("best_E"), r.get("best_err_mHa")
        if be is not None and bm is not None and np.isfinite(be) and np.isfinite(bm):
            E0 = float(be) - float(bm) / 1000.0
    lag = {}
    for r in rows:
        fe = r.get("frontier_E")
        if fe is not None and np.isfinite(E0):
            lag[int(r["iter"])] = (float(fe) - E0) * 1000.0
    return lag, E0


def block(name, rof, rfci, moved):
    """Statistics on the transitions where the ENERGY actually moved (see the flatness note above).

    `flat`    energy identical at logged precision AND both rewards zero — a genuine plateau step.
    `res_lim` energy identical but the rewards disagree about that, i.e. one of the two fields kept a
              sub-precision wiggle. These are logging noise and are excluded from every statistic;
              they are reported so the reader can see how much of the stream they are.
    """
    keep = np.isfinite(rof) & np.isfinite(rfci)
    rof, rfci, moved = rof[keep], rfci[keep], moved[keep]
    if len(rof) < 3:
        return None
    flat = ~moved
    res_lim = float(np.mean(flat & ((rof != 0) | (rfci != 0))))
    if moved.sum() < 3:
        return dict(name=name, n=len(rof), n_moved=int(moved.sum()), rho=float("nan"),
                    sign=float("nan"), dr_mean=float("nan"), dr_sd=float("nan"),
                    dr_p95=float("nan"), flat=float(np.mean(flat)), res_lim=res_lim)
    a, b = rof[moved], rfci[moved]
    dr = a - b
    return dict(name=name, n=len(rof), n_moved=int(moved.sum()), rho=_spearman(a, b),
                sign=float(np.mean(np.sign(a) == np.sign(b))),
                dr_mean=float(dr.mean()), dr_sd=float(dr.std(ddof=1)),
                dr_p95=float(np.percentile(np.abs(dr), 95)),
                flat=float(np.mean(flat)), res_lim=res_lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-eps", type=int, default=0, help="0 = all episodes")
    a = ap.parse_args()
    print("=" * 116)
    print("ORACLE-FREE vs FCI REWARD — SAME-SAMPLE PAIRED AUDIT  (appendix §5; read-only, no new VQE)")
    print("  r_OF  = S(E_t) - S(E_t+1)      empirical-frontier signed-log score (what training consumed)")
    print("  r_FCI = log10(E_t-E0) - log10(E_t+1-E0)   canonical E0 log-error reward, recomputed offline")
    print("  Both on the SAME real rollout transitions of the SAME oracle-free runs. Positive = energy fell.")
    print("  All statistics are over transitions where the ENERGY MOVED at logged precision.")
    print("  rho uses AVERAGE ranks for ties. 'flat' = energy unchanged. 'res-lim' = energy unchanged")
    print("  but one reward field kept a sub-precision wiggle (logging noise; excluded everywhere).")
    print(f"  LiH6q truncated at the locked {CAP['LiH6q']}-iteration reporting budget.")
    print("=" * 116)
    summary = {}
    for mol in MOLS:
        ds = sorted(glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_of"))[: a.seeds]
        per_seed, lags = [], []
        for d in ds:
            R = load_run(d, mol, a.max_eps or None)
            if R is None:
                continue
            lag, E0 = frontier_lag(d, mol)
            it = R["it"]
            lo, hi = it.min(), it.max()
            thirds = [("early", it <= lo + (hi - lo) / 3),
                      ("mid", (it > lo + (hi - lo) / 3) & (it <= lo + 2 * (hi - lo) / 3)),
                      ("late", it > lo + 2 * (hi - lo) / 3)]
            row = {"seed": os.path.basename(d).split("_s")[-1].split("_")[0],
                   "all": block("all", R["rof"], R["rfci"], R["moved"]),
                   "E0": E0,
                   "lag_early": np.median([v for k, v in lag.items() if k <= lo + (hi - lo) / 3] or [np.nan]),
                   "lag_late": np.median([v for k, v in lag.items() if k > lo + 2 * (hi - lo) / 3] or [np.nan])}
            for nm, m in thirds:
                row[nm] = block(nm, R["rof"][m], R["rfci"][m], R["moved"][m])
            # distortion vs distance to frontier: bin the ORIGIN state's score S(E_t), on MOVED
            # transitions only. Quartile edges are computed on that same subset — using the full
            # stream's quantiles collapsed whole bins on plateau tasks, where the origin score is
            # massively tied, and printed a bare `nan`.
            mv = R["moved"]
            s0 = R["s0"][mv]; dr = (R["rof"] - R["rfci"])[mv]
            if len(s0) >= 8:
                qs = np.percentile(s0, [25, 50, 75])
                bins = (s0 <= qs[0], (s0 > qs[0]) & (s0 <= qs[1]),
                        (s0 > qs[1]) & (s0 <= qs[2]), s0 > qs[2])
                row["by_dist"] = [float(np.mean(np.abs(dr[m]))) if m.sum() else float("nan")
                                  for m in bins]
            else:
                row["by_dist"] = [float("nan")] * 4
            per_seed.append(row); lags.append(row["lag_late"])
        if not per_seed:
            print(f"\n--- {DISP[mol]}: NO USABLE DATA"); continue
        summary[mol] = per_seed
        print(f"\n--- {DISP[mol]}   {len(per_seed)} seeds   E0 = {per_seed[0]['E0']:.6f} Ha")
        print(f"{'stage':7}{'moved/seed':>12}{'rho(r_OF,r_FCI)':>18}{'sign agree':>12}"
              f"{'mean dr':>10}{'sd dr':>9}{'p95|dr|':>10}{'flat':>8}{'res-lim':>9}")
        for stage in ("early", "mid", "late", "all"):
            B = [s[stage] for s in per_seed if s.get(stage)]
            if not B:
                continue
            g = lambda k: np.array([b[k] for b in B], float)
            rh = g("rho")
            print(f"{stage:7}{int(np.mean(g('n_moved'))):>12,}"
                  f"{f'{np.nanmean(rh):.3f}±{np.nanstd(rh, ddof=1):.3f}':>18}"
                  f"{np.nanmean(g('sign')):>12.4f}"
                  f"{np.nanmean(g('dr_mean')):>+10.4f}{np.nanmean(g('dr_sd')):>9.4f}"
                  f"{np.nanmean(g('dr_p95')):>10.4f}{np.mean(g('flat')):>8.3f}"
                  f"{np.mean(g('res_lim')):>9.3f}")
        L = np.array([s["lag_early"] for s in per_seed], float)
        Ll = np.array([s["lag_late"] for s in per_seed], float)
        print(f"  frontier lag (F_adopted - E0), median over iters: early {np.nanmean(L):8.2f} mHa"
              f"   late {np.nanmean(Ll):8.2f} mHa")
        D = np.nanmean(np.array([s["by_dist"] for s in per_seed], float), 0)
        print(f"  mean |dr| by distance-to-frontier quartile of the ORIGIN state "
              f"(Q1 nearest -> Q4 furthest): "
              + "  ".join("     —" if not np.isfinite(v) else f"{v:.4f}" for v in D))

    print("\n" + "=" * 116)
    print("CROSS-TASK: does reward distortion track the quality cost of going oracle-free?")
    print("  quality cost from oracle_free_deviation.txt (independent campaigns; SECOND-LEVEL evidence)")
    print("=" * 116)
    COST = {"LiH4q": "+42%", "BeH2": "+0% (saturated)", "LiH6q": "+33%",
            "BeH2_8q": "+0% (saturated)", "BeH2_10q": "-40%"}
    print(f"{'task':12}{'rho late':>12}{'sd dr late':>12}{'p95|dr| late':>14}{'flat late':>11}"
          f"{'lag late (mHa)':>16}   Full OF-vs-canonical")
    for mol in MOLS:
        if mol not in summary:
            continue
        B = [s["late"] for s in summary[mol] if s.get("late")]
        if not B:
            continue
        print(f"{DISP[mol]:12}{np.nanmean([b['rho'] for b in B]):>12.3f}"
              f"{np.nanmean([b['dr_sd'] for b in B]):>12.4f}"
              f"{np.nanmean([b['dr_p95'] for b in B]):>14.4f}"
              f"{np.mean([b['flat'] for b in B]):>11.3f}"
              f"{np.nanmean([s['lag_late'] for s in summary[mol]]):>16.2f}   {COST[mol]}")

    print("\n" + "=" * 116)
    print("HOW TO READ / WHAT IS MISSING")
    print("=" * 116)
    print("  * This is the PRIMARY diagnostic for the oracle-free reward: same transitions, paired.")
    print("    The cross-campaign quality gap (oracle_free_deviation.txt) is SECOND-LEVEL evidence and")
    print("    must never be presented as a paired reward audit — the two campaigns differ in every")
    print("    stochastic draw, not only in the reward.")
    print("  * Sign agreement is expected to be EXACTLY 1.000 and is printed as a CHECK, not a finding: both")
    print("    S and log10(E-E0) are strictly increasing in E, so on any real energy change the two")
    print("    rewards agree by construction. Anything below 1.000 indicates a logging/precision defect")
    print("    in this audit, not a property of the reward. What the oracle-free signal can distort is")
    print("    the MAGNITUDE (dr, and the rho below 1), never the direction.")
    print("  * MISSING, and not recoverable without new VQE:")
    print("      - top-1 / top-k action agreement between r_OF and r_FCI;")
    print("      - FCI regret of the action selected by r_OF.")
    print("    Both need several actions evaluated from the SAME prefix. Rollout logs hold one taken")
    print("    action per state, and t1a_action_ranking.py enumerates candidates but persists only")
    print("    per-prefix aggregates (rho / regret / recall), never the per-candidate energies. The")
    print("    fix for future runs is one line in that probe: dump (action, energy) per candidate.")
    print("  * 'flat' counts transitions where the energy did not move at all (an ansatz-plateau step).")
    print("    On the hard tasks it is 63-90% of the stream, which is the same plateau that")
    print("    plateau_diagnostic.txt reports — quote them together, and note that the reward audit")
    print("    therefore describes a MINORITY of the transitions on those molecules.")
    print("  * Residual sign disagreement of ~1e-3 is float rounding of the 6-dp logged score on")
    print("    near-zero energy changes, not a reward property. It is left visible rather than")
    print("    thresholded away so the check stays a check.")


if __name__ == "__main__":
    main()
