"""End-to-end WALL-CLOCK breakdown: how much of training time is real-VQE vs WM-training vs imagination?

VQE-call counts alone cannot answer "does imagination's extra compute eat the VQE savings?" — that needs
per-phase seconds. Training runs launched with `--timing 1` log cumulative per-phase seconds into
metrics.jsonl (see Runner._tick/_tock):

    t_vqe_real      real-VQE optimisation charged to the acceleration budget (rollout + DAgger verify)
    t_vqe_calib     diagnostic-only calibration VQE (NoDAG arm; not in the budget)
    t_rollout_other real-rollout cost minus its VQE = actor forward + WM forward + buffer
    t_imagine       imagined rollouts + imagined-loss construction (the imagination "tax")
    t_wm_train      fidelity check + WM refresh (GRU ensemble training)
    t_actor         REINFORCE loss + backward + optimiser step
    t_other         wall minus the above (env setup, logging, checkpointing)

Reports (a) the phase share per arm, and (b) the END-TO-END time to reach a matched error level, i.e.
    time_to(y) = t_wall at the first iteration whose 100-episode moving-mean episode-best <= y,
which is the honest analogue of the matched-error VQE gap: if Full needs fewer VQE calls but pays for
imagination, this is where that trade-off actually lands.

Usage: python analysis/timing_table.py [--levels 5,2,1] > outputs/main_results/oracle_free_timing.txt
"""
import argparse
import glob
import json
import os

import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1/timing"
PHASES = ["t_vqe_real", "t_rollout_other", "t_wm_train", "t_imagine", "t_actor", "t_vqe_calib", "t_other"]
LABEL = {"t_vqe_real": "real VQE", "t_rollout_other": "rollout(actor+WM fwd)", "t_wm_train": "WM training",
         "t_imagine": "imagination", "t_actor": "actor update", "t_vqe_calib": "calib VQE(diag)",
         "t_other": "other"}
ARMS = [("Full", "gru_energy_surrogate_{m}_s*_tfull"), ("No-imag", "gru_energy_none_{m}_s*_tnoimag"),
        ("Full(short)", "gru_energy_surrogate_{m}_s*_tshort"),
        ("No-imag(short)", "gru_energy_none_{m}_s*_tshortni")]
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
W = 100


def rows_of(d):
    f = f"{d}/metrics.jsonl"
    if not os.path.exists(f):
        return []
    return [json.loads(l) for l in open(f) if l.strip()]


def phase_share(rows, since=0):
    """-> (dict phase->seconds, wall, n_iter) over iterations [since, end].

    The counters are cumulative, so a window is a difference of two rows. `since>0` matters because the
    run is NOT stationary: for LiH-4q the mean episode depth is ~10/40 gates for the first ~125 iters and
    only jumps to ~35 once the policy stops dead-ending — measured, see `oracle_free_timing.txt` header.
    Real-VQE cost scales with that depth, so a whole-run average silently mixes a cheap shallow warm-up
    into the steady state. Report the steady-state window for any per-phase claim.
    """
    rows = [r for r in rows if "t_wall" in r]
    if not rows:
        return None
    last = rows[-1]
    base = None
    for r in rows:
        if r["iter"] >= since:
            base = r
            break
    if base is None or base is last:
        return None
    d = {p: float(last.get(p, 0.0)) - float(base.get(p, 0.0)) for p in PHASES}
    wall = float(last["t_wall"]) - float(base["t_wall"])
    if wall <= 0:
        return None
    return d, wall, int(last["iter"]) - int(base["iter"])


def ep_curve(d):
    """per-episode best error -> W=100 trailing mean, paired with the iteration's cumulative t_wall."""
    tp, mp = f"{d}/trajectory.jsonl", f"{d}/metrics.jsonl"
    if not (os.path.exists(tp) and os.path.exists(mp)):
        return None
    it_t = {}
    for l in open(mp):
        if l.strip():
            r = json.loads(l)
            if "t_wall" in r:
                it_t[r["iter"]] = float(r["t_wall"])
    eps, cur, cmin, cit = [], None, np.inf, None
    for l in open(tp):
        if not l.strip():
            continue
        r = json.loads(l)
        k = (r["iter"], r["ep"])
        if k != cur:
            if cur is not None:
                eps.append((cit, cmin))
            cur, cmin, cit = k, np.inf, r["iter"]
        te = r.get("true_error")
        if te is not None:
            cmin = min(cmin, te)
    if cur is not None:
        eps.append((cit, cmin))
    eps = [(it, b) for it, b in eps if it in it_t and np.isfinite(b)]
    if len(eps) <= W:
        return None
    b = np.array([e[1] for e in eps], float)
    t = np.array([it_t[e[0]] for e in eps], float)
    cs = np.cumsum(np.insert(b, 0, 0.0))
    i = np.arange(len(b))
    lo = np.maximum(0, i - W + 1)
    return t, (cs[i + 1] - cs[lo]) / (i + 1 - lo)


def time_to(curve, y):
    t, m = curve
    hit = np.where(m <= y)[0]
    return float(t[hit[0]]) if len(hit) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="5,2,1")
    ap.add_argument("--since", type=int, default=300,
                    help="steady-state window start iter (default 300: past the ~125-iter warm-up where "
                         "mean episode depth jumps from ~10/40 to ~35/40 and imagination switches on)")
    a = ap.parse_args()
    levels = [float(x) for x in a.levels.split(",")]
    print("=" * 108)
    print("END-TO-END WALL-CLOCK BREAKDOWN (oracle-free, --timing 1 runs)")
    print("  ⚠ PRELIMINARY: measured while the Q2 horizon training occupied the same node. ABSOLUTE seconds")
    print("  are inflated and must NOT be quoted as wall-clock speed. The per-phase SHARES and the")
    print("  Full-vs-No-imag RATIO are the usable quantities (both arms ran concurrently under the same load).")
    print("  Phase timing forces a cuda.synchronize at each boundary, so a timed run is also slightly")
    print("  slower than an untimed one — another reason to read shares, not absolutes.")
    print("=" * 108)
    print(f"  ⚠ NON-STATIONARY per-iteration cost: mean episode depth on LiH-4q is ~10/40 gates for the")
    print(f"  first ~125 iterations and only reaches ~35-38/40 afterwards, so early iterations are ~3x")
    print(f"  cheaper in VQE. Shares are therefore reported over a STEADY-STATE window (iter >= {a.since})")
    print("  as well as whole-run; quote the steady-state row. (Depth measured from trajectory.jsonl.)")
    print("=" * 108)
    for mol in MOLS:
        got = False
        for arm, pat in ARMS:
            dirs = sorted(glob.glob(f"{C}/{pat.format(m=mol)}"))
            for tag, since in (("whole-run", 0), (f"iter>={a.since}", a.since)):
                per = [phase_share(rows_of(d), since) for d in dirs]
                per = [p for p in per if p]
                if not per:
                    continue
                if not got:
                    print(f"\n--- {mol}")
                    print(f"{'arm':16s}{'window':>12}{'seeds':>6}{'iters':>7}{'wall(s)':>10}  " +
                          "".join(f"{LABEL[p][:13]:>15}" for p in PHASES))
                    got = True
                shares = np.array([[d[p] / w * 100 for p in PHASES] for d, w, _n in per])
                wall = np.mean([w for _d, w, _n in per])
                nit = int(np.mean([n for _d, _w, n in per]))
                print(f"{arm:16s}{tag:>12}{len(per):>6}{nit:>7}{wall:>10.0f}  " +
                      "".join(f"{shares[:, i].mean():>14.1f}%" for i in range(len(PHASES))))
        if not got:
            print(f"\n--- {mol}: no timed runs yet")
    # ---- DAgger verification share: reconstructed, NOT separately timed ----
    # `_vqe_circuit(count_budget=True)` bills to t_vqe_real together with the rollout VQE, because in the
    # accounting they are the same budget. The verification CALL count is measurable (calibration.jsonl
    # candidate lengths), and the two populations have similar circuit depth, so a linear split of
    # t_vqe_real is a fair reconstruction — labelled as such, never as a measurement.
    print("\n" + "=" * 108)
    print("DAgger VERIFICATION SHARE  [call count = measured; time = RECONSTRUCTED by linear split]")
    print("  Trigger: `imagination == surrogate and it % calib_every == 0 and len(buf) >= 8` — note this is")
    print("  NOT gated on imag_on, so verification already spends VQE during the imagination warm-up.")
    print("  Cost is structural: calib_n_top+calib_n_disagree = 10 imagined circuits, each REPLAYED from")
    print("  the empty circuit to full depth, i.e. ~depth real VQE calls apiece, every calib_every iters.")
    print("=" * 108)
    print(f"{'mol':10s}{'seeds':>6}{'verify/total calls':>20}{'verify depth':>14}{'rollout depth':>15}"
          f"{'t_vqe_real%':>13}{'=> verify % of wall':>21}")
    for mol in MOLS:
        rows = []
        for d in sorted(glob.glob(f"{C}/gru_energy_surrogate_{mol}_s*_tfull")):
            m, cal = f"{d}/metrics.jsonl", f"{d}/calibration.jsonl"
            if not (os.path.exists(m) and os.path.exists(cal)):
                continue
            mr = [json.loads(l) for l in open(m) if l.strip()]
            if not mr or "t_wall" not in mr[-1]:
                continue
            tot = mr[-1]["vqe_calls"]
            vl = [int(json.loads(l)["len"]) for l in open(cal)
                  if l.strip() and json.loads(l).get("selection") != "SUMMARY" and json.loads(l).get("len")]
            if not tot or not vl:
                continue
            rows.append((sum(vl) / tot, float(np.mean(vl)), mr[-1]["t_vqe_real"] / mr[-1]["t_wall"]))
        if not rows:
            continue
        fr = float(np.mean([r[0] for r in rows])); vd = float(np.mean([r[1] for r in rows]))
        tv = float(np.mean([r[2] for r in rows]))
        print(f"{mol:10s}{len(rows):>6}{fr * 100:>19.1f}%{vd:>14.1f}{'(see above)':>15}"
              f"{tv * 100:>12.1f}%{fr * tv * 100:>20.1f}%")
    print("  The remainder of t_vqe_real is the training rollout's own VQE.")

    # ---- what the WM apparatus actually costs -------------------------------------------------
    # Folding verification into "real VQE" understates the price of the WM: DAgger VQE is paid ONLY by
    # the imagination pipeline (No-imag never calls calibrate()), so it belongs on the WM's bill even
    # though it is spent on real VQE. Report both an ABSOLUTE cost (everything the WM apparatus consumes)
    # and the DIFFERENTIAL vs No-imag (what imagination + verification ADD), because the WM's GRU is the
    # actor's state encoder in every variant — its training and forward passes are NOT differential.
    print("\n" + "=" * 108)
    print("COST OF THE WM APPARATUS  — verification listed separately, not hidden inside 'real VQE'")
    print("=" * 108)
    print(f"{'mol':10s}{'DAgger verify':>15}{'imagination':>13}{'WM training':>13}{'rollout WM fwd':>16}"
          f"{'ABS total':>11}{'DIFF vs No-imag':>17}")
    for mol in MOLS:
        f_rows = [phase_share(rows_of(d), a.since)
                  for d in sorted(glob.glob(f"{C}/gru_energy_surrogate_{mol}_s*_tfull"))]
        n_rows = [phase_share(rows_of(d), a.since)
                  for d in sorted(glob.glob(f"{C}/gru_energy_none_{mol}_s*_tnoimag"))]
        f_rows = [r for r in f_rows if r]; n_rows = [r for r in n_rows if r]
        if not f_rows:
            continue
        def sh(rows, key):
            return float(np.mean([d[key] / w * 100 for d, w, _n in rows]))
        imag, wmt, rof = sh(f_rows, "t_imagine"), sh(f_rows, "t_wm_train"), sh(f_rows, "t_rollout_other")
        # verification share (reconstructed, as above)
        vs = []
        for d in sorted(glob.glob(f"{C}/gru_energy_surrogate_{mol}_s*_tfull")):
            m, cal = f"{d}/metrics.jsonl", f"{d}/calibration.jsonl"
            if not (os.path.exists(m) and os.path.exists(cal)):
                continue
            mr = [json.loads(l) for l in open(m) if l.strip()]
            if not mr or "t_wall" not in mr[-1] or not mr[-1]["vqe_calls"]:
                continue
            vl = [int(json.loads(l)["len"]) for l in open(cal)
                  if l.strip() and json.loads(l).get("selection") != "SUMMARY" and json.loads(l).get("len")]
            if vl:
                vs.append(sum(vl) / mr[-1]["vqe_calls"] * mr[-1]["t_vqe_real"] / mr[-1]["t_wall"] * 100)
        ver = float(np.mean(vs)) if vs else float("nan")
        absolute = ver + imag + wmt + rof
        diff = ver + imag        # No-imag pays neither; it DOES pay WM training and the WM forward pass
        print(f"{mol:10s}{ver:>14.1f}%{imag:>12.1f}%{wmt:>12.1f}%{rof:>15.1f}%{absolute:>10.1f}%{diff:>16.1f}%")
    print("\n  ABS  = every phase the WM apparatus consumes, including the real VQE spent verifying it.")
    print("  DIFF = what imagination + verification ADD over No-imag. WM training and the rollout WM")
    print("         forward pass are NOT in DIFF: the WM's GRU is the actor's state encoder in EVERY")
    print("         variant, so No-imag pays those too (measured: its shares match Full's to <1 point).")
    print("  Read DIFF against the matched-error VQE gap: that is the price paid for the VQE saving.")
    print("  ⚠ VERIFICATION IS THE LARGEST SINGLE ITEM — larger than imagination. Since DAgger has no")
    print("    CI-clean standalone VQE-efficiency gain, present it as the price of reliability")
    print("    (collapse-prevention + keeping the WM honest), never as part of a speed claim.")

    print("\n" + "=" * 108)
    print("END-TO-END TIME TO A MATCHED ERROR LEVEL (seconds of training wall-clock, median over seeds)")
    print("  y = training episode-best, W=100 trailing mean — the same curve as the matched-error VQE gap,")
    print("  with the x-axis swapped from VQE calls to seconds. ratio = No-imag / Full (>1 = Full faster).")
    print("=" * 108)
    print(f"{'mol':10s}{'y(mHa)':>8}{'Full(s)':>12}{'No-imag(s)':>12}{'ratio':>9}   note")
    for mol in MOLS:
        cf = [ep_curve(d) for d in sorted(glob.glob(f"{C}/gru_energy_surrogate_{mol}_s*_tfull"))]
        cn = [ep_curve(d) for d in sorted(glob.glob(f"{C}/gru_energy_none_{mol}_s*_tnoimag"))]
        cf = [c for c in cf if c]; cn = [c for c in cn if c]
        if not cf or not cn:
            print(f"{mol:10s}   (needs FULL-budget timed runs for both arms — short probes cannot reach these levels)")
            continue
        for y in levels:
            tf = np.array([time_to(c, y) for c in cf]); tn = np.array([time_to(c, y) for c in cn])
            rf, rn = tf[np.isfinite(tf)], tn[np.isfinite(tn)]
            if not len(rf) or not len(rn):
                print(f"{mol:10s}{y:>8.2f}{'-':>12}{'-':>12}{'-':>9}   censored: reach {len(rf)}/{len(cf)} vs {len(rn)}/{len(cn)}")
                continue
            mf, mn = float(np.median(rf)), float(np.median(rn))
            print(f"{mol:10s}{y:>8.2f}{mf:>12.0f}{mn:>12.0f}{mn / mf:>8.2f}x   reach {len(rf)}/{len(cf)} vs {len(rn)}/{len(cn)}")
    print("\nREAD: the imagination share is the price paid for the VQE savings. Compare it against the")
    print("matched-error VQE gap (oracle_free_gap_table.txt): if imagination costs x% of wall-clock while")
    print("cutting real VQE by a factor g, the end-to-end win survives only if the VQE share dominates.")
    print("Report the phase shares WITH the caveat above; do not convert them into a hardware claim —")
    print("these are state-vector simulations, so the VQE share here is simulation cost, not QPU time.")


if __name__ == "__main__":
    main()
