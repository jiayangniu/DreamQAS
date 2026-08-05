"""Matched-error VQE gap (parallel-descent gap) for the ORACLE-FREE speed figures.

At a fixed training-error level y (mHa), read off each method's cumulative real VQE calls to first reach y
(sustained, 3 consecutive reported points) from the SAME W=100 curves the figures draw, and report

    gap(y) = cum_VQE_baseline(y) / cum_VQE_Full(y)          (>1 = DreamQAS needs fewer real VQE)

This is the horizontal distance between the two descending curves, NOT the seed-level rq2_speedup (which
targets a single Q_b = baseline final-window floor). Reported over a LADDER of y so the reader can see
whether the curves are parallel (gap ~ constant) or converging/diverging; the y used in the figure arrow
(GAP_Y) is starred. Median-curve based (same aggregation as the figure).

Read-only: reuses plot_rq2_w100_of (oracle-free dq_dirs override + CAP_ITERS + cross()). No training, no VQE.
Usage: python analysis/of_gap_table.py > outputs/main_results/oracle_free_gap_table.txt
"""
import os
import numpy as np
import plot_rq2_w100_of as P

MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
# error ladder per molecule (mHa), coarse->fine; only levels both curves actually reach are reported
LADDER = {
    "LiH4q":    [10.0, 5.0, 2.0, 1.0, 0.5],
    "BeH2":     [10.0, 5.0, 2.0, 1.0, 0.5],
    "LiH6q":    [50.0, 30.0, 20.0, 15.0, 12.0],
    "BeH2_8q":  [20.0, 10.0, 5.0, 3.0, 2.5],
    "BeH2_10q": [30.0, 15.0, 10.0, 5.0, 3.0],
}
BASES = ["No-imag", "RLQAS"]


def seed_level_gap(rng, n_boot=3000):
    """The headline gaps above are read off the CROSS-SEED MEDIAN curve — a point estimate with no
    uncertainty attached. This block recomputes the same matched-error quantity with the SEED as the
    experimental unit: per seed, the cumulative real VQE to first sustainably reach y; then the ratio
    of medians with an unpaired percentile bootstrap over the seeds that actually REACHED y.

    Reporting convention (locked): right-censored seeds are COUNTED and excluded from the median,
    never imputed. A cell whose reach is below 3/5 is not a reportable speedup.
    """
    print("\n" + "=" * 104)
    print("SEED-LEVEL matched-error gap — the same quantity with the SEED as the experimental unit")
    print("  median over REACHED seeds, 3000x unpaired bootstrap. reach = seeds that ever reach y.")
    print("  ⚠ Cite THIS for uncertainty; the median-curve table above is the point estimate only.")
    print("=" * 104)
    print(f"{'task':10s}{'baseline':9s}{'y(mHa)':>8}{'base VQE':>11}{'reach':>7}"
          f"{'Full VQE':>11}{'reach':>7}{'gap':>8}{'95% CI':>16}")
    for mol in MOLS:
        cf = P.seed_curves("Full", mol)
        if len(cf) < 2:
            continue
        for b in BASES:
            cb = P.seed_curves(b, mol)
            if len(cb) < 2:
                continue
            for y in LADDER[mol]:
                vf = np.array([P.cross(c, y) for c in cf])
                vb = np.array([P.cross(c, y) for c in cb])
                rf, rb = vf[np.isfinite(vf)], vb[np.isfinite(vb)]
                if len(rf) < 2 or len(rb) < 2:
                    continue
                mf, mb = float(np.median(rf)), float(np.median(rb))
                bs = [np.median(rng.choice(rb, len(rb), True)) / np.median(rng.choice(rf, len(rf), True))
                      for _ in range(n_boot)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                flag = "" if (len(rf) >= 3 and len(rb) >= 3) else "  <3 reached: not reportable"
                print(f"{mol:10s}{b:9s}{y:>8g}{mb:>11,.0f}{f'{len(rb)}/{len(cb)}':>7}"
                      f"{mf:>11,.0f}{f'{len(rf)}/{len(cf)}':>7}{mb / mf:>7.2f}x"
                      f"{f'[{lo:.2f},{hi:.2f}]':>16}{flag}")
        print()
    print("  A CI spanning 1.00 means the gap is NOT established at 5 seeds — report it as such.")


def cap_sensitivity():
    """Why the oracle-free figure annotates a matched-error gap instead of the Q_b seed-level speedup:
    on LiH6q the Q_b target itself moves with the displayed training budget, and at the full budget the
    comparison becomes heavily right-censored (few Full seeds ever reach the deeper baseline floor)."""
    rng = np.random.default_rng(7)
    keep = dict(P.CAP_ITERS)
    print("\n" + "=" * 104)
    print("LiH-6q — Q_b-based seed-level speedup vs the DISPLAYED training-budget cap (why §1b uses the gap)")
    print("  Q_b = median baseline final-window mean -> it MOVES with the cap; 'reach' = seeds that ever")
    print("  cross Q_b (right-censored seeds excluded from the median, per the reporting convention).")
    print("=" * 104)
    print(f"{'cap(iters)':>11}{'baseline':>10}{'Q_b(mHa)':>10}{'Full VQE':>11}{'reach':>7}"
          f"{'base VQE':>11}{'reach':>7}{'speedup':>9}{'95% CI':>15}")
    for cap in (3750, 5000, 6000, 7500):
        P.CAP_ITERS = {"LiH6q": cap}
        for b in ("No-imag", "RLQAS"):
            r = P.speedup_row("LiH6q", b, rng)
            if r is None or not np.isfinite(r["sp"]):
                print(f"{cap:>11}{b:>10}   (unavailable / never reaches Q_b)"); continue
            reach_f = f"{r['rf']}/{r['nf']}"
            reach_b = f"{r['rb']}/{r['nb']}"
            ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]"
            print(f"{cap:>11}{b:>10}{r['Q']:>10.2f}{r['mf']:>11.0f}{reach_f:>7}"
                  f"{r['mb']:>11.0f}{reach_b:>7}{r['sp']:>8.2f}x{ci:>15}")
    P.CAP_ITERS = keep
    print("\nREADING: the 5000/6000 rows agree (1.8x, Full 5/5 reach); the 7500 row's 0.81x is a CENSORING")
    print("artifact, not a budget-shopping difference — extending to 7500 lowers Q_b to No-imag's deeper")
    print("floor (11.4 -> 10.6 mHa), which only 2/5 Full seeds ever reach, so the median is computed over 2")
    print("seeds. Per the project's reporting convention (report only achieved crossings, never censored")
    print("guesses), a 2/5-reach cell is not a reportable speedup. The matched-error gap above avoids the")
    print("problem entirely by fixing the error LEVEL instead of a moving target.")


def main():
    print("=" * 104)
    print("ORACLE-FREE matched-error VQE gap (parallel-descent gap) — DreamQAS vs baseline at a FIXED error")
    print("  curves = the SAME W=100 training-time moving-mean median curves as the oracle-free speed figures")
    print("  x(y) = cumulative real training VQE calls to first reach error y, sustained 3 reported points")
    print("  gap(y) = x_baseline(y) / x_Full(y)   (>1 = DreamQAS reaches the SAME error with fewer real VQE)")
    print("  *  = the level annotated by the arrow in the figure (GAP_Y).  '-' = that curve never reaches y.")
    print("  NOT the seed-level rq2_speedup (which uses one target Q_b = baseline final-window floor).")
    print(f"  LiH6q training budget capped at {P.CAP_ITERS.get('LiH6q')} iters (=20000 episodes) for this figure.")
    print("=" * 104)
    for mol in MOLS:
        aF = P.aggregate("Full", mol)
        if aF is None:
            print(f"\n{mol}: no Full curves"); continue
        aggs = {b: P.aggregate(b, mol) for b in BASES}
        ns = f"nFull={aF[4]}" + "".join(f" n{b}={aggs[b][4]}" for b in BASES if aggs[b])
        print(f"\n--- {P.DISP[mol]}  [{ns}]")
        print(f"{'y (mHa)':>9}{'Full VQE':>12}" + "".join(f"{b+' VQE':>14}{'gap':>8}" for b in BASES))
        for y in LADDER[mol]:
            xF = P.cross((aF[0], aF[1]), y)
            star = "*" if abs(P.GAP_Y.get(mol, -1) - y) < 1e-9 else " "
            row = f"{y:>8.1f}{star}" + (f"{xF:>12.0f}" if np.isfinite(xF) else f"{'-':>12}")
            for b in BASES:
                a = aggs[b]
                xB = P.cross((a[0], a[1]), y) if a else np.nan
                row += f"{xB:>14.0f}" if np.isfinite(xB) else f"{'-':>14}"
                row += (f"{xB / xF:>7.1f}x" if np.isfinite(xB) and np.isfinite(xF) and xF > 0 else f"{'-':>8}")
            print(row)
    print("\n" + "=" * 104)
    print("READING: a roughly CONSTANT gap down the ladder = the two curves descend in parallel, i.e. a")
    print("sustained VQE-efficiency factor rather than a one-off head start. A gap that shrinks toward the")
    print("bottom rows = the baseline is catching up near its floor (report the level with the number).")
    print("Saturated molecules (BeH2-6q/8q Full at the ansatz floor) give few valid rows by construction.")
    print("Figure arrows use the starred level only; cite THIS table for any numeric matched-error claim.")
    seed_level_gap(np.random.default_rng(7))
    cap_sensitivity()


if __name__ == "__main__":
    main()
