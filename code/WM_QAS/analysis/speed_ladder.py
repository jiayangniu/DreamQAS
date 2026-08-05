"""Four-stage ladder on the SPEED axis — where does the VQE-efficiency gain actually come from?

The existing ladder (`ladder_table.txt`) decomposes final POLICY QUALITY into
    RLQAS -> No-imag  (representation + replay + dense reward)
          -> NoDAG    (imagination)
          -> Full     (DAgger verification)
It says imagination contributes little quality on the hard molecules (imag x1.1 on BeH2-8q). But the
headline VQE-efficiency numbers are Full-vs-No-imag matched-error GAPS, which bundle imagination AND
DAgger — and on BeH2-8q that gap is 9.0x at 5 mHa while the quality gain is ~nil. A quality ladder
cannot attribute a speed number. This does the same decomposition on the axis the claim lives on.

    gap_stage(y) = cum_real_VQE_lower(y) / cum_real_VQE_upper(y)      (>1 = the upper stage is faster)

read off the SAME W=100 training curves as Figure 2 (median over seeds, cumulative real VQE including
DAgger verification for the arms that pay it). Stage gaps multiply to the end-to-end gap by
construction, so the decomposition is exact at each error level, not a regression.

WHY THIS IS THE RIGHT ATTRIBUTION FOR BeH2-8q / 10q
---------------------------------------------------
Those two tasks are exactly where the cross-circuit WM probes are NOT MEASURABLE (their true error is
pinned at the ansatz plateau — `plateau_diagnostic.txt`), so "the WM ranks well/badly there" cannot be
established either way. What CAN be established is which stage of the pipeline buys the speed. That is
a controlled comparison between trained arms and needs no probe at all.

⚠ Accounting asymmetry that must stay in any caption: `dagger=1` (Full) counts verification VQE inside
`vqe_calls`; `dagger=0` (NoDAG) books the same probe into a separate `calib_vqe_calls`. So the
NoDAG->Full stage charges Full for verification while NoDAG is not charged for its diagnostic — the
DAgger stage gap is therefore a LOWER bound on DAgger's cost, i.e. conservative against DAgger.

Usage: python analysis/speed_ladder.py > outputs/main_results/speed_ladder.txt
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_rq2_w100_of as P          # W=100 curve machinery (curve/cross/dq_episodes_true_vqe)

OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
LADDER = [("RLQAS",   f"{OF}/dreamqas_rlqas/baseline_analysis/G0_v2_{{m}}/seed_*"),
          ("No-imag", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag"),
          ("NoDAG",   f"{OF}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag"),
          ("Full",    f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of")]
STAGE = [("repr+replay", "RLQAS", "No-imag"), ("imagination", "No-imag", "NoDAG"),
         ("DAgger", "NoDAG", "Full")]
LEVELS = {"LiH4q": [10.0, 5.0, 2.0, 1.0], "BeH2": [10.0, 5.0, 2.0, 1.0],
          "LiH6q": [50.0, 30.0, 20.0, 15.0], "BeH2_8q": [20.0, 10.0, 5.0, 3.0],
          "BeH2_10q": [30.0, 15.0, 10.0, 5.0]}
CAP = {"LiH6q": 3750}                 # locked common reporting budget (15,000 episodes)


def median_curve(pattern, mol):
    cvs = []
    for d in sorted(glob.glob(pattern.format(m=mol))):
        eps = P.dq_episodes_true_vqe(d)
        cap = CAP.get(mol)
        if cap and eps:
            eps = eps[: cap * 4]
        if eps and len(eps) > P.W:
            cvs.append(P.curve(eps))
    if len(cvs) < 2:
        return None
    L = min(len(x) for x, _ in cvs)
    return np.median([x[:L] for x, _ in cvs], 0), np.median([y[:L] for _, y in cvs], 0)


def main():
    print("=" * 118)
    print("FOUR-STAGE LADDER ON THE SPEED AXIS — which stage buys the real-VQE efficiency?")
    print("  cell = cumulative real training VQE to first sustainably reach that error (W=100 median curve)")
    print("  stage gap = VQE(lower stage) / VQE(upper stage).  >1 = the upper stage gets there cheaper.")
    print("  Stage gaps multiply to the end-to-end RLQAS->Full gap exactly. '—' = never reaches that level.")
    print("  ⚠ NoDAG->Full charges Full for verification VQE but not NoDAG for its diagnostic probe, so the")
    print("    DAgger column is CONSERVATIVE AGAINST DAgger (a lower bound on its cost).")
    print("=" * 118)
    for mol in MOLS:
        curves = {n: median_curve(p, mol) for n, p in LADDER}
        miss = [n for n, c in curves.items() if c is None]
        print(f"\n--- {DISP[mol]}" + (f"   (missing curves: {', '.join(miss)})" if miss else ""))
        print(f"{'error y (mHa)':>15}" + "".join(f"{n:>13}" for n, _ in LADDER)
              + "  |" + "".join(f"{s:>15}" for s, _, _ in STAGE) + f"{'end-to-end':>13}")
        for y in LEVELS[mol]:
            x = {}
            for n, _ in LADDER:
                x[n] = P.cross(curves[n], y) if curves[n] else np.nan
            cells = "".join(f"{('—' if not np.isfinite(x[n]) else f'{x[n]:,.0f}'):>13}" for n, _ in LADDER)
            gaps = []
            for _, lo, hi in STAGE:
                g = x[lo] / x[hi] if (np.isfinite(x[lo]) and np.isfinite(x[hi]) and x[hi] > 0) else np.nan
                gaps.append("—" if not np.isfinite(g) else f"{g:.2f}x")
            e2e = (x["RLQAS"] / x["Full"] if (np.isfinite(x["RLQAS"]) and np.isfinite(x["Full"])
                                              and x["Full"] > 0) else np.nan)
            print(f"{y:>15g}{cells}  |" + "".join(f"{g:>15}" for g in gaps)
                  + f"{('—' if not np.isfinite(e2e) else f'{e2e:.2f}x'):>13}")
    print("\n" + "=" * 118)
    print("HOW TO READ")
    print("=" * 118)
    print("  * This attributes SPEED, which is what the paper's VQE-efficiency claim is about. Do not")
    print("    read it as a quality decomposition — `ladder_table.txt` is that, and the two disagree on")
    print("    purpose by design (imagination can buy speed while buying almost no converged quality;")
    print("    that is the dose finding in IMAGINATION_STRENGTH.md §3S restated on the ladder).")
    print("  * A stage gap near 1.0 means that stage is not what makes the pipeline fast at that error")
    print("    level. A gap below 1.0 means the stage COSTS real VQE there — report it, do not hide it.")
    print("  * Gaps are level-dependent (see the rows). Never quote a single number without its y.")
    print("  * Median-curve readings, no CI. For seed-level CIs use the Q_b-based rq2_speedup pipeline;")
    print("    this table is a decomposition, not a significance test.")


if __name__ == "__main__":
    main()
