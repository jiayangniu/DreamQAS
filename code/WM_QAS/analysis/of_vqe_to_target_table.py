"""APPENDIX TABLE: cumulative training VQE calls to first sustainably reach a FIXED error target,
for the five VQE-comparable methods, reported per seed and aggregated as median [min-max] (reach k/5).

WHY THIS EXISTS (and how it differs from the two speed numbers already in outputs/)
-----------------------------------------------------------------------------------
Two different estimators are already in the outputs directory and they are NOT interchangeable:

  rq2_speedup_w100.txt   target = Q_b, i.e. each BASELINE's OWN final-window level (a different
                         target per row);  per-seed sustained crossing;  median over REACHED seeds;
                         speedup = median(base)/median(Full) = ratio of medians, unpaired;
                         3000x unpaired bootstrap CI.
  speed_ladder.txt       target = FIXED error levels;  crossing read off the CROSS-SEED MEDIAN CURVE
                         (one curve, no per-seed crossing);  no reach counts, no CI.

Reading a threshold off the median curve is not the same statistic as the median of the per-seed
thresholds, and the median curve cannot show that a level was reached by only 2 of 5 seeds. This
table therefore uses the estimator with explicit censoring -- per-seed crossing -- but on the FIXED
targets, so that all five methods are compared against one line per task, which is what an appendix
supporting a speedup claim has to show.

TARGETS ARE NOT CHOSEN HERE. `LEVELS` is imported verbatim from speed_ladder.py, where the per-task
ladders were fixed before this table existed. No post-hoc threshold picking.

ESTIMATOR
  per seed:  training episode-best (min true post-VQE error over the episode's prefixes)
             -> trailing W=100-episode mean -> downsample every 50 episodes
             -> first reported point after which the smoothed curve stays <= target for 3
                consecutive points (sustained crossing; `plot_rq2_w100_of.cross`)
  x-axis:    cumulative REAL training VQE calls. DreamQAS-family reads metrics.jsonl `vqe_calls`,
             which INCLUDES DAgger verification VQE (trajectory step counts undercount Full by ~11%).
             Frozen-evaluation VQE is excluded (separate counter).
  aggregate: median over the seeds that REACHED, plus [min-max] of those seeds, plus reach k/n.
             Right-censored seeds are NEVER imputed to the budget and never enter the median.
  suppress:  a cell with reach < MIN_REACH is printed as `—` (the project's censoring convention:
             report only what was actually achieved). `—` means "not reached on enough seeds within
             the budget", which is a RESULT, not missing data.
  left edge: reported points whose trailing window is NOT YET FULL are DISCARDED before the crossing
             test. `plot_rq2_w100_of.curve` divides by `min(W, i+1)`, so the first reported point
             (episode index 0) is a 1-episode mean and the second (index 50) a 51-episode mean --
             not W=100 statistics at all. Left in, a single lucky opening episode "crosses" any loose
             target: the first version of this table reported CRLQAS reaching 50 mHa on LiH(6q) in
             **50** VQE calls, with [min-max] = [50-50], because a PSQAS run's cumulative count after
             its first depth-capped episode is exactly 50. DreamQAS-family rows hid the artifact only
             because their x[0] is already thousands of calls (warm-up). Dropping the first
             ceil((W-1)/DOWN) = 2 points makes every method's earliest admissible crossing a genuine
             100-episode mean. Consequence: no crossing can be reported before ~100 episodes have
             elapsed, which is the honest floor of a W=100 estimator.
             ⚠ speed_ladder.txt and rq2_speedup_w100.txt share the un-guarded `cross` and are NOT
             regenerated here. Their targets are tighter (fixed ladders read off a median curve /
             per-baseline Q_b) so they are far less exposed, but any cell of theirs that crosses at
             the first reported point is suspect for the same reason.

PROVENANCE — two run sets in one table, labelled per row group:
  Full / No-imag / DreamQAS-RL  oracle_free_v1  (ORACLE-FREE training signal; the paper's method)
  CRLQAS / HyRLQAS              campaign_v1     (each under its own native reward/stopping protocol;
                                                 HyRLQAS from the psqas_hyrlqas_std tree, which is
                                                 the row policy_quality_table.py:59 reports)

WHY THESE FIVE AND NOT ALL EIGHT: VQE counting is only comparable across these five. All five count
"number of prefixes variationally optimized", and the per-call inner budget was aligned (COBYLA
global_iters=1000; rotosolve_sweeps forced to 1, down from the baselines' native 2). GQE / TF-QAS /
QuantumDARTS use native accounting that is not this unit -> quality comparison only, never a VQE
speedup. (The trailing line of rq2_speedup_w100.txt that lumps CRLQAS/HyRLQAS in with them is stale;
of_main_table.py's caption 3 is the correct statement.)

NOT AVAILABLE: a `vqe_nfev` (inner energy-evaluation) column cannot be produced for all five --
DreamQAS writes best_circuit.json with `vqe_nfev`, the PSQAS baselines write no best_circuit.json at
all. Marked MISSING rather than filled for a subset.

Outputs
  outputs/main_results/of_vqe_to_target.txt        the appendix table
  outputs/main_results/of_vqe_to_target_seeds.csv  every per-seed crossing (supplementary bundle)

Usage: python analysis/of_vqe_to_target_table.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_rq2_w100_of as P                       # W=100 curve machinery + per-seed `cross`
from speed_ladder import LEVELS                    # per-task error ladders, fixed elsewhere

MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
# (internal key used by seed_curves, display name, provenance tag)
METHODS = [("Full",    "DreamQAS (Full)", "oracle-free"),
           ("No-imag", "No-imag",         "oracle-free"),
           ("RLQAS",   "DreamQAS-RL",     "oracle-free"),
           ("CRLQAS",  "CRLQAS",          "canonical"),
           ("HyRLQAS", "HyRLQAS",         "canonical")]
MIN_REACH = 3            # cells with fewer reached seeds are suppressed to '—'
# Discard reported points whose trailing window is not yet full (see module docstring: this is what
# removed the spurious "50 VQE calls" cells). ceil((100-1)/50) = 2 points.
FIRST_FULL = int(np.ceil((P.W - 1) / P.DOWN))
OUT_TXT = f"{HERE}/outputs/main_results/of_vqe_to_target.txt"
OUT_CSV = f"{HERE}/outputs/main_results/of_vqe_to_target_seeds.csv"

lines = []


def W(s=""):
    lines.append(s)
    print(s)


def cell(vals, n_seeds, n_left):
    """vals = per-seed crossing VQE (nan = right-censored, i.e. never reached).
    n_left = how many of the REACHED seeds crossed at the first admissible reported point, which makes
    the value an UPPER BOUND: the target was already met before a full W=100 window could resolve when.
    Those cells get a `≤` so the appendix never claims a cost that was not actually measured.
    -> '≤median  k/n' | 'median  k/n' | '—  k/n'."""
    v = np.asarray(vals, float)
    r = v[np.isfinite(v)]
    reach = f"{len(r)}/{n_seeds}"
    if len(r) < MIN_REACH:
        return f"{'—':>9}  {reach:>5}"
    med = np.median(r)
    pre = "≤" if n_left else ""
    return f"{pre + format(med, ',.0f'):>9}  {reach:>5}"


def rng_cell(vals):
    v = np.asarray(vals, float)
    r = v[np.isfinite(v)]
    if len(r) < MIN_REACH:
        return ""
    return f"[{r.min():,.0f}-{r.max():,.0f}]"


def main():
    rows_csv = []
    n_left_total = [0]      # how many per-seed crossings landed on the first admissible point
    W("=" * 132)
    W("APPENDIX — CUMULATIVE TRAINING VQE CALLS TO FIRST SUSTAINABLY REACH A FIXED ERROR TARGET")
    W("  cell = median over the seeds that REACHED the target  |  k/n = seeds reached / seeds available")
    W("  estimator = PER-SEED sustained crossing (W=100-episode trailing mean, 3 consecutive reported")
    W("              points, stride 50).  Censored seeds are excluded from the median, never imputed.")
    W(f"  '—' = reached by fewer than {MIN_REACH}/n seeds within the budget -> not reportable (a RESULT, not")
    W("        missing data).  Error ladders are imported verbatim from speed_ladder.LEVELS.")
    W("  x = cumulative REAL training VQE calls; DreamQAS-family INCLUDES DAgger verification VQE;")
    W("      frozen-evaluation VQE excluded.  LiH(6q) capped at the locked 15,000-episode budget.")
    W(f"  Reported points whose W=100 window is not yet full are discarded (first {FIRST_FULL}), so the")
    W("      earliest admissible crossing is a genuine 100-episode mean, not one lucky opening episode.")
    W("  '≤' on a cell = at least one contributing seed already met the target at that first admissible")
    W("      point, so the number is an UPPER BOUND on the cost, not a measurement of it (LEFT-censored:")
    W("      the target is simply loose for that method and this smoothing cannot resolve when it fell).")
    W("  ⚠ Two provenances, labelled per row: oracle-free (our arms) vs canonical (externals, native")
    W("    reward/stopping).  All five share the VQE-counting unit (prefixes variationally optimized,")
    W("    COBYLA global_iters=1000 / rotosolve_sweeps=1); GQE/TF-QAS/QuantumDARTS do NOT and are absent.")
    W("=" * 132)

    for mol in MOLS:
        levels = LEVELS[mol]
        W("")
        W(f"--- {DISP[mol]}      targets (mHa): " + ", ".join(f"{y:g}" for y in levels))
        hdr = f"{'method':<17}{'source':<13}" + "".join(f"{f'≤ {y:g} mHa':>18}" for y in levels)
        W(hdr)
        W("-" * len(hdr))
        for key, disp, prov in METHODS:
            cvs = P.seed_curves(key, mol)
            n = len([1 for x, _ in cvs if len(x) > FIRST_FULL])
            if n == 0:
                W(f"{disp:<17}{prov:<13}" + "".join(f"{'(no runs)':>18}" for _ in levels))
                continue
            trimmed = [(x[FIRST_FULL:], yy[FIRST_FULL:]) for x, yy in cvs
                       if len(x) > FIRST_FULL]
            x_first = [x[0] for x, _ in trimmed]        # first admissible reported point per seed
            cells, ranges = [], []
            for y in levels:
                vals = [P.cross(cv, y) for cv in trimmed]
                left = [bool(np.isfinite(v) and v <= xf) for v, xf in zip(vals, x_first)]
                cells.append(cell(vals, n, sum(left)))
                ranges.append(rng_cell(vals))
                n_left_total[0] += sum(left)
                for i, (v, lf) in enumerate(zip(vals, left)):
                    rows_csv.append(dict(task=DISP[mol], method=disp, source=prov, target_mHa=y,
                                         seed_index=i, vqe_calls=("" if not np.isfinite(v) else int(v)),
                                         reached=int(np.isfinite(v)), left_censored=int(lf)))
            W(f"{disp:<17}{prov:<13}" + "".join(f"{c:>18}" for c in cells))
            if any(ranges):
                W(f"{'':<17}{'[min-max]':<13}" + "".join(f"{r:>18}" for r in ranges))

    W("")
    W("=" * 132)
    W("HOW TO QUOTE THIS TABLE")
    W("  * A smaller number is only better AT EQUAL REACH. A method that reaches a target on 3/5 seeds")
    W("    with few VQE calls is NOT faster than one reaching it on 5/5 with more -- the median over")
    W("    reached seeds is subject to survivor bias, and HyRLQAS in particular has deterministic")
    W("    mode-collapse seeds. Always quote k/n next to the number.")
    W("  * This is a TRAINING-TIME statistic: it says the training curve reached that error, not that")
    W("    the frozen policy DEPLOYS there. HyRLQAS is the worked example -- it reaches low targets")
    W("    cheaply here on BeH2(8q) yet its frozen best-of-1 in the main table is 18.4 mHa. Deployable")
    W("    quality is the frozen best-of-1 table; this table is only about cost-to-reach in training.")
    W("  * Do NOT combine these numbers with the ratios in speed_ladder.txt (median-curve estimator)")
    W("    or rq2_speedup_w100.txt (per-baseline target Q_b). Three different estimators; pick one per")
    W("    claim and name it.")
    W("  * vqe_nfev (inner energy-evaluation count) column: MISSING -- DreamQAS logs it in")
    W("    best_circuit.json, the PSQAS baselines write no best_circuit.json.")
    W(f"  * Left-censored per-seed crossings (crossed at the first admissible point): {n_left_total[0]}")
    W("    of the reached ones. Their cells carry '≤'. Quote those as upper bounds only.")
    W(f"  * Per-seed values for every cell (incl. a left_censored flag): {os.path.basename(OUT_CSV)}")
    W("=" * 132)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "source", "target_mHa",
                                          "seed_index", "vqe_calls", "reached", "left_censored"])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\n[wrote] {OUT_TXT}")
    print(f"[wrote] {OUT_CSV}  ({len(rows_csv)} per-seed rows)")


if __name__ == "__main__":
    main()
