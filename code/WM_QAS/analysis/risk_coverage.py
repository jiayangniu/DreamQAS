"""RQ4 experiment C — risk–coverage: does ensemble disagreement actually let us reject the bad predictions?

DreamQAS leans on the ensemble disagreement sigma in three places: the pessimistic value
phi = -(mean + beta*sigma) — CONTINUOUS, applied to every candidate at every step, and the primary use —
the imagination confidence truncation (stop once sigma > imag_conf_tau=0.60) — DISCRETE, a safety valve —
and DAgger candidate selection. When reading the retention numbers below, keep the two apart: a threshold
that rarely fires does NOT mean sigma is unused, because the pessimism term acts regardless. All three assume "large sigma => large prediction error". So far we only
have a CORRELATION (Spearman 0.03-0.75), which cannot say whether THRESHOLDING at the operating point
helps. A risk-coverage curve answers exactly that:

    coverage(tau) = Pr(sigma <= tau)                     how much of the candidate set we keep
    risk(tau)     = E[ |y_hat - y|  |  sigma <= tau ]     mean prediction error on what we kept

If sigma carries information, risk falls as coverage shrinks. Three rejection rules are compared on the
SAME sample: by disagreement (ours), by predicted value (does the prediction alone already tell us?), and
random (the flat-line control). Only "disagreement clearly below value AND random" licenses the claim that
uncertainty monitoring works.

A fourth curve, `value-inv`, KEEPS the predicted-WORST instead. It is not a deployable rule — nobody ships
"prefer the candidates the model dislikes" — it is a DIAGNOSTIC of over-optimism: if value-inv beats value,
the model's error is concentrated exactly on the candidates it rates best. That is the failure mode DAgger
was introduced to fix, and it is the mechanism behind the WM-Greedy / WM-Beam collapse (both pick the
predicted-best, i.e. they walk straight into the region where the WM is least reliable).

⚠ SELECTION BIAS AND HOW IT IS HANDLED. `calibration.jsonl` is not a random sample: `dagger_select="mix"`
verifies 5 candidates chosen by predicted value and 5 chosen by HIGHEST disagreement, so sigma is directly
manipulated. The sigma-neutral stratum is the value-selected five, which in the log is
`selection in {"top", "both"}` — "both" means a candidate was in the top-5 by value AND happened to also be
in the top-5 by disagreement, so it is still part of the value-selected set. Keeping only "top" would DROP
exactly the high-sigma members of that set and bias sigma downward — the opposite error. The
disagreement-selected candidates (`selection == "disagree"`) are excluded entirely.
Consequence: this is a conservative test. If sigma still separates risk inside a stratum that was chosen
without looking at sigma, that is stronger evidence than a curve drawn on the manipulated full sample.

Usage: python analysis/risk_coverage.py [--mols LiH4q,LiH6q,BeH2_8q] [--half]
       > outputs/main_results/risk_coverage.txt
"""
import argparse
import glob
import json
import os

import zlib

import numpy as np

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)", "BeH2": "BeH2(6q)",
        "BeH2_10q": "BeH2(10q)"}
TAU_CONF = 0.60                      # cfg.imag_conf_tau — the deployed truncation threshold
GRID = np.arange(0.10, 1.001, 0.05)  # coverage levels to report


def load_run(d, half=False):
    """-> (sigma, pred, abs_err) over the SIGMA-NEUTRAL stratum only."""
    f = f"{d}/calibration.jsonl"
    if not os.path.exists(f):
        return None
    sig, pred, err, it = [], [], [], []
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("selection") not in ("top", "both"):     # value-selected five == sigma-neutral
            continue
        a = r.get("score_abs_err", r.get("abs_logerr_err"))
        if a is None or r.get("disagree") is None or r.get("pred_score") is None:
            continue
        sig.append(float(r["disagree"])); pred.append(float(r["pred_score"]))
        err.append(float(a)); it.append(int(r["iter"]))
    if len(sig) < 50:
        return None
    sig, pred, err, it = map(np.asarray, (sig, pred, err, it))
    if half:                                              # steady state only (2nd half of training)
        m = it >= np.median(it)
        sig, pred, err = sig[m], pred[m], err[m]
    return sig, pred, err


def curve(score, err, grid, rng=None):
    """risk at each coverage level when KEEPING the lowest-score fraction (score = the reject signal)."""
    n = len(err)
    if rng is not None:                                   # random control: shuffle the ranking
        order = rng.permutation(n)
    else:
        order = np.argsort(score, kind="stable")
    e = err[order]
    out = []
    for c in grid:
        k = max(1, int(round(c * n)))
        out.append(float(e[:k].mean()))
    return np.asarray(out)


def aurc(risks, grid):
    return float(np.trapezoid(risks, grid) / (grid[-1] - grid[0]))


def _per_seed_block(A, rng, n_boot=10000):
    """AURC as a PER-SEED statistic, plus paired CIs against the two controls.

    The curve block above averages the four rules' curves across seeds and reports the AURC of that
    mean curve — a point estimate with no uncertainty attached. The experimental unit here is the
    SEED (each contributes ~10^3 candidates), so this block recomputes AURC per seed and does the
    comparison the paper's claim actually needs: is disagreement-based rejection better than value or
    random ON THE SEEDS, not on the pooled candidate set.
    """
    ks = ["disagree", "value", "value-inv", "random"]
    print(f"\n  per-seed AURC (n={len(A['disagree'])} seeds; the experimental unit)")
    print("  " + "".join(f"{k:>16}" for k in ks))
    print("  " + "".join(f"{f'{A[k].mean():.4f}±{A[k].std(ddof=1):.4f}':>16}" for k in ks))
    print(f"\n  {'paired contrast':28s}{'Δ AURC':>10}{'95% CI':>22}{'better':>9}   verdict")
    for other in ("value", "random"):
        d = A["disagree"] - A[other]                       # negative = disagreement rejects better
        bs = [rng.choice(d, len(d), True).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        clean = lo > 0 or hi < 0
        v = ("disagreement BETTER" if (clean and d.mean() < 0)
             else "disagreement WORSE" if clean else "n.s. — not distinguishable")
        print(f"  {'disagreement − ' + other:28s}{d.mean():>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{f'{int((d < 0).sum())}/{len(d)}':>9}   {v}")
    print("  Δ<0 means disagreement-based rejection achieves LOWER risk at matched coverage.")


def rule_rng(mol):
    """Deterministic RNG for the RANDOM-rejection baseline, keyed on the molecule.

    It used to come from one generator consumed across molecules in argument order, which made the
    baseline depend on WHICH molecules were requested and in what order: `--mols BeH2_8q` alone gave a
    different random curve than `--mols LiH4q,LiH6q,BeH2_8q`, and any downstream figure that recomputed
    the curves could not reproduce the table (it did not — the mismatch is what surfaced this).
    Keying on a stable content hash of the molecule name makes the baseline order-independent and
    identical in every caller.
    """
    return np.random.default_rng(zlib.crc32(mol.encode()) & 0xFFFFFFFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", default="LiH4q,LiH6q,BeH2_8q")
    ap.add_argument("--half", action="store_true", help="restrict to the 2nd half of training")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    print("=" * 104)
    print("RISK–COVERAGE OF ENSEMBLE DISAGREEMENT  (oracle-free Full runs; RQ4 experiment C)")
    print("  risk = mean |predicted − real| in the WM's own target space (frontier score S);")
    print("  coverage = fraction of candidates KEPT after rejecting the worst by each rule.")
    print("  Sample = the SIGMA-NEUTRAL stratum only: candidates selected by predicted value")
    print("  (`selection in {top, both}`); disagreement-selected ones are excluded, so sigma was")
    print("  never used to build this sample. Conservative by construction.")
    print(f"  Deployed truncation threshold: imag_conf_tau = {TAU_CONF}.")
    if a.half:
        print("  WINDOW: second half of training only.")
    print("=" * 104)
    for mol in a.mols.split(","):
        runs = [load_run(d, a.half) for d in sorted(glob.glob(f"{C}/dreamqas/*_{mol}_s*_of"))]
        runs = [r for r in runs if r]
        if not runs:
            print(f"\n--- {DISP.get(mol, mol)}: no data"); continue
        cd, cv, ci_, cr, taus, ns = [], [], [], [], [], []
        for sig, pred, err in runs:
            cd.append(curve(sig, err, GRID))                     # ours: reject high disagreement
            cv.append(curve(pred, err, GRID))                    # control: reject predicted-bad
            ci_.append(curve(-pred, err, GRID))                  # diagnostic: reject predicted-GOOD
            cr.append(curve(pred, err, GRID, rng=rule_rng(mol)))  # control: random (order-independent)
            taus.append(float((sig <= TAU_CONF).mean()))
            ns.append(len(err))
        # per-SEED AURC before averaging the curves. The experimental unit is the seed, not the
        # candidate: a run contributes ~10^3 candidates, so pooling them would inflate n by 3 orders
        # of magnitude and any CI built on the pooled sample would be meaningless.
        A = {k: np.array([aurc(c, GRID) for c in v])
             for k, v in (("disagree", cd), ("value", cv), ("value-inv", ci_), ("random", cr))}
        cd, cv, ci_, cr = map(lambda x: np.mean(np.asarray(x), 0), (cd, cv, ci_, cr))
        print(f"\n--- {DISP.get(mol, mol)}   {len(runs)} seeds, {int(np.mean(ns))} candidates/seed")
        print(f"{'coverage':>10}{'disagreement':>15}{'value':>10}{'value-inv':>12}{'random':>10}"
              f"   risk (lower better)")
        for i, c in enumerate(GRID):
            if abs(c * 100 % 10) > 1e-6 and abs(c - 0.95) > 1e-6:
                continue
            print(f"{c:>10.2f}{cd[i]:>15.4f}{cv[i]:>10.4f}{ci_[i]:>12.4f}{cr[i]:>10.4f}")
        print(f"{'AURC':>10}{aurc(cd, GRID):>15.4f}{aurc(cv, GRID):>10.4f}{aurc(ci_, GRID):>12.4f}"
              f"{aurc(cr, GRID):>10.4f}   (area under the curve; lower = better rejection)")
        _per_seed_block(A, rng)
        full = cr[-1]
        print(f"\n  risk at full coverage (no rejection) = {full:.4f}")
        i50 = int(np.argmin(np.abs(GRID - 0.5)))
        for lab, cc in (("disagreement", cd), ("value", cv), ("value-inv", ci_), ("random", cr)):
            ch = cc[i50] / full - 1.0
            verd = "risk DOWN" if ch < -0.02 else ("risk UP" if ch > 0.02 else "no change")
            print(f"  keep 50% by {lab:13s}: risk {cc[i50]:.4f}  -> {verd} {abs(ch) * 100:.0f}%")
        # how often the DEPLOYED truncation actually fires. The value-selected stratum may
        # under-represent high-sigma candidates, so also bound it on the disagreement-selected
        # stratum (which over-represents them): the true rollout retention lies between.
        hi = []
        for d in sorted(glob.glob(f"{C}/dreamqas/*_{mol}_s*_of")):
            f = f"{d}/calibration.jsonl"
            if not os.path.exists(f):
                continue
            sg = [float(json.loads(l)["disagree"]) for l in open(f)
                  if l.strip() and json.loads(l).get("selection") == "disagree"
                  and json.loads(l).get("disagree") is not None]
            if sg:
                hi.append(float((np.asarray(sg) <= TAU_CONF).mean()))
        lo_ret = 100 * np.mean(hi) if hi else float("nan")
        # over-optimism diagnostic: is the model's error concentrated on what it RATES BEST?
        av, ai = aurc(cv, GRID), aurc(ci_, GRID)
        if ai < av:
            print(f"  ⚠ OVER-OPTIMISM: keeping the predicted-WORST gives LOWER risk than keeping the")
            print(f"    predicted-best (AURC {ai:.4f} vs {av:.4f}, ratio {av / ai:.2f}x). The WM's error")
            print(f"    concentrates on the candidates it rates best — which is precisely what greedy /")
            print(f"    beam deployment selects, and what DAgger verification exists to correct.")
        else:
            print(f"  no over-optimism by this test: predicted-best is the safer half "
                  f"(AURC {av:.4f} vs inverse {ai:.4f}).")
        print(f"  deployed threshold sigma <= {TAU_CONF}: retains {100 * np.mean(taus):.0f}% of the")
        print(f"    value-selected stratum and {lo_ret:.0f}% of the disagreement-selected stratum")
        print(f"    -> true rollout retention is bracketed by these; the truncation is "
              f"{'ESSENTIALLY INACTIVE' if 100 * np.mean(taus) > 95 and lo_ret > 60 else 'active'} here.")
    print("\n" + "=" * 104)
    print("READ: the claim 'ensemble disagreement identifies where the WM is wrong' requires the")
    print("disagreement curve to sit BELOW both controls. If it tracks `value`, sigma adds nothing beyond")
    print("the prediction itself; if it tracks `random`, sigma is uninformative and the pessimism term,")
    print("the confidence truncation and the DAgger selection rule all lose their stated justification.")
    print("Risk is in frontier-score units (oracle-free target space) — not mHa, not comparable to the")
    print("canonical log10-error calibration MAE.")


if __name__ == "__main__":
    main()
