"""WM ranking accuracy vs CIRCUIT DEPTH — the quantity historically (mis)named "horizon fidelity".

⚠ READ THIS BEFORE USING THE WORD "HORIZON"
-------------------------------------------
Under DreamQAS's architecture this probe cannot measure a horizon effect, because there is nothing
horizon-like left to measure:

  1. Imagined circuit dynamics are EXACT — appending action a to sequence s yields s+[a], the same
     circuit reality yields. There is no learned transition.
  2. The WM's prediction is a PURE FUNCTION OF THE FULL SEQUENCE: `_wm_pred_logerr` calls
     `_encode_seeds`, which restarts from `wm.init_state` and replays the whole prefix every time
     (imagine.py:34-44). Predictions are never chained, so nothing accumulates across steps.
  3. Every prefix is drawn at a FIXED depth sd = num_layers // 4 (LiH-4q 10; LiH-6q / BeH2-8q 12),
     then H more gates are appended -> endpoint depth == sd + H exactly.

  => fidelity(H) IS the WM's ranking accuracy on circuits of depth sd+H. That is an identity, not a
     confound to be controlled away. Report it as a DEPTH curve.

  Corollary — a claim that must NOT be made: the canonical Figure 3a argues "ranking does not degrade
  with H, therefore there is no step-wise accumulation of learned-transition error". Accumulation is
  architecturally impossible here (points 1-2), so the probe could never have detected it whatever
  the slope came out to be. That argument is an architectural tautology dressed as an empirical
  finding, and it is RETRACTED.

  Also note the H steps walk the REAL environment (`env.step`, one real VQE per step), not the
  imagination loop — so "imagined horizon" was doubly a misnomer.

WHAT IT DOES MEASURE, AND WHY THAT IS WORTH HAVING
--------------------------------------------------
Not the end-to-end horizon experiment (that is `q2_horizon_table.py`, which measures POLICY quality at
H in {1,5,15} under a matched transition budget). This measures a WORLD-MODEL property:

    how far down the circuit does the WM's ranking stay trustworthy?

Measured by `t1a_action_ranking.py::horizon_fidelity` (v2, 2026-07-18) and already stored per run in
`t1a_probe.json`. Per held-out shallow prefix: normalise the prefix to env-replayable form, then walk ONE
actor-sampled path forward under the real env's legality mask with frozen-eval termination semantics,
recording at H in {1,5,10,15} the pair (WM-predicted endpoint log-error, real post-VQE endpoint
log-error). The torch seed is fixed PER PREFIX, not per H, so the four endpoints are NESTED points on the
same path — that removes sampling noise from the depth comparison. Spearman over prefixes at each depth
gives fidelity(depth); its slope is the test statistic.

Why the paper cares (the SURVIVING reason): it tells the reader HOW DEEP the WM can be trusted. The
imagination loop, the DAgger verification budget and the sigma-pessimism gate all act on circuits the
policy is building; knowing that ranking quality falls toward the deep end is an operational fact about
where the model's guidance is worth least — and where real VQE verification is worth most. Reporting
convention (locked): lead with the SLOPE, never with individual per-depth points (seed-noisy, std 0.2-0.4).

WHY THIS SCRIPT NEEDS NO CHECKPOINTS
------------------------------------
The per-H fidelities are already inside every `t1a_probe.json` (keys `wmstart` / `onset` / `quarter`,
each with a `horizon` block). This is pure post-processing of files that ARE in the curated backup, so
the test does NOT become unrunnable when the checkpoint grid is dropped. (Re-probing a NEW task or a NEW
checkpoint would need the grid — that part is genuinely last-chance.)

Statistic: per-seed OLS slope of fidelity on H, then mean +- sample std (ddof=1) across seeds and a
two-sided one-sample t-test against 0. Reported per task and pooled, for oracle-free and canonical side
by side. Rank-based fidelity is invariant to the monotone reparameterisation between the two training
targets, so the two signals ARE comparable here (unlike absolute calibration MAE).

TWO THINGS THIS TEST CANNOT DO — both are printed with the results so they travel with any quote:

  1. It cannot say the two SIGNALS differ. Each slope is tested against 0 separately; "significant on
     one arm, n.s. on the other" is NOT evidence that the arms differ (the classic
     difference-of-significance fallacy). The two-sample block below runs the test that actually
     answers it — Welch's t on the 15 vs 15 per-seed slopes — and it comes out n.s. An earlier draft
     of the docs claimed the signals "reach opposite conclusions"; that claim is RETRACTED.

  2. It cannot say anything about HORIZON as distinct from depth — see the header: they are the same
     quantity here, by construction. Never write "imagining further degrades ranking".
     What IS still open is whether ENDPOINT DEPTH ALONE explains the curve, or whether the circuit
     POPULATION matters too: a depth-25 endpoint reached as (sd=10, +15 gates) is drawn from a
     different distribution than one reached as (sd=20, +5 gates), because `_gen_prefix` (stateless
     Shadow legality) and the real-env rollout compose circuits differently. Re-running the probe
     from several starting depths would answer that. It needs the checkpoint grid -> LAST-CHANCE.
     (Measured, for the record: per-depth training VISIT counts are near-flat across the probed
     depths — LiH-4q 2.86% -> 2.42%, LiH-6q/8q ~2.0% throughout — so the naive "less training data
     deeper" story does not hold. What does still grow with depth is the size of the reachable
     circuit space, hence coverage DENSITY falls. Untested.)

Usage: python analysis/of_horizon_slope.py > outputs/main_results/oracle_free_horizon_slope.txt
"""
import glob
import json
import os

import numpy as np
from scipy import stats

OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CV = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas"
HS = [1, 5, 10, 15]
CKPT = "quarter"                  # the 1/4-budget operating point — the locked convention for this probe
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
SRC = {"oracle-free": f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of/t1a_probe.json",
       "canonical":   f"{CV}/gru_energy_surrogate_{{m}}_s*_q/t1a_probe.json"}


def _degenerate(f):
    """True if this run's cross-prefix ranking target is (near-)constant, making every Spearman on it
    UNDEFINED rather than low. Measured on the stored advfid traces (plateau_diagnostic.py); absent
    traces -> treated as non-degenerate, since we then have no evidence either way."""
    d = json.load(open(f)).get(CKPT, {}).get("advfid")
    if not d or not d.get("traces"):
        return False
    return float(np.std([t["phi_true"][-1] for t in d["traces"]])) <= 1e-6


def per_seed(pattern):
    """-> (slopes[], per-H fidelity matrix [seed, H]). Skips runs whose probe lacks the horizon block,
    and runs whose ranking target is degenerate (ansatz plateau)."""
    sl, curves = [], []
    for f in sorted(glob.glob(pattern)):
        if _degenerate(f):
            continue
        hz = json.load(open(f)).get(CKPT, {}).get("horizon")
        if not hz:
            continue
        xs = [H for H in HS if str(H) in hz]
        ys = [hz[str(H)]["fidelity"] for H in xs]
        if len(xs) < 3:
            continue
        sl.append(float(np.polyfit(xs, ys, 1)[0]))
        curves.append([hz.get(str(H), {}).get("fidelity", np.nan) for H in HS])
    return np.array(sl), np.array(curves, float)


def block(label, pat_tmpl):
    """-> {task: per-seed slopes}. Prints the within-arm test (slope vs 0)."""
    print(f"\n{label.upper()}  (checkpoint = {CKPT})")
    print("-" * 112)
    print(f"{'task':10s}{'n':>3}{'slope /H':>11}{'± std':>9}{'p':>9}{'vs 0':>14}   "
          f"{'per-H fidelity (cross-seed mean)':s}")
    got = {}
    for m in MOLS:
        s, C = per_seed(pat_tmpl.format(m=m))
        if len(s) < 3:
            print(f"{DISP[m]:10s}{len(s):>3}   (fewer than 3 seeds with a horizon block)")
            continue
        got[m] = s
        p = float(stats.ttest_1samp(s, 0.0).pvalue)
        v = "DEGRADES" if (p < 0.05 and s.mean() < 0) else ("improves" if p < 0.05 else "n.s.")
        curve = "  ".join(f"H{H}:{v2:+.3f}" for H, v2 in zip(HS, np.nanmean(C, 0)))
        print(f"{DISP[m]:10s}{len(s):>3}{s.mean():>11.4f}{s.std(ddof=1):>9.4f}{p:>9.3f}{v:>14}   {curve}")
    if got:
        s = np.concatenate(list(got.values()))
        p = float(stats.ttest_1samp(s, 0.0).pvalue)
        v = "DEGRADES" if (p < 0.05 and s.mean() < 0) else ("improves" if p < 0.05 else "n.s.")
        print("-" * 112)
        print(f"{'POOLED':10s}{len(s):>3}{s.mean():>11.4f}{s.std(ddof=1):>9.4f}{p:>9.4f}{v:>14}")
    return got


def between_signals(A, B):
    """The test the within-arm block does NOT perform: do the two signals' slopes differ?

    Comparing "significant on one arm, n.s. on the other" is the difference-of-significance fallacy.
    Welch's two-sample t on the per-seed slopes is the test that answers the question.
    """
    print("\n" + "=" * 112)
    print("BETWEEN-SIGNAL TEST — do the oracle-free and canonical slopes actually DIFFER?")
    print("  Welch's two-sample t on the per-seed slopes. This is NOT answered by the blocks above:")
    print("  'significant vs 0 on one arm, n.s. on the other' does not imply the arms differ.")
    print("=" * 112)
    print(f"{'task':10s}{'oracle-free':>13}{'canonical':>12}{'difference':>12}{'t':>8}{'p':>9}   verdict")
    print("-" * 112)
    for m in MOLS + ["POOLED"]:
        a = np.concatenate(list(A.values())) if m == "POOLED" else A.get(m)
        b = np.concatenate(list(B.values())) if m == "POOLED" else B.get(m)
        if a is None or b is None or len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        name = "POOLED" if m == "POOLED" else DISP[m]
        v = "DIFFER" if p < 0.05 else "n.s. — indistinguishable"
        print(f"{name:10s}{a.mean():>13.4f}{b.mean():>12.4f}{a.mean() - b.mean():>12.4f}"
              f"{t:>8.2f}{p:>9.3f}   {v}")
    a = np.concatenate(list(A.values())); b = np.concatenate(list(B.values()))
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    n_need = int(np.ceil(2 * (1.96 + 0.84) ** 2 / d ** 2)) if d else 0
    print("-" * 112)
    print(f"  Cohen's d = {d:.2f}. To detect an effect this size at alpha=0.05 / power=0.8 would need")
    print(f"  n ~ {n_need} seeds per arm; we have {len(a)} (mol,seed) pairs. The study is UNDERPOWERED")
    print("  for the between-signal comparison — report it as 'not distinguishable', never as 'the same'")
    print("  and never as 'opposite conclusions'.")


def main():
    print("=" * 112)
    print("WM RANKING ACCURACY vs CIRCUIT DEPTH  (historically mis-named 'horizon fidelity')")
    print("  fidelity(H) = Spearman(WM-predicted endpoint log-error, real post-VQE endpoint log-error)")
    print("    over held-out shallow prefixes; the H sweep walks ONE nested path per prefix (seed fixed")
    print("    per prefix, not per H), so the trend is not sampling noise.")
    print("  ⚠ H is NOT a horizon here: dynamics are exact and the WM re-encodes the whole sequence, so")
    print("    the H-step endpoint is simply a circuit of depth sd+H (sd = num_layers//4: 4q 10, 6q/8q 12).")
    print("    fidelity(H) IS accuracy-vs-depth. 'No accumulation of transition error' is architecturally")
    print("    true and UNTESTABLE by this probe — that argument is retracted, not supported.")
    print("  statistic  = per-seed OLS slope of fidelity on H; mean ± sample std (ddof=1) over 5 seeds;")
    print("    two-sided one-sample t-test vs 0. LEAD WITH THE SLOPE — the per-H points are seed-noisy.")
    print("  source     = per-run t1a_probe.json (already written; no checkpoint, no new VQE).")
    print("  ⚠ DEGENERACY GATE: seeds whose cross-prefix ranking target is pinned at the ansatz plateau")
    print("    are DROPPED (a Spearman on a constant is undefined, not low) — see plateau_diagnostic.txt.")
    print("    ⚠ ASYMMETRY: the gate needs advfid traces, which exist only on the ORACLE-FREE arm. The")
    print("    canonical arm is therefore UNGATED, so its numbers may still contain degenerate seeds and")
    print("    the between-arm comparison is not like-for-like. It was already n.s., so no conclusion")
    print("    turns on it — but do not quote the canonical column as a gated measurement.")
    print("  NOT the end-to-end horizon experiment — that is q2_horizon.txt (policy quality at matched T).")
    print("=" * 112)
    arms = {label: block(label, pat) for label, pat in SRC.items()}
    between_signals(arms["oracle-free"], arms["canonical"])
    print("\n" + "=" * 112)
    print("HOW TO READ — three limits, all load-bearing")
    print("=" * 112)
    print("  1. WITHIN an arm the slope is a real measurement: on the oracle-free arm the pooled slope")
    print("     is negative with p<0.01. BETWEEN arms the difference is NOT resolvable (see the block")
    print("     above). So the honest sentence is 'we cannot distinguish the two training signals on")
    print("     this statistic', NOT 'they reach opposite conclusions'.")
    print("  2. H IS circuit depth (endpoint = sd + H), not a horizon: exact dynamics + full-sequence")
    print("     re-encoding leave nothing horizon-like to measure. Write the result as 'WM ranking")
    print("     accuracy falls on deeper circuits'. NEVER 'imagining further degrades ranking', and")
    print("     never cite this as evidence about accumulated transition error.")
    print("  3. Rank-based, so the two arms' columns ARE comparable in principle (the training targets")
    print("     differ by a strictly monotone reparameterisation). Absolute calibration MAE is not.")
    print("  * A negative slope does NOT mean long-horizon imagination is harmful for policy learning:")
    print("    the imagined policy gradient uses lambda-returns along the WHOLE path plus a pessimism")
    print("    term in sigma, so it does not rest on the endpoint valuation. q2_horizon.txt in fact")
    print("    shows H=15 beating H=1 end-to-end on LiH-4q. Report both; do not merge them.")
    print("  * Do not quote individual per-H points as a result; they are printed only to show the shape.")


if __name__ == "__main__":
    main()
