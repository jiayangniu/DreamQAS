"""Does selective real-VQE verification actually CORRECT the world model? (RQ4, oracle-free)

WHAT CAN AND CANNOT BE ASKED OF THE LOGS
-----------------------------------------
The claim "verification corrects errors" invites a per-candidate before/after test: verify candidate
c at iteration t, then show the WM's error ON c (or on circuits like c) drops afterwards.
**That test is NOT possible here.** `calibration.jsonl` records `iter`, `pred_score`, `real_score`,
`disagree`, `len` and `selection`, but NOT the identity of the circuit — no action sequence, no id.
Without identity a candidate cannot be followed across events. Fabricating a proxy (e.g. matching on
`len`) would not be the same test, so it is not attempted.

What the logs DO support is a clean controlled comparison, and it is arguably the stronger design:

    Full  (dagger=1) runs the verification probe AND feeds the verified samples back into replay.
    NoDAG (dagger=0) runs the IDENTICAL probe and throws the result away (it is booked to a separate
                     `calib_vqe_calls` counter and never enters training).

So the two arms see the same probe on the same schedule and differ ONLY in whether the correction is
applied. If verification corrects the model, Full's prediction error on subsequent probes should fall
below NoDAG's. This is an event-aligned population comparison, not a per-candidate one — state it that
way, and if it comes out null, narrow the paper's claim to a "targeted correction ROUTE" (a mechanism
that exists and is exercised) rather than a demonstrated correction effect.

METRIC: per calibration event, mean |pred_score − real_score| over that event's candidates, in the
WM's own target space (frontier score S — both arms are oracle-free, so the space matches). Compared
over a LATE window (last 25% of the common budget) so early transients do not dominate, seed-paired.
Also split by `selection`, because the two strata are different questions: `top`/`both` are the
value-selected candidates the actor would actually commit to, `disagree` are the blind-spot probes.

Usage: python analysis/verification_correction.py > outputs/main_results/verification_correction.txt
"""
import glob
import json
import os

import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
CAP = {"LiH6q": 3750}                       # locked reporting budget
ARMS = {"Full": f"{C}/dreamqas/gru_energy_surrogate_{{m}}_s*_of",
        "NoDAG": f"{C}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag"}
LATE = 0.75                                  # window start, as a fraction of the budget


def events(run_dir, mol, sel=None):
    """-> {iter: mean |pred_score - real_score|} over that event's candidates."""
    f = f"{run_dir}/calibration.jsonl"
    if not os.path.exists(f):
        return {}
    cap = CAP.get(mol)
    acc = {}
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("selection") == "SUMMARY":
            continue
        it = int(r.get("iter", -1))
        if it < 0 or (cap and it >= cap):
            continue
        if sel and r.get("selection") not in sel:
            continue
        p, q = r.get("pred_score"), r.get("real_score")
        if p is None or q is None:
            continue
        acc.setdefault(it, []).append(abs(float(p) - float(q)))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def late_mean(run_dir, mol, sel=None):
    ev = events(run_dir, mol, sel)
    if len(ev) < 4:
        return None
    its = np.array(sorted(ev))
    cut = its.min() + LATE * (its.max() - its.min())
    v = [ev[i] for i in its if i >= cut]
    return float(np.mean(v)) if v else None


def by_seed(arm, mol, sel=None):
    out = {}
    for d in sorted(glob.glob(ARMS[arm].format(m=mol))):
        v = late_mean(d, mol, sel)
        if v is not None:
            out[os.path.basename(d).split("_s")[-1].split("_")[0]] = v
    return out


def paired(a, b, rng, n=10000):
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    d = np.array([a[k] - b[k] for k in ks])          # Full − NoDAG; negative = Full is better
    bs = [rng.choice(d, len(d), True).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(ks), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                wins=int((d < 0).sum()))


def main():
    rng = np.random.default_rng(7)
    print("=" * 112)
    print("DOES VERIFICATION CORRECT THE WORLD MODEL?  Full vs NoDAG — identical probe, only the")
    print("feedback differs (NoDAG books the same VQE to calib_vqe_calls and never trains on it).")
    print(f"  metric: mean |pred_score − real_score| per calibration event, LATE window (last "
          f"{int((1 - LATE) * 100)}% of the budget), seed-paired.")
    print("  ⚠ This is a POPULATION comparison. A per-candidate before/after test is impossible:")
    print("    calibration.jsonl logs no circuit identity, so a candidate cannot be followed.")
    print("=" * 112)
    for label, sel in (("all verified candidates", None),
                       ("value-selected stratum {top, both}", {"top", "both"}),
                       ("blind-spot stratum {disagree}", {"disagree"})):
        print(f"\n--- {label}")
        print(f"{'task':11s}{'Full':>12}{'NoDAG':>12}{'n':>4}{'Full−NoDAG':>13}{'95% CI':>24}"
              f"{'Full better':>13}   verdict")
        for m in MOLS:
            A, B = by_seed("Full", m, sel), by_seed("NoDAG", m, sel)
            r = paired(A, B, rng)
            if r is None:
                print(f"{DISP[m]:11s}   (fewer than 3 shared seeds)")
                continue
            clean = r["lo"] > 0 or r["hi"] < 0
            v = ("verification CORRECTS" if (clean and r["mean"] < 0) else
                 "verification WORSENS" if clean else "n.s. — not demonstrated")
            print(f"{DISP[m]:11s}{np.mean(list(A.values())):>12.4f}{np.mean(list(B.values())):>12.4f}"
                  f"{r['n']:>4}{r['mean']:>+13.4f}{f'[{r[chr(108)+chr(111)]:+.4f}, {r[chr(104)+chr(105)]:+.4f}]':>24}"
                  f"{f'{r[chr(119)+chr(105)+chr(110)+chr(115)]}/{r[chr(110)]}':>13}   {v}")
    print("\n" + "=" * 112)
    print("HOW TO READ / WHAT MAY BE CLAIMED")
    print("=" * 112)
    print("  * A null result here does NOT mean verification is useless — it means the CORRECTION")
    print("    EFFECT on the model's own predictions is not demonstrated at 5 seeds. DAgger's")
    print("    documented value is elsewhere: collapse-prevention (ladder_table.txt) and keeping the")
    print("    imagination gate honest. Its cost is real and measured (§2.4: 8.5-11% of wall-clock).")
    print("  * If this is null, the main text must say 'selective verification provides a targeted")
    print("    correction ROUTE' — a mechanism that exists and is exercised — and must NOT say")
    print("    'verification corrects model errors', which is the claim this table would have to")
    print("    support and does not.")
    print("  * The two strata answer different questions: the value-selected stratum is what the actor")
    print("    would commit to; the disagreement stratum is the blind-spot probe. A correction effect")
    print("    could plausibly appear in one and not the other — read them separately, never pooled.")


if __name__ == "__main__":
    main()
