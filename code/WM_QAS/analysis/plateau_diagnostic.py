"""Ansatz-plateau diagnostic — the confound that silently invalidates several ranking probes.

WHY THIS EXISTS
---------------
Every cross-circuit ranking probe in this project (endpoint fidelity, advantage fidelity, the Q2
horizon table) asks the WM to order a set of circuits by their real post-VQE error. That question is
only well posed if the circuits actually DIFFER in real error. On the harder molecules they often do
not: the actor's rollout reaches an ansatz plateau within a step or two and the true error is then
constant for the rest of the trajectory and identical across prefixes. Spearman on a constant is
undefined, so those probes return values near zero — which reads as "the world model ranks at chance"
when the truth is "there was nothing to rank".

This was found (2026-07-28) after BeH2-8q's advantage fidelity came out at +0.089 (p=0.35) and its
raw traces turned out to sit at exactly 2.175 mHa from the FIRST recorded step — which is precisely
the value the main table reports for Full on that task. 3 of its 5 seeds have exactly zero
cross-prefix variance. The same root cause is already visible elsewhere: `q2_horizon.txt`'s
saturation guard fires on BeH2-8q because H1 and H5 are both pinned at 2.175.

WHAT IT REPORTS (all from the stored advfid traces — ZERO new VQE)
------------------------------------------------------------------
  plateau onset      first step after which the true error is constant to the end of the trace,
                     converted to CIRCUIT DEPTH (= seed_len + step). "0 steps" means the probe never
                     observed any change at all.
  plateau value      the constant the trajectory settles on, in mHa. Cross-referencing it against the
                     paper's own numbers is the point: if it equals Full's best-of-1, the probe has
                     been ranking circuits that are all already at the reported optimum.
  distinct levels    number of distinct endpoint true-error values across the 15 prefixes of a seed.
                     1 = the ranking target is a constant and every correlation on it is undefined.
  usable seeds       seeds whose cross-prefix spread exceeds TOL — the only ones any cross-circuit
                     correlation may be computed on.

HOW TO USE IT: any cross-circuit ranking number must be reported together with its usable-seed count.
A near-zero correlation on a degenerate target is NOT evidence about the world model.

Usage: python analysis/plateau_diagnostic.py > outputs/main_results/plateau_diagnostic.txt
"""
import glob
import json
import os

import numpy as np

OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOL = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
       "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
CKPT = "final"
TOL = 1e-6
RTOL = 1e-9          # "constant" = successive true errors equal to within this relative tolerance
# paper reference values to cross-check the plateau against (mHa, oracle-free Full best-of-1 / greedy)
REF = {"LiH4q": ("Full BO1", 0.053), "BeH2": ("Full BO1", 0.058), "LiH6q": ("WM-Greedy", 36.9),
       "BeH2_8q": ("Full BO1", 2.175), "BeH2_10q": ("Full BO1", 1.03)}


def err_trace(t):
    """phi_true = -log10(err in mHa)  ->  err in mHa."""
    return np.array([10.0 ** (-v) for v in t["phi_true"]], float)


def plateau_onset(e):
    """First index k with e[k:] all equal (to RTOL). len(e)-1 if it never settles."""
    for k in range(len(e)):
        seg = e[k:]
        if len(seg) < 2:
            return k
        if np.allclose(seg, seg[0], rtol=RTOL, atol=0.0):
            return k
    return len(e) - 1


def main():
    print("=" * 116)
    print("ANSATZ-PLATEAU DIAGNOSTIC — is there anything for the world model to rank?")
    print(f"  source: stored advfid traces at the `{CKPT}` checkpoint (15 prefixes/seed). No new VQE.")
    print("  A cross-circuit correlation is only defined on seeds whose ranking target actually varies.")
    print("=" * 116)
    print(f"{'task':11s}{'seeds':>6}{'plateau onset':>15}{'onset depth':>13}{'plateau mHa':>14}"
          f"{'distinct lvls':>14}{'usable':>9}   paper reference")
    print("-" * 116)
    for m, disp in MOL.items():
        onset, depth, val, lvls, usable, n = [], [], [], [], 0, 0
        for f in sorted(glob.glob(f"{OF}/dreamqas/gru_energy_surrogate_{m}_s*_of/t1a_probe.json")):
            d = json.load(open(f)).get(CKPT, {}).get("advfid")
            if not d or not d.get("traces"):
                continue
            n += 1
            ends = []
            for t in d["traces"]:
                e = err_trace(t)
                k = plateau_onset(e)
                onset.append(k)
                depth.append(t["seed_len"] + k)
                val.append(e[-1])
                ends.append(e[-1])
            ends = np.array(ends)
            lvls.append(len(np.unique(np.round(ends, 9))))
            if ends.std() > TOL:
                usable += 1
        if not n:
            continue
        rname, rval = REF.get(m, ("", float("nan")))
        match = "== " if abs(np.median(val) - rval) / max(rval, 1e-9) < 0.02 else "vs "
        print(f"{disp:11s}{n:>6}{np.median(onset):>13.0f} 步{np.median(depth):>13.0f}"
              f"{np.median(val):>14.3f}{np.median(lvls):>14.0f}{f'{usable}/{n}':>9}   "
              f"{match}{rname} {rval:g}")
    print("-" * 116)
    print("READING")
    print("  * 'plateau onset 0 步' = the true error never changed anywhere in the probed rollout, so")
    print("    every cross-circuit correlation on that seed is undefined, not low.")
    print("  * 'distinct lvls 1' = the 15 prefixes all end at the SAME real error. Same conclusion.")
    print("  * A plateau value that EQUALS the paper's Full best-of-1 means the probe was ranking")
    print("    circuits that had already reached the reported optimum — the probe is measuring the")
    print("    ansatz limit, not the world model.")
    print("  * Consequence, and it is load-bearing: any near-zero cross-circuit correlation on a task")
    print("    whose 'usable' count is below the seed count must be reported as NOT MEASURABLE. It is")
    print("    not evidence that the world model ranks at chance. This retroactively voids the earlier")
    print("    readings of BeH2-8q (and the absolute-value column of BeH2-10q).")
    print("  * The same root cause explains q2_horizon.txt's saturation guard on BeH2-8q (H1 and H5")
    print("    both pinned at 2.175) — one confound, three probes, not three separate findings.")


if __name__ == "__main__":
    main()
