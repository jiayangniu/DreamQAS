"""Unified oracle-free MAIN policy-quality table (storyline §11 item 5).

Combines two provenances into one table and labels every row, because they are NOT the same campaign:

  internal arms  (Full / No-imag / DreamQAS-RL)  -> oracle_free_v1, E0-FREE training signal,
                 reported at the LOCKED COMMON BUDGET of 15,000 episodes. LiH6q is the only task
                 configured with 2x, so its cell comes from the ep15000 re-evaluation
                 (`eval_traces_ep15000.jsonl`, 100 eval episodes) rather than its 30k final checkpoint.
  external       (CRLQAS / HyRLQAS / GQE / TFQAS / QuantumDARTS) -> campaign_v1, each under its OWN
                 native reward and stopping protocol. Numbers are taken verbatim from the existing SSOT
                 loader in `policy_quality_table.py` (imported, not reimplemented) so they cannot drift.

⚠ Three asymmetries that MUST stay in the caption:
  1. Training signal differs: internal arms are oracle-free; the externals have their own rewards, and
     the *canonical* internal arms are a different run set entirely — never mix their numbers in.
  2. Budget: our LiH6q uses 15,000 episodes while several externals run longer under native protocols.
     That is unfavourable to us, so it may be stated but never exploited in reverse.
  3. VQE comparability: internal arms + CRLQAS/HyRLQAS share the VQE counting convention;
     GQE/TFQAS/QuantumDARTS use native accounting -> quality comparison only, no matched-VQE speedup.

Metric = frozen best-of-1, mean +- sample std (ddof=1) over 5 seeds. qdarts delivers ONE architecture
per run, so best-of-1 == best-of-10 by construction and its +- is run-to-run search variance.

Usage: python analysis/of_main_table.py > outputs/main_results/of_main_table.txt
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_quality_table as PQ          # SSOT for the external baselines (import only; no side effects)

OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
COMMON_EP = 15000                                   # the locked common budget
INTERNAL = [("DreamQAS (Full)", f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of"),
            ("No-imag", f"{OF}/ablations/gru_energy_none_{{m}}_s*_of_noimag"),
            ("DreamQAS-RL", f"{OF}/dreamqas_rlqas/baseline_analysis/G0_v2_{{m}}/seed_*")]
EXTERNAL = ["CRLQAS", "HyRLQAS", "GQE", "TFQAS", "qdarts"]
CHEM_ACC = 1.6


def internal_bo1(pattern, mol):
    """best-of-1 per seed at the COMMON budget. LiH6q -> the ep15000 re-eval file; others -> final."""
    out = []
    for d in sorted(glob.glob(pattern.format(m=mol))):
        f = f"{d}/eval_traces_ep{COMMON_EP}.jsonl" if mol == "LiH6q" else f"{d}/eval_traces.jsonl"
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [r for r in rows if r.get("ep_best")]
        if not rows:
            continue
        r = max(rows, key=lambda r: r.get("_ck_ep", r["episode"]))
        out.append(float(np.mean(r["ep_best"])))
    return out


def fmt(vals):
    if not vals:
        return "—"
    x = np.asarray(vals, float)
    return f"{x.mean():.3g}±{x.std(ddof=1):.2g}" if len(x) > 1 else f"{x.mean():.3g}"


def main():
    print("=" * 122)
    print("UNIFIED MAIN POLICY-QUALITY TABLE — oracle-free internal arms + external baselines")
    print("  frozen best-of-1, mHa, mean ± sample std (ddof=1), 5 seeds. Lower is better.")
    print(f"  Internal arms: ORACLE-FREE training, reported at the common {COMMON_EP:,}-episode budget")
    print("    (LiH6q from its ep15000 re-evaluation, 100 eval episodes — not its 2x final checkpoint).")
    print("  External baselines: CANONICAL campaign, each under its own native reward/stopping protocol.")
    print("  ⚠ Two provenances in one table — every row is labelled; do not merge the vocabularies.")
    print("=" * 122)
    hdr = f"{'method':22s}{'source':12s}" + "".join(f"{DISP[m]:>17s}" for m in MOLS)
    print(hdr)
    print("-" * 122)
    rows = {}
    for name, pat in INTERNAL:
        cells = [internal_bo1(pat, m) for m in MOLS]
        rows[name] = cells
        print(f"{name:22s}{'oracle-free':12s}" + "".join(f"{fmt(c):>17s}" for c in cells))
    print("-" * 122)
    for meth in EXTERNAL:
        cells = []
        for m in MOLS:
            a = PQ.agg(meth, m)
            cells.append(None if a is None else a["B1_1"])
        disp = "QuantumDARTS †" if meth == "qdarts" else meth
        cs = ["—" if c is None else f"{c[0]:.3g}±{c[1]:.2g}" for c in cells]
        print(f"{disp:22s}{'canonical':12s}" + "".join(f"{x:>17s}" for x in cs))
    print("-" * 122)
    print("† QuantumDARTS is a deterministic differentiable NAS delivering ONE architecture per run, so")
    print("  best-of-1 ≡ best-of-10 by construction; its ± is variance across 5 independent search runs.")
    # chemical accuracy + placement, computed rather than asserted
    print(f"\nChemical accuracy ({CHEM_ACC} mHa) reached by DreamQAS (Full):")
    full = rows["DreamQAS (Full)"]
    for m, c in zip(MOLS, full):
        if c:
            v = float(np.mean(c))
            print(f"  {DISP[m]:12s} {v:8.3f}  {'YES' if v < CHEM_ACC else 'no'}")
    print("\n⚠ CAPTION REQUIREMENTS (do not drop):")
    print("  1. Training-signal asymmetry: internal arms are oracle-free; externals use native rewards;")
    print("     the canonical internal arms are a SEPARATE run set and must not be mixed in.")
    print("  2. Budget asymmetry: our LiH6q uses the 15,000-episode common budget while several externals")
    print("     run longer natively. This is UNFAVOURABLE to us — state it, never exploit it in reverse.")
    print("  3. VQE comparability: internal arms + CRLQAS/HyRLQAS share the VQE counting convention;")
    print("     GQE/TFQAS/QuantumDARTS use native accounting -> quality comparison ONLY, no speedup claim.")
    print("  4. Collapse cells stay in the MEAN (HyRLQAS's large means are real deterministic mode")
    print("     collapse on 1-of-5 seeds); the median is a footnote diagnostic, never a substitute.")


if __name__ == "__main__":
    main()
