"""Four-stage causal ladder (RQ4 + T0.6 policy half): RLQAS -> No-imag -> noDAG -> Full.

best-of-1 = mean of the FINAL checkpoint's ep_best over its eval episodes (deploy-faithful, no
cross-rollout selection), per seed, then mean±std over seeds. Incremental ratios are reported in
log10 space (the metric spans orders of magnitude); they are implementation-matched incremental
comparisons, NOT an additive decomposition (components interact).

  RLQAS  -> No-imag : WM representation-and-replay pipeline
  No-imag-> noDAG   : imagination increment
  noDAG  -> Full    : DAgger increment  (T0.6 policy half)
"""
import glob
import json
import os
import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "BeH2_6q", "LiH6q", "BeH2_8q", "BeH2_10q"]   # BeH2_6q noDAG backfilled 2026-07-19
DISP = {"LiH4q": "LiH(4q)", "BeH2_6q": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
# BeH2-6q run dirs use the bare "BeH2" molecule key (8q/10q use "BeH2_8q"/"BeH2_10q").
MOLKEY = {"BeH2_6q": "BeH2"}
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results/ladder_table.txt"


def dirs(stage, mol):
    k = MOLKEY.get(mol, mol)
    if stage == "Full":    return glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{k}_s*_q")
    if stage == "No-imag": return glob.glob(f"{C}/ablations/gru_energy_none_{k}_s*_ab_noimag")
    if stage == "noDAG":   return glob.glob(f"{C}/ablations/gru_energy_surrogate_{k}_s*_ab_noDAG")
    if stage == "RLQAS":   return glob.glob(f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{k}/seed_*")
    return []


def bo1_per_seed(d):
    """best-of-1 = mean of the final checkpoint's ep_best; final = record with the most eval eps."""
    p = f"{d}/eval_traces.jsonl"
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("ep_best")]
    if not recs:
        return None
    fin = max(recs, key=lambda r: (r.get("n_eval_episodes", 0), r.get("_ck_ep", 0)))
    b = np.array(fin["ep_best"], float)
    b = b[np.isfinite(b)]
    return float(b.mean()) if b.size else None


def agg(stage, mol):
    v = [bo1_per_seed(d) for d in dirs(stage, mol)]
    v = [x for x in v if x is not None]
    return (np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0), len(v), np.array(v)) if v else (np.nan, np.nan, 0, np.array([]))


lines = []
def P(s=""):
    lines.append(s); print(s)

STAGES = ["RLQAS", "No-imag", "noDAG", "Full"]
P("=" * 104)
P("FOUR-STAGE CAUSAL LADDER — frozen best-of-1 error (mHa, lower=better), mean±std over 5 seeds")
P("  RLQAS -> No-imag = representation+replay ; No-imag -> noDAG = imagination ; noDAG -> Full = DAgger")
P("  ratios in log10 space; implementation-matched incremental comparisons, NOT additive.")
P("=" * 104)
P(f"{'molecule':<11}{'RLQAS':>14}{'No-imag':>14}{'noDAG':>14}{'Full':>14}   |  repr×   imag×   DAgger×")
P("-" * 104)
res = {}
for mol in MOLS:
    a = {s: agg(s, mol) for s in STAGES}
    res[mol] = a
    def cell(s):
        m, sd, n, _ = a[s]
        return f"{m:.3f}±{sd:.3f}" if n else "   -   "
    def ratio(s1, s2):  # s1 worse (higher) / s2 better (lower) -> factor >1 = improvement
        m1, m2 = a[s1][0], a[s2][0]
        return (m1 / m2) if (a[s1][2] and a[s2][2] and m2 > 0) else float("nan")
    r_repr, r_imag, r_dag = ratio("RLQAS", "No-imag"), ratio("No-imag", "noDAG"), ratio("noDAG", "Full")
    P(f"{DISP[mol]:<11}{cell('RLQAS'):>14}{cell('No-imag'):>14}{cell('noDAG'):>14}{cell('Full'):>14}   |  "
      f"{r_repr:5.1f}  {r_imag:5.1f}  {r_dag:5.1f}")
P("-" * 104)
# collapse guard: any cell whose std exceeds its mean is mean-poisoned by a single collapsed seed
# -> report the per-seed values + median so the ratios are read correctly (mirrors HyRLQAS handling).
flagged = []
for mol in MOLS:
    for s in STAGES:
        m, sd, n, arr = res[mol][s]
        if n >= 3 and sd > m and m > 0:
            flagged.append((mol, s, arr))
if flagged:
    P("⚠ MEAN-POISONED CELLS (std>mean = one collapsed seed; use MEDIAN for these — ratios above are mean-based):")
    for mol, s, arr in flagged:
        P(f"  {DISP[mol]} {s}: per-seed {np.round(arr,3).tolist()}  mean={arr.mean():.3f}  MEDIAN={np.median(arr):.3f}")
    P()
P("T0.6 POLICY VERDICT: compare noDAG vs Full best-of-1 per molecule (does DAgger help the POLICY?).")
for mol in MOLS:
    nd, fu = res[mol]["noDAG"], res[mol]["Full"]
    if nd[2] and fu[2]:
        r = nd[0] / fu[0] if fu[0] > 0 else float("nan")
        tag = "DAgger helps" if r > 1.3 else ("~neutral" if r > 0.77 else "DAgger HURTS")
        extra = ""
        if nd[1] > nd[0]:                        # noDAG mean-poisoned -> add the median verdict
            ndm = np.median(nd[3]); rm = ndm / fu[0] if fu[0] > 0 else float("nan")
            extra = f"  |  MEDIAN noDAG {ndm:.3f} -> {rm:.2f}x [collapse-rescue, not median gain]"
        P(f"  {DISP[mol]:<11} noDAG {nd[0]:.3f} vs Full {fu[0]:.3f}  -> {r:.2f}x  [{tag}]{extra}")
P()
P("READ (locked interpretation):")
P("  * On LiH(4q)/LiH(6q)/BeH2(8q)/BeH2(10q): DAgger is ~neutral on the median best-of-1 (0.6–1.25x).")
P("  * BeH2(6q) is the informative easy case, read on the MEDIAN: representation+replay HURTS the")
P("    easy molecule (RLQAS 0.094 -> No-imag 0.341); imagination RECOVERS it (median 0.341 -> 0.058,")
P("    ~5.9x); DAgger is median-neutral (0.058 -> 0.058) BUT its real-VQE verification RESCUES the 1/5")
P("    seed where imagination derailed (noDAG s0=6.09). So DAgger = a rare-catastrophe safety net on")
P("    easy molecules (variance reduction), NOT a systematic median-quality gain. The 21.9x mean ratio")
P("    is entirely that one rescued seed.")
P("  * Combined with T0.6 calibration half (Full≈noDAG calibration): DAgger does not heal point-")
P("    calibration; its value is verification/efficiency + occasional collapse-prevention.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
