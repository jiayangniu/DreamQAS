"""T0.3 (WM-diagnostic aggregation) + T0.6 (Full-vs-noDAG longitudinal control), read-only.

Reads the campaign runs' fidelity.jsonl (pairwise ranking accuracy) and calibration.jsonl
(calib_MAE = |pred_logerr - real_logerr| on the DAgger-VERIFIED circuits, i.e. a *biased*
selected sample; disagree_vs_err_corr = Spearman(ensemble std, |error|)). Aggregates per
molecule over 5 seeds and emits the RQ3 base table + the Full-vs-noDAG calibration-growth
comparison that sets the RQ3(b) wording tier.

Honest labels (locked plan):
  - pairwise fidelity = RANKING quality (buffer-val pairs) -> the "good-enough ranker" evidence.
  - calib_MAE(log10) = ABSOLUTE calibration on DAgger-SELECTED circuits (biased) -> "not an oracle".
    Unbiased held-out calibration + signed bias + Spearman/top-k come from the T1.a probe, NOT here.
  - RMSE / signed bias are NOT recoverable from these logs (calibration.jsonl stores aggregate MAE
    only, no per-circuit pred/real arrays) -> deferred to T1.a.
"""
import glob
import json
import os
import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results/wm_diagnostics.txt"

FULL_MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
ABL_MOLS = ["LiH4q", "LiH6q", "BeH2_8q", "BeH2_10q"]          # surrogate ablations have no 6q
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}


def full_dirs(mol):
    return glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q")


def abl_dirs(mol, tag):
    return glob.glob(f"{C}/ablations/gru_energy_surrogate_{mol}_s*_ab_{tag}")


def _last_fidelity(d):
    p = f"{d}/fidelity.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows[-1]["pairwise"] if rows else None


def _summary_calib(d):
    """-> (final calibration MAE, final disagree_corr, trajectory of (frac, MAE)).
    Legacy runs: calib_MAE_all = |pred - log10(true error)| (E0-based log-error MAE).
    oracle_free runs: calib_score_mae = |pred - real frontier score| (S-space MAE) — same role,
    DIFFERENT unit/reference; do not mix the two generations in one table without labeling."""
    p = f"{d}/calibration.jsonl"
    if not os.path.exists(p):
        return None
    sm = [json.loads(l) for l in open(p) if l.strip() and '"SUMMARY"' in l]
    if not sm:
        return None
    key = "calib_MAE_all" if "calib_MAE_all" in sm[-1] else "calib_score_mae"
    n = len(sm)
    traj = [(i / max(1, n - 1), r.get(key)) for i, r in enumerate(sm)]
    return sm[-1].get(key), sm[-1]["disagree_vs_err_corr"], traj


def _at_frac(traj, f):
    """value nearest a training-progress fraction f in [0,1]."""
    if not traj:
        return np.nan
    fr = np.array([t[0] for t in traj]); v = np.array([t[1] for t in traj])
    return float(v[np.argmin(np.abs(fr - f))])


def agg(dirs):
    fid, mae, corr, early, mid, late = [], [], [], [], [], []
    for d in dirs:
        f = _last_fidelity(d); c = _summary_calib(d)
        if f is not None:
            fid.append(f)
        if c is not None:
            mae.append(c[0]); corr.append(c[1])
            early.append(_at_frac(c[2], 0.1)); mid.append(_at_frac(c[2], 0.5)); late.append(_at_frac(c[2], 0.95))
    def ms(x):
        return (np.mean(x), (np.std(x, ddof=1) if len(x) > 1 else 0.0), len(x)) if x else (np.nan, np.nan, 0)
    return dict(fid=ms(fid), mae=ms(mae), corr=ms(corr),
               early=ms(early), mid=ms(mid), late=ms(late))


def fmt(t):
    m, s, n = t
    return f"{m:.3f}±{s:.3f}" if n else "  -  "


lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 96)
P("T0.3  WM DIAGNOSTIC — campaign DreamQAS-Full  (mean±std over seeds)")
P("  fidelity = pairwise RANKING accuracy (held-out buffer pairs) -> decision-useful ranker")
P("  calib-MAE = |pred-real| log10-error on DAgger-SELECTED circuits (biased sample; NOT an oracle)")
P("  10^MAE = multiplicative error factor;  disagree-corr = Spearman(ensemble std, |err|)")
P("=" * 96)
P(f"{'molecule':<12}{'pairwise fid':>16}{'calib-MAE(log10)':>20}{'(=  x factor)':>16}{'disagree-corr':>16}{'n':>4}")
P("-" * 96)
for mol in FULL_MOLS:
    a = agg(full_dirs(mol))
    xf = 10 ** a["mae"][0] if a["mae"][2] else np.nan
    P(f"{DISP[mol]:<12}{fmt(a['fid']):>16}{fmt(a['mae']):>20}{('%.1fx' % xf):>16}{fmt(a['corr']):>16}{a['fid'][2]:>4}")
P("-" * 96)
P("READ: ranking is strong (0.77–0.95) but absolute calibration is poor and molecule-dependent")
P("      (1.7x–5.3x); note the INVERSE pattern — LiH(4q) ranks best yet calibrates worst.")
P("      Reconciliation: earlier notes cited 0.22–0.26 log-MAE (≈1.7x); that is the LiH(6q)/causal-")
P("      chain regime, not a single campaign number. Campaign spans 0.22–0.72 log-MAE across molecules.")
P()

P("=" * 96)
P("T0.6  Full vs noDAG — calibration growth over training  (does DAgger limit error accumulation?)")
P("  calib-MAE(log10) at early(10%) / mid(50%) / late(95%) of training, mean±std over seeds.")
P("  Full CORRECTS via DAgger feedback; noDAG measures the same calibration but never feeds back.")
P("=" * 96)
P(f"{'molecule':<12}{'variant':<9}{'early':>14}{'mid':>14}{'late':>14}{'late-early':>13}{'n':>4}")
P("-" * 96)
for mol in ABL_MOLS:
    for name, dirs in (("Full", full_dirs(mol)), ("noDAG", abl_dirs(mol, "noDAG"))):
        a = agg(dirs)
        grow = a["late"][0] - a["early"][0] if a["late"][2] else np.nan
        P(f"{DISP[mol]:<12}{name:<9}{fmt(a['early']):>14}{fmt(a['mid']):>14}{fmt(a['late']):>14}{('%+.3f' % grow) if not np.isnan(grow) else '  -  ':>13}{a['late'][2]:>4}")
    P("-" * 96)
P("VERDICT RULE (locked plan): if noDAG's late calibration (and/or best-of-1 from T0.1) degrades")
P("  MORE than Full's -> RQ3(b) = 'DAgger limits the accumulation of policy-induced model error'.")
P("  Otherwise weaken to 'selective VQE verification detects and corrects prediction errors on the")
P("  imagined candidates used for policy improvement'. (best-of-1 half needs T0.1 frozen eval.)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
