"""T0.4 — diagnostic on DAgger-SELECTED candidates (read-only, selection-bias acknowledged).

calibration.jsonl non-SUMMARY rows log every DAgger-verified imagined candidate:
  pred_logerr, real_logerr, real_err_mHa, abs_logerr_err, disagree, selection in {top,disagree,both}.
These are the circuits the policy WANTED to use (top-value) or was most UNSURE about (top-disagree)
-> a biased sample. We therefore report ONLY selected-candidate diagnostics and make NO
random-candidate comparison (that is the matched-random arm of the T1.a probe).

Metrics (per campaign-Full run, then mean±std over 5 seeds):
  - signed bias = mean(pred_logerr - real_logerr); NEGATIVE = OPTIMISTIC (WM thinks circuit is
    better/lower-error than it is).  |log10| units.
  - pred-vs-real Spearman (pooled over the run's selected candidates; labelled as pooled, NOT the
    clean per-prefix action-ranking of T1.a).
  - large-error rate = P(|pred-real| > log10(3)) i.e. off by >3x.
  - by selection type: do top-VALUE candidates verify better than top-DISAGREEMENT ones?
"""
import glob
import json
import os
import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results/dagger_diagnostic.txt"
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
LOG3 = np.log10(3.0)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def run_stats(d):
    rows = [json.loads(l) for l in open(f"{d}/calibration.jsonl") if l.strip() and '"SUMMARY"' not in l]
    rows = [r for r in rows if "pred_logerr" in r and "real_logerr" in r]
    if len(rows) < 5:
        return None
    pred = np.array([r["pred_logerr"] for r in rows]); real = np.array([r["real_logerr"] for r in rows])
    bias = float(np.mean(pred - real))                    # <0 = optimistic
    rho = spearman(pred, real)
    big = float(np.mean(np.abs(pred - real) > LOG3))
    out = {"n": len(rows), "bias": bias, "rho": rho, "big": big}
    for sel in ("top", "disagree"):
        rr = [r for r in rows if r["selection"] == sel]
        out[f"mae_{sel}"] = float(np.mean([r["abs_logerr_err"] for r in rr])) if rr else np.nan
    return out


def agg(mol):
    S = [run_stats(d) for d in glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q")]
    S = [s for s in S if s]
    if not S:
        return None
    def ms(k):
        v = [s[k] for s in S if not (isinstance(s[k], float) and np.isnan(s[k]))]
        return (np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)) if v else (np.nan, np.nan)
    return {k: ms(k) for k in ("n", "bias", "rho", "big", "mae_top", "mae_disagree")}, len(S)


lines = []
def P(s=""):
    lines.append(s); print(s)

P("=" * 100)
P("T0.4  DIAGNOSTIC ON DAgger-SELECTED CANDIDATES  (campaign DreamQAS-Full, mean±std over seeds)")
P("  selection-biased sample (top-value + top-disagreement); NO random comparison here (-> T1.a).")
P("  bias = mean(pred-real) log10;  NEGATIVE = OPTIMISTIC.   rho = pred-vs-real Spearman (pooled).")
P("  >3x rate = P(|pred-real|>log10 3).   mae_top / mae_dis = |pred-real| by selection type.")
P("=" * 100)
P(f"{'molecule':<12}{'n/run':>8}{'signed bias':>14}{'rho(pooled)':>13}{'>3x rate':>10}{'MAE top':>10}{'MAE dis':>10}{'seeds':>6}")
P("-" * 100)
for mol in MOLS:
    r = agg(mol)
    if not r:
        P(f"{DISP[mol]:<12}{'  (no data)':>8}"); continue
    a, ns = r
    def f(k, p="%.3f"):
        m, s = a[k]; return (p % m) if not np.isnan(m) else "  -  "
    P(f"{DISP[mol]:<12}{f('n','%.0f'):>8}{f('bias'):>14}{f('rho'):>13}{f('big'):>10}{f('mae_top'):>10}{f('mae_disagree'):>10}{ns:>6}")
P("-" * 100)
P("READ (fill after run): sign & size of the optimistic bias; whether pred still RANKS real well")
P("  on selected candidates (rho); and whether top-value candidates verify better than top-disagree.")
P("  This is the T0.4 half of RQ3(b): verification observes real error on the candidates it selects.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\n[written] {OUT}")
