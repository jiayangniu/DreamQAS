"""DAgger VQE-efficiency (Full vs noDAG) on the RQ2 W=100 training-time basis, ALL 5 tasks. READ-ONLY.

Unifies the DAgger-efficiency window with Figure 2 (W=100). Reuses the RQ2 W=100 curve machinery
(`plot_rq2_w100.dq_episodes_true_vqe` = per-episode (cum_real_vqe, episode-best); the x-axis is
`metrics.jsonl` vqe_calls, which INCLUDES DAgger verification VQE for Full and EXCLUDES the diagnostic
calibration VQE for noDAG — verified: noDAG vqe_calls == trajectory steps, calib_vqe_calls uncounted).
Same trailing moving-mean + DOWN=50 + left-truncated early window; only W changes. No frozen eval_traces.

Per task: Q_Full = median_j seed final-window (last 10% reported points) mean; Q_noDAG likewise;
common target Q = max(Q_Full, Q_noDAG) (larger error = the level BOTH can reach). Per-seed sustained
crossing = first cum-VQE after which the W-smoothed curve stays <= Q*1.0001 for 3 consecutive reported
points (stride 50). Right-censored seeds excluded from the median (counted, not imputed).
S_DAG = median C_noDAG / median C_Full  (>1 = DAgger reaches Q with FEWER VQE). 95% CI = 3000x unpaired
bootstrap over reached seeds (fixed rng seed 7), ratio of bootstrap medians, 2.5/97.5 pct.

Computed for W in {100, 400} on the SAME (DAgger-inclusive) machinery so the comparison isolates the
window. (The committed `dagger_efficiency.txt` is the OLD W=400 with a DAgger-EXCLUDED x — kept as-is.)

Outputs: dagger_efficiency_w100.txt, dagger_efficiency_w100.csv, DAGGER_EFFICIENCY_W100_AUDIT.md.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_rq2_w100 import dq_episodes_true_vqe          # per-episode (cum_real_vqe incl DAgger, ebest)

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
OUT = f"{HERE}/outputs/main_results"
DOWN = 50
TASKS = ["LiH4q", "BeH2_6q", "LiH6q", "BeH2_8q", "BeH2_10q"]
KEY = {"LiH4q": "LiH4q", "BeH2_6q": "BeH2", "LiH6q": "LiH6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
DISP = {"LiH4q": "LiH(4q)", "BeH2_6q": "BeH2(6q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
# committed OLD W=400 reference (dagger_efficiency.txt; DAgger-EXCLUDED x) + BeH2-6q from the audit
OLD_W400 = {"LiH4q": (0.146, 1.03), "BeH2_6q": (0.837, 0.94), "LiH6q": (10.044, 1.48),
            "BeH2_8q": (2.513, 1.28), "BeH2_10q": (1.639, 1.13)}


def arm_dirs(arm, key):
    if arm == "Full":
        ds = glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{key}_s*_q")
        return sorted(d for d in ds if re.search(rf"_{key}_s\d_q$", d))
    ds = glob.glob(f"{C}/ablations/gru_energy_surrogate_{key}_s*_ab_noDAG")
    return sorted(d for d in ds if re.search(rf"_{key}_s\d_ab_noDAG$", d))


def curve_w(eps, w):
    """[(cum_vqe, ebest)] -> (cum_vqe_reported[], trailing-mean-over-w ebest[]) downsampled by DOWN.
    Same logic as plot_rq2_w100.curve, W-parametric (left-truncated early window)."""
    x = np.array([e[0] for e in eps], float)
    b = np.array([e[1] for e in eps], float)
    csum = np.cumsum(np.insert(b, 0, 0.0))
    idx = np.arange(len(b))
    lo = np.maximum(0, idx - w + 1)
    win = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    sel = np.arange(0, len(b), DOWN)
    return x[sel], win[sel]


def seed_curves(arm, key, w):
    out = []
    for d in arm_dirs(arm, key):
        eps = dq_episodes_true_vqe(d)
        if eps and len(eps) > w:
            out.append(curve_w(eps, w))
    return out


def final_window_q(cvs):
    return [float(np.mean(y[int(len(y) * 0.9):])) for _, y in cvs]


def cross(cv, Q, sustain=3):
    x, y = cv
    for i in range(len(y) - sustain + 1):
        if all(y[i + r] <= Q * 1.0001 for r in range(sustain)):
            return float(x[i])
    return np.nan


def analyse(task, w, rng):
    key = KEY[task]
    cf, cn = seed_curves("Full", key, w), seed_curves("noDAG", key, w)
    qf_seeds = final_window_q(cf); qn_seeds = final_window_q(cn)
    Qf, Qn = float(np.median(qf_seeds)), float(np.median(qn_seeds))
    Q = max(Qf, Qn)
    vf = np.array([cross(c, Q) for c in cf]); vn = np.array([cross(c, Q) for c in cn])
    rf, rn = vf[np.isfinite(vf)], vn[np.isfinite(vn)]
    res = dict(task=task, Qf=Qf, Qn=Qn, Q=Q, vf=vf, vn=vn,
               nf=len(cf), nn=len(cn), rf=len(rf), rn=len(rn),
               mf=(float(np.median(rf)) if len(rf) else np.nan),
               mn=(float(np.median(rn)) if len(rn) else np.nan))
    if len(rf) and len(rn):
        res["sp"] = res["mn"] / res["mf"]
        bs = [np.median(rng.choice(rn, len(rn), True)) / np.median(rng.choice(rf, len(rf), True))
              for _ in range(3000)]
        res["ci"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    else:
        res["sp"] = np.nan; res["ci"] = (np.nan, np.nan)
    return res


def verdict(r):
    if not np.isfinite(r["sp"]):
        return "inconclusive (an arm never reaches Q)"
    lo, hi = r["ci"]
    censored = (r["nf"] - r["rf"]) + (r["nn"] - r["rn"])
    if censored >= 3:
        return f"heavily censored ({censored} seeds) — inconclusive"
    if lo > 1.0:
        return "clear efficiency gain (DAgger faster; CI>1)"
    if hi < 1.0:
        return "NoDAG faster (CI<1)"
    if 0.9 <= r["sp"] <= 1.1:
        return "neutral (~1x)"
    return f"point estimate {r['sp']:.2f}x but CI crosses 1 — not conclusive"


def main():
    rng = np.random.default_rng(7)
    R100 = {t: analyse(t, 100, rng) for t in TASKS}
    R400 = {t: analyse(t, 400, rng) for t in TASKS}

    # ---- txt ----
    L = ["=" * 116,
         "DAgger VQE-EFFICIENCY (Full vs noDAG) — RQ2 W=100 training-time basis, all 5 tasks",
         "  curve = min-prefix episode-best -> W=100 trailing mean (DOWN=50); x = real VQE (Full incl. DAgger; noDAG excl. calib diag)",
         "  Q = max(median Q_Full, median Q_noDAG); sustained crossing (3 pts, stride 50); censored excluded from median.",
         "  gain S_DAG = median C_noDAG / median C_Full  (>1 = DAgger reaches Q with FEWER VQE); 95% CI = 3000x unpaired bootstrap.",
         "=" * 116,
         f"{'task':<10}{'Q_Full':>9}{'Q_noDAG':>9}{'Q(mHa)':>9}{'Full med':>11}{'noDAG med':>11}{'F reach':>9}{'nD reach':>9}{'gain':>7}{'95% CI':>15}",
         "-" * 116]
    for t in TASKS:
        r = R100[t]
        gain = f"{r['sp']:.2f}x" if np.isfinite(r["sp"]) else "  -  "
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]" if np.isfinite(r["sp"]) else "   -   "
        mf = f"{r['mf']:.0f}" if np.isfinite(r["mf"]) else "-"
        mn = f"{r['mn']:.0f}" if np.isfinite(r["mn"]) else "-"
        freach = f"{r['rf']}/{r['nf']}"; nreach = f"{r['rn']}/{r['nn']}"
        L.append(f"{DISP[t]:<10}{r['Qf']:>9.3f}{r['Qn']:>9.3f}{r['Q']:>9.3f}{mf:>11}{mn:>11}"
                 f"{freach:>9}{nreach:>9}{gain:>7}{ci:>15}")
    L += ["-" * 116, "", "VERDICT per task (W=100):"]
    for t in TASKS:
        L.append(f"  {DISP[t]:<10} {verdict(R100[t])}")
    L += ["", "W=100 vs W=400 SENSITIVITY (both on the DAgger-inclusive x; committed dagger_efficiency.txt used a DAgger-EXCLUDED x):",
          f"{'task':<10}{'W400 Q':>9}{'W400 gain':>11}{'W100 Q':>9}{'W100 gain':>11}   direction changed?"]
    for t in TASKS:
        a, b = R400[t], R100[t]
        ga = f"{a['sp']:.2f}x" if np.isfinite(a["sp"]) else "-"
        gb = f"{b['sp']:.2f}x" if np.isfinite(b["sp"]) else "-"
        chg = "no"
        if np.isfinite(a["sp"]) and np.isfinite(b["sp"]):
            chg = "YES (crosses 1)" if (a["sp"] - 1) * (b["sp"] - 1) < 0 else "no"
        L.append(f"{DISP[t]:<10}{a['Q']:>9.3f}{ga:>11}{b['Q']:>9.3f}{gb:>11}   {chg}")
    open(f"{OUT}/dagger_efficiency_w100.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- csv ----
    with open(f"{OUT}/dagger_efficiency_w100.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "Q_full", "Q_nodag", "Q_common", "full_crossing_median", "nodag_crossing_median",
                    "full_reach", "full_total", "nodag_reach", "nodag_total", "speedup", "ci_low", "ci_high",
                    "full_censored_seeds", "nodag_censored_seeds"])
        for t in TASKS:
            r = R100[t]
            w.writerow([DISP[t], round(r["Qf"], 4), round(r["Qn"], 4), round(r["Q"], 4),
                        (round(r["mf"], 1) if np.isfinite(r["mf"]) else ""),
                        (round(r["mn"], 1) if np.isfinite(r["mn"]) else ""),
                        r["rf"], r["nf"], r["rn"], r["nn"],
                        (round(r["sp"], 3) if np.isfinite(r["sp"]) else ""),
                        (round(r["ci"][0], 3) if np.isfinite(r["sp"]) else ""),
                        (round(r["ci"][1], 3) if np.isfinite(r["sp"]) else ""),
                        r["nf"] - r["rf"], r["nn"] - r["rn"]])

    # ---- audit md (per-seed crossings incl NaN) ----
    A = ["# DAgger VQE-efficiency (W=100) — audit", "",
         "Read-only recompute on the RQ2 W=100 training-time basis, all 5 tasks. Full x includes DAgger",
         "verification VQE (`metrics.jsonl` vqe_calls); noDAG x excludes the diagnostic calibration VQE",
         "(verified: noDAG vqe_calls == trajectory steps; calib_vqe_calls uncounted). No frozen eval_traces.",
         "Reuses `plot_rq2_w100.dq_episodes_true_vqe` + a W-parametric `curve_w` (same trailing-mean, DOWN=50,",
         "left-truncated early window). Q = max(median Q_Full, median Q_noDAG). Sustained crossing 3 pts, stride 50.",
         "Bootstrap: 3000 unpaired resamples over reached seeds, rng seed 7, ratio of medians, 2.5/97.5 pct.", "",
         "## Per-seed crossing VQE (W=100)  — NaN = right-censored (never sustains ≤ Q)"]
    for t in TASKS:
        r = R100[t]
        fv = ", ".join(("NaN" if not np.isfinite(v) else f"{v:.0f}") for v in r["vf"])
        nv = ", ".join(("NaN" if not np.isfinite(v) else f"{v:.0f}") for v in r["vn"])
        A += [f"### {DISP[t]}  —  Q_Full={r['Qf']:.3f}, Q_noDAG={r['Qn']:.3f}, **Q={r['Q']:.3f} mHa**",
              f"- Full  seeds: [{fv}]  reached {r['rf']}/{r['nf']}, censored {r['nf']-r['rf']}; median {('%.0f'%r['mf']) if np.isfinite(r['mf']) else '—'}",
              f"- noDAG seeds: [{nv}]  reached {r['rn']}/{r['nn']}, censored {r['nn']-r['rn']}; median {('%.0f'%r['mn']) if np.isfinite(r['mn']) else '—'}",
              f"- **gain S_DAG = {('%.2f'%r['sp']) if np.isfinite(r['sp']) else '—'}×**"
              f"{('  95%% CI [%.2f, %.2f]'%r['ci']) if np.isfinite(r['sp']) else ''}  →  {verdict(r)}", ""]
    A += ["## W=100 vs W=400 sensitivity (both DAgger-inclusive x)",
          "| task | W=400 Q | W=400 gain | W=100 Q | W=100 gain | direction changed? |",
          "|---|---|---|---|---|---|"]
    for t in TASKS:
        a, b = R400[t], R100[t]
        ga = f"{a['sp']:.2f}×" if np.isfinite(a["sp"]) else "—"
        gb = f"{b['sp']:.2f}×" if np.isfinite(b["sp"]) else "—"
        chg = "no"
        if np.isfinite(a["sp"]) and np.isfinite(b["sp"]):
            chg = "**YES**" if (a["sp"] - 1) * (b["sp"] - 1) < 0 else "no"
        A.append(f"| {DISP[t]} | {a['Q']:.3f} | {ga} | {b['Q']:.3f} | {gb} | {chg} |")
    A += ["", "Note: the committed `dagger_efficiency.txt` (old W=400) used a DAgger-EXCLUDED x, so its",
          "gains (LiH6q 1.48×, BeH2-8q 1.28×) differ from the DAgger-inclusive W=400 above; kept as-is.",
          "", "## LaTeX-ready (W=100)",
          "| Task | Q (mHa) | Full VQE | NoDAG VQE | VQE gain | 95% CI | Reach F/N |",
          "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        r = R100[t]
        mf = f"{r['mf']:.0f}" if np.isfinite(r["mf"]) else "—"
        mn = f"{r['mn']:.0f}" if np.isfinite(r["mn"]) else "—"
        gain = f"{r['sp']:.2f}×" if np.isfinite(r["sp"]) else "—"
        ci = f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if np.isfinite(r["sp"]) else "—"
        A.append(f"| {DISP[t]} | {r['Q']:.2f} | {mf} | {mn} | {gain} | {ci} | {r['rf']}/{r['rn']} |")
    A += ["", "## Conclusion (strictly from W=100; the old 1.48×/1.28× are NOT assumed)"]
    for t in TASKS:
        r = R100[t]
        detail = ""
        if np.isfinite(r["sp"]):
            detail = f" (S_DAG={r['sp']:.2f}×, CI [{r['ci'][0]:.2f},{r['ci'][1]:.2f}], reach {r['rf']}/{r['rn']})"
        A.append(f"- **{DISP[t]}**: {verdict(r)}{detail}")
    open(f"{OUT}/DAGGER_EFFICIENCY_W100_AUDIT.md", "w").write("\n".join(A) + "\n")
    print(f"\n[written] dagger_efficiency_w100.{{txt,csv}} + DAGGER_EFFICIENCY_W100_AUDIT.md in {OUT}")


if __name__ == "__main__":
    main()
