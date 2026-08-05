"""Imagination-strength dose-response (LiH4q, 1/4-budget checkpoint).

The two missing imagination-strength axes (imagination horizon was the third, already done):
  Axis A — imagination BREADTH: imag_n_seeds ∈ {8, 32, 64, 256}  (64 = canonical Full).
  Axis B — imagination gradient WEIGHT: imag_loss_weight λ ∈ {1, 4, 10, 30}  (1 = canonical Full).

y = frozen best-of-1 at the **1/4-budget checkpoint** (ep~3748), MEAN±std over 5 seeds (no mean/median
mixing — locked convention; the median is not used here). Source = eval_traces_ep3748.jsonl written by
eval_policy_traces.py --only_ep 3750. Anchors (n_seeds=64, λ=1) come from the canonical Full retrain
(horizon_rerun/dreamqas, which kept checkpoints).

Axis B x-value = MEASURED imagined-gradient share, not nominal λ: `imag_grad_frac` =
   ‖∇imag‖/(‖∇real‖+‖∇imag‖) IS logged per run in `imag.jsonl` (runner.py:427, every wm_refresh_every
   iters when imagination is on). We report the mean over iters ≤ the 1/4-budget point (iter 937), matching
   the best-of-1 checkpoint. Measured shares: λ=1→13.4%, λ=4→37.8%, λ=10→60.4%, λ=30→79.2% (per-λ std
   0.4–1.7%). (The historical "≈7%" was a single-point P1-era value; the 1/4-budget-window mean here is
   13.4%.)

Outputs: imag_strength_dose.{png,pdf} + imag_strength_dose.csv
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/outputs/main_results"
ABL = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1/ablations"
HR = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas"
CM = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1/dreamqas"   # campaign main Full (LiH4q full-budget anchor)

# Per-molecule axis globs. 64 / lambda=1 = canonical Full (anchor).
#   LiH4q anchor = horizon_rerun retrain (kept ckpts); BeH2 anchor = the freshly-trained _ab_anchor
#   (campaign LiH4q/BeH2 Full ckpts were deleted, so each molecule uses its own kept-ckpt anchor).
def mol_axes(mol, budget="quarter"):
    if mol == "LiH4q":
        # LiH4q anchor source is budget-specific: horizon_rerun has the ¼-budget ckpt (ep3748) but
        # only reached ~45%; the full-budget anchor is the campaign main Full's stored final eval
        # (ckpts deleted, but eval_traces_ep15000.jsonl + imag.jsonl are present).
        anchor = (f"{CM}/gru_energy_surrogate_LiH4q_s*_q" if budget == "full"
                  else f"{HR}/gru_energy_surrogate_LiH4q_s*_q")
        pre = f"{ABL}/gru_energy_surrogate_LiH4q_s*_ab"
    elif mol == "BeH2":
        anchor = f"{ABL}/gru_energy_surrogate_BeH2_s*_ab_anchor"    # freshly trained; has both ep3748 & ep15000
        pre = f"{ABL}/gru_energy_surrogate_BeH2_s*_ab"
    else:
        raise SystemExit(f"unknown molecule {mol}")
    nseed = [(8, f"{pre}_nseed8"), (32, f"{pre}_nseed32"), (64, anchor), (256, f"{pre}_nseed256")]
    lam = [(1, anchor), (4, f"{pre}_lam4"), (10, f"{pre}_lam10"), (30, f"{pre}_lam30")]
    return nseed, lam


# Budget-dependent read points (set by main via --budget). Both molecules: 3750 iters total.
#   quarter (¼-budget, paper operating point): eval ep3748, grad-share iters<=937.
#   full    (converged):                       eval ep15000, grad-share iters<=3750.
EVAL_EP = 3748     # frozen-eval checkpoint episode to read best-of-1 from
QITER = 937        # grad-share averaging window (iters <=)


def bo1_quarter(run_dir):
    """frozen best-of-1 = mean of ep_best over eval episodes, at the EVAL_EP checkpoint (budget-set).
    Reads eval_traces_ep<EVAL_EP>.jsonl EXPLICITLY (not sorted-first — both ¼ and full files coexist)."""
    f = f"{run_dir}/eval_traces_ep{EVAL_EP}.jsonl"
    if not os.path.exists(f):                              # fall back to the nearest available eval file
        fs = glob.glob(f"{run_dir}/eval_traces_ep*.jsonl")
        if not fs:
            return None
        f = min(fs, key=lambda p: abs(int(os.path.basename(p)[len('eval_traces_ep'):-len('.jsonl')]) - EVAL_EP))
    try:
        txt = open(f).read().strip()
        if not txt:
            return None
        rec = json.loads(txt.split("\n")[0])
    except (ValueError, OSError):
        return None
    b = np.array([x for x in rec.get("ep_best", []) if np.isfinite(x)], float)
    return float(b.mean()) if b.size else None


def grad_frac(run_dir, upto=None):
    """MEASURED imagined-gradient share = mean of imag_grad_frac over iters <= QITER (budget window).
    imag_grad_frac = ‖∇imag‖/(‖∇real‖+‖∇imag‖), logged in imag.jsonl every wm_refresh_every iters
    when imagination is on (runner.py:427)."""
    upto = QITER if upto is None else upto
    p = f"{run_dir}/imag.jsonl"
    if not os.path.exists(p):
        return None
    vs = []
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l); g = r.get("imag_grad_frac")
        if g is not None and r.get("iter", 10**9) <= upto:
            vs.append(float(g))
    return float(np.mean(vs)) if vs else None


def agg(pattern):
    vals = [bo1_quarter(d) for d in sorted(glob.glob(pattern))]
    vals = [v for v in vals if v is not None]
    a = np.array(vals, float)
    if a.size == 0:
        return None
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0), a.size, a


def collect(axis):
    rows = []
    for xval, pat in axis:
        r = agg(pat)
        if r:
            rows.append((xval, *r))
    return rows


def collect_gradshare(axis):
    """per variant: (lambda, grad_share_mean, grad_share_std, bo1_mean, bo1_std, n)."""
    rows = []
    for lam, pat in axis:
        ds = sorted(glob.glob(pat))
        gs = np.array([g for g in (grad_frac(d) for d in ds) if g is not None], float)
        bs = np.array([b for b in (bo1_quarter(d) for d in ds) if b is not None], float)
        if gs.size and bs.size:
            rows.append((lam, gs.mean(), gs.std(ddof=1) if gs.size > 1 else 0.0,
                         bs.mean(), bs.std(ddof=1) if bs.size > 1 else 0.0, bs.size))
    return rows


def main(mol="LiH4q", budget="quarter"):
    global EVAL_EP, QITER
    EVAL_EP, QITER = (15000, 3750) if budget == "full" else (3748, 937)
    os.makedirs(OUT, exist_ok=True)
    NSEED, LAM = mol_axes(mol, budget)
    disp = {"LiH4q": "LiH-4q", "BeH2": "BeH$_2$-6q"}[mol]
    bl = "full-budget" if budget == "full" else "¼-budget"
    suf = ("" if mol == "LiH4q" else f"_{mol}") + ("" if budget == "quarter" else "_full")
    A = collect(NSEED)          # (n_seeds, mean, std, n, arr)
    Bg = collect_gradshare(LAM)  # (lambda, gs_mean, gs_std, bo1_mean, bo1_std, n)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw=dict(wspace=0.30))
    CANON = "#d62728"

    def panel(ax, rows, anchor_x, xlabel, title, tag):
        xs = [r[0] for r in rows]; mu = [r[1] for r in rows]; sd = [r[2] for r in rows]
        ax.errorbar(xs, mu, yerr=sd, color="#333", marker="o", ms=5, lw=1.6, capsize=3,
                    elinewidth=1.1, zorder=3)
        # highlight the canonical-Full anchor
        ai = xs.index(anchor_x)
        ax.scatter([anchor_x], [mu[ai]], s=90, facecolor=CANON, edgecolor="k", lw=0.8, zorder=5,
                   label="canonical Full")
        ax.set_xscale("log", base=2 if tag == "a" else 10)
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel(f"best-of-1 error @{bl} (mHa)", fontsize=9)
        ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
        ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.text(-0.20, 1.03, f"({tag})", transform=ax.transAxes, fontweight="bold",
                fontsize=11, va="bottom")
        ax.legend(fontsize=7.5, frameon=True, loc="best", handlelength=1.2)
        for r in rows:
            print(f"    {tag} x={r[0]:<4} bo1={r[1]:.3f}±{r[2]:.3f} (n={r[3]})")

    panel(axA, A, 64, "imagination seeds per update  ($n_{\\mathrm{seeds}}$)",
          f"Breadth of imagination · {disp}", "a")

    # ---- panel (b): best-of-1 vs MEASURED imagined-gradient share (x = %, not nominal λ) ----
    gx = np.array([r[1] for r in Bg]) * 100; gxe = np.array([r[2] for r in Bg]) * 100
    by = np.array([r[3] for r in Bg]); bye = np.array([r[4] for r in Bg])
    axB.errorbar(gx, by, xerr=gxe, yerr=bye, color="#333", marker="o", ms=5, lw=1.6,
                 capsize=3, elinewidth=1.1, zorder=3)
    lam1 = [r for r in Bg if r[0] == 1][0]
    axB.scatter([lam1[1] * 100], [lam1[3]], s=90, facecolor=CANON, edgecolor="k", lw=0.8, zorder=5,
                label="canonical Full ($\\lambda{=}1$)")
    for lam, gm, _gs, bm, _bs, _n in Bg:                       # annotate each point with its λ
        axB.annotate(f"$\\lambda{{=}}{lam}$", (gm * 100, bm), textcoords="offset points",
                     xytext=(6, 6), fontsize=7.5, color="#555")
    axB.set_yscale("log")
    axB.set_xlabel("measured imagined-gradient share  (%)", fontsize=9.5)
    axB.set_ylabel(f"best-of-1 error @{bl} (mHa)", fontsize=9)
    axB.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    axB.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    axB.set_title(f"Imagined-gradient weight · {disp}", fontsize=9.5, pad=4)
    axB.text(-0.20, 1.03, "(b)", transform=axB.transAxes, fontweight="bold", fontsize=11, va="bottom")
    axB.legend(fontsize=7.5, frameon=True, loc="lower left", handlelength=1.2)
    for lam, gm, gs, bm, bs, n in Bg:
        print(f"    b λ={lam:<3} grad_share={gm*100:.1f}%±{gs*100:.1f}  bo1={bm:.3f}±{bs:.3f} (n={n})")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/imag_strength_dose{suf}.{ext}", dpi=(300 if ext == "png" else None),
                    bbox_inches="tight")
    plt.close(fig)

    with open(f"{OUT}/imag_strength_dose{suf}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "lambda_nominal", "grad_share_mean_pct", "grad_share_std_pct",
                    "best_of_1_mean", "best_of_1_std", "n_seeds", "best_of_1_per_seed"])
        for x, m, s, n, arr in A:
            w.writerow(["n_seeds", f"n={x}", "", "", round(m, 4), round(s, 4), n,
                        ";".join(f"{v:.4f}" for v in arr)])
        for lam, gm, gs, bm, bs, n in Bg:
            w.writerow(["grad_share", lam, round(gm * 100, 2), round(gs * 100, 2),
                        round(bm, 4), round(bs, 4), n, ""])
    print(f"[written] imag_strength_dose{suf}.{{png,pdf,csv}} in {OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="LiH4q", choices=["LiH4q", "BeH2"])
    ap.add_argument("--budget", default="quarter", choices=["quarter", "full"])
    a = ap.parse_args()
    main(a.molecule, a.budget)
