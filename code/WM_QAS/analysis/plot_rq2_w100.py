"""RQ2 unified on TRAINING-TIME 100-episode moving mean (Figure 2 + VQE-to-target speedup).

Supersedes the split pipeline (frozen eval_traces figure + W=400 speedup). BOTH the Figure-2 curves
AND every speedup/CI here come from ONE per-seed training-time curve:

    training episode-best e_i = min_t ε_{i,t}   (min true post-VQE error over the episode's prefixes,
        from trajectory.jsonl for DreamQAS-family; from episode_traces.txt for PSQAS externals)
    -> trailing W=100-episode arithmetic moving mean  ē_i^(100)
    -> downsample every DOWN=50 episodes (adjacent reported windows overlap 50%)
    -> cross-seed MEDIAN line + IQR (25-75%) band, aligned by common episode index.

x-axis = cumulative REAL training VQE calls. For DreamQAS-family this is `metrics.jsonl` `vqe_calls`
(which INCLUDES DAgger verification VQE — trajectory step counts do NOT, they undercount Full by ~11%);
frozen-policy evaluation VQE is excluded (it lives in a separate eval_vqe counter). PSQAS externals use
their native per-step count under their native stopping protocol.

Q_b = median over baseline seeds of each seed's final-window (last 10% reported points) mean.
Sustained crossing = first reported point after which ē^(100) ≤ Q_b for 3 consecutive points (stride 50).
Right-censored seeds excluded from the median, reach reported. speedup = median C_b / median C_Full
(>1 = DreamQAS uses fewer real VQE). 95% CI = 3000x unpaired bootstrap over reached seeds, percentile.

Outputs: speed_main_3mol_w100.{png,pdf}, rq2_speedup_w100.txt.  Does NOT overwrite the W=400 canon.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_training_window import dq_dirs, ps_dirs, episodes_from_traces  # reuse dir globs + PSQAS parser

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
OUT = f"{HERE}/outputs/main_results"
W, DOWN = 100, 50
MAIN3 = ["LiH4q", "LiH6q", "BeH2_8q"]
ALL = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH (4q)", "LiH6q": "LiH (6q)", "BeH2_8q": "BeH$_2$ (8q)"}
FNAME = {"LiH4q": "LiH_4q", "LiH6q": "LiH_6q", "BeH2_8q": "BeH2_8q"}
# (display, color, linewidth, zorder, linestyle). DreamQAS-family solid; PSQAS dashed low-sat.
STYLE = {"Full":    ("DreamQAS",      "#d62728", 1.6, 5, "-"),   # legend unified to "DreamQAS" (main table + text)
         "No-imag": ("No-imag",       "#ff7f0e", 1.1, 4, "-"),
         "RLQAS":   ("DreamQAS-RL",   "#1f77b4", 1.1, 3, "-"),   # legend label unified to DreamQAS-RL
         "CRLQAS":  ("CRLQAS",        "#8fbc8f", 0.9, 2, "--"),
         "HyRLQAS": ("HyRLQAS",       "#b09cc9", 0.9, 2, "--")}
METHODS = list(STYLE)
DARK = {"No-imag": "#b35900", "RLQAS": "#12557f"}


def dq_episodes_true_vqe(run_dir):
    """DreamQAS-family: per-episode (cum_real_vqe, episode_best_mHa).
    episode_best from trajectory.jsonl (min true_error); cum_real_vqe from metrics.jsonl `vqe_calls`
    at that episode's iter -> INCLUDES DAgger verification VQE (trajectory step counts do not)."""
    tp, mp = f"{run_dir}/trajectory.jsonl", f"{run_dir}/metrics.jsonl"
    if not (os.path.exists(tp) and os.path.exists(mp)):
        return None
    iter_vqe = {}
    for l in open(mp):
        if l.strip():
            r = json.loads(l); iter_vqe[r["iter"]] = float(r["vqe_calls"])
    eps, cur, cmin, cit = [], None, np.inf, None
    for l in open(tp):
        if not l.strip():
            continue
        r = json.loads(l); k = (r["iter"], r["ep"])
        if k != cur:
            if cur is not None:
                eps.append((cit, cmin))
            cur, cmin, cit = k, np.inf, r["iter"]
        te = r.get("true_error")
        if te is not None:
            cmin = min(cmin, te)
    if cur is not None:
        eps.append((cit, cmin))
    out = [(iter_vqe[it], eb) for it, eb in eps if it in iter_vqe and np.isfinite(eb)]
    return out or None


def curve(eps):
    """[(cum_vqe, ebest)] -> (cum_vqe_reported[], W=100 trailing-mean ebest[]) downsampled by DOWN."""
    x = np.array([e[0] for e in eps], float)
    b = np.array([e[1] for e in eps], float)
    csum = np.cumsum(np.insert(b, 0, 0.0))
    idx = np.arange(len(b))
    lo = np.maximum(0, idx - W + 1)
    win = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)          # trailing mean over min(W, i+1) eps
    sel = np.arange(0, len(b), DOWN)
    return x[sel], win[sel]


def seed_curves(method, mol):
    """list of (cum_vqe[], smoothed_ebest[]) per seed at W=100/DOWN=50."""
    out = []
    if method in ("Full", "No-imag", "RLQAS"):
        for d in dq_dirs(method, mol):
            eps = dq_episodes_true_vqe(d)
            if eps and len(eps) > W:
                out.append(curve(eps))
    else:                                                       # PSQAS: native per-step count
        for d in ps_dirs(method, mol):
            e2 = episodes_from_traces(d)
            if e2 and len(e2) > W:
                eps = list(zip(np.cumsum([n for n, _ in e2]).tolist(), [b for _, b in e2]))
                out.append(curve(eps))
    return out


def aggregate(method, mol, min_seeds=2):
    runs = seed_curves(method, mol)
    if len(runs) < min_seeds:
        return None
    L = min(len(x) for x, _ in runs)                           # align by common episode index
    X = np.array([x[:L] for x, _ in runs]); Y = np.array([y[:L] for _, y in runs])
    return np.median(X, 0), np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0), len(runs)


# ---------------------------------------------------------------- speedup

def final_window_q(cvs):
    return [float(np.mean(y[int(len(y) * 0.9):])) for _, y in cvs]


def cross(cv, Q, sustain=3):
    x, y = cv
    for i in range(len(y) - sustain + 1):
        if all(y[i + r] <= Q * 1.0001 for r in range(sustain)):
            return float(x[i])
    return np.nan


def speedup_row(mol, base, rng):
    cf, cb = seed_curves("Full", mol), seed_curves(base, mol)
    if len(cf) < 2 or len(cb) < 2:
        return None
    Q = float(np.median(final_window_q(cb)))
    vf = np.array([cross(c, Q) for c in cf]); vb = np.array([cross(c, Q) for c in cb])
    rf, rb = vf[np.isfinite(vf)], vb[np.isfinite(vb)]
    if len(rf) == 0 or len(rb) == 0:
        return dict(Q=Q, mf=np.nan, mb=np.nan, rf=len(rf), nf=len(cf), rb=len(rb), nb=len(cb),
                    sp=np.nan, ci=(np.nan, np.nan))
    mf, mb = float(np.median(rf)), float(np.median(rb))
    bs = [np.median(rng.choice(rb, len(rb), True)) / np.median(rng.choice(rf, len(rf), True))
          for _ in range(3000)]
    return dict(Q=Q, mf=mf, mb=mb, rf=len(rf), nf=len(cf), rb=len(rb), nb=len(cb),
                sp=mb / mf, ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))


def compute_speedups():
    rng = np.random.default_rng(7)
    rows = {}
    for mol in ALL:
        for b in ("No-imag", "RLQAS", "CRLQAS", "HyRLQAS"):
            rows[(mol, b)] = speedup_row(mol, b, rng)
    return rows


# ---------------------------------------------------------------- figure

def draw_panel(ax, mol, sp):
    handles, ys, xmax = {}, [], W * 1e3
    for m in METHODS:
        agg = aggregate(m, mol)
        if agg is None:
            continue
        x, y, lo, hi, _ns = agg
        disp, col, lw, z, ls = STYLE[m]
        y = np.clip(y, 1e-4, None); lo = np.clip(lo, 1e-4, None); hi = np.clip(hi, 1e-4, None)
        ln, = ax.plot(x, y, color=col, lw=lw, ls=ls, zorder=z, label=disp)
        ax.fill_between(x, lo, hi, color=col, alpha=(0.13 if ls == "-" else 0.07), zorder=1, lw=0)
        handles[m] = ln
        vis = x >= 1e4
        if vis.any():
            ys.append(y[vis]); xmax = max(xmax, float(x.max()))
    # headline arrow: Full vs No-imag (matched accounting); number = seed-level rq2 speedup
    rec = sp.get((mol, "No-imag"))
    if rec and np.isfinite(rec["sp"]) and "Full" in handles:
        Q = rec["Q"]
        xF = cross(aggregate("Full", mol)[:2], Q); xB = cross(aggregate("No-imag", mol)[:2], Q)
        if np.isfinite(xF) and np.isfinite(xB) and xB > xF:
            col = DARK["No-imag"]
            ax.annotate("", xy=(xF, Q), xytext=(xB, Q), zorder=6,
                        arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.5, shrinkA=0, shrinkB=0))
            for cx in (xF, xB):
                ax.plot([cx], [Q], marker="|", color=col, ms=8, mew=1.5, zorder=6)
        ha, va, bx, by = ("left", "bottom", 0.03, 0.05) if mol == "LiH4q" else ("right", "top", 0.97, 0.95)
        ax.text(bx, by, "seed-level speedup", transform=ax.transAxes, ha=ha, va=va, fontsize=7.5, color="#666")
        ax.text(bx, by - (0.072 if va == "top" else -0.072), f"{rec['sp']:.1f}× vs No-imag",
                transform=ax.transAxes, ha=ha, va=va, fontsize=9.5, fontweight="bold", color=DARK["No-imag"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e4, xmax * 1.35)
    if ys:
        yall = np.concatenate(ys); y0, y1 = float(yall.min()) / 1.8, float(yall.max()) * 1.8
        ax.set_ylim(y0, y1)
        if y0 <= 1.6 <= y1:
            ax.axhline(1.6, color="gray", ls=":", lw=1, zorder=0)
            ax.annotate("chem. acc.", xy=(0.02, 1.6), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color="gray", va="bottom", ha="left")
    ax.set_title(DISP[mol], fontsize=12, fontweight="bold")
    ax.set_xlabel("cumulative real VQE calls", fontsize=10)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    return handles


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = compute_speedups()
    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    handles = {}
    for ax, mol in zip(axes, MAIN3):
        handles.update(draw_panel(ax, mol, sp))
    axes[0].set_ylabel("training episode-best error (mHa, 100-ep moving mean)", fontsize=10)
    hs = [handles[m] for m in METHODS if m in handles]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=5, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/speed_main_3mol_w100.{ext}", dpi=(160 if ext == "png" else None), bbox_inches="tight")
    plt.close(fig)

    # ---- rq2_speedup_w100.txt ----
    L = ["=" * 112,
         "RQ2 W=100 SEED-LEVEL VQE SPEEDUP — unified training-time 100-episode moving-mean pipeline",
         "  curve = min-prefix episode-best -> W=100 trailing mean -> cross-seed median; x = real VQE (incl. DAgger for Full)",
         "  Q_b = median baseline final-window (last 10%) mean; sustained crossing (3 pts, stride 50); censored excluded.",
         "  speedup = median C_baseline / median C_Full (>1 = DreamQAS fewer VQE); 95% CI = 3000x unpaired bootstrap.",
         "=" * 112,
         f"{'task':<10}{'baseline':<11}{'Q_b(mHa)':>10}{'Full VQE med':>14}{'F reach':>9}{'base VQE med':>14}{'b reach':>9}{'speedup':>10}{'95% CI':>16}",
         "-" * 112]
    for mol in ALL:
        tname = DISP[mol].replace("$", "").replace("_", "")
        for b in ("No-imag", "RLQAS", "CRLQAS", "HyRLQAS"):
            r = sp.get((mol, b)); bl = STYLE[b][0]
            if r is None:
                L.append(f"{tname:<10}{bl:<11}{'(missing / <2 seeds)':>10}")
                continue
            if not np.isfinite(r["sp"]):
                L.append(f"{tname:<10}{bl:<11}{r['Q']:>10.3f}{'  (never reaches Q_b)':>14}")
                continue
            freach = f"{r['rf']}/{r['nf']}"; breach = f"{r['rb']}/{r['nb']}"
            ci = f"[{r['ci'][0]:.1f},{r['ci'][1]:.1f}]"
            L.append(f"{tname:<10}{bl:<11}{r['Q']:>10.3f}{r['mf']:>14.0f}{freach:>9}"
                     f"{r['mb']:>14.0f}{breach:>9}{r['sp']:>8.1f}x{ci:>16}")
        L.append("-" * 112)
    L += ["HEADLINE = Full vs No-imag (matched accounting). Full vs DreamQAS-RL reported with CI + reach.",
          "CRLQAS/HyRLQAS shown for quality-cost reference only (native accounting) — no matched-speedup claim."]
    open(f"{OUT}/rq2_speedup_w100.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[written] speed_main_3mol_w100.{{png,pdf}}, rq2_speedup_w100.txt in {OUT}")
    return sp


if __name__ == "__main__":
    main()
