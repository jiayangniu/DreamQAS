"""SPEED figures: frozen best-of-1 policy quality vs cumulative TRAINING VQE.

Outputs:
  speed_<mol>.png       one panel per molecule (all 5 tasks; appendix use)
  speed_main_3mol.png   * paper figure: LiH4q | LiH6q | BeH2_8q in ONE row with a single
                        shared legend (the three main-text molecules).

Style contract (locked 2026-07-18):
  - five curves: DreamQAS-Full / No-imag / RLQAS  = SOLID (VQE accounting strictly matched);
                 CRLQAS / HyRLQAS                 = DASHED + low-saturation (external
                 reference: they answer "where does DreamQAS sit in quality-cost space",
                 their accounting differs -> NO speedup claims against them).
  - speedup annotations ONLY vs No-imag and vs RLQAS, using the SEED-LEVEL numbers parsed
    from rq2_speedup.txt (sustained-crossing protocol): arrow at y = Q_b between the two
    median-VQE crossings, labeled with the seed-level multiplier.

y = frozen best-of-1 = mean over eval episodes of episode-best (min-prefix) -- SAME quantity
as the main table. x = cumulative training VQE at each saved checkpoint. Seeds aggregated at
each checkpoint: median line + IQR (25-75%) band.
"""
import re
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
MAIN3 = ["LiH4q", "LiH6q", "BeH2_8q"]          # the three main-text molecules
DISP = {"LiH4q": "LiH (4q)", "BeH2": "BeH$_2$ (6q)", "LiH6q": "LiH (6q)",
        "BeH2_8q": "BeH$_2$ (8q)", "BeH2_10q": "BeH$_2$ (10q)"}
# (display, color, linewidth, zorder, linestyle) -- matched methods solid, external dashed+desaturated
STYLE = {"Full":    ("DreamQAS-Full", "#d62728", 2.6, 5, "-"),
         "No-imag": ("No-imag",       "#ff7f0e", 1.8, 4, "-"),
         "RLQAS":   ("RLQAS",         "#1f77b4", 1.8, 3, "-"),
         "CRLQAS":  ("CRLQAS",        "#8fbc8f", 1.5, 2, "--"),
         "HyRLQAS": ("HyRLQAS",       "#b09cc9", 1.5, 2, "--")}
METHODS = list(STYLE)
FNAME = {"LiH4q": "LiH_4q", "BeH2": "BeH2_6q", "LiH6q": "LiH_6q",
         "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}


def bo1(errs):
    a = np.asarray([e for e in errs if np.isfinite(e)], float)
    return float(a.mean()) if a.size else np.nan


def series(method, mol):
    """per seed: list of (episode, cum_vqe, best-of-1)."""
    out = []
    if method in ("Full", "No-imag", "RLQAS"):
        pat = {"Full": f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q/eval_traces.jsonl",
               "No-imag": f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag/eval_traces.jsonl",
               "RLQAS": f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{mol}/seed_*/eval_traces.jsonl"}[method]
        for f in glob.glob(pat):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            s = [(r["episode"], r["cum_train_vqe_calls"], bo1(r["ep_best"]))
                 for r in rows if r.get("ep_best") and r.get("cum_train_vqe_calls") is not None]
            if s:
                out.append(sorted(s))
    else:
        pdir = method.lower()
        # HyRLQAS: fixed good-regime runs in psqas_hyrlqas_std (old psqas/hyrlqas is crippled)
        root = f"{C}/psqas_hyrlqas_std" if method == "HyRLQAS" else f"{C}/psqas"
        for d in glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*"):
            if not re.search(r"/seed\d+$", d):   # exclude backups like seed1_collapsed_bak
                continue
            p = f"{d}/eval.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p) if l.strip()]
            s = [(r["episode"], r["cum_train_vqe_calls"], bo1(r["episode_bests"]))
                 for r in rows if r.get("episode_bests") and r.get("cum_train_vqe_calls") is not None]
            if s:
                out.append(sorted(s))
    return out


AGG = "median"       # "median" -> line=median, band=IQR ; "mean" -> line=mean, band=±std (matches table)
FLOOR = 1e-4         # clip the lower band edge for the log axis (mean-std can go <=0)


def aggregate(method, mol, min_seeds=3):
    """group by checkpoint episode -> (central_cum_vqe, central, lo, hi) across seeds.
    AGG='median': median + IQR (robust; typical-seed trajectory).
    AGG='mean':   mean + ±std (expected error; shows the collapse-driven variance — matches the
                  mean±std main table). x uses the median cum-VQE either way (x is just the grid)."""
    per = {}
    for s in series(method, mol):
        for ep, v, b in s:
            per.setdefault(ep, []).append((v, b))
    rows = []
    for ep in sorted(per):
        vb = per[ep]
        if len(vb) < min_seeds:
            continue
        vs = np.array([x[0] for x in vb]); bs = np.array([x[1] for x in vb])
        if AGG == "mean":
            m = float(bs.mean()); sd = float(bs.std(ddof=1)) if bs.size > 1 else 0.0
            mu, lo, hi = m, max(m - sd, FLOOR), m + sd
        else:
            mu = np.median(bs); lo = np.percentile(bs, 25); hi = np.percentile(bs, 75)
        rows.append((np.median(vs), mu, lo, hi))
    if not rows:
        return None
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# ---------------------------------------------------------------- rq2 seed-level speedups

RQ2_TXT = f"{HERE}/outputs/main_results/rq2_speedup.txt"
TASK2MOL = {"LiH(4q)": "LiH4q", "LiH(6q)": "LiH6q", "BeH2(8q)": "BeH2_8q"}


def parse_rq2():
    """{(mol, baseline): dict(Qb, full_med, base_med, mult, ci)} from rq2_speedup.txt.
    Only the accounting-matched baselines (No-imag, RLQAS) are kept -- the figure never
    annotates a speedup against CRLQAS/HyRLQAS (different VQE accounting)."""
    out = {}
    if not os.path.exists(RQ2_TXT):
        return out
    for line in open(RQ2_TXT):
        t = line.split()
        if len(t) >= 9 and t[0] in TASK2MOL and t[1] in ("No-imag", "RLQAS") and t[7].endswith("x"):
            out[(TASK2MOL[t[0]], t[1])] = dict(
                Qb=float(t[2]), full_med=float(t[3]), base_med=float(t[5]),
                mult=t[7][:-1], ci=t[8])
    return out


DARK = {"No-imag": "#b35900", "RLQAS": "#12557f"}    # darker shades for the speedup "ladders"
# per-panel corner for the speedup number box: (ha, va, x, y) in axes fraction
BOXPOS = {"LiH4q": ("left", "bottom", 0.03, 0.05)}
BOXPOS_DEFAULT = ("right", "top", 0.97, 0.95)


def crossing(x, y, Q):
    """first x at which the running-min of the plotted median curve reaches <= Q."""
    ym = np.minimum.accumulate(np.asarray(y, float))
    idx = np.where(ym <= Q * 1.0000001)[0]
    return float(np.asarray(x, float)[idx[0]]) if idx.size else None


def draw_rq2_arrows(ax, mol, rq2, curves):
    """speedup gauges + number box (matched baselines only).
    - the NUMBER is always the seed-level rq2 multiplier, listed color-coded in one corner
      box (never floats over the curves);
    - the ARROW is drawn at y=Q_b between the points where the two PLOTTED median curves
      actually cross Q_b (so it always sits on the curves); if the plotted crossings do not
      exist in the right order, the arrow is skipped and only the number is shown."""
    # Annotate ONLY the matched-accounting Full-vs-No-imag speedup (the locked RQ2 headline).
    # vs-RLQAS is dropped from the figure: on LiH6q it is intrinsically fragile (Q_b sits in RLQAS's
    # flat 6q plateau, so a ~5% Q_b shift swings the crossing ~10x: 3.5x@25.7mHa -> 0.9x@27.0mHa),
    # and elsewhere it is high-variance; the full vs-RLQAS table (with CIs) lives in rq2_speedup.txt.
    lines = []
    for base in ("No-imag",):
        rec = rq2.get((mol, base))
        if not rec:
            continue
        col = DARK[base]
        lines.append((f"{rec['mult']}× vs {STYLE[base][0]}", col))
        if "Full" not in curves or base not in curves:
            continue
        Qb = rec["Qb"]
        xF = crossing(*curves["Full"], Qb)
        xB = crossing(*curves[base], Qb)
        if xF is None or xB is None or xF >= xB * 0.95:
            continue                                   # no clean on-curve gauge -> number only
        ax.annotate("", xy=(xF, Qb), xytext=(xB, Qb), zorder=6,
                    arrowprops=dict(arrowstyle="<|-|>", color=col, lw=2.2, shrinkA=0, shrinkB=0))
        for cx in (xF, xB):
            ax.plot([cx], [Qb], marker="|", color=col, ms=10, mew=2.0, zorder=6)
    if lines:
        ha, va, bx, by = BOXPOS.get(mol, BOXPOS_DEFAULT)
        step = 0.072 if va == "top" else -0.072
        ax.text(bx, by, "seed-level speedup", transform=ax.transAxes, ha=ha, va=va,
                fontsize=7.5, color="#666", zorder=7)
        for i, (txt, col) in enumerate(lines):
            ax.text(bx, by - (i + 1) * step, txt, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=9.5, fontweight="bold", color=col, zorder=7)


# ---------------------------------------------------------------- panels

XMIN = 1e4          # x starts at 10^4: the pre-descent plateau carries no information


def draw_panel(ax, mol, rq2, annotate=True):
    """draw the five curves (+ seed-level arrows) on ax; return handles.
    x is clipped to >= XMIN; the y-range is set from the VISIBLE (x >= XMIN) data only,
    and the chem-acc line is drawn only when 1.6 mHa falls inside that natural range
    (the axis is never stretched just to show it)."""
    handles = {}
    curves = {}
    ys = []
    xmax = XMIN * 10
    for m in METHODS:
        agg = aggregate(m, mol)
        if agg is None:
            continue
        x, y, lo, hi = agg
        disp, col, lw, z, ls = STYLE[m]
        y = np.clip(y, 1e-4, None); lo = np.clip(lo, 1e-4, None); hi = np.clip(hi, 1e-4, None)
        ln, = ax.plot(x, y, color=col, lw=lw, ls=ls, zorder=z, label=disp,
                      marker="o", ms=(3.2 if ls == "-" else 2.4), mew=0)
        ax.fill_between(x, lo, hi, color=col, alpha=(0.13 if ls == "-" else 0.07), zorder=1, lw=0)
        handles[m] = ln
        curves[m] = (np.asarray(x, float), np.asarray(y, float))
        vis = np.asarray(x, float) >= XMIN
        if vis.any():
            ys.append(np.asarray(y, float)[vis])
            xmax = max(xmax, float(np.max(np.asarray(x, float))))
    if annotate:
        draw_rq2_arrows(ax, mol, rq2, curves)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(XMIN, xmax * 1.35)
    if ys:
        yall = np.concatenate(ys)
        y0, y1 = float(np.min(yall)) / 1.8, float(np.max(yall)) * 1.8
        ax.set_ylim(y0, y1)
        if y0 <= 1.6 <= y1:
            ax.axhline(1.6, color="gray", ls=":", lw=1, zorder=0)
            ax.annotate("chem. acc.", xy=(0.02, 1.6), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color="gray", va="bottom", ha="left")
    ax.set_title(DISP[mol], fontsize=12, fontweight="bold")
    ax.set_xlabel("cumulative training VQE calls", fontsize=10)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    return handles


FOOT_MEDIAN = "median + IQR (5 seeds) · solid = matched accounting · dashed = external reference · speedup: Full vs No-imag, seed-level at Q$_b$"
FOOT_MEAN = "mean ± std (5 seeds; band shows collapse-driven variance) · solid = matched accounting · dashed = external reference · speedup: Full vs No-imag, seed-level at Q$_b$"


def foot():
    return FOOT_MEAN if AGG == "mean" else FOOT_MEDIAN


def suffix():
    return "_mean" if AGG == "mean" else ""


def plot_one(mol, rq2, outdir):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    draw_panel(ax, mol, rq2, annotate=True)
    ax.set_ylabel("best-of-1 policy error (mHa)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10,
              frameon=True, title="method", title_fontsize=10, handlelength=2.4)
    fig.text(0.99, 0.02, foot(), ha="right", va="bottom", fontsize=6.8, color="#666")
    fig.tight_layout()
    out = f"{outdir}/speed_{FNAME[mol]}{suffix()}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")     # vector twin for the paper
    plt.close(fig)
    return out


def plot_main3(rq2, outdir):
    """paper figure: the three main-text molecules in one row, one shared legend."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    handles = {}
    for ax, mol in zip(axes, MAIN3):
        h = draw_panel(ax, mol, rq2, annotate=True)
        handles.update(h)
    axes[0].set_ylabel("best-of-1 policy error (mHa)", fontsize=11)
    hs = [handles[m] for m in METHODS if m in handles]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=5, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.6, columnspacing=1.6)
    fig.text(0.99, 0.01, foot(), ha="right", va="bottom", fontsize=6.8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = f"{outdir}/speed_main_3mol{suffix()}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")     # vector twin for the paper
    plt.close(fig)
    return out


def main():
    global AGG
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agg", choices=["median", "mean"], default="median",
                    help="median+IQR (default; typical-seed) or mean±std (matches the mean main table)")
    AGG = ap.parse_args().agg
    d = f"{HERE}/outputs/main_results"
    os.makedirs(d, exist_ok=True)
    rq2 = parse_rq2()
    print(f"AGG={AGG}   rq2 seed-level speedups loaded: {sorted(rq2)}")
    for mol in MOLS:
        cov = {m: (len(aggregate(m, mol)[0]) if aggregate(m, mol) is not None else 0) for m in METHODS}
        out = plot_one(mol, rq2, d)
        print(f"[written] {out}   coverage: " + " ".join(f"{m}={cov[m]}" for m in METHODS))
    out = plot_main3(rq2, d)
    print(f"[written] {out}   (paper Figure: {' | '.join(DISP[m] for m in MAIN3)})")


if __name__ == "__main__":
    main()
