"""TRAINING-WINDOW speed figure (Path 2): training-time episode-best (min-prefix), sliding-window
smoothed, vs cumulative training VQE. Complement to the frozen figure (plot_speed_curves.py):
  frozen  = deploy-faithful policy quality at checkpoints (no exploration)
  training-window = exploration-time policy quality, dense (every episode)

Per-episode episode-best e_j = min_t |E(prefix_t) - E_FCI| (mHa), computed uniformly from per-step
traces:  DreamQAS-family -> trajectory.jsonl (true_error per step); PSQAS -> episode_traces.txt
(energies_ha per step, minus result_seed.npz exact_energy). Sliding window W episodes -> best-of-1
(= mean over the window). Seeds aggregated: mean +/- SEM (matches the table); x = median cum-VQE.

Usage: python analysis/plot_training_window.py [Full No-imag CRLQAS ...]   (default: available set)
"""
import json, glob, os, sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH (4q)", "BeH2": "BeH$_2$ (6q)", "LiH6q": "LiH (6q)",
        "BeH2_8q": "BeH$_2$ (8q)", "BeH2_10q": "BeH$_2$ (10q)"}
FNAME = {"LiH4q": "LiH_4q", "BeH2": "BeH2_6q", "LiH6q": "LiH_6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
STYLE = {"Full":    ("DreamQAS-Full", "#d62728", 2.6, 5),
         "No-imag": ("No-imag",       "#ff7f0e", 1.8, 4),
         "RLQAS":   ("RLQAS",         "#1f77b4", 1.8, 3),
         "CRLQAS":  ("CRLQAS",        "#2ca02c", 1.8, 2),
         "HyRLQAS": ("HyRLQAS",       "#9467bd", 1.8, 2),
         "GQE":     ("GQE",           "#8c564b", 1.6, 2)}
W = 400          # sliding window in episodes (~100 iters x 4 eps, matches frozen W=100 iters)
DOWN = 50        # plot every DOWN-th episode


def dq_dirs(method, mol):
    if method == "Full":    return glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q")
    if method == "No-imag": return glob.glob(f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag")
    if method == "RLQAS":   return glob.glob(f"{C}/dreamqas_rlqas_traj/baseline_analysis/G0_v2_{mol}/seed_*")
    return []


def ps_dirs(method, mol):
    pdir = {"GQE": "gqeqas"}.get(method, method.lower())
    # HyRLQAS: use the FIXED "good-regime" runs (masking-bug fix + accept_err=2.45 etc.) in the
    # separate psqas_hyrlqas_std tree; the old psqas/hyrlqas runs are the crippled reward regime.
    root = f"{C}/psqas_hyrlqas_std" if method == "HyRLQAS" else f"{C}/psqas"
    # match only exact seed<N> dirs -> excludes backups like seed1_collapsed_bak
    return [d for d in glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*")
            if re.search(r"/seed\d+$", d)]


def episodes_from_trajectory(run_dir):
    """DreamQAS/RLQAS trajectory.jsonl -> ordered [(n_steps, episode_best_mHa)] per episode."""
    p = f"{run_dir}/trajectory.jsonl"
    if not os.path.exists(p):
        return None
    eps, cur, cmin, cn = [], None, np.inf, 0
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["iter"], r["ep"])
        if key != cur:
            if cur is not None:
                eps.append((cn, cmin))
            cur, cmin, cn = key, np.inf, 0
        te = r.get("true_error")
        if te is not None:
            cmin = min(cmin, te)
        cn += 1
    if cur is not None:
        eps.append((cn, cmin))
    return eps if eps else None


_EN = re.compile(r"energies_ha\s*=\s*\[(.*)\]")


def episodes_from_traces(run_dir):
    """PSQAS episode_traces.txt -> ordered [(n_steps, episode_best_mHa)] per episode."""
    p = f"{run_dir}/episode_traces.txt"
    rz = glob.glob(f"{run_dir}/result_seed*.npz")
    if not os.path.exists(p) or not rz:
        return None
    z = np.load(rz[0], allow_pickle=True)
    exact = float(z["exact_energy"] if "exact_energy" in z else z["exact_energy_ha"])
    eps = []
    for line in open(p):
        m = _EN.search(line)
        if m:
            arr = np.fromstring(m.group(1), sep=",")
            if arr.size:
                eb = float(np.min(np.abs(arr - exact))) * 1000.0
                eps.append((int(arr.size), eb))
    return eps if eps else None


def run_curve(eps):
    """[(n_steps, ebest)] -> (cum_vqe[], window-best-of-1[]) downsampled."""
    n = np.array([e[0] for e in eps], float)
    b = np.array([e[1] for e in eps], float)
    cum = np.cumsum(n)
    csum = np.cumsum(np.insert(b, 0, 0.0))
    idx = np.arange(len(b))
    lo = np.maximum(0, idx - W + 1)
    win = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)     # rolling mean over up to W episodes
    sel = np.arange(0, len(b), DOWN)
    return cum[sel], win[sel]


def series(method, mol):
    dirs = dq_dirs(method, mol); parse = episodes_from_trajectory
    if not dirs:
        dirs = ps_dirs(method, mol); parse = episodes_from_traces
    out = []
    for d in dirs:
        eps = parse(d)
        if eps and len(eps) > W:
            out.append(run_curve(eps))
    return out


def aggregate(method, mol, min_seeds=2):
    runs = series(method, mol)
    if len(runs) < min_seeds:
        return None
    L = min(len(x) for x, _ in runs)                       # align by episode index (common grid)
    X = np.array([x[:L] for x, _ in runs]); Y = np.array([y[:L] for _, y in runs])
    # MEDIAN + IQR over seeds: robust to the deterministic 1-seed HyRLQAS mode-collapse (matches the
    # main table's median convention for HyRLQAS; for collapse-free methods median ~= mean).
    mu = np.median(Y, 0)
    lo = np.percentile(Y, 25, axis=0); hi = np.percentile(Y, 75, axis=0)
    return np.median(X, 0), mu, lo, hi, Y.shape[0]


def plot_one(mol, methods, outdir):
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ncov = {}
    xmax, ymin = 0.0, np.inf
    curves = {}
    for m in methods:
        agg = aggregate(m, mol)
        if agg is None:
            continue
        x, y, lo, hi, ns = agg; ncov[m] = ns
        disp, col, lw, z = STYLE[m]
        y = np.clip(y, 1e-4, None); lo = np.clip(lo, 1e-4, None); hi = np.clip(hi, 1e-4, None)
        ax.plot(x, y, color=col, lw=lw, zorder=z, label=disp)
        ax.fill_between(x, lo, hi, color=col, alpha=0.13, zorder=1, lw=0)
        xmax = max(xmax, float(x.max())); ymin = min(ymin, float(y.min()))
        curves[m] = (x, y)
    if ymin < 1.6:                                 # chem-acc line only where a method approaches it
        ax.axhline(1.6, color="gray", ls=":", lw=1, zorder=0)
        ax.annotate("chem. acc. 1.6 mHa", xy=(0.015, 1.6), xycoords=("axes fraction", "data"),
                    fontsize=8, color="gray", va="bottom", ha="left")
    # ---- explicit VQE-savings: Full reaches EACH baseline's converged quality N× sooner ----
    # one color-coded arrow per baseline (at that baseline's floor); the "Nx vs X" figures are
    # listed under the legend. Makes the horizontal speedup legible despite the log x-axis.
    def _cross(xy, Q, sustain=3):
        # SUSTAINED crossing: first cumulative-VQE after which the (already-smoothed) curve stays
        # <= Q for `sustain` consecutive reported windows -> robust to single-window dips (NOT a
        # running-minimum, so it is consistent with the paper's no-best-found stance).
        xx, yy = xy; n = len(yy)
        for i in range(n - sustain + 1):
            if all(yy[i + r] <= Q * 1.0001 for r in range(sustain)):
                return float(xx[i])
        return None
    speedups = []                                            # (label, color, ratio)
    if "Full" in curves:
        for b in ("No-imag", "RLQAS", "CRLQAS", "HyRLQAS"):
            if b not in curves:
                continue
            yb = curves[b][1]
            Q = float(np.mean(yb[int(len(yb) * 0.9):]))       # baseline's FINAL-WINDOW quality = mean over last 10%
            cf, cb = _cross(curves["Full"], Q), _cross(curves[b], Q)
            if cf and cb and cb > cf * 1.15:
                col = STYLE[b][1]
                ax.annotate("", xy=(cf, Q), xytext=(cb, Q), zorder=7,
                            arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.6, shrinkA=0, shrinkB=0))
                ax.plot([cf, cb], [Q, Q], marker="|", color=col, ms=8, mew=1.6, ls="none", zorder=7)
                speedups.append((STYLE[b][0], col, cb / cf))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e4, (xmax if xmax > 0 else 1e6) * 1.15)   # from 10^4, tight right bound (no empty gap)
    ax.set_title(f"{DISP[mol]}   [PRELIMINARY]", fontsize=13, fontweight="bold")
    ax.set_xlabel("cumulative training VQE calls", fontsize=11)
    ax.set_ylabel("training-window best-of-1 error (mHa)", fontsize=11)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=True,
              title="method", title_fontsize=10)
    # VQE-savings figures under the legend, color-coded per baseline
    if speedups:
        ax.text(1.03, 0.46, "VQE-call speedup to each\nbaseline's final-window quality:",
                transform=ax.transAxes, fontsize=8.3, va="top", ha="left", fontweight="bold", color="#333")
        for i, (name, col, r) in enumerate(speedups):
            rs = f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"
            ax.text(1.03, 0.33 - i * 0.075, f"{rs}  vs {name}", transform=ax.transAxes,
                    fontsize=9.5, va="top", ha="left", color=col, fontweight="bold")
    fig.text(0.99, 0.02, f"training-time episode-best, sliding window {W} eps · median+IQR · lower=better",
             ha="right", va="bottom", fontsize=7.5, color="#666")
    fig.tight_layout()
    out = f"{outdir}/trainwin_{FNAME[mol]}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")     # vector twin for the paper
    plt.close(fig)
    return out, ncov


def main():
    want = sys.argv[1:] or ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS"]   # RLQAS auto-appears when its traj is ready
    d = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results"
    os.makedirs(d, exist_ok=True)
    for mol in MOLS:
        out, ncov = plot_one(mol, want, d)
        print(f"[written] {out}   " + " ".join(f"{m}={ncov.get(m,0)}s" for m in want))


if __name__ == "__main__":
    main()
