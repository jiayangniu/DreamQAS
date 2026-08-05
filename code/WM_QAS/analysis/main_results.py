"""Main-experiment checkpoint analysis (frozen-policy quality vs training-search quality).

Read-only over campaign outputs; does NOT modify training code. Subcommands:
  coverage : enumerate runs / checkpoints / eval-episode counts / per-episode-stat source
  build    : produce datasets A-E (parquet/csv + json)
  render   : frozen-policy (mean+best) curves + training-best curves + summary/milestone tables

Two separated quantities:
  (1) frozen-policy quality  -> Full/RLQAS from frozen eval (analysis/eval_policy_traces.py);
                                PSQAS from training-trace proxy (episode_traces.txt window).
  (2) training-search quality-> running-min best_err over cumulative training VQE (all methods).
Molecule-relative milestones (initial-error ratios); NO chemical-accuracy threshold.
"""
import glob
import json
import os
import re
import sys

CAMP = os.environ.get("CAMP", "/home/USER/DreamQAS/experiments/campaign_v1")

# ---- molecule + method identity ----
MOLS = ["LiH4q", "LiH6q", "BeH2", "BeH2_8q", "BeH2_10q"]        # internal keys (BeH2 = 6q)
MOL_DISP = {"LiH4q": "LiH4q", "LiH6q": "LiH6q", "BeH2": "BeH2_6q",
            "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
MOL_TOKENS = ["BeH2_10q", "BeH2_8q", "BeH2", "LiH4q", "LiH6q"]  # longest-first for basename parse
OUTER2MOL = {"DQ_LiH_4q": "LiH4q", "DQ_LiH_6q": "LiH6q", "DQ_BeH2_6q": "BeH2",
             "T5_BeH2_631G_8q": "BeH2_8q", "T5_BeH2_6311G_10q": "BeH2_10q"}
METHODS = ["Full", "RLQAS", "crlqas", "hyrlqas", "qdarts", "gqeqas"]
METHOD_DISP = {"Full": "DreamQAS-Full", "RLQAS": "RLQAS", "crlqas": "CRLQAS",
               "hyrlqas": "HyRLQAS", "qdarts": "QuantumDARTS", "gqeqas": "GQE"}
COMPARABLE_VQE = {"Full", "RLQAS", "crlqas", "hyrlqas"}          # per-step VQE unit; qdarts/gqe not
FCI_HA = {"LiH4q": -7.789089, "BeH2": -14.861589, "BeH2_8q": -15.761504, "BeH2_10q": -15.765124}


def resolve(run_dir):
    """Full path of a run dir -> (method, mol) or None (skip)."""
    p = run_dir.rstrip("/")
    base = os.path.basename(p)
    if "/psqas/" in p:
        m = re.search(r"/psqas/([^/]+)/([^/]+)/", p)
        if not m:
            return None
        method, outer = m.group(1), m.group(2)
        mol = OUTER2MOL.get(outer)
        return (method, mol) if (method in METHODS and mol in MOLS) else None
    if "/dreamqas_rlqas/" in p:
        m = re.search(r"/G0_v2_([^/]+)/", p)                    # mol lives in parent G0_v2_<mol>
        mol = m.group(1) if m else None
        return ("RLQAS", mol) if mol in MOLS else None
    if base.startswith("gru_energy_surrogate_") and base.endswith("_q"):
        mol = next((t for t in MOL_TOKENS if f"_{t}_" in base), None)
        return ("Full", mol) if mol in MOLS else None
    return None


def find_runs():
    """Return {(method,mol): [run_dir,...]} across Full / RLQAS / PSQAS for the 4 molecules."""
    dirs = []
    dirs += glob.glob(f"{CAMP}/dreamqas/gru_energy_surrogate_*_q")
    dirs += glob.glob(f"{CAMP}/dreamqas_rlqas/baseline_analysis/G0_v2_*/seed_*")
    dirs += glob.glob(f"{CAMP}/psqas/*/*/DreamQAS_EXP/*/seed*")
    groups = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        key = resolve(d)
        if key:
            groups.setdefault(key, []).append(d)
    return groups


def _seed_of(run_dir):
    m = re.search(r"seed_?(\d+)", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else (int(m.group(1)) if (m := re.search(r"_s(\d+)_", run_dir)) else -1)


def n_ckpts(run_dir):
    return len(glob.glob(f"{run_dir}/ckpt/ep*.pt"))


def eval_episode_counts(run_dir):
    """(n_inter, n_final) from a run's eval.jsonl, else (None,None)."""
    ejl = f"{run_dir}/eval.jsonl"
    if not os.path.exists(ejl):
        return (None, None)
    lines = [l for l in open(ejl) if l.strip()]
    if not lines:
        return (None, None)
    try:
        ni = json.loads(lines[0]).get("n_eval_episodes")
        nf = json.loads(lines[-1]).get("n_eval_episodes")
        return (ni, nf)
    except Exception:
        return (None, None)


def training_done(run_dir, method):
    """Best-effort: has the run reached its full episode budget?"""
    if method in ("Full", "RLQAS"):
        f = f"{run_dir}/metrics.jsonl"
        if not os.path.exists(f):
            return None
        last = None
        for l in open(f):
            if l.strip():
                last = l
        try:
            d = json.loads(last)
            return d.get("iter")
        except Exception:
            return None
    ts = f"{run_dir}/episode_summary.tsv"
    if os.path.exists(ts):
        return sum(1 for _ in open(ts)) - 1
    return None


def coverage():
    groups = find_runs()
    lines = ["=" * 88, "COVERAGE REPORT — main-experiment checkpoint analysis", "=" * 88, ""]
    lines.append(f"CAMP = {CAMP}\n")
    lines.append(f"{'method':14s} {'molecule':10s} {'seeds':6s} {'ckpts(med)':11s} "
                 f"{'eval n(int/fin)':16s} {'train_progress':14s} {'per-episode source':22s}")
    lines.append("-" * 100)
    for method in METHODS:
        for mol in MOLS:
            runs = sorted(groups.get((method, mol), []), key=_seed_of)
            n = len(runs)
            if n == 0:
                lines.append(f"{METHOD_DISP[method]:14s} {MOL_DISP[mol]:10s} {'0':6s} "
                             f"{'-':11s} {'-':16s} {'MISSING':14s}")
                continue
            cks = sorted(n_ckpts(r) for r in runs)
            ck_med = cks[len(cks) // 2]
            nis, nfs = zip(*[eval_episode_counts(r) for r in runs])
            ni = next((x for x in nis if x), None)
            nf = next((x for x in nfs if x), None)
            evalstr = f"{ni}/{nf}" if ni else "none-yet"
            progs = [training_done(r, method) for r in runs]
            progs = [p for p in progs if p is not None]
            prog = f"{min(progs)}..{max(progs)}" if progs else "?"
            if method in ("Full", "RLQAS"):
                src = "frozen(traces:new)" if not ni else "frozen(eval.jsonl)"
            else:
                src = "train-trace proxy"
            lines.append(f"{METHOD_DISP[method]:14s} {MOL_DISP[mol]:10s} {str(n):6s} "
                         f"{str(ck_med):11s} {evalstr:16s} {str(prog):14s} {src:22s}")
        lines.append("")
    lines.append("NOTES:")
    lines.append("  - Full/RLQAS per-episode stats (A) come from a NEW frozen-eval pass (eval_policy_traces.py):")
    lines.append("    every intermediate ckpt = 20 fresh episodes, final = 100; run to max depth; no side effects.")
    lines.append("  - PSQAS eval.jsonl carry only episode-BEST distribution (mean/median/std/best) + episode_bests[]")
    lines.append("    -> per-episode within-rollout stats NOT in eval; taken from training-trace episode_traces.txt")
    lines.append("    (per-step energy_errors) windowed around each checkpoint's episode number (labeled proxy).")
    lines.append("  - Training-best (D/E) available for ALL methods (metrics.jsonl / episode_summary.tsv / npz).")
    lines.append("  - eval-episode N differs across PSQAS (crlqas 20/100, hyrlqas 20/20, qdarts 30/30, gqeqas 50/50):")
    lines.append("    headline uses N-unbiased MEAN of episode-best; best-of-N subsampled to a common N when compared.")
    return "\n".join(lines)


import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_protocol import _boot_ci                                   # cross-seed median + 95% CI

COMMON_N = 20                                                        # fair best-of-N across methods
OUT = "/home/USER/DreamQAS/code/WM_QAS/analysis/outputs/main_results"


# ---------------- loaders ----------------
def training_best_traj(run_dir, method):
    """(cum_vqe, running-min best_err mHa) sorted by cum_vqe, or None."""
    try:
        if method in ("Full", "RLQAS"):
            rows = [json.loads(l) for l in open(f"{run_dir}/metrics.jsonl") if l.strip()]
            v = np.array([r["vqe_calls"] for r in rows], float)
            if "best_err_mHa" in rows[0]:                               # DreamQAS Full (mHa)
                b = np.array([r["best_err_mHa"] for r in rows], float)
            else:                                                       # RLQAS baseline: 'best_err' in Ha
                b = np.array([r["best_err"] for r in rows], float) * 1000.0
            return v, np.minimum.accumulate(b)
        if method in ("crlqas", "hyrlqas"):
            rows = list(csv.DictReader(open(f"{run_dir}/episode_summary.tsv"), delimiter="\t"))
            err = np.array([float(r["final_energy_error_ha"]) for r in rows]) * 1000.0
            cum = np.cumsum([float(r["n_steps"]) for r in rows])
            return cum, np.minimum.accumulate(err)
        if method in ("qdarts", "gqeqas"):
            npz = glob.glob(f"{run_dir}/result_seed*.npz")
            if not npz:
                return None
            d = np.load(npz[0], allow_pickle=True)
            cum = np.asarray(d["eval_cum_vqe_circuits"], float).ravel()
            be = np.asarray(d["eval_best_error_mha"], float).ravel()
            if cum.size == 0 or be.size == 0:
                return None
            o = np.argsort(cum)
            return cum[o], np.minimum.accumulate(be[o])
    except Exception as e:
        print(f"  [warn] training_best_traj failed {run_dir}: {e}")
    return None


def frozen_ebest_ckpts(run_dir, method):
    """[(cum_vqe, ep_best_array)] per checkpoint sorted by cum_vqe, or None.
    Full/RLQAS <- eval_traces.jsonl (new frozen pass); PSQAS <- eval.jsonl episode_bests."""
    f = f"{run_dir}/eval_traces.jsonl" if method in ("Full", "RLQAS") else f"{run_dir}/eval.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    key = "ep_best" if method in ("Full", "RLQAS") else "episode_bests"
    out = []
    for r in sorted(rows, key=lambda r: r["cum_train_vqe_calls"]):
        eb = np.asarray([x for x in r.get(key, []) if x is not None and np.isfinite(x)], float)
        if eb.size:
            out.append((r["cum_train_vqe_calls"], eb))
    return out or None


# ---------------- aggregation (seed-level; never pool episodes) ----------------
def _step_interp(cum, val, grid):
    """running-min (monotone) sampled onto grid: value = last val at cum<=g (else nan)."""
    idx = np.searchsorted(cum, grid, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, len(val) - 1)], np.nan)
    return out


def agg_training_best(runs, method, npoints=60):
    trajs = [t for t in (training_best_traj(r, method) for r in runs) if t is not None]
    if not trajs:
        return None
    lo = min(t[0][t[0] > 0].min() for t in trajs if (t[0] > 0).any())
    hi = max(t[0].max() for t in trajs)
    grid = np.unique(np.round(np.geomspace(max(lo, 1), hi, npoints)))
    curve = []
    for g in grid:
        vals = [float(_step_interp(c, b, np.array([g]))[0]) for c, b in trajs]
        med, clo, chi = _boot_ci([v for v in vals if np.isfinite(v)])
        ns = sum(np.isfinite(v) for v in vals)
        if med is not None:
            curve.append({"cum_vqe": float(g), "median": med, "ci_lo": clo, "ci_hi": chi, "n_seeds": ns})
    return {"curve": curve, "n_seeds": len(trajs)}


def agg_frozen(runs, method):
    """Per checkpoint (aligned by index): cross-seed of per-seed mean(ep_best) and best-of-COMMON_N."""
    per = [frozen_ebest_ckpts(r, method) for r in runs]
    per = [p for p in per if p]
    if not per:
        return None
    nck = min(len(p) for p in per)
    curve = []
    for i in range(nck):
        xs = [p[i][0] for p in per]
        seed_mean = [float(p[i][1].mean()) for p in per]
        seed_best = [float(p[i][1][:COMMON_N].min()) for p in per]         # common-N best-of-N
        mmed, mlo, mhi = _boot_ci(seed_mean)
        bmed, blo, bhi = _boot_ci(seed_best)
        curve.append({"cum_vqe": float(np.mean(xs)),
                      "mean": mmed, "mean_lo": mlo, "mean_hi": mhi,
                      "best": bmed, "best_lo": blo, "best_hi": bhi, "n_seeds": len(per)})
    return {"curve": curve, "n_seeds": len(per)}


def milestones(groups):
    """E: per molecule, target = init_err * {0.75,0.5,0.25}; reachers + conditional median VQE."""
    RATIOS = [0.75, 0.50, 0.25]
    out = {}
    for mol in MOLS:
        # initial error = HF-floor = median first running-best of the from-HF-start methods
        # (Full/RLQAS begin at HF; PSQAS begin at random circuits, so they are not a fair reference).
        inits = []
        for method in ("Full", "RLQAS"):
            for r in groups.get((method, mol), []):
                t = training_best_traj(r, method)
                if t is not None and t[1].size:
                    inits.append(float(t[1][0]))
        if not inits:                                                   # fallback: any method
            for method in METHODS:
                for r in groups.get((method, mol), []):
                    t = training_best_traj(r, method)
                    if t is not None and t[1].size:
                        inits.append(float(t[1][0]))
        if not inits:
            continue
        # initial error = the hard warmup starting point (from-HF methods' worst first-best);
        # milestones are coarse fractional reductions {75,50,25}% from it. (Warmup is bimodal
        # HF-floor vs worse-random-circuit, so median/percentile are unstable -> use max.)
        init = float(np.max(inits))
        targets = [init * rr for rr in RATIOS]
        rows = {}
        for method in METHODS:
            runs = groups.get((method, mol), [])
            trajs = [training_best_traj(r, method) for r in runs]
            trajs = [t for t in trajs if t is not None]
            cells = []
            for tg in targets:
                vqe_hits = []
                for c, b in trajs:
                    idx = np.where(b <= tg)[0]
                    if idx.size:
                        vqe_hits.append(float(c[idx[0]]))
                if vqe_hits:
                    cells.append((len(vqe_hits), len(trajs), float(np.median(vqe_hits))))
                else:
                    cells.append((0, len(trajs), None))
            rows[method] = cells
        out[mol] = {"init": init, "ratios": RATIOS, "targets": targets, "rows": rows}
    return out


# ---------------- rendering ----------------
COLORS = {"Full": "#1f77b4", "RLQAS": "#ff7f0e", "crlqas": "#2ca02c",
          "hyrlqas": "#d62728", "qdarts": "#9467bd", "gqeqas": "#8c564b"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_training_best(groups):
    plt = _plt()
    for mol in MOLS:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        any_line = False
        for method in METHODS:
            runs = groups.get((method, mol), [])
            if not runs:
                continue
            agg = agg_training_best(runs, method)
            if not agg or not agg["curve"]:
                continue
            c = agg["curve"]
            x = [p["cum_vqe"] for p in c]; y = [p["median"] for p in c]
            lo = [p["ci_lo"] for p in c]; hi = [p["ci_hi"] for p in c]
            ls = "--" if method not in COMPARABLE_VQE else "-"
            ax.plot(x, y, ls, color=COLORS[method], label=f"{METHOD_DISP[method]} (n={agg['n_seeds']})")
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.15)
            any_line = True
        if not any_line:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cumulative training VQE calls"); ax.set_ylabel("training best error (mHa)")
        ax.set_title(f"Training-search best-seen — {MOL_DISP[mol]}")
        ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(f"{OUT}/trainbest_{MOL_DISP[mol]}.png", dpi=150)
        plt.close(fig)
        print(f"[fig] trainbest_{MOL_DISP[mol]}.png")


def render_frozen(groups):
    plt = _plt()
    for mol in MOLS:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        any_line = False
        for method in METHODS:
            runs = groups.get((method, mol), [])
            if not runs:
                continue
            agg = agg_frozen(runs, method)
            if not agg or not agg["curve"]:
                continue
            c = agg["curve"]; x = [p["cum_vqe"] for p in c]
            # frozen data for ALL methods (Full/RLQAS from new eval, PSQAS from own eval.jsonl);
            # style solid vs dashed only by VQE-comparability (qdarts/gqe not per-step comparable).
            ls = "-" if method in COMPARABLE_VQE else "--"
            lab = METHOD_DISP[method] + ("" if method in COMPARABLE_VQE else " [VQE n/c]")
            ax.plot(x, [p["mean"] for p in c], ls, color=COLORS[method], label=f"{lab} mean")
            ax.plot(x, [p["best"] for p in c], ls, color=COLORS[method], alpha=0.45, lw=1)
            ax.fill_between(x, [p["mean_lo"] for p in c], [p["mean_hi"] for p in c],
                            color=COLORS[method], alpha=0.12)
            any_line = True
        if not any_line:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cumulative training VQE calls")
        ax.set_ylabel("frozen episode-best error (mHa)  [solid=mean, faint=best-of-%d]" % COMMON_N)
        ax.set_title(f"Frozen-policy quality — {MOL_DISP[mol]}")
        ax.legend(fontsize=6.5); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(f"{OUT}/frozen_{MOL_DISP[mol]}.png", dpi=150)
        plt.close(fig)
        print(f"[fig] frozen_{MOL_DISP[mol]}.png")


def render_milestone_table(ms):
    lines = ["# Training milestones (target = initial_error x {0.75,0.50,0.25})", ""]
    for mol, d in ms.items():
        lines.append(f"## {MOL_DISP[mol]}  (initial error = {d['init']:.3f} mHa)")
        hdr = "| method | " + " | ".join(f"<={t:.3f}mHa ({int(r*100)}%)"
                                         for r, t in zip(d["ratios"], d["targets"])) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(d["ratios"]) + 1))
        for method in METHODS:
            if method not in d["rows"]:
                continue
            cells = []
            for (nr, nt, med) in d["rows"][method]:
                cells.append("not reached" if nr == 0 else f"{med/1000:.1f}k VQE ({nr}/{nt})")
            lines.append(f"| {METHOD_DISP[method]} | " + " | ".join(cells) + " |")
        lines.append("")
    txt = "\n".join(lines)
    open(f"{OUT}/milestones.md", "w").write(txt)
    print("[table] milestones.md\n"); print(txt)


# ---------------- PSQAS per-episode proxy (dataset A) from episode_traces.txt ----------------
_ERRS_RE = re.compile(r"energy_errors_ha\s*=\s*\[([^\]]*)\]")


def parse_episode_traces(path):
    """{episode:int -> per-prefix error array (mHa)} from a PSQAS episode_traces.txt."""
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path).read()
    for block in txt.split("[episode ")[1:]:
        try:
            ep = int(block[:block.index("]")])
            m = _ERRS_RE.search(block)
            if not m:
                continue
            errs = np.fromstring(m.group(1), sep=",") * 1000.0
            if errs.size:
                out[ep] = errs
        except Exception:
            continue
    return out


def psqas_proxy_ckpts(run_dir):
    """PSQAS dataset-A proxy: per checkpoint, window of N training episodes ENDING at the
    checkpoint episode; per-episode best/mean/median/var/final from episode_traces.txt.
    Returns [(cum_vqe, ep_best[], ep_mean[], ep_median[], ep_var[], ep_final[])]."""
    ejl = f"{run_dir}/eval.jsonl"
    tr = f"{run_dir}/episode_traces.txt"
    if not (os.path.exists(ejl) and os.path.exists(tr)):
        return None
    ck = [json.loads(l) for l in open(ejl) if l.strip()]
    traces = parse_episode_traces(tr)
    if not traces:
        return None
    eps_sorted = np.array(sorted(traces))
    out = []
    for c in sorted(ck, key=lambda r: r["cum_train_vqe_calls"]):
        E = int(c["episode"]); N = int(c.get("n_eval_episodes", 20))
        win = eps_sorted[eps_sorted <= E][-N:]                        # N training eps ending at E
        if win.size == 0:
            win = eps_sorted[:N]
        reds = [_episode_reductions_np(traces[e]) for e in win]
        out.append({"cum_vqe": c["cum_train_vqe_calls"],
                    "ep_best": [r["best"] for r in reds], "ep_mean": [r["mean"] for r in reds],
                    "ep_median": [r["median"] for r in reds], "ep_var": [r["var"] for r in reds],
                    "ep_final": [r["final"] for r in reds]})
    return out


def _episode_reductions_np(errs):
    return dict(best=float(errs.min()), mean=float(errs.mean()), median=float(np.median(errs)),
                var=float(errs.var(ddof=1)) if errs.size > 1 else 0.0, final=float(errs[-1]))


def write_datasets(groups):
    """Write datasets A (per-episode) + B (per-checkpoint summary) as CSV. Source per method:
    Full/RLQAS = frozen eval_traces.jsonl; PSQAS = training-trace proxy (episode_traces.txt)."""
    fa = open(f"{OUT}/A_policy_episode_metrics.csv", "w")
    fa.write("method,molecule,seed,cum_train_vqe,episode_idx,best,mean,median,var,final,source\n")
    fb = open(f"{OUT}/B_policy_checkpoint_summary.csv", "w")
    fb.write("method,molecule,seed,cum_train_vqe,n_episodes,ebest_mean,ebest_median,ebest_var,best_of_N,source\n")
    for (method, mol) in sorted(groups):
        for rd in sorted(groups[(method, mol)], key=_seed_of):
            seed = _seed_of(rd)
            if method in ("Full", "RLQAS"):
                f = f"{rd}/eval_traces.jsonl"
                if not os.path.exists(f):
                    continue
                recs = [json.loads(l) for l in open(f) if l.strip()]
                src = "frozen"
                getter = lambda r: (r["cum_train_vqe_calls"], r["ep_best"], r["ep_mean"],
                                    r["ep_median"], r["ep_var"], r["ep_final"])
            else:
                recs = psqas_proxy_ckpts(rd)
                if not recs:
                    continue
                src = "train-proxy"
                getter = lambda r: (r["cum_vqe"], r["ep_best"], r["ep_mean"],
                                    r["ep_median"], r["ep_var"], r["ep_final"])
            for r in recs:
                cv, eb, em, emd, ev, ef = getter(r)
                for i in range(len(eb)):
                    fa.write(f"{METHOD_DISP[method]},{MOL_DISP[mol]},{seed},{cv},{i},"
                             f"{eb[i]:.6g},{em[i]:.6g},{emd[i]:.6g},{ev[i]:.6g},{ef[i]:.6g},{src}\n")
                eba = np.array(eb, float)
                fb.write(f"{METHOD_DISP[method]},{MOL_DISP[mol]},{seed},{cv},{len(eb)},"
                         f"{eba.mean():.6g},{np.median(eba):.6g},{eba.var(ddof=1) if eba.size>1 else 0:.6g},"
                         f"{eba[:COMMON_N].min():.6g},{src}\n")
    fa.close(); fb.close()
    print(f"[data] A_policy_episode_metrics.csv + B_policy_checkpoint_summary.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "coverage"
    os.makedirs(OUT, exist_ok=True)
    if cmd == "coverage":
        rep = coverage(); print(rep); open(f"{OUT}/coverage.txt", "w").write(rep)
        print(f"\n[written] {OUT}/coverage.txt")
    elif cmd == "track2":
        g = find_runs()
        render_training_best(g)
        render_milestone_table(milestones(g))
    elif cmd == "frozen":
        render_frozen(find_runs())
    elif cmd == "datasets":
        write_datasets(find_runs())
    elif cmd == "all":
        g = find_runs()
        render_training_best(g); render_frozen(g); render_milestone_table(milestones(g))
        write_datasets(g)
        rep = coverage(); open(f"{OUT}/coverage.txt", "w").write(rep)
        print("[done] all figures + datasets + coverage in", OUT)
