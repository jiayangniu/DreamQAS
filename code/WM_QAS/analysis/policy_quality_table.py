import re
import json, glob, os
import numpy as np
from scipy.stats import t as student_t

from analysis_paths import campaign_root, require_campaign, OUTPUTS

C = campaign_root("campaign_v1")
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH4q", "BeH2": "BeH2_6q", "LiH6q": "LiH6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
METH = ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS", "GQE", "TFQAS", "qdarts"]
K_CKPT = 3
def _mean_std_ci(x):
    x = np.asarray(x, float)
    n = x.size
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    ci = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return float(x.mean()), sd, ci


def bestofK(errs, K):
    """Expected minimum of K draws from the empirical error distribution."""
    e = np.sort(np.asarray([x for x in errs if np.isfinite(x)], float))
    n = e.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    w = ((n - i + 1) / n) ** K - ((n - i) / n) ** K
    return float((e * w).sum())


def run_series(method, mol):
    out = []
    if method in ("Full", "No-imag", "RLQAS"):
        pat = {"Full": f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q/eval_traces.jsonl",
               "No-imag": f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag/eval_traces.jsonl",
               "RLQAS": f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{mol}/seed_*/eval_traces.jsonl"}[method]
        for f in glob.glob(pat):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            s = [(r.get("_ck_ep", r.get("episode", 0)), r["ep_best"]) for r in rows if r.get("ep_best")]
            if s:
                out.append(sorted(s))
    else:
        pdir = {"GQE": "gqeqas", "qdarts": "qdarts"}.get(method, method.lower())
        root = f"{C}/psqas_hyrlqas_std" if method == "HyRLQAS" else f"{C}/psqas"
        for d in glob.glob(f"{root}/{pdir}/{PMK[mol]}/*/*/seed*"):
            if not re.search(r"/seed\d+$", d):
                continue
            p = f"{d}/eval.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p) if l.strip()]
            s = [(r.get("episode", 0), r["episode_bests"]) for r in rows if r.get("episode_bests")]
            if s:
                out.append(sorted(s))
    return out


def qdarts_delivered(mol):
    import numpy as _np
    vals = []
    for d in glob.glob(f"{C}/psqas/qdarts/{PMK[mol]}/*/*/seed*"):
        f = glob.glob(f"{d}/result_seed*.npz")
        if not f:
            continue
        z = _np.load(f[0], allow_pickle=True)
        vals.append(float(z["energy_error_ha"]) * 1000.0)
    return vals


def agg(method, mol):
    if method == "qdarts":
        v = qdarts_delivered(mol)
        if not v:
            return None
        x = np.asarray(v, float)
        t = _mean_std_ci(x)
        return {"n": len(v), "B1_1": t, "B1_10": t, "B2_1": t, "B2_10": t, "single": True}
    b1_1, b1_10, b2_1, b2_10 = [], [], [], []
    for series in run_series(method, mol):
        b1_1.append(bestofK(series[-1][1], 1))
        b1_10.append(bestofK(series[-1][1], 10))
        tail = series[-K_CKPT:]
        b2_1.append(np.mean([bestofK(e, 1) for _, e in tail]))
        b2_10.append(np.mean([bestofK(e, 10) for _, e in tail]))
    if not b1_1:
        return None

    ms = _mean_std_ci

    return {"n": len(b1_1), "B1_1": ms(b1_1), "B1_10": ms(b1_10),
            "B2_1": ms(b2_1), "B2_10": ms(b2_10)}


def cell(t):
    return f"{t[0]:.3g}±{t[1]:.2g}"


DOC = """DreamQAS policy-quality tables (clean canonical campaign).
Metrics are in mHa. Per-checkpoint best-of-K values are computed within each seed,
then aggregated across seeds; episodes are not pooled between seeds.
B1: final checkpoint. B2: average over the last three available checkpoints.
K=1 is the mean per-episode best-prefix error; K=10 is an empirical order statistic.
The target protocol uses five seeds. Each cell reports its actual sample count.
Missing inputs remain missing; partial campaigns are not complete paper reproductions.
External methods require separately supplied, protocol-matched native outputs.
QuantumDARTS reports one delivered circuit, not a sampled-policy distribution.
"""

POLICY_METHODS = ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS", "GQE", "TFQAS"]


def _tableA(out, key, title):
    out.append(f"\n{'='*100}\n{title}   [mean +/- sample std, actual n, mHa]\n{'='*100}")
    out.append(f"{'method':10s} | " + " | ".join(f"{DISP[m]:^15s}" for m in MOLS))
    out.append("-" * 100)
    for m in POLICY_METHODS:
        cells = []
        for mol in MOLS:
            a = agg(m, mol)
            cells.append(f"{cell(a[key])} (n={a['n']})" if a else "-")
        out.append(f"{m:10s} | " + " | ".join(f"{c:^15s}" for c in cells))
    cells = []
    for mol in MOLS:
        a = agg("qdarts", mol)
        cells.append(f"{cell(a[key])} (n={a['n']})" if a else "-")
    out.append(f"{'qdarts †':10s} | " + " | ".join(f"{c:^15s}" for c in cells))
    out.append("  † QuantumDARTS = deterministic differentiable NAS: value is its SINGLE DELIVERED circuit")
    out.append("    (best over its search, result_seed.npz); best-of-1 == best-of-10 by construction. Not a")
    out.append("    sampled-policy value -> not directly comparable to the sampled best-of-K columns above.")


def main():
    require_campaign(C)
    out = [DOC]
    _tableA(out, "B1_1", "TABLE A1 -- best-of-1  (AVERAGE single-deploy policy quality)  [HEADLINE]")
    _tableA(out, "B1_10", "TABLE A2 -- best-of-10 (OPTIMAL over 10 deploys)              [secondary]")
    out.append(f"\n{'='*100}\nDETAIL -- all methods, Block-1 (converged) vs Block-2 (last-{K_CKPT}, timing-robust)\n{'='*100}")
    for mol in MOLS:
        out.append(f"\n### {DISP[mol]}")
        hdr = (f"  {'method':9s} | {'B1 avg(bo1)':>14s} {'B1 opt(bo10)':>14s} | "
               f"{'B2 avg(bo1)':>14s} {'B2 opt(bo10)':>14s}")
        out.append(hdr); out.append("  " + "-" * (len(hdr) - 2))
        for m in METH:
            a = agg(m, mol)
            if not a:
                out.append(f"  {m:9s} | missing input"); continue
            note = "  <qdarts: single delivered circuit>" if m == "qdarts" else ""
            out.append(f"  {m:9s} (n={a['n']}) | {cell(a['B1_1']):>14s} {cell(a['B1_10']):>14s} | "
                       f"{cell(a['B2_1']):>14s} {cell(a['B2_10']):>14s}{note}")
    rep = "\n".join(out)
    print(rep)
    d = str(OUTPUTS / "main_results")
    os.makedirs(d, exist_ok=True)
    open(f"{d}/policy_quality_table.txt", "w").write(rep)
    print(f"\n[written] {d}/policy_quality_table.txt")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Read campaign data under DREAMQAS_RESULTS_ROOT; see REPRODUCE.md").parse_args()
    main()
