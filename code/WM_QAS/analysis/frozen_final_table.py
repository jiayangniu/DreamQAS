"""Cross-method FROZEN converged-policy quality table on BOTH reductions:
  final-circuit (errs[-1], the committed circuit) and episode-best (min-over-prefix).
DreamQAS-family (Full/No-imag/RLQAS) read analysis eval_traces.jsonl; PSQAS (CRLQAS/HyRLQAS)
read eval_traces_final.jsonl produced by PSQASBench/offline_final_eval.py. Per run we take the
FINAL (converged) checkpoint = max _ck_ep; cross-seed mean±std. Writes outputs/main_results/."""
import json, glob, os
import numpy as np

C = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
PMK = {"LiH4q": "DQ_LiH_4q", "BeH2": "DQ_BeH2_6q", "LiH6q": "DQ_LiH_6q",
       "BeH2_8q": "T5_BeH2_631G_8q", "BeH2_10q": "T5_BeH2_6311G_10q"}
MOLS = ["LiH4q", "BeH2", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH4q", "BeH2": "BeH2_6q", "LiH6q": "LiH6q", "BeH2_8q": "BeH2_8q", "BeH2_10q": "BeH2_10q"}
FINEP = {"LiH4q": 15000, "BeH2": 15000, "LiH6q": 30000, "BeH2_8q": 15000, "BeH2_10q": 15000}
METH = ["Full", "No-imag", "RLQAS", "CRLQAS", "HyRLQAS"]


def runs(method, mol):
    if method == "Full":    return glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{mol}_s*_q/eval_traces.jsonl")
    if method == "No-imag": return glob.glob(f"{C}/ablations/gru_energy_none_{mol}_s*_ab_noimag/eval_traces.jsonl")
    if method == "RLQAS":   return glob.glob(f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{mol}/seed_*/eval_traces.jsonl")
    if method == "CRLQAS":  return glob.glob(f"{C}/psqas/crlqas/{PMK[mol]}/*/*/seed*/eval_traces_final.jsonl")
    if method == "HyRLQAS": return glob.glob(f"{C}/psqas_hyrlqas_std/hyrlqas/{PMK[mol]}/*/*/seed*/eval_traces_final.jsonl")
    return []


def final_rec(f):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("ep_final") and r.get("ep_best")]
    return max(rows, key=lambda r: r.get("_ck_ep", r.get("episode", 0))) if rows else None


def agg(method, mol):
    fin, eb, conv = [], [], 0
    for f in runs(method, mol):
        r = final_rec(f)
        if not r:
            continue
        fin.append(np.mean(r["ep_final"])); eb.append(np.mean(r["ep_best"]))
        if r.get("_ck_ep", r.get("episode", 0)) >= FINEP[mol] * 0.9:
            conv += 1
    if not fin:
        return None
    return len(fin), conv, np.mean(fin), np.std(fin), np.mean(eb), np.std(eb)


def main():
    out = ["FROZEN converged-policy quality — cross-seed mean±std (mHa). final-circuit | episode-best",
           "(DreamQAS-family: analysis eval_traces; PSQAS: offline_final_eval eval_traces_final; final ckpt)"]
    for mol in MOLS:
        out.append(f"\n### {DISP[mol]}")
        out.append(f"  {'method':9s} {'n(conv)':>8s} {'FINAL-CIRCUIT':>20s} {'EPISODE-BEST':>20s}")
        base_fin = {}
        for m in METH:
            a = agg(m, mol)
            if not a:
                out.append(f"  {m:9s}   pending"); continue
            n, cv, fm, fs, bm, bs = a
            base_fin[m] = fm
            out.append(f"  {m:9s} {n}({cv}c){'':>2s} {fm:>10.3g}±{fs:<8.2g} {bm:>10.3g}±{bs:<8.2g}")
        if "Full" in base_fin:
            edges = [f"{m} {base_fin[m] / base_fin['Full']:.0f}x" for m in METH
                     if m in base_fin and m != "Full" and base_fin["Full"] > 0]
            out.append(f"  Full final-circuit edge: " + "  ".join(edges))
    rep = "\n".join(out)
    print(rep)
    d = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/frozen_final_vs_episodebest.txt", "w").write(rep)
    print(f"\n[written] {d}/frozen_final_vs_episodebest.txt")


if __name__ == "__main__":
    main()
