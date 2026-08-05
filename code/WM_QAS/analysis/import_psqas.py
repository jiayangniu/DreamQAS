"""PSQASBench importers (merged 2026-07-18 from import_psqas_baselines.py + import_psqas_molecules.py).

Subcommand `baselines` — import PSQASBench baseline runs into DreamQAS metrics.jsonl format.
  Converts the 4 baselines (crlqas, hyrlqas, qdarts, gqeqas) run on the DreamQAS molecules
  (DQ_LiH_4q / DQ_LiH_6q / DQ_BeH2_6q) into per-run `metrics.jsonl` with the phase2 schema
  (iter, vqe_calls, best_err_mHa, mean_ep_err_mHa) so DreamQAS analysis scripts read them as
  extra baselines on the same (vqe_calls, error) axes.
  VQE-call convention (aligned with DreamQAS's per-circuit-optimization unit):
    - RL (crlqas/hyrlqas): cum_vqe_calls = cumsum(n_steps) from episode_summary.tsv
      (one circuit VQE-optimization per gate-placement step, exactly DreamQAS's unit).
    - qdarts/gqeqas: cum_vqe_circuits from result_seed*.npz (direct_opt counter),
      per eval-checkpoint curve + final endpoint.
  Usage: python analysis/import_psqas.py baselines <psqas_results_root> [<out_root>]
  Default out root: <repo>/experiments/psqasbench_baselines

Subcommand `molecules` — import PSQASBench BeH2 8q/10q Hamiltonians into DreamQAS mol_data (B4).
  DreamQAS's env loads `mol_data/{ham_type}_{n}q_geom_{geometry}_{mapping}.npz` and reads keys
  `hamiltonian` (complex128 dense), `weights`, `eigvals`, `energy_shift`. The PSQASBench npz
  already stores exactly those, so this is a straight copy into a DreamQAS-named file.
  min_energy = min(eigvals)+energy_shift is the TRUE ground state; fake_min_energy (curriculum
  offset) is set in the cfg, not here.
  Usage: python analysis/import_psqas.py molecules
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

# ------------------------------------------------------------------ baselines

RL_METHODS = {"crlqas", "hyrlqas"}
NONRL_METHODS = {"qdarts", "gqeqas"}
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUT = os.path.join(_REPO, "experiments", "psqasbench_baselines")
REAL_EPS_PER_ITER = 4  # DreamQAS convention: 1 iteration aggregates 4 real episodes.


def _write_metrics(out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path, len(rows)


def convert_rl(run_dir):
    """episode_summary.tsv -> per-ITERATION metrics (aggregate 4 episodes/iter, like DreamQAS).

    Aggregating by 4 makes the iteration count match DreamQAS exactly
    (BeH2 2000 / LiH4q 5000 / LiH6q 7500 iters) so the last-W=100-iter
    pool is apples-to-apples, and each row's mean_ep_err_mHa is a 4-episode mean
    (same variance structure as DreamQAS's mean_ep_error)."""
    tsv = os.path.join(run_dir, "episode_summary.tsv")
    if not os.path.exists(tsv):
        return None
    # TRUE ground-state energy (robust to the env's fake_min_energy reference):
    # error is computed as |final_energy_ha - exact_energy|, NOT the logged
    # final_energy_error_ha (which is measured vs fake_min_energy).
    exact = None
    meta = os.path.join(run_dir, "run_meta.txt")
    if os.path.exists(meta):
        for line in open(meta):
            if line.strip().startswith("exact_energy_ha"):
                exact = float(line.split("=")[1])
                break
    with open(tsv) as f:
        eps = list(csv.DictReader(f, delimiter="\t"))
    if not eps or exact is None:
        return None
    cum_vqe, best, rows = 0, float("inf"), []
    n_iter = (len(eps) + REAL_EPS_PER_ITER - 1) // REAL_EPS_PER_ITER
    for it in range(n_iter):
        grp = eps[it * REAL_EPS_PER_ITER:(it + 1) * REAL_EPS_PER_ITER]
        errs = [abs(float(g["final_energy_ha"]) - exact) * 1000.0 for g in grp]
        steps = [int(float(g["n_steps"])) for g in grp]
        cum_vqe += sum(steps)
        best = min(best, min(errs))
        rows.append({"iter": it, "vqe_calls": cum_vqe,
                     "best_err_mHa": round(best, 6),
                     "mean_ep_err_mHa": round(sum(errs) / len(errs), 6),
                     "mean_ep_steps": round(sum(steps) / len(steps), 3)})
    return rows


def convert_nonrl(run_dir):
    """result_seed*.npz eval arrays -> per-checkpoint curve + final endpoint row."""
    hits = glob.glob(os.path.join(run_dir, "result_seed*.npz"))
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True)
    ecv = list(d["eval_cum_vqe_circuits"]) if "eval_cum_vqe_circuits" in d else []
    ebe = list(d["eval_best_error_mha"]) if "eval_best_error_mha" in d else []
    eme = list(d["eval_mean_error_mha"]) if "eval_mean_error_mha" in d else ebe
    best, rows = float("inf"), []
    for i, (v, e) in enumerate(zip(ecv, ebe)):
        best = min(best, float(e))
        m = float(eme[i]) if i < len(eme) else float(e)
        rows.append({"iter": i, "vqe_calls": int(v),
                     "best_err_mHa": best, "mean_ep_err_mHa": m})
    # final endpoint (total cum_vqe, final best circuit error)
    final_err = float(d["energy_error_ha"]) * 1000.0 if "energy_error_ha" in d else best
    total_vqe = int(d["cum_vqe_circuits"]) if "cum_vqe_circuits" in d else (int(ecv[-1]) if ecv else 0)
    best = min(best, final_err)
    rows.append({"iter": len(rows), "vqe_calls": total_vqe,
                 "best_err_mHa": best, "mean_ep_err_mHa": final_err})
    return rows


def run_baselines(src_root, out_root):
    n_runs = 0
    for method in sorted(RL_METHODS | NONRL_METHODS):
        # PSQAS layout: <root>/<method>/<mol>/<config_tag.../>/seed<N>/  (config_tag may be nested)
        for run_dir in sorted(glob.glob(os.path.join(src_root, method, "**", "seed*"), recursive=True)):
            if not os.path.isdir(run_dir):
                continue
            rel = os.path.relpath(run_dir, os.path.join(src_root, method)).split(os.sep)
            if len(rel) < 2:
                continue
            mol = rel[0]
            cfg_tag = "_".join(rel[1:-1]) or "run"       # config_tag path -> flat label
            seed = rel[-1].replace("seed", "")
            rows = convert_rl(run_dir) if method in RL_METHODS else convert_nonrl(run_dir)
            if not rows:
                continue
            out_dir = os.path.join(out_root, method, mol, cfg_tag, f"seed_{seed}")
            _, n = _write_metrics(out_dir, rows)
            n_runs += 1
            print(f"  {method:8s} {mol:12s} {cfg_tag:24s} seed{seed}: {n:5d} rows  "
                  f"final vqe_calls={rows[-1]['vqe_calls']:>8d}  best={rows[-1]['best_err_mHa']:.3f} mHa")
    print(f"\nDone: {n_runs} runs -> {out_root}")


# ------------------------------------------------------------------ molecules

PB = "/home/sh0/S4068570/NeurIPS2026/PSQASBench/mol_data"
DQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "data", "mol_data")
GEOM = "Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326"   # underscored; matches cfg [problem] geometry

SRC = {
    8:  "L6_BeH2_631G_8q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    10: "L6_BeH2_6311G_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
}


def run_molecules():
    os.makedirs(DQ, exist_ok=True)
    for nq, src in SRC.items():
        d = np.load(os.path.join(PB, src), allow_pickle=True)
        out = {
            "hamiltonian": np.asarray(d["hamiltonian"], dtype=np.complex128),
            "weights":     np.asarray(d["weights"]),
            "eigvals":     np.asarray(d["eigvals"]),
            "energy_shift": d["energy_shift"],
        }
        if "pauli_strings" in d.files:
            out["paulis"] = d["pauli_strings"]        # optional (env doesn't read it)
        dst = os.path.join(DQ, f"BEH2_{nq}q_geom_{GEOM}_jordan_wigner.npz")
        np.savez(dst, **out)
        me = float(np.min(out["eigvals"]) + out["energy_shift"])
        print(f"[{nq}q] wrote {os.path.basename(dst)}  H={out['hamiltonian'].shape}  "
              f"min_energy={me:.6f}  fake(-2.28)={me - 2.28:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines", help="PSQAS baseline runs -> metrics.jsonl")
    b.add_argument("src_root")
    b.add_argument("out_root", nargs="?", default=DEFAULT_OUT)
    sub.add_parser("molecules", help="PSQAS BeH2 8q/10q Hamiltonians -> mol_data npz")
    args = ap.parse_args()
    run_baselines(args.src_root, args.out_root) if args.cmd == "baselines" else run_molecules()
