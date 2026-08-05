"""RQ3 evidence recomputed on the ORACLE-FREE run set (no retraining, no new VQE).

Part A of the oracle-free RQ3 pass — the two quantities that training already logged per run:
  - pairwise ranking fidelity  (fidelity.jsonl `pairwise`, held-out buffer pairs)
  - disagreement vs prediction error (calibration.jsonl `disagree_vs_err_corr`,
    Spearman(ensemble std, |pred - real|) on the DAgger-verified sample)
  - the WM's own absolute calibration MAE, reported in ITS OWN target space:
      canonical runs  -> `calib_MAE_all`   = |pred − log10(true error)|      [log10 mHa]
      oracle-free run -> `calib_score_mae` = |pred − S(E)|                    [S units]
    ** The two MAE columns are NOT comparable across training signals ** (different target
    space); fidelity and the disagreement correlation ARE (both rank-based / scale-free).

Part B (per-prefix Spearman, normalized regret, WM-vs-matched-random) needs counterfactual VQE and
is produced by `t1a_action_ranking.py <run_dir> --ckpts onset,quarter`, then tabled by `t1a_table.py`.

Usage: python analysis/of_rq3_tables.py > outputs/main_results/oracle_free_rq3.txt
"""
import glob
import json
import os

import numpy as np

OF = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"
CANON = "/research/data/s4068570/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
MOLS = ["LiH4q", "LiH6q", "BeH2_8q"]
DISP = {"LiH4q": "LiH(4q)", "LiH6q": "LiH(6q)", "BeH2_8q": "BeH2(8q)"}
# label -> (glob template, is_oracle_free)
ARMS = {
    "oracle-free Full":   (f"{OF}/dreamqas/gru_energy_surrogate_{{m}}_s*_of", True),
    "oracle-free NoDAG":  (f"{OF}/nodag/gru_energy_surrogate_{{m}}_s*_of_nodag", True),
    "canonical Full":     (f"{CANON}/dreamqas/gru_energy_surrogate_{{m}}_s*_q", False),
}


def _last(path, keys):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in reversed(rows):
        if all(r.get(k) is not None for k in keys):
            return r
    return None


def run_row(d):
    """-> (pairwise_fidelity, calib_mae_own_space, disagree_corr) for one run dir."""
    f = _last(f"{d}/fidelity.jsonl", ["pairwise"])
    c = _last(f"{d}/calibration.jsonl", ["disagree_vs_err_corr"])
    mae = None
    if c is not None:
        mae = c.get("calib_MAE_all", c.get("calib_score_mae"))
    return (f["pairwise"] if f else None, mae, c["disagree_vs_err_corr"] if c else None)


def agg(pattern, mol):
    vals = [run_row(d) for d in sorted(glob.glob(pattern.format(m=mol)))]
    out = []
    for i in range(3):
        x = [v[i] for v in vals if v[i] is not None]
        out.append((float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, len(x))
                   if x else (None, None, 0))
    return out


def fmt(t):
    m, s, n = t
    return "        n/a" if m is None else f"{m:.3f}±{s:.3f}"


def main():
    print("=" * 100)
    print("RQ3 on the ORACLE-FREE checkpoints — Part A (logged during training; NO new VQE, no retraining)")
    print("  fidelity  = pairwise ranking accuracy on held-out buffer pairs  -> 'decision-useful ranker'")
    print("  calib-MAE = |pred − real| in the WM's OWN target space:")
    print("              canonical  = log10(true error)  [log10 mHa]   |   oracle-free = frontier score S")
    print("              ** NOT comparable across the two training signals — same role, different units **")
    print("  disagree  = Spearman(ensemble std, |prediction error|) on the DAgger-verified sample")
    print("  mean±std (ddof=1) over seeds; final logged value per run.")
    print("=" * 100)
    print(f"{'molecule':<10}{'arm':<20}{'fidelity':>14}{'calib-MAE(own)':>18}{'disagree-corr':>16}{'seeds':>7}")
    for mol in MOLS:
        print("-" * 100)
        for label, (pat, _of) in ARMS.items():
            a = agg(pat, mol)
            if a[0][2] == 0 and a[2][2] == 0:
                continue
            print(f"{DISP[mol]:<10}{label:<20}{fmt(a[0]):>14}{fmt(a[1]):>18}{fmt(a[2]):>16}"
                  f"{max(a[0][2], a[2][2]):>7}")
    print("=" * 100)
    print("READ: compare oracle-free vs canonical WITHIN the fidelity and disagree-corr columns only.")
    print("A preserved fidelity under the E0-free signal means the WM stays a usable RANKER when the")
    print("exact ground state is unavailable — which is the property the method actually relies on.")
    print("The calib-MAE columns document each WM's absolute accuracy in its own space; a smaller S-space")
    print("MAE does NOT mean better calibration than a log10-error MAE.")
    print("\nPart B (per-prefix Spearman / normalized regret / WM-vs-matched-random) requires counterfactual")
    print("VQE: run `t1a_action_ranking.py <run_dir> --ckpts onset,quarter` on the same runs, then")
    print("`t1a_table.py`. Ground-truth error there uses E0 as an EVAL-ONLY diagnostic (same licence as")
    print("best-of-1); no E0 enters action selection, pruning or checkpoint choice.")


if __name__ == "__main__":
    main()
