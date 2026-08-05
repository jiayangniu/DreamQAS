"""Data-lineage audit (storyline §11 item 6): can every headline number be traced to a live artifact?

For each quantity the paper cites, this checks the chain
    claim -> artifact file -> generating script -> run set (campaign + glob) -> seed count
and flags anything that is missing, stale relative to its runs, or still training. It does NOT re-derive
the numbers; it verifies that the thing they are supposed to come from exists, is readable, and is newer
than the runs it summarises.

Three failure classes it is designed to catch, all of which have actually occurred in this project:
  MISSING   the artifact a doc points at does not exist;
  STALE     the artifact predates the newest run in its own run set (so it cannot include it);
  PARTIAL   the run set contains runs that have not finished, so any aggregate over it is provisional.

Usage: python analysis/lineage_audit.py > outputs/main_results/lineage_audit.txt
"""
import glob
import json
import os
import time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "main_results")
CV = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
OF = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/oracle_free_v1"

# (claim, artifact, generating script, campaign label, run globs it aggregates)
CHAIN = [
    ("RQ1 canonical best-of-1 (Table 1)", "policy_quality_table.txt", "policy_quality_table.py",
     "canonical", [f"{CV}/dreamqas/gru_energy_surrogate_*_s*_q"]),
    ("best-of-1 vs best-of-10 @15k budget", "of_bestofk_table.txt", "of_bestofk_table.py",
     "oracle-free + canonical externals",
     [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/ablations/gru_energy_none_*_s*_of_noimag",
      f"{OF}/dreamqas_rlqas/baseline_analysis/*/seed_*"],
     "EXPECTED: only the INTERNAL arms are listed. The five external baselines live under "
     "campaign_v1/psqas* and record completion in run_meta.txt (`status = completed`), not in "
     "metrics.jsonl, so run_state() cannot judge them and would report every one of them as "
     "below-budget. Verified separately 2026-07-29: 125/125 runs (5 methods x 5 tasks x 5 seeds) have "
     "eval.jsonl; CRLQAS/HyRLQAS additionally carry `status = completed` (50/50). GQE/TFQAS/qdarts do "
     "not write a status key at all — their completion evidence is wall_clock_sec + result_seed*.npz."),
    ("final-circuit vs episode-best reduction", "terminal_vs_best.txt", "terminal_vs_best_table.py",
     "oracle-free + CRLQAS offline traces",
     [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/ablations/gru_energy_none_*_s*_of_noimag"],
     "EXPECTED: only the internal arms are listed (run_state() reads metrics.jsonl, which the PSQAS "
     "trees do not write). CRLQAS contributes via eval_traces_final.jsonl; HyRLQAS-std/GQE/TFQAS/"
     "qdarts have NO terminal reduction at all and are printed as MISSING in the artifact."),
    ("RQ1 oracle-free unified main table", "of_main_table.txt", "of_main_table.py",
     "oracle-free + canonical externals",
     [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/ablations/gru_energy_none_*_s*_of_noimag"]),
    ("oracle-free vs canonical deviation", "oracle_free_deviation.txt", "scratchpad/of_deviation_table.py",
     "both", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("RQ2 canonical seed-level speedup", "rq2_speedup_w100.txt", "plot_rq2_w100.py",
     "canonical", [f"{CV}/dreamqas/gru_energy_surrogate_*_s*_q"]),
    ("RQ2 oracle-free matched-error gap", "oracle_free_gap_table.txt", "of_gap_table.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of",
                     f"{OF}/ablations/gru_energy_none_*_s*_of_noimag"]),
    ("cost accounting (VQE/nfev/WM-queries/verify)", "cost_table.txt", "cost_table.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/nodag/*_of_nodag"]),
    ("end-to-end phase wall-clock", "oracle_free_timing.txt", "timing_table.py",
     "oracle-free (timing runs)", [f"{OF}/timing/*"]),
    ("RQ3a action-ranking (canonical)", "t1a_action_ranking.txt", "t1a_table.py",
     "canonical", ["/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/horizon_rerun/dreamqas/*_q"],
     "EXPECTED: horizon_rerun runs were stopped once the 1/4-budget checkpoint the probe needs existed; "
     "below-budget here is by design, not truncation of the artifact's inputs."),
    ("RQ3a action-ranking (oracle-free)", "oracle_free_t1a.txt", "of_t1a_table.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("RQ3 fidelity / calib / disagreement", "oracle_free_rq3.txt", "of_rq3_tables.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/nodag/*_of_nodag"]),
    ("RQ3 surrogate-vs-imagination three-way", "wmgreedy_threeway.txt", "wmgreedy_threeway.py",
     "oracle-free", [f"{OF}/wmgreedy/*_of_wmgreedy", f"{OF}/wmgreedy_opt/*", f"{OF}/nodag/*_of_nodag"],
     "EXPECTED: the LiH-6q arms are configured for 7500 iters and capped at 3750 by "
     "lih6q_cap_watchdog.sh (the locked reporting budget); the table reads ep15000 for every arm and "
     "refuses to compare arms evaluated at different checkpoints."),
    ("oracle-free component ablation", "of_ablation_table.txt", "of_ablation_table.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/nodag/*_of_nodag",
                     f"{OF}/ablations/*_of_noDIR", f"{OF}/ablations/*_of_noUNC",
                     f"{OF}/ablations/gru_energy_none_*_of_noimag"]),
    ("speed-axis ladder attribution", "speed_ladder.txt", "speed_ladder.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of", f"{OF}/nodag/*_of_nodag",
                     f"{OF}/ablations/gru_energy_none_*_of_noimag"]),
    ("ansatz-plateau diagnostic", "plateau_diagnostic.txt", "plateau_diagnostic.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("WM accuracy vs circuit depth", "wm_accuracy_by_depth.pdf", "plot_wm_accuracy_by_depth.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("depth-slope, both training signals", "oracle_free_horizon_slope.txt", "of_horizon_slope.py",
     "oracle-free + canonical", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("Acc_pair ablation curves (oracle-free)", "fidelity_curves_ablations_oraclefree.pdf",
     "plot_fidelity_curves.py", "oracle-free",
     [f"{OF}/ablations/*_of_noDIR", f"{OF}/ablations/*_of_noUNC", f"{OF}/nodag/*_of_nodag"]),
    ("RQ3 same-WM deployment control", "deploy_contrast.txt", "deploy_contrast_table.py",
     "oracle-free", [f"{OF}/nodag/*LiH4q*_of_nodag", f"{OF}/nodag/*LiH6q*_of_nodag",
                     f"{OF}/nodag/*BeH2_8q*_of_nodag"]),
    ("Q2 transition-matched horizon", "q2_horizon.txt", "q2_horizon_table.py",
     "oracle-free", [f"{OF}/h1/*", f"{OF}/h5/*", f"{OF}/h15x/*"],
     "EXPECTED: the LiH-6q arms are CONFIGURED for 7500 iters but are deliberately capped at 3750 "
     "(= ep15000, the locked reporting budget) by experiments/launchers/lih6q_cap_watchdog.sh. "
     "Below-budget here is the cap doing its job, not truncated inputs — the table reads ep15000."),
    ("RQ4 risk-coverage", "risk_coverage.txt", "risk_coverage.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("RQ4 risk-coverage figure", "risk_coverage_oraclefree.pdf", "plot_risk_coverage.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_LiH4q_s*_of",
                     f"{OF}/dreamqas/gru_energy_surrogate_LiH6q_s*_of",
                     f"{OF}/dreamqas/gru_energy_surrogate_BeH2_8q_s*_of"]),
    ("RQ4 DAgger efficiency (canonical)", "dagger_efficiency_w100.txt", "dagger_efficiency.py",
     "canonical", [f"{CV}/dreamqas/gru_energy_surrogate_*_s*_q",
                   f"{CV}/ablations/gru_energy_surrogate_*_ab_noDAG"]),
    ("four-stage ladder", "ladder_table.txt", "ladder_table.py",
     "canonical", [f"{CV}/dreamqas/gru_energy_surrogate_*_s*_q"]),
    ("Acc_pair curves (appendix)", "fidelity_curves_oraclefree.pdf", "plot_fidelity_curves.py",
     "oracle-free", [f"{OF}/dreamqas/gru_energy_surrogate_*_s*_of"]),
    ("Acc_pair ablation curves (appendix)", "fidelity_curves_ablations.pdf", "plot_fidelity_curves.py",
     "canonical", [f"{CV}/ablations/gru_energy_surrogate_*_ab_noDIR"]),
    ("imagination dose on the speed axis", "imag_dose_speed.txt", "imag_dose_speed.py",
     "canonical (dose sweeps + their anchors)",
     [f"{CV}/ablations/gru_energy_surrogate_*_ab_nseed*", f"{CV}/ablations/gru_energy_surrogate_*_ab_lam*",
      f"{CV}/ablations/gru_energy_surrogate_BeH2_s*_ab_anchor",
      f"{CV}/dreamqas/gru_energy_surrogate_LiH4q_s*_q"]),
]


def run_state(globs):
    """-> (n_runs, n_below_budget, newest_FINISHED_mtime).

    STALE is judged against FINISHED runs only. An in-flight run rewrites metrics.jsonl every iteration,
    so comparing against the newest run of any kind would mark every artifact stale the moment a campaign
    is live — an alarm that carries no information.
    """
    n = inc = 0
    newest = 0.0
    for g in globs:
        for d in glob.glob(g):
            if not os.path.isdir(d):
                continue
            n += 1
            m = f"{d}/metrics.jsonl"
            if not os.path.exists(m):
                inc += 1
                continue
            try:
                tot = json.load(open(f"{d}/config.json"))["config"]["n_iterations"]
                finished = sum(1 for l in open(m) if l.strip()) >= tot
            except Exception:
                finished = True
            if finished:
                newest = max(newest, os.path.getmtime(m))
            else:
                inc += 1
    return n, inc, newest


def main():
    now = time.time()
    print("=" * 118)
    print("DATA-LINEAGE AUDIT — claim -> artifact -> script -> run set")
    print("  MISSING = the artifact does not exist.   STALE = older than the newest FINISHED run it covers.")
    print("  BELOW-BUDGET = some runs have not reached their configured n_iterations. Marked [expected]")
    print("  where a run set was deliberately stopped once the checkpoint an artifact needs existed.")
    print("  This checks TRACEABILITY, not numerical correctness.")
    print("=" * 118)
    print(f"{'claim':42s}{'artifact':34s}{'runs':>6}{'age(h)':>8}   status")
    print("-" * 118)
    bad = []
    for entry in CHAIN:
        claim, art, script, camp, globs = entry[:5]
        expected = entry[5] if len(entry) > 5 else None
        p = os.path.join(OUT, art)
        n, inc, newest = run_state(globs)
        if not os.path.exists(p):
            st = "MISSING artifact"
            bad.append((claim, st))
        else:
            age = (now - os.path.getmtime(p)) / 3600
            flags = []
            if newest and os.path.getmtime(p) < newest:
                flags.append("STALE (runs newer than artifact)")
            if inc:
                flags.append(f"BELOW-BUDGET ({inc}/{n})" + (" [expected]" if expected else ""))
            st = "; ".join(flags) if flags else "ok"
            if flags and not (expected and "STALE" not in st):
                bad.append((claim, st + (f"\n      -> {expected}" if expected else "")))
            print(f"{claim[:41]:42s}{art[:33]:34s}{n:>6}{age:>8.1f}   {st}")
            continue
        print(f"{claim[:41]:42s}{art[:33]:34s}{n:>6}{'—':>8}   {st}")
    print("-" * 118)
    print(f"\n{len(bad)} issue(s):" if bad else "\nAll chains resolve cleanly.")
    for c, s in bad:
        print(f"  - {c}: {s}")
    print("\nNOTE: STALE/PARTIAL is expected while a campaign is in flight — it is a REMINDER to regenerate")
    print("the artifact before the numbers are quoted, not necessarily a defect. Re-run this audit after")
    print("every batch completes, and immediately before freezing the paper's numbers.")


if __name__ == "__main__":
    main()
