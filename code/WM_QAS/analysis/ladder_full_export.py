"""COMPLETE four-stage ladder export — every (task × rung): 5 raw seed values + mean±std +
median + IQR, and every rung ratio reported THREE ways: mean-based, median-based, collapse-flag.

Motivation (2026-07-19): the compact ladder_table mixes conventions (cells = mean±std; mechanism
text switches to median on collapse; but the printed repr/imag/DAgger multipliers are mean-based).
That is unsafe where a single/double outlier seed poisons the mean — e.g. BeH2(10q) RLQAS mean
13.568 vs median 1.596, so the mean-based repr x = 13.568/2.753 ≈ 4.9 CANNOT be cited as
representation/replay scaling evidence. This export lays out both conventions side by side so the
writing agent picks the defensible number per cell.

Rungs (best-of-1 mHa, lower=better):  RLQAS -> No-imag -> noDAG -> Full.
Ratios (>1 = the later rung is better):
  repr   = RLQAS / No-imag     (representation + replay)
  imag   = No-imag / noDAG     (imagination increment)
  DAgger = noDAG / Full        (DAgger increment)

Outputs (in outputs/main_results/):
  ladder_full_table.txt   human-readable
  ladder_full_cells.csv   one row per (task, rung): seed0..4, mean, std, median, q1, q3, iqr, collapse
  ladder_full_ratios.csv  one row per (task, transition): mean_ratio, median_ratio, collapse_flag
"""
import csv
import glob
import json
import os
import re

import numpy as np

C = "/data/RUN_ROOT/DreamQAS_transfer/2_current/dreamqas_campaign/campaign_v1"
TASKS = ["LiH4q", "BeH2_6q", "LiH6q", "BeH2_8q", "BeH2_10q"]
DISP = {"LiH4q": "LiH(4q)", "BeH2_6q": "BeH2(6q)", "LiH6q": "LiH(6q)",
        "BeH2_8q": "BeH2(8q)", "BeH2_10q": "BeH2(10q)"}
MOLKEY = {"BeH2_6q": "BeH2"}                         # 6q run dirs use the bare "BeH2" key
RUNGS = ["RLQAS", "No-imag", "noDAG", "Full"]
OUTDIR = f"{os.path.dirname(os.path.abspath(__file__))}/outputs/main_results"


def dirs(rung, task):
    k = MOLKEY.get(task, task)
    if rung == "Full":    return glob.glob(f"{C}/dreamqas/gru_energy_surrogate_{k}_s*_q")
    if rung == "No-imag": return glob.glob(f"{C}/ablations/gru_energy_none_{k}_s*_ab_noimag")
    if rung == "noDAG":   return glob.glob(f"{C}/ablations/gru_energy_surrogate_{k}_s*_ab_noDAG")
    if rung == "RLQAS":   return glob.glob(f"{C}/dreamqas_rlqas/baseline_analysis/G0_v2_{k}/seed_*")
    return []


def seed_of(path):
    m = re.search(r"(?:_s|seed_?)(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def bo1(d):
    """best-of-1 = mean of the FINAL checkpoint's ep_best over its eval episodes."""
    p = f"{d}/eval_traces.jsonl"
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("ep_best")]
    if not recs:
        return None
    fin = max(recs, key=lambda r: (r.get("n_eval_episodes", 0), r.get("_ck_ep", 0)))
    b = np.array(fin["ep_best"], float); b = b[np.isfinite(b)]
    return float(b.mean()) if b.size else None


def seed_values(rung, task):
    """dict seed->bo1 (only present seeds)."""
    out = {}
    for d in dirs(rung, task):
        v = bo1(d)
        if v is not None:
            out[seed_of(d)] = v
    return out


def stats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return dict(n=0)
    q1, q3 = np.percentile(a, [25, 75])
    mean = float(a.mean()); std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return dict(n=a.size, mean=mean, std=std, median=float(np.median(a)),
               q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
               collapse=bool(std > mean and mean > 0))


# -------- gather --------
cells = {}   # (task, rung) -> {seedvals, stats}
for t in TASKS:
    for r in RUNGS:
        sv = seed_values(r, t)
        cells[(t, r)] = dict(seeds=sv, st=stats(list(sv.values())))

SEED_IDS = sorted({s for v in cells.values() for s in v["seeds"]})

# -------- text --------
L = []
def P(s=""):
    L.append(s)

P("=" * 118)
P("COMPLETE FOUR-STAGE LADDER — best-of-1 (mHa, lower=better). Per (task, rung): 5 seed values +")
P("mean±std (ddof=1) + median + IQR.  collapse = std>mean (a single/few outlier seeds poison the mean).")
P("Rungs: RLQAS -> No-imag -> noDAG -> Full.   Read COLLAPSE cells via the MEDIAN.")
P("=" * 118)
hdr = f"{'task':<10}{'rung':<9}" + "".join(f"{'s'+str(s):>9}" for s in SEED_IDS) + \
      f"{'mean':>9}{'std':>8}{'median':>9}{'IQR(q1-q3)':>18}  collapse"
P(hdr); P("-" * len(hdr))
for t in TASKS:
    for r in RUNGS:
        c = cells[(t, r)]; st = c["st"]
        seedcells = "".join((f"{c['seeds'][s]:>9.3f}" if s in c["seeds"] else f"{'-':>9}") for s in SEED_IDS)
        if st["n"]:
            P(f"{DISP[t]:<10}{r:<9}{seedcells}{st['mean']:>9.3f}{st['std']:>8.3f}{st['median']:>9.3f}"
              f"{st['q1']:>8.3f}-{st['q3']:<9.3f}  {'⚠YES' if st['collapse'] else 'no'}")
        else:
            P(f"{DISP[t]:<10}{r:<9}{seedcells}{'(no data)':>44}")
    P("-" * len(hdr))

# -------- ratios --------
TRANS = [("repr", "RLQAS", "No-imag"), ("imag", "No-imag", "noDAG"), ("DAgger", "noDAG", "Full")]
P()
P("RUNG RATIOS — reported THREE ways (>1 = later rung better). USE MEDIAN-BASED where collapse=YES.")
P(f"{'task':<10}{'transition':<22}{'mean-based':>12}{'median-based':>14}   collapse (which endpoint)")
P("-" * 96)
ratio_rows = []
for t in TASKS:
    for name, hi, lo in TRANS:
        chi, clo = cells[(t, hi)]["st"], cells[(t, lo)]["st"]
        if not (chi["n"] and clo["n"]):
            continue
        mean_r = chi["mean"] / clo["mean"] if clo["mean"] > 0 else float("nan")
        med_r = chi["median"] / clo["median"] if clo["median"] > 0 else float("nan")
        flags = [n for n, cc in ((hi, chi), (lo, clo)) if cc["collapse"]]
        flag = ("⚠ " + "+".join(flags)) if flags else "no"
        P(f"{DISP[t]:<10}{name+' ('+hi+'/'+lo+')':<22}{mean_r:>12.2f}{med_r:>14.2f}   {flag}")
        ratio_rows.append(dict(task=DISP[t], transition=name, numerator=hi, denominator=lo,
                               mean_ratio=round(mean_r, 3), median_ratio=round(med_r, 3),
                               collapse=";".join(flags) if flags else ""))
P("-" * 96)
P("REPORTING CONVENTION (locked by user 2026-07-19): the MAIN TABLE and all rung ratios use MEAN±std")
P("CONSISTENTLY — no mean/median mixing. The MEAN is the expected best-of-1 error and it correctly")
P("REFLECTS a method's collapse/instability as a REAL property: a method that collapses on some seeds")
P("SHOULD score worse. The median is a DIAGNOSTIC only (this file) to separate 'typical-seed improvement'")
P("from 'collapse-avoidance'; it is NEVER the headline.")
P("How to read the mean ratios (collapse=YES cells): the advantage there is EXPECTED-ERROR / reliability,")
P("driven by collapse-avoidance, NOT by the typical seed. Examples:")
P("  * repr 10q mean 4.93: representation+replay avoids the model-free collapse (RLQAS 2/5 seeds ->16.6/47.2);")
P("    the median (1.04) shows the NON-collapsed seed is comparable, so cite 4.93 as 'lower expected error /")
P("    avoids model-free instability at scale', NOT 'the typical seed is 4.9x better'.")
P("  * DAgger 6q mean 21.9: DAgger's real-VQE verification prevents the 1/5 imagination collapse (6.09->0.058)")
P("    -> a QUANTIFIED safety-net value in expected error. (median 1.00 = the 4 non-collapsed seeds unchanged.)")

os.makedirs(OUTDIR, exist_ok=True)
open(f"{OUTDIR}/ladder_full_table.txt", "w").write("\n".join(L) + "\n")

with open(f"{OUTDIR}/ladder_full_cells.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "rung"] + [f"seed{s}" for s in SEED_IDS] +
               ["mean", "std", "median", "q1", "q3", "iqr", "collapse"])
    for t in TASKS:
        for r in RUNGS:
            c = cells[(t, r)]; st = c["st"]
            row = [DISP[t], r] + [(round(c["seeds"][s], 4) if s in c["seeds"] else "") for s in SEED_IDS]
            if st["n"]:
                row += [round(st["mean"], 4), round(st["std"], 4), round(st["median"], 4),
                        round(st["q1"], 4), round(st["q3"], 4), round(st["iqr"], 4),
                        "YES" if st["collapse"] else "no"]
            else:
                row += [""] * 7
            w.writerow(row)

with open(f"{OUTDIR}/ladder_full_ratios.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["task", "transition", "numerator", "denominator",
                                       "mean_ratio", "median_ratio", "collapse"])
    w.writeheader(); w.writerows(ratio_rows)

print("\n".join(L))
print(f"\n[written] ladder_full_table.txt, ladder_full_cells.csv, ladder_full_ratios.csv in {OUTDIR}")
