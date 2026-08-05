"""Fail loudly if the two embedded copies of noise_ptm have drifted apart.

WHY THIS EXISTS
---------------
The package is embedded in two trees:

    DreamQAS/code/noise_ptm          <- consumed by DreamQAS Full / No-imag / DreamQAS-RL
    PSQASBench/noise_ptm             <- consumed by CRLQAS / HyRLQAS

Those five arms are compared *against each other* in the paper's noise section. If the
copies diverge, the comparison silently becomes "five methods under two different noise
models", which is not a result — it is a bug that looks like one. Nothing else in either
repo would notice: both copies import fine, both produce plausible energies.

So: run this before launching a noise campaign, and after editing either copy.

    python code/noise_ptm/verify_sync.py            # exits non-zero on drift
    python code/noise_ptm/verify_sync.py --update   # copy DreamQAS -> PSQASBench

Only the modules that affect numbers are compared. tests/ and sweeps/ live in the
DreamQAS tree only and are deliberately not mirrored.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

DREAMQAS_COPY = Path("/home/USER/DreamQAS/code/noise_ptm")
PSQAS_COPY = Path("/home/USER/NeurIPS2026/PSQASBench/noise_ptm")

# Everything whose contents can change a computed energy.
TRACKED = (
    "__init__.py", "spec.py", "channels.py", "ptm_utils.py",
    "compile.py", "readout.py", "forward_4q.py", "evaluator.py", "integration.py",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="overwrite the PSQASBench copy from the DreamQAS one")
    a = ap.parse_args()

    if a.update:
        for name in TRACKED:
            shutil.copyfile(DREAMQAS_COPY / name, PSQAS_COPY / name)
        print(f"[sync] copied {len(TRACKED)} files  {DREAMQAS_COPY} -> {PSQAS_COPY}")

    bad, missing = [], []
    for name in TRACKED:
        src, dst = DREAMQAS_COPY / name, PSQAS_COPY / name
        if not src.exists():
            missing.append(f"missing in DreamQAS: {src}")
            continue
        if not dst.exists():
            missing.append(f"missing in PSQASBench: {dst}")
            continue
        hs, hd = _sha(src), _sha(dst)
        status = "OK " if hs == hd else "DRIFT"
        print(f"  {status}  {name:<16} {hs[:16]}")
        if hs != hd:
            bad.append(name)

    print()
    if missing:
        for m in missing:
            print(f"  !! {m}")
    if bad:
        print(f"!! {len(bad)} file(s) differ between the two copies: {', '.join(bad)}")
        print("   The five noise arms would NOT share a noise model. Fix before running.")
        print("   Re-sync with:  python code/noise_ptm/verify_sync.py --update")
        return 1
    if missing:
        return 1
    print(f"[sync] all {len(TRACKED)} tracked files identical — the five arms share one "
          f"noise model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
