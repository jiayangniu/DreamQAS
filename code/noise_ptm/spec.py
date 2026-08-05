"""The noise specification DreamQAS's noise experiments use.

Deliberately narrower than NAHyRLQAS's five-channel ``NoiseSpec``: this project's
protocol is exactly three numbers.

    p1q   depolarizing after every single-qubit rotation gate
    p2q   depolarizing after every CNOT
    p_ro  symmetric readout assignment error, P(1̃|0) = P(0̃|1) = p_ro

There is no T1/T2 (amplitude damping / dephasing) here by design, so the circuit has no
per-moment channel and the PTM forward pass has no TIME slot.

Every ⟨H⟩ is the exact expectation on the noisy density matrix — no measurement
sampling, no shots. Consequences the rest of the code relies on:

* energy is **deterministic** given (circuit, θ), so COBYLA sees no sampling variance
  and the oracle-free frontier cannot be ratcheted down by fluctuations;
* noise still changes the energy landscape, the optimal parameters, and the ordering of
  architectures — which is the thing the experiment measures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseSpec4q:
    """Frozen, hashable noise specification. Hashability matters: it keys the PTM
    compile cache, which is the expensive one-time step."""

    p1q: float = 0.0
    p2q: float = 0.0
    p_ro: float = 0.0
    # Model the depolarizing error of the measurement basis-change gates (H for an X
    # support, S†H for a Y support). Exact and free — see readout.py.
    model_basis_change: bool = True

    def __post_init__(self) -> None:
        for name, v in (("p1q", self.p1q), ("p2q", self.p2q), ("p_ro", self.p_ro)):
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float; got {type(v)}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{name} out of [0, 1]: {v}")
        # p_ro > 0.5 inverts the assignment matrix; (1-2p) would go negative.
        if float(self.p_ro) > 0.5:
            raise ValueError(f"p_ro must be <= 0.5; got {self.p_ro}")

    @property
    def is_noiseless(self) -> bool:
        return self.p1q == 0.0 and self.p2q == 0.0 and self.p_ro == 0.0

    @property
    def has_gate_noise(self) -> bool:
        """True iff the PTM matrices themselves differ from the noiseless ones.

        Readout and basis-change noise do NOT count here — they are folded into the
        Hamiltonian's Pauli vector at compile time and never touch the forward pass.
        """
        return self.p1q > 0.0 or self.p2q > 0.0

    def to_tag(self) -> str:
        """Stable string for run names / provenance."""
        if self.is_noiseless:
            return "noiseless"
        bc = "" if self.model_basis_change else "_nobc"
        return f"p1q{self.p1q:.3e}_p2q{self.p2q:.3e}_pro{self.p_ro:.3e}{bc}"


# ---------------------------------------------------------------------------
# Device-calibrated tiers — the end-to-end experiment's noise levels
# ---------------------------------------------------------------------------
# Derived from real IBM calibration CSVs (2026-05-20) by the sibling NAHyRLQAS project
# (`nahyrlqas/noise/strengths.py`, device-wide medians over all operational qubits and
# all CZ pairs). Using measured device numbers rather than round figures means the noise
# level is citable rather than chosen.
#
# ⚠ SUBSET, THEREFORE A LOWER BOUND. Those anchors also carry amp_damp and dephasing
# (T1/T2, applied per moment per qubit); this project's spec has no time-like noise, so
# only the gate + readout part is taken. A real device is WORSE than these numbers. Say
# so in the paper.
#
# Measured noise penalty on LiH-4q (mHa above the noiseless energy of the same circuit),
# 2026-08-05, at 12 / 24 / 36 / 44 gates:
#     BOSTON   8.98  14.46  18.47  15.59
#     FEZ     24.29  38.20  48.10  40.61
#     MIAMI   36.02  65.77  90.16  70.66
# Chemical accuracy is 1.6 mHa, so NO tier reaches it in the noisy metric — the
# noiseless re-evaluation is the primary metric at every level. That is a property of
# the physics, not of the tier choice.
BOSTON = NoiseSpec4q(p1q=1.5e-4, p2q=1.1e-3, p_ro=5.4e-3)   # Heron r3 — best current
FEZ    = NoiseSpec4q(p1q=3.3e-4, p2q=2.9e-3, p_ro=1.5e-2)   # Heron r2 — typical current
MIAMI  = NoiseSpec4q(p1q=7.0e-4, p2q=7.8e-3, p_ro=2.0e-2)   # Nighthawk r1 — harshest

DEVICE_TIERS = {"boston": BOSTON, "fez": FEZ, "miami": MIAMI}

# For reference only — CRLQAS (arXiv:2402.03500 §H) uses p1q=1e-3, p2q=5e-3 generically
# and IBM Mumbai (2023-08-08) medians p1q=2.44e-4 / p2q=8.25e-3 / p_ro=2.25e-2 for its
# device runs. Measured above, its generic setting sits at the MIAMI end. Kept so the
# comparison to the baseline paper's noise level is explicit.
CRLQAS_GENERIC = NoiseSpec4q(p1q=1.0e-3, p2q=5.0e-3, p_ro=2.25e-2)

# The two tiers the end-to-end experiment runs. 3 arms x 5 seeds x 2 tiers = 30 runs.
# Chosen 2026-08-05 to span the widest realistic range (~4-5x in noise penalty). MIAMI
# additionally lands almost exactly on CRLQAS's generic setting (see CRLQAS_GENERIC
# below: 37.50/61.30/77.98/67.05 vs MIAMI 36.02/65.77/90.16/70.66), so the high tier is
# directly comparable to the baseline paper's noise level without extra argument.
# These are two ends of the CURRENT-HARDWARE range, not two named devices — BOSTON is
# Heron r3 and MIAMI is Nighthawk r1, different processor lines. Describe them that way.
LOW, HIGH = BOSTON, MIAMI
TIER_NAMES = {"low": "boston", "high": "miami"}
COMPOSITE = MIAMI        # single-tier default, kept for back-compat

# The three offline sweeps. Each varies one channel with the others held at zero.
SWEEP_1Q = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p1q=1e-4),
    "medium": NoiseSpec4q(p1q=5e-4),
    "strong": NoiseSpec4q(p1q=1e-3),
}
SWEEP_2Q = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p2q=1e-3),
    "medium": NoiseSpec4q(p2q=5e-3),
    "strong": NoiseSpec4q(p2q=1e-2),
}
SWEEP_RO = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p_ro=5e-3),
    "medium": NoiseSpec4q(p_ro=1e-2),
    "strong": NoiseSpec4q(p_ro=2e-2),
}

__all__ = ["NoiseSpec4q", "COMPOSITE", "LOW", "HIGH", "BOSTON", "FEZ", "MIAMI",
           "DEVICE_TIERS", "TIER_NAMES", "CRLQAS_GENERIC", "SWEEP_1Q", "SWEEP_2Q", "SWEEP_RO"]
