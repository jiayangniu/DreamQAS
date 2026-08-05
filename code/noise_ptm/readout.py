"""Readout error + measurement basis-change noise, as a rescale of the Hamiltonian.

THE POINT
---------
Both effects act *only at measurement*, so in the Pauli-Liouville picture neither one
touches the state ρ. They multiply each Pauli term of the Hamiltonian by a constant.
That means we can fold them into ``h_pauli`` **once, at compile time**, and the hot
COBYLA loop pays nothing at all.

THE PHYSICS
-----------
A hardware experiment measures ``H = Σ_α h_α σ_α`` term by term. For one Pauli string
``σ_α``:

1. Apply a basis-change unitary on each non-Z support — ``H`` for an X support,
   ``S†H`` for a Y support — to rotate it into the Z basis. Each of those is a
   single-qubit gate and therefore picks up 1q depolarizing, which in the Heisenberg
   picture multiplies that qubit's Pauli by ``(1 - 4·p1q/3)``.
2. Measure the resulting Z-string. Each measured bit is flipped with probability
   ``p_ro``, so each qubit in the support contributes a factor ``(1 - 2·p_ro)``.

Hence

    h_α  ->  h_α · (1 - 2·p_ro)^{weight(α)} · (1 - 4·p1q/3)^{#(X or Y supports in α)}

Identity terms (weight 0) are untouched, as they must be — the identity coefficient is
a constant offset, and no measurement is performed for it.

VALIDATED (2026-08-05) against an explicit simulation that literally applies the
basis-change gates, depolarizes, takes the Z-basis diagonal and pushes it through the
2-qubit confusion matrix: agreement to 8e-17 on a weight-2 ``XY`` string and on a
weight-1 ``ZI`` string.

WHY NOT NAHyRLQAS's ``kraus_readout``
-------------------------------------
That one applies a channel to ρ, which damps Z by ``(1-2p)`` but X/Y by ``(1-p)``, even
for symmetric ``p``. See the note in :mod:`noise_ptm.channels`.
"""

from __future__ import annotations

import numpy as np

from .ptm_utils import _index_to_pauli_digits
from .spec import NoiseSpec4q


def measurement_scale_vector(spec: NoiseSpec4q, n_qubits: int) -> np.ndarray:
    """Per-Pauli-index multiplicative factor, shape ``(4**n,)`` float64.

    All-ones when ``p_ro == 0`` and basis-change modelling is off or ``p1q == 0``.
    """
    n = int(n_qubits)
    n_basis = 4 ** n
    scale = np.ones(n_basis, dtype=np.float64)

    ro = float(spec.p_ro)
    p1q = float(spec.p1q)
    bc = bool(spec.model_basis_change) and p1q > 0.0
    if ro == 0.0 and not bc:
        return scale

    f_ro = 1.0 - 2.0 * ro          # per measured qubit
    f_bc = 1.0 - 4.0 * p1q / 3.0   # per basis-change gate (non-Z support)

    for idx in range(n_basis):
        digits = _index_to_pauli_digits(idx, n)   # digits[q] in {0=I,1=X,2=Y,3=Z}
        weight = 0
        n_xy = 0
        for dgt in digits:
            if dgt != 0:
                weight += 1
                if dgt in (1, 2):
                    n_xy += 1
        f = 1.0
        if ro != 0.0 and weight:
            f *= f_ro ** weight
        if bc and n_xy:
            f *= f_bc ** n_xy
        scale[idx] = f
    return scale


def apply_measurement_noise(
    h_pauli: np.ndarray, spec: NoiseSpec4q, n_qubits: int
) -> np.ndarray:
    """Return ``h_pauli`` rescaled for readout + basis-change noise.

    Pure function; does not mutate the input.
    """
    h = np.asarray(h_pauli, dtype=np.float64)
    n_basis = 4 ** int(n_qubits)
    if h.shape != (n_basis,):
        raise ValueError(
            f"apply_measurement_noise: h_pauli must have shape ({n_basis},); got {h.shape}"
        )
    return h * measurement_scale_vector(spec, n_qubits)


__all__ = ["measurement_scale_vector", "apply_measurement_noise"]
