"""Kraus operators for the two gate-noise channels DreamQAS's noise spec uses.

VENDORED (trimmed) from NAHyRLQAS ``nahyrlqas/noise/channels.py`` @ commit bd763a4
(2026-06-29). The two functions below are byte-identical to upstream.

WHAT WAS DELIBERATELY *NOT* VENDORED, and why
---------------------------------------------
``kraus_amplitude_damping`` / ``kraus_dephasing``
    T1/T2 relaxation. The DreamQAS noise spec has no time-like noise, so there are no
    per-moment channels at all. Dropping them means the PTM forward pass never emits a
    TIME slot, which roughly halves the scan length (upstream emits one TIME slot per
    circuit moment).

``kraus_readout``
    Upstream models readout as a **quantum channel applied to ρ** before ``Tr(Hρ)``.
    Measured on 2026-08-05, that channel damps ``Z`` by ``(1-2p)`` but ``X``/``Y`` by
    only ``(1-p)`` — and this holds for *symmetric* ``p01 = p10 = p`` too. (Upstream's
    own ``PROJECT_NOTES.md §5.10.1`` asserts the two factors coincide when readout is
    symmetric; that assertion is wrong — ``sqrt((1-p)(1-p)) = 1-p``, not ``1-2p``.)

    DreamQAS's spec defines readout as a **measurement assignment matrix**,
    ``P(1̃|0) = P(0̃|1) = p_ro``, which scales *every* Pauli string of weight k by
    ``(1-2p)^k`` regardless of whether its supports are X, Y or Z. That is what real
    hardware Pauli tomography does, and at ``p_ro = 1%`` it differs from the channel
    form by 1.02% on every X/Y term.

    So readout lives in :mod:`noise_ptm.readout` instead, as a compile-time diagonal
    rescale of the Hamiltonian's Pauli vector — both more faithful to the spec and
    free at run time.
"""

from __future__ import annotations

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kraus_1q_depolarizing(p: float) -> list[np.ndarray]:
    """Symmetric single-qubit depolarizing: ρ → (1-p)ρ + p/3 (XρX+YρY+ZρZ).

    In the Heisenberg picture this multiplies every non-identity single-qubit Pauli by
    ``(1 - 4p/3)`` — the factor :mod:`noise_ptm.readout` reuses for basis-change gates.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"depolarizing p out of [0,1]: {p}")
    k0 = np.sqrt(1.0 - p) * _I2
    s = np.sqrt(p / 3.0)
    return [k0, s * _X, s * _Y, s * _Z]


def kraus_2q_depolarizing(p: float) -> list[np.ndarray]:
    """Symmetric 2q depolarizing on the 4-dim subspace.

    ρ → (1-p) ρ + (p/15) Σ_{P ≠ I⊗I} P ρ P
    where the sum runs over the 15 non-identity 2-qubit Pauli operators.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"2q_depol p out of [0,1]: {p}")
    paulis = [_I2, _X, _Y, _Z]
    kraus: list[np.ndarray] = []
    # K0 = √(1-p) (I⊗I)
    kraus.append(np.sqrt(1.0 - p) * np.kron(_I2, _I2))
    # 15 non-identity Pauli products, each weighted √(p/15)
    coef = np.sqrt(p / 15.0)
    for i, A in enumerate(paulis):
        for j, B in enumerate(paulis):
            if i == 0 and j == 0:
                continue
            kraus.append(coef * np.kron(A, B))
    return kraus


__all__ = ["kraus_1q_depolarizing", "kraus_2q_depolarizing"]
