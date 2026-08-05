"""Compile a :class:`NoiseSpec4q` into numpy PTM tables. Host-side, no JAX.

ADAPTED from NAHyRLQAS ``nahyrlqas/backends/ptm_compile.py`` @ commit bd763a4.
Two deliberate differences from upstream:

1. **No ``time_moment_composite`` and no ``readout_composite``.** This project's spec
   has no T1/T2, and readout is handled as a Hamiltonian rescale (see
   :mod:`noise_ptm.readout`). Upstream needs both because it models IBM device noise.
2. Only the three gate-level tables survive, which is what the forward pass indexes.

Forward-pass contract (column-vector convention, ``r' = M @ r``):

* CNOT(c, t):            ``r <- cnot_with_noise[(c, t)] @ r``
* Rotation(axis, q, θ):  ``M_R = cos²(θ/2)·B0 - cos(θ/2)sin(θ/2)·B1 + sin²(θ/2)·B2``
                         ``r <- per_rotation_noise[q] @ M_R @ r``

The rotation blocks follow the **qulacs** angle convention ``R(θ) = exp(+iθA/2)``
(note the minus sign on the middle term) — see ``ptm_utils.rotation_3blocks``. This is
what makes the PTM path agree with DreamQAS's own qulacs circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import kraus_1q_depolarizing, kraus_2q_depolarizing
from .ptm_utils import cnot_ptm, embed_1q_op, embed_2q_op, kraus_to_ptm
from .ptm_utils import rotation_3blocks
from .spec import NoiseSpec4q


@dataclass
class CompiledPTM:
    """Pure-numpy compiled gate tables for one (NoiseSpec4q, n_qubits)."""

    n_qubits: int
    d4: int
    rotation_blocks: dict           # (axis_idx, qubit) -> (3, d4, d4) float64
    cnot_with_noise: dict           # (control, target) -> (d4, d4) float64
    per_rotation_noise: dict        # qubit -> (d4, d4) float64
    spec_tag: str


def compile_ptm(spec: NoiseSpec4q, n_qubits: int) -> CompiledPTM:
    """Build every rotation block and CNOT PTM once.

    Cost is dominated by the Kraus->PTM conversions. At 4q (d4 = 256) this is a couple
    of seconds and happens once per run, not per step.
    """
    n = int(n_qubits)
    d4 = 4 ** n
    identity = np.eye(d4, dtype=np.float64)

    # --- Rotation 3-blocks (gate only, noiseless; noise is a separate left factor) ---
    rotation_blocks: dict = {}
    for axis in (0, 1, 2):
        for q in range(n):
            m_i, m_pc, m_p = rotation_3blocks(axis, q, n)
            rotation_blocks[(axis, q)] = np.stack([m_i, m_pc, m_p]).astype(np.float64)

    # --- CNOT, with 2q depolarizing composed on the LEFT (i.e. acting after the gate) ---
    cnot_with_noise: dict = {}
    if spec.p2q > 0.0:
        k2 = kraus_2q_depolarizing(float(spec.p2q))
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            m_cnot = cnot_ptm(c, t, n)
            if spec.p2q > 0.0:
                embedded = [embed_2q_op(k, c, t, n) for k in k2]
                cnot_with_noise[(c, t)] = kraus_to_ptm(embedded, n) @ m_cnot
            else:
                cnot_with_noise[(c, t)] = m_cnot

    # --- 1q depolarizing after every rotation ---
    per_rotation_noise: dict = {}
    if spec.p1q > 0.0:
        k1 = kraus_1q_depolarizing(float(spec.p1q))
        for q in range(n):
            per_rotation_noise[q] = kraus_to_ptm(
                [embed_1q_op(k, q, n) for k in k1], n
            )
    else:
        for q in range(n):
            per_rotation_noise[q] = identity

    return CompiledPTM(
        n_qubits=n,
        d4=d4,
        rotation_blocks=rotation_blocks,
        cnot_with_noise=cnot_with_noise,
        per_rotation_noise=per_rotation_noise,
        spec_tag=spec.to_tag(),
    )


__all__ = ["CompiledPTM", "compile_ptm"]
