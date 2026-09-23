from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import kraus_1q_depolarizing, kraus_2q_depolarizing
from .ptm_utils import cnot_ptm, embed_1q_op, embed_2q_op, kraus_to_ptm
from .ptm_utils import rotation_3blocks
from .spec import NoiseSpec4q


@dataclass
class CompiledPTM:

    n_qubits: int
    d4: int
    rotation_blocks: dict
    cnot_with_noise: dict
    per_rotation_noise: dict
    spec_tag: str


def compile_ptm(spec: NoiseSpec4q, n_qubits: int) -> CompiledPTM:
    n = int(n_qubits)
    d4 = 4 ** n
    identity = np.eye(d4, dtype=np.float64)

    rotation_blocks: dict = {}
    for axis in (0, 1, 2):
        for q in range(n):
            m_i, m_pc, m_p = rotation_3blocks(axis, q, n)
            rotation_blocks[(axis, q)] = np.stack([m_i, m_pc, m_p]).astype(np.float64)

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
