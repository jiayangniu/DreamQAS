from __future__ import annotations

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kraus_1q_depolarizing(p: float) -> list[np.ndarray]:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"depolarizing p out of [0,1]: {p}")
    k0 = np.sqrt(1.0 - p) * _I2
    s = np.sqrt(p / 3.0)
    return [k0, s * _X, s * _Y, s * _Z]


def kraus_2q_depolarizing(p: float) -> list[np.ndarray]:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"2q_depol p out of [0,1]: {p}")
    paulis = [_I2, _X, _Y, _Z]
    kraus: list[np.ndarray] = []
    kraus.append(np.sqrt(1.0 - p) * np.kron(_I2, _I2))
    coef = np.sqrt(p / 15.0)
    for i, A in enumerate(paulis):
        for j, B in enumerate(paulis):
            if i == 0 and j == 0:
                continue
            kraus.append(coef * np.kron(A, B))
    return kraus


__all__ = ["kraus_1q_depolarizing", "kraus_2q_depolarizing"]
