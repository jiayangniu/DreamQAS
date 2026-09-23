from __future__ import annotations

import numpy as np

from .ptm_utils import _index_to_pauli_digits
from .spec import NoiseSpec4q


def measurement_scale_vector(spec: NoiseSpec4q, n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    n_basis = 4 ** n
    scale = np.ones(n_basis, dtype=np.float64)

    ro = float(spec.p_ro)
    p1q = float(spec.p1q)
    bc = bool(spec.model_basis_change) and p1q > 0.0
    if ro == 0.0 and not bc:
        return scale

    f_ro = 1.0 - 2.0 * ro
    f_bc = 1.0 - 4.0 * p1q / 3.0

    for idx in range(n_basis):
        digits = _index_to_pauli_digits(idx, n)
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
    h = np.asarray(h_pauli, dtype=np.float64)
    n_basis = 4 ** int(n_qubits)
    if h.shape != (n_basis,):
        raise ValueError(
            f"apply_measurement_noise: h_pauli must have shape ({n_basis},); got {h.shape}"
        )
    return h * measurement_scale_vector(spec, n_qubits)


__all__ = ["measurement_scale_vector", "apply_measurement_noise"]
