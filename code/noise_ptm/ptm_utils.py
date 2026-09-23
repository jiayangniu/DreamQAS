from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI_1Q: tuple[np.ndarray, ...] = (_I, _X, _Y, _Z)
_PAULI_LABELS: tuple[str, ...] = ("I", "X", "Y", "Z")


def _index_to_pauli_digits(index: int, n_qubits: int) -> tuple[int, ...]:
    digits = []
    x = int(index)
    for _ in range(int(n_qubits)):
        digits.append(x % 4)
        x //= 4
    return tuple(digits)


def pauli_label(index: int, n_qubits: int) -> str:
    digits = _index_to_pauli_digits(index, n_qubits)
    return "".join(_PAULI_LABELS[d] for d in digits)


@lru_cache(maxsize=8)
def pauli_basis(n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    n_basis = 4 ** n
    d = 2 ** n
    out = np.empty((n_basis, d, d), dtype=np.complex128)
    for idx in range(n_basis):
        digits = _index_to_pauli_digits(idx, n)
        P = _PAULI_1Q[digits[n - 1]]
        for q in range(n - 2, -1, -1):
            P = np.kron(P, _PAULI_1Q[digits[q]])
        out[idx] = P
    return out


def kraus_to_ptm(kraus_ops: Iterable[np.ndarray], n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    d = 2 ** n
    basis = pauli_basis(n)
    K_list = [np.asarray(K, dtype=np.complex128) for K in kraus_ops]
    if not K_list:
        raise ValueError("kraus_to_ptm: need at least one Kraus operator")
    if any(K.shape != (d, d) for K in K_list):
        raise ValueError(
            f"kraus_to_ptm: all Kraus ops must be ({d}, {d}); "
            f"got shapes {[K.shape for K in K_list]}"
        )

    transformed = np.zeros((basis.shape[0], d, d), dtype=np.complex128)
    for K in K_list:
        Kd = K.conj().T
        transformed += K @ basis @ Kd

    ptm_complex = np.einsum("aij,bji->ab", basis, transformed) / d

    max_imag = float(np.max(np.abs(ptm_complex.imag)))
    if max_imag > 1e-9:
        raise AssertionError(
            f"kraus_to_ptm: PTM has imaginary residual {max_imag:.3e}; "
            "channel may not be hermiticity-preserving."
        )
    return ptm_complex.real.astype(np.float64)


def unitary_to_ptm(U: np.ndarray, n_qubits: int) -> np.ndarray:
    return kraus_to_ptm([U], n_qubits)


def state_to_pauli_vector(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    d = 2 ** n
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (d, d):
        raise ValueError(
            f"state_to_pauli_vector: rho must be ({d}, {d}); got {rho.shape}"
        )
    basis = pauli_basis(n)
    r = np.einsum("aij,ji->a", basis, rho)
    if np.max(np.abs(r.imag)) > 1e-9:
        raise AssertionError(
            "state_to_pauli_vector: ρ has imaginary trace residuals; "
            "ρ is not Hermitian."
        )
    return r.real.astype(np.float64)


def pauli_vector_to_state(r: np.ndarray, n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    d = 2 ** n
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (4 ** n,):
        raise ValueError(
            f"pauli_vector_to_state: r must be length {4**n}; got {r.shape}"
        )
    basis = pauli_basis(n)
    rho = np.einsum("a,aij->ij", r, basis) / d
    return rho


def hamiltonian_to_pauli_vector(hamiltonian: np.ndarray, n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    d = 2 ** n
    H = np.asarray(hamiltonian, dtype=np.complex128)
    if H.shape != (d, d):
        raise ValueError(
            f"hamiltonian_to_pauli_vector: H must be ({d}, {d}); got {H.shape}"
        )
    basis = pauli_basis(n)
    h = np.einsum("aij,ji->a", basis, H)
    if np.max(np.abs(h.imag)) > 1e-8 * max(1.0, np.max(np.abs(h.real))):
        raise AssertionError(
            f"hamiltonian_to_pauli_vector: H has imaginary trace residuals "
            f"(max |imag| = {np.max(np.abs(h.imag)):.3e}); H is not Hermitian."
        )
    return h.real.astype(np.float64)


def expectation_via_ptm(h_pauli: np.ndarray, rho_pauli: np.ndarray, n_qubits: int) -> float:
    d = 2 ** int(n_qubits)
    if h_pauli.shape != rho_pauli.shape:
        raise ValueError(
            f"expectation_via_ptm: h and r shapes differ: "
            f"{h_pauli.shape} vs {rho_pauli.shape}"
        )
    return float(np.dot(h_pauli, rho_pauli) / d)


def zero_state_pauli_vector(n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    r = np.zeros(4 ** n, dtype=np.float64)
    for idx in range(4 ** n):
        digits = _index_to_pauli_digits(idx, n)
        if all(d in (0, 3) for d in digits):
            r[idx] = 1.0
    return r


def embed_1q_op(K_1q: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    n = int(n_qubits)
    factors = [np.eye(2, dtype=np.complex128)] * n
    factors[int(qubit)] = np.asarray(K_1q, dtype=np.complex128)
    result = factors[n - 1]
    for q in range(n - 2, -1, -1):
        result = np.kron(result, factors[q])
    return result


def embed_2q_op(K_4x4: np.ndarray, q_a: int, q_b: int, n_qubits: int) -> np.ndarray:
    K_4x4 = np.asarray(K_4x4, dtype=np.complex128)
    n = int(n_qubits)
    q_a = int(q_a)
    q_b = int(q_b)
    d = 2 ** n
    out = np.zeros((d, d), dtype=np.complex128)
    other_qubits = [q for q in range(n) if q not in (q_a, q_b)]
    for i_ab in range(4):
        bit_a_i = (i_ab >> 1) & 1
        bit_b_i = i_ab & 1
        for j_ab in range(4):
            bit_a_j = (j_ab >> 1) & 1
            bit_b_j = j_ab & 1
            v = K_4x4[i_ab, j_ab]
            if v == 0:
                continue
            for other_bits in range(2 ** len(other_qubits)):
                full_i = 0
                full_j = 0
                ob = other_bits
                for q in range(n):
                    if q == q_a:
                        bi, bj = bit_a_i, bit_a_j
                    elif q == q_b:
                        bi, bj = bit_b_i, bit_b_j
                    else:
                        b = ob & 1
                        ob >>= 1
                        bi = bj = b
                    pos = q
                    full_i |= bi << pos
                    full_j |= bj << pos
                out[full_i, full_j] = v
    return out


_AXIS_PAULI = (_X, _Y, _Z)


def rotation_3blocks(axis_idx: int, qubit: int, n_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis_idx not in (0, 1, 2):
        raise ValueError(f"axis_idx must be 0(X)/1(Y)/2(Z); got {axis_idx}")
    n = int(n_qubits)
    d = 2 ** n
    A_1q = _AXIS_PAULI[axis_idx]
    A = embed_1q_op(A_1q, qubit, n)

    M_I = np.eye(4 ** n, dtype=np.float64)
    M_P = unitary_to_ptm(A, n)

    basis = pauli_basis(n)
    G = np.einsum("bij,jk->bik", basis, A) - np.einsum("ij,bjk->bik", A, basis)
    X = np.einsum("aij,bji->ab", basis, G) / d
    if np.max(np.abs(X.real)) > 1e-8:
        raise AssertionError(
            f"rotation_3blocks: X has real residual {np.max(np.abs(X.real)):.3e}; "
            "expected purely imaginary."
        )
    M_PC = (-X.imag).astype(np.float64)

    return M_I, M_PC, M_P


def rotation_ptm_from_blocks(
    M_I: np.ndarray, M_PC: np.ndarray, M_P: np.ndarray, theta: float
) -> np.ndarray:
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    return c * c * M_I - c * s * M_PC + s * s * M_P


def cnot_ptm(control: int, target: int, n_qubits: int) -> np.ndarray:
    cnot_4x4 = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    U_full = embed_2q_op(cnot_4x4, int(control), int(target), int(n_qubits))
    return unitary_to_ptm(U_full, int(n_qubits))


__all__ = [
    "pauli_basis",
    "pauli_label",
    "kraus_to_ptm",
    "unitary_to_ptm",
    "state_to_pauli_vector",
    "pauli_vector_to_state",
    "hamiltonian_to_pauli_vector",
    "expectation_via_ptm",
    "zero_state_pauli_vector",
    "embed_1q_op",
    "embed_2q_op",
    "rotation_3blocks",
    "rotation_ptm_from_blocks",
    "cnot_ptm",
]
