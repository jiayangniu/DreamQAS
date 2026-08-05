"""Pauli Transfer Matrix utilities — numpy reference math.

VENDORED VERBATIM from NAHyRLQAS ``nahyrlqas/backends/ptm_utils.py`` @ commit bd763a4
(2026-06-29). Copied byte-for-byte apart from this header: the math is validated there
against qulacs DensityMatrix, and re-typing it would risk silent transcription errors.
Do not edit — if upstream changes, re-copy and re-run ``tests/test_equivalence.py``.


PTM (a.k.a. Pauli-Liouville) representation: a state ρ is written as a
real vector ``r`` in the n-qubit Pauli basis, gates and noise channels
are 4^n × 4^n real matrices, and expectation values are dot products.

Conventions
-----------
* **Qubit endianness**: qubit 0 is the **least-significant** bit (LSB) of
  the basis-state index, matching the qulacs / qiskit convention. In a
  tensor product written with ``np.kron``, qubit 0 is therefore the
  **rightmost** factor (``np.kron(σ_{n-1}, σ_{n-2}, ..., σ_1, σ_0)``).
* **Pauli basis index**: ``α = α_0 + 4·α_1 + 16·α_2 + ... + 4^(n-1)·α_{n-1}``
  with ``α_q ∈ {0=I, 1=X, 2=Y, 3=Z}`` the Pauli on qubit ``q``.
* **Normalization**: ``Tr[σ_α σ_β] = d δ_αβ`` with ``d = 2^n``.
* **PTM of channel E**: ``Λ_αβ = (1/d) Tr[σ_α E(σ_β)]``.
* **State as Pauli vector**: ``r_α = Tr[σ_α ρ]``.
  Recovery: ``ρ = (1/d) Σ_α r_α σ_α``.
* **Expectation value**: ``⟨H⟩ = Tr[H ρ] = (1/d) Σ_α h_α r_α``,
  where ``h_α = Tr[σ_α H]`` is the Hamiltonian in Pauli vector form.

This module is **numpy only** so we can validate the math line-by-line
against qulacs DensityMatrix outputs before porting to JAX (Sprint 1
Track B).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

# Single-qubit Pauli matrices in the {I, X, Y, Z} ordering.
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI_1Q: tuple[np.ndarray, ...] = (_I, _X, _Y, _Z)
_PAULI_LABELS: tuple[str, ...] = ("I", "X", "Y", "Z")


def _index_to_pauli_digits(index: int, n_qubits: int) -> tuple[int, ...]:
    """Decompose a Pauli-basis index into per-qubit Pauli digits."""
    digits = []
    x = int(index)
    for _ in range(int(n_qubits)):
        digits.append(x % 4)
        x //= 4
    return tuple(digits)


def pauli_label(index: int, n_qubits: int) -> str:
    """Human-readable Pauli label, e.g. index=5 (n=4) -> 'IXIY' ... etc.

    qubit 0 is the leftmost character (matches our tensor convention).
    """
    digits = _index_to_pauli_digits(index, n_qubits)
    return "".join(_PAULI_LABELS[d] for d in digits)


@lru_cache(maxsize=8)
def pauli_basis(n_qubits: int) -> np.ndarray:
    """Return all 4^n Pauli strings stacked along axis 0.

    Output shape: ``(4**n, 2**n, 2**n)`` complex128.

    Cached up to ``n_qubits = 7`` (4^7 = 16384 PTM basis matrices, each
    128×128 complex = 4 GB total — borderline; cap to keep the LRU sane).
    """
    n = int(n_qubits)
    n_basis = 4 ** n
    d = 2 ** n
    out = np.empty((n_basis, d, d), dtype=np.complex128)
    for idx in range(n_basis):
        digits = _index_to_pauli_digits(idx, n)
        # LSB convention: qubit n-1 is the leftmost factor in kron, qubit 0
        # is the rightmost. So kron from digits[n-1] down to digits[0].
        P = _PAULI_1Q[digits[n - 1]]
        for q in range(n - 2, -1, -1):
            P = np.kron(P, _PAULI_1Q[digits[q]])
        out[idx] = P
    return out


def kraus_to_ptm(kraus_ops: Iterable[np.ndarray], n_qubits: int) -> np.ndarray:
    """Build the PTM of a Kraus channel ``E(ρ) = Σ_k K_k ρ K_k†``.

    Output shape: ``(4**n, 4**n)``, dtype float64.

    For a CPTP map (Σ_k K_k† K_k = I), the PTM is real; we drop tiny
    imaginary residuals from numerical roundoff at the end.
    """
    n = int(n_qubits)
    d = 2 ** n
    basis = pauli_basis(n)  # (4^n, d, d)
    K_list = [np.asarray(K, dtype=np.complex128) for K in kraus_ops]
    if not K_list:
        raise ValueError("kraus_to_ptm: need at least one Kraus operator")
    if any(K.shape != (d, d) for K in K_list):
        raise ValueError(
            f"kraus_to_ptm: all Kraus ops must be ({d}, {d}); "
            f"got shapes {[K.shape for K in K_list]}"
        )

    # transformed[β] = Σ_k K_k · σ_β · K_k†
    # Shape: (4^n, d, d). Loop over Kraus (usually 1–16), vectorize over β.
    transformed = np.zeros((basis.shape[0], d, d), dtype=np.complex128)
    for K in K_list:
        Kd = K.conj().T
        # K @ basis[β] @ Kd for each β
        transformed += K @ basis @ Kd

    # Λ_αβ = (1/d) Tr[σ_α · transformed[β]]
    #      = (1/d) Σ_ij σ_α[i,j] · transformed[β][j,i]
    #      = (1/d) einsum('aij,bji->ab', basis, transformed)
    ptm_complex = np.einsum("aij,bji->ab", basis, transformed) / d

    max_imag = float(np.max(np.abs(ptm_complex.imag)))
    if max_imag > 1e-9:
        raise AssertionError(
            f"kraus_to_ptm: PTM has imaginary residual {max_imag:.3e}; "
            "channel may not be hermiticity-preserving."
        )
    return ptm_complex.real.astype(np.float64)


def unitary_to_ptm(U: np.ndarray, n_qubits: int) -> np.ndarray:
    """PTM of a unitary channel ``E(ρ) = U ρ U†``. Single-Kraus shortcut."""
    return kraus_to_ptm([U], n_qubits)


def state_to_pauli_vector(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a density matrix to its Pauli-basis vector ``r_α = Tr[σ_α ρ]``.

    Output: real 1-D array, length ``4^n``.
    """
    n = int(n_qubits)
    d = 2 ** n
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (d, d):
        raise ValueError(
            f"state_to_pauli_vector: rho must be ({d}, {d}); got {rho.shape}"
        )
    basis = pauli_basis(n)
    # r_α = Tr[σ_α ρ] = Σ_ij σ_α[i,j] · ρ[j,i]
    r = np.einsum("aij,ji->a", basis, rho)
    if np.max(np.abs(r.imag)) > 1e-9:
        raise AssertionError(
            "state_to_pauli_vector: ρ has imaginary trace residuals; "
            "ρ is not Hermitian."
        )
    return r.real.astype(np.float64)


def pauli_vector_to_state(r: np.ndarray, n_qubits: int) -> np.ndarray:
    """Inverse of :func:`state_to_pauli_vector`.

    ``ρ = (1/d) Σ_α r_α σ_α``. Output: complex128 (d, d) density matrix.
    """
    n = int(n_qubits)
    d = 2 ** n
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (4 ** n,):
        raise ValueError(
            f"pauli_vector_to_state: r must be length {4**n}; got {r.shape}"
        )
    basis = pauli_basis(n)
    # ρ = (1/d) Σ_α r_α σ_α
    rho = np.einsum("a,aij->ij", r, basis) / d
    return rho


def hamiltonian_to_pauli_vector(hamiltonian: np.ndarray, n_qubits: int) -> np.ndarray:
    """Decompose a dense Hamiltonian into its Pauli vector ``h_α = Tr[σ_α H]``.

    Output: real 1-D array, length ``4^n``. The Hermitian Hamiltonian
    guarantees real entries.
    """
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
    """``⟨H⟩ = Tr[H ρ] = (1/d) h · r``.

    The factor ``1/d`` matches our normalization
    ``Tr[σ_α σ_β] = d δ_αβ`` (see module docstring).
    """
    d = 2 ** int(n_qubits)
    if h_pauli.shape != rho_pauli.shape:
        raise ValueError(
            f"expectation_via_ptm: h and r shapes differ: "
            f"{h_pauli.shape} vs {rho_pauli.shape}"
        )
    return float(np.dot(h_pauli, rho_pauli) / d)


def zero_state_pauli_vector(n_qubits: int) -> np.ndarray:
    """Pauli vector of |0...0⟩⟨0...0|.

    Closed form: ``r_α = 1`` iff α corresponds to a Pauli string made of
    only I and Z on each qubit; else 0. (Because |0⟩⟨0| = (I+Z)/2 per qubit
    and Tr[σ ρ] picks out matching diagonal Paulis.)
    """
    n = int(n_qubits)
    r = np.zeros(4 ** n, dtype=np.float64)
    for idx in range(4 ** n):
        digits = _index_to_pauli_digits(idx, n)
        # Pauli string of only I (0) and Z (3) → r_α = 1
        if all(d in (0, 3) for d in digits):
            r[idx] = 1.0
    return r


def embed_1q_op(K_1q: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a 2×2 operator on ``qubit`` into the n-qubit (2^n, 2^n) operator.

    LSB convention: ``qubit`` 0 is the rightmost factor in the kron product
    (matches qulacs / qiskit basis-index endianness — qubit 0 is bit 0 of
    the basis-state index).
    """
    n = int(n_qubits)
    factors = [np.eye(2, dtype=np.complex128)] * n
    factors[int(qubit)] = np.asarray(K_1q, dtype=np.complex128)
    # LSB convention: kron from qubit n-1 down to qubit 0
    result = factors[n - 1]
    for q in range(n - 2, -1, -1):
        result = np.kron(result, factors[q])
    return result


def embed_2q_op(K_4x4: np.ndarray, q_a: int, q_b: int, n_qubits: int) -> np.ndarray:
    """Embed a 4×4 operator on ``(q_a, q_b)`` into the n-qubit (2^n, 2^n) operator.

    ``q_a`` is the **upper** factor of the 4×4 (i.e. row/col index of K is
    ``2·bit(q_a) + bit(q_b)``). Qubit ``q`` is at bit position ``q`` of the
    n-qubit basis-state index (LSB convention).
    """
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
                    pos = q  # qubit q = bit q (LSB convention)
                    full_i |= bi << pos
                    full_j |= bj << pos
                out[full_i, full_j] = v
    return out


# Axis index → 2x2 Pauli (used for rotation generator). 0=X, 1=Y, 2=Z.
_AXIS_PAULI = (_X, _Y, _Z)


def rotation_3blocks(axis_idx: int, qubit: int, n_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (M_I, M_PC, M_P) building blocks of a Pauli rotation gate.

    **Qulacs convention** (``add_RX_gate(q, θ)`` etc.): ``R(θ) = exp(+i θ A / 2)``
    where ``A`` is the rotation-axis Pauli. This is the *opposite-sign*
    convention from the physics ``exp(-iθA/2)``; qulacs's RX(π/2) sends
    ``Y → -Z`` (not ``+Z``).

    Reconstruction (column-vector convention, ``r' = M_R(θ) @ r``):

        M_R(θ) = cos²(θ/2) · M_I − cos(θ/2)·sin(θ/2) · M_PC + sin²(θ/2) · M_P

    Note the **minus sign** in front of the ``cs M_PC`` term — this is
    what flips the angle convention from physics to qulacs.

    where:

    * ``M_I`` is the identity PTM (no-op channel).
    * ``M_P`` is the PTM of conjugation by the generator Pauli
      ``A`` (a Pauli string with the rotation axis on ``qubit``, I elsewhere):
      ``M_P[α,β] = (1/d) Tr[σ_α · A · σ_β · A]``.
    * ``M_PC = -imag(X)`` where ``X[α,β] = (1/d) Tr[σ_α (σ_β A - A σ_β)]``.
      ``X`` is purely imaginary because the commutator is anti-hermitian;
      we store the real-valued ``M_PC`` so the reconstruction is all-real.

    Cross-check: ``M_R(0) = M_I`` (cos=1, sin=0), and qulacs's
    ``add_RX_gate(q, π/2)`` sends ``Y → -Z`` so ``M_R(π/2)[Z, Y] = -1``
    (validated by ``validate_ptm_pipeline_4q.py``).
    """
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
    """Reconstruct ``M_R(θ)`` from its 3 building blocks (numpy only).

    Qulacs convention: ``R(θ) = exp(+i θ A / 2)``, so the cs term has a
    **minus sign** (see :func:`rotation_3blocks` docstring).
    """
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    return c * c * M_I - c * s * M_PC + s * s * M_P


def cnot_ptm(control: int, target: int, n_qubits: int) -> np.ndarray:
    """PTM of CNOT(control, target) on n qubits.

    Matches qulacs ``add_CNOT_gate(control, target)`` semantics.
    """
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
