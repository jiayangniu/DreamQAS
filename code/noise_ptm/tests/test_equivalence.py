"""Validation of the PTM noisy backend against DreamQAS's own qulacs engines.

Four independent checks:

1. ``test_noiseless_matches_qulacs_statevector``
   With all noise at zero the PTM path must reproduce DreamQAS's existing noiseless
   qulacs statevector. This is the guard against the **TF32 trap** (GPU matmuls default
   to ~10-bit mantissa, which would show up here as ~1e-3 Ha) and against any
   state-tensor parsing / parameter-ordering mismatch.

2. ``test_gate_noise_matches_qulacs_density_matrix``
   1q+2q depolarizing vs DreamQAS's *existing* qulacs DensityMatrix path
   (``VQE.get_exp_val(..., phys_noise=True)``). That path is far too slow for training
   but is a completely independent implementation, which is exactly what we want here.

3. ``test_measurement_noise_matches_explicit_simulation``
   Readout + basis-change depolarizing vs a literal hardware-style simulation: apply the
   basis-change unitaries, depolarize, take the Z-basis diagonal, push it through the
   per-bit confusion matrix. This is the check on the *new* physics in ``readout.py``
   (the part that deliberately differs from NAHyRLQAS).

4. ``test_measurement_scale_vector_invariants``
   Cheap structural properties of the scale vector.

Run standalone:  python code/noise_ptm/tests/test_equivalence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, "..", ".."))          # .../DreamQAS/code
_ROOT = os.path.abspath(os.path.join(_CODE, ".."))                # .../DreamQAS
for _p in (_CODE, os.path.join(_CODE, "WM_QAS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# JAX on CPU by default: these tests must not fight the training fleet for GPU memory,
# and CPU is where float32 vs TF32 differences cannot hide.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch  # noqa: E402

from noise_ptm import NoiseSpec4q, NoisyEvaluator          # noqa: E402
from noise_ptm.readout import measurement_scale_vector     # noqa: E402

N_QUBITS = 4
MAX_LAYERS = 64
HAM_PATH = os.path.join(
    _ROOT, "data", "mol_data",
    "LiH_4q_geom_Li_.0_.0_.0;_H_.0_.0_3.4_parity.npz",
)


def _load_lih4q():
    d = np.load(HAM_PATH, allow_pickle=True)
    return np.asarray(d["hamiltonian"]).astype(np.complex128), float(d["energy_shift"])


def _random_state_tensor(rng, n_gates, n=N_QUBITS, layers=MAX_LAYERS, cnot_ratio=0.4):
    """Build a state tensor exactly the way environment.py:236-253 does.

    Rotation axes are stored 1-based (1=X, 2=Y, 3=Z) in one-hot row ``n + axis - 1``,
    with the angle in row ``n + 3 + axis - 1``.
    """
    st = torch.zeros((layers, n + 6, n), dtype=torch.float32)
    moments = [0] * n
    for _ in range(n_gates):
        if rng.random() < cnot_ratio:
            c = int(rng.integers(n))
            t = (c + 1 + int(rng.integers(n - 1))) % n
            lay = max(moments[c], moments[t])
            if lay >= layers:
                break
            st[lay][t][c] = 1.0
            moments[c] = moments[t] = lay + 1
        else:
            q = int(rng.integers(n))
            ax = int(rng.integers(1, 4))
            lay = moments[q]
            if lay >= layers:
                break
            st[lay][n + ax - 1][q] = 1.0
            st[lay][n + 3 + ax - 1][q] = float(rng.uniform(-np.pi, np.pi))
            moments[q] = lay + 1
    return st


def _angles_from_state(st, n=N_QUBITS):
    """The parameter order DreamQAS's optimizer uses (environment.py:537-538)."""
    rot_pos = (st[:, n:n + 3] == 1).nonzero(as_tuple=True)
    return st[:, n + 3:][rot_pos].numpy().astype(np.float64)


def _qulacs_energy(st, ham, shift, noise_models=(), noise_values=()):
    import VQE as vc
    pc = vc.Parametric_Circuit(
        n_qubits=N_QUBITS,
        noise_models=list(noise_models),
        noise_values=list(noise_values),
    )
    circ = pc.construct_ansatz(st)
    phys = bool(noise_values)
    return vc.get_exp_val(N_QUBITS, circ, ham, phys_noise=phys) + shift


# ---------------------------------------------------------------------------
# 1. noiseless equivalence
# ---------------------------------------------------------------------------

def test_noiseless_matches_qulacs_statevector(n_circuits=200, tol=1e-5, verbose=False):
    ham, shift = _load_lih4q()
    ev = NoisyEvaluator(NoiseSpec4q(), ham, N_QUBITS, shift, num_layers=48)
    rng = np.random.default_rng(20260805)
    worst = 0.0
    for i in range(n_circuits):
        n_gates = int(rng.integers(1, 41))
        st = _random_state_tensor(rng, n_gates)
        thetas = _angles_from_state(st)
        e_ptm = ev.energy_of(st, thetas)
        e_qul = _qulacs_energy(st, ham, shift)
        worst = max(worst, abs(e_ptm - e_qul))
    if verbose:
        print(f"  noiseless: {n_circuits} circuits, worst |Δ| = {worst:.3e} Ha "
              f"({worst * 1000:.3e} mHa)")
    assert worst < tol, f"PTM vs qulacs statevector differ by {worst:.3e} Ha (> {tol})"
    return worst


# ---------------------------------------------------------------------------
# 2. gate noise vs qulacs density matrix
# ---------------------------------------------------------------------------

def test_gate_noise_matches_qulacs_density_matrix(n_circuits=40, tol=1e-5, verbose=False):
    """Gate noise ONLY.

    ``model_basis_change=False`` is essential here, not incidental. DreamQAS's qulacs
    reference computes a bare ``Tr(Hρ)``: it has no notion of measurement basis-change
    gates, so it cannot see their depolarizing. Leaving the (default-on) basis-change
    modelling enabled makes the PTM path legitimately disagree with it — by 8e-5 Ha at
    the composite point and 1.8e-3 Ha at p1q=1e-2, which looks exactly like a bug and
    is not one. The measurement-side physics is validated in test 3 instead, against a
    reference that *does* model it.
    """
    ham, shift = _load_lih4q()
    rng = np.random.default_rng(4242)
    overall = 0.0
    # The paper's composite point plus exaggerated ones — a bug invisible at p=5e-4
    # will not stay invisible at p=5e-2 — and each channel in isolation.
    for p1q, p2q in ((5e-4, 5e-3), (1e-2, 5e-2), (0.0, 5e-3), (1e-3, 0.0)):
        spec = NoiseSpec4q(p1q=p1q, p2q=p2q, p_ro=0.0, model_basis_change=False)
        ev = NoisyEvaluator(spec, ham, N_QUBITS, shift, num_layers=48)
        worst = 0.0
        for _ in range(n_circuits):
            st = _random_state_tensor(rng, int(rng.integers(1, 31)))
            thetas = _angles_from_state(st)
            e_ptm = ev.energy_of(st, thetas)
            e_qul = _qulacs_energy(
                st, ham, shift,
                noise_models=("depolarizing", "two_depolarizing"),
                noise_values=(p1q, p2q),
            )
            worst = max(worst, abs(e_ptm - e_qul))
        if verbose:
            print(f"  gate noise p1q={p1q:.0e} p2q={p2q:.0e}: worst |Δ| = {worst:.3e} Ha")
        assert worst < tol, (
            f"PTM vs qulacs DM differ by {worst:.3e} Ha (> {tol}) "
            f"at p1q={p1q}, p2q={p2q}"
        )
        overall = max(overall, worst)
    return overall


# ---------------------------------------------------------------------------
# 3. measurement noise vs an explicit hardware-style simulation
# ---------------------------------------------------------------------------

def _explicit_measured_pauli(rho, digits, p1q, p_ro, n=N_QUBITS):
    """⟨σ⟩ as a real experiment would obtain it.

    Basis-change unitary on each non-Z support (with 1q depolarizing after it), then a
    Z-basis measurement whose every bit flips with probability ``p_ro``.
    """
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1]).astype(complex)
    Hd = (X + Z) / np.sqrt(2)
    Sdg = np.diag([1, -1j])

    def embed(op, q):
        facs = [I2] * n
        facs[q] = op
        out = facs[n - 1]
        for k in range(n - 2, -1, -1):
            out = np.kron(out, facs[k])
        return out

    def depol(r, q, p):
        out = (1 - p) * r
        for P in (X, Y, Z):
            E = embed(P, q)
            out = out + (p / 3.0) * (E @ r @ E.conj().T)
        return out

    r = rho
    support = [q for q, d in enumerate(digits) if d != 0]
    for q, d in enumerate(digits):
        if d == 1:                       # X -> Z
            U = embed(Hd, q)
            r = U @ r @ U.conj().T
            r = depol(r, q, p1q)
        elif d == 2:                     # Y -> Z
            U = embed(Hd @ Sdg, q)
            r = U @ r @ U.conj().T
            r = depol(r, q, p1q)

    probs = np.real(np.diag(r))
    # per-bit confusion matrix on every measured qubit
    conf = np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    full = np.array([[1.0]])
    for q in range(n):                   # qubit 0 is the LSB
        full = np.kron(conf, full) if q else conf.copy()
    probs_meas = full.T @ probs
    idx = np.arange(2 ** n)
    parity = np.zeros(2 ** n)
    for q in support:
        parity += ((idx >> q) & 1)
    return float(np.sum(probs_meas * (-1.0) ** parity))


def test_measurement_noise_matches_explicit_simulation(tol=1e-9, verbose=False):
    from noise_ptm.ptm_utils import _index_to_pauli_digits, pauli_basis

    rng = np.random.default_rng(11)
    # a random pure state of 4 qubits
    A = rng.normal(size=(2 ** N_QUBITS,) * 2) + 1j * rng.normal(size=(2 ** N_QUBITS,) * 2)
    Q, _ = np.linalg.qr(A)
    psi = Q[:, 0]
    rho = np.outer(psi, psi.conj())

    basis = pauli_basis(N_QUBITS)
    worst = 0.0
    for p1q, p_ro in ((0.0, 1e-2), (5e-4, 1e-2), (1e-3, 2e-2)):
        spec = NoiseSpec4q(p1q=p1q, p_ro=p_ro)
        scale = measurement_scale_vector(spec, N_QUBITS)
        for idx in rng.choice(4 ** N_QUBITS, size=40, replace=False):
            idx = int(idx)
            digits = _index_to_pauli_digits(idx, N_QUBITS)
            exact = float(np.real(np.trace(basis[idx] @ rho)))
            predicted = exact * scale[idx]
            explicit = _explicit_measured_pauli(rho, digits, p1q, p_ro)
            worst = max(worst, abs(predicted - explicit))
        if verbose:
            print(f"  measurement p1q={p1q:.0e} p_ro={p_ro:.0e}: worst |Δ| = {worst:.3e}")
    assert worst < tol, f"per-Pauli scaling vs explicit simulation differ by {worst:.3e}"
    return worst


# ---------------------------------------------------------------------------
# 4. structural invariants
# ---------------------------------------------------------------------------

def test_measurement_scale_vector_invariants():
    n = N_QUBITS
    # noiseless -> all ones
    s = measurement_scale_vector(NoiseSpec4q(), n)
    assert np.allclose(s, 1.0)

    # identity term is never rescaled (no measurement is performed for it)
    s = measurement_scale_vector(NoiseSpec4q(p1q=1e-3, p_ro=2e-2), n)
    assert s[0] == 1.0

    # readout only: every weight-k string gets exactly (1-2p)^k, X/Y/Z alike
    p = 1e-2
    s = measurement_scale_vector(NoiseSpec4q(p_ro=p), n)
    from noise_ptm.ptm_utils import _index_to_pauli_digits
    for idx in (1, 2, 3, 5, 10, 15, 4 ** n - 1):
        w = sum(1 for d in _index_to_pauli_digits(idx, n) if d != 0)
        assert abs(s[idx] - (1 - 2 * p) ** w) < 1e-12, f"idx={idx}"

    # basis-change only affects X/Y supports
    s = measurement_scale_vector(NoiseSpec4q(p1q=1e-3), n)
    assert abs(s[3] - 1.0) < 1e-12          # 'Z' on qubit 0 -> no basis change
    assert abs(s[1] - (1 - 4e-3 / 3)) < 1e-12   # 'X' on qubit 0 -> one basis change

    # and it can be switched off
    s_off = measurement_scale_vector(
        NoiseSpec4q(p1q=1e-3, model_basis_change=False), n
    )
    assert np.allclose(s_off, 1.0)
    return True


if __name__ == "__main__":
    print("noise_ptm validation")
    print("=" * 70)
    w1 = test_noiseless_matches_qulacs_statevector(verbose=True)
    print(f"[1] noiseless vs qulacs statevector      OK   worst {w1:.3e} Ha")
    w2 = test_gate_noise_matches_qulacs_density_matrix(verbose=True)
    print(f"[2] gate noise vs qulacs density matrix  OK   worst {w2:.3e} Ha")
    w3 = test_measurement_noise_matches_explicit_simulation(verbose=True)
    print(f"[3] measurement noise vs explicit sim    OK   worst {w3:.3e}")
    test_measurement_scale_vector_invariants()
    print("[4] scale-vector invariants              OK")
    print("=" * 70)
    print("all checks passed")
