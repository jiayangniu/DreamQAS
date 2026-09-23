from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compile import CompiledPTM, compile_ptm
from .ptm_utils import hamiltonian_to_pauli_vector, zero_state_pauli_vector
from .readout import apply_measurement_noise
from .spec import NoiseSpec4q

OP_CNOT = 0
OP_ROT = 1
OP_NOOP = 2

_MAX_DENSE_QUBITS = 5

_JAX = None
_JNP = None


def _ensure_jax():
    global _JAX, _JNP
    if _JAX is None:
        import jax
        import jax.numpy as jnp
        _JAX = jax
        _JNP = jnp
    return _JAX, _JNP


@dataclass
class PaddedCompiled:

    rot_blocks_table: object
    cnot_table: object
    h_pauli_jnp: object
    r0_jnp: object
    n_qubits: int
    d4: int
    spec_tag: str


def compile_padded(
    spec: NoiseSpec4q, hamiltonian: np.ndarray, n_qubits: int
) -> PaddedCompiled:
    _, jnp = _ensure_jax()
    n = int(n_qubits)
    if not 1 <= n <= _MAX_DENSE_QUBITS:
        raise NotImplementedError(
            f"dense PTM supports n_qubits 1..{_MAX_DENSE_QUBITS}; got {n}. "
            "For 6q+ port NAHyRLQAS's local/sparse backend (ptm_local_jax_padded.py)."
        )
    d4 = 4 ** n

    compiled: CompiledPTM = compile_ptm(spec, n)

    rot_blocks_table = np.zeros((3 * n, 3, d4, d4), dtype=np.float32)
    for (axis, q), blocks in compiled.rotation_blocks.items():
        noise_q = compiled.per_rotation_noise[q]
        rot_blocks_table[axis * n + q] = np.einsum(
            "ij,kjl->kil", noise_q, blocks
        ).astype(np.float32)

    cnot_table = np.tile(np.eye(d4, dtype=np.float32), (n * n, 1, 1))
    for (c, t), m in compiled.cnot_with_noise.items():
        cnot_table[c * n + t] = m.astype(np.float32)

    h_pauli = hamiltonian_to_pauli_vector(
        np.asarray(hamiltonian, dtype=np.complex128), n
    )
    h_pauli = apply_measurement_noise(h_pauli, spec, n)

    return PaddedCompiled(
        rot_blocks_table=jnp.asarray(rot_blocks_table, dtype=jnp.float32),
        cnot_table=jnp.asarray(cnot_table, dtype=jnp.float32),
        h_pauli_jnp=jnp.asarray(h_pauli, dtype=jnp.float32),
        r0_jnp=jnp.asarray(zero_state_pauli_vector(n), dtype=jnp.float32),
        n_qubits=n,
        d4=d4,
        spec_tag=compiled.spec_tag,
    )


@dataclass
class PaddedSlots:
    op_type: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    axis: np.ndarray
    theta_idx: np.ndarray
    n_active_slots: int
    n_rotations: int


def parse_state_tensor(state_tensor, n_qubits: int):
    n = int(n_qubits)
    st = state_tensor
    if hasattr(st, "detach"):
        st = st.detach().cpu().numpy()
    st = np.asarray(st, dtype=np.float64)
    if st.ndim != 3 or st.shape[2] != n:
        raise ValueError(
            f"state_tensor must be (layers, {n}+6, {n}); got {st.shape}"
        )

    ops: list = []
    angles: list[float] = []
    if st.shape[0] == 0:
        return ops, np.empty(0, dtype=np.float32)

    relevant = np.abs(st[:, : n + 3, :]).sum(axis=(1, 2))
    nz = np.nonzero(relevant > 0)[0]
    if nz.size == 0:
        return ops, np.empty(0, dtype=np.float32)
    max_layer = int(nz[-1]) + 1

    for layer in range(max_layer):
        local = st[layer]
        tgts, ctrls = np.nonzero(local[:n] == 1)
        for tgt, ctrl in zip(tgts.tolist(), ctrls.tolist()):
            ops.append((OP_CNOT, int(ctrl), int(tgt), 0))
        axes, qubits = np.nonzero(local[n : n + 3] == 1)
        for axis, q in zip(axes.tolist(), qubits.tolist()):
            ops.append((OP_ROT, int(q), 0, int(axis)))
            angles.append(float(local[n + 3 + int(axis), int(q)]))

    return ops, np.asarray(angles, dtype=np.float32)


def state_tensor_to_padded_slots(
    state_tensor, n_qubits: int, max_slots: int
) -> tuple[PaddedSlots, np.ndarray]:
    ops, angles = parse_state_tensor(state_tensor, n_qubits)
    n_ops = len(ops)
    if n_ops > max_slots:
        raise ValueError(
            f"circuit has {n_ops} ops but max_slots={max_slots}; raise max_slots"
        )

    op_type = np.full(max_slots, OP_NOOP, dtype=np.int32)
    q0 = np.zeros(max_slots, dtype=np.int32)
    q1 = np.zeros(max_slots, dtype=np.int32)
    axis = np.zeros(max_slots, dtype=np.int32)
    theta_idx = np.full(max_slots, -1, dtype=np.int32)

    rot_count = 0
    for i, (op_t, a, b, ax) in enumerate(ops):
        op_type[i] = op_t
        if op_t == OP_CNOT:
            q0[i] = a
            q1[i] = b
        else:
            q0[i] = a
            axis[i] = ax
            theta_idx[i] = rot_count
            rot_count += 1

    slots = PaddedSlots(
        op_type=op_type, q0=q0, q1=q1, axis=axis, theta_idx=theta_idx,
        n_active_slots=n_ops, n_rotations=rot_count,
    )
    return slots, angles


_FWD_CACHE: dict = {}

SCAN_BUCKET = 8


def bucket_len(n_active: int, max_slots: int) -> int:
    n = max(1, int(n_active))
    return int(min(max_slots, ((n + SCAN_BUCKET - 1) // SCAN_BUCKET) * SCAN_BUCKET))


def _build_forward(n_qubits: int, scan_len: int):
    jax, jnp = _ensure_jax()
    HIGHEST = jax.lax.Precision.HIGHEST

    def _forward(angles, op_type, q0, q1, axis, theta_idx,
                 rot_blocks_table, cnot_table, r0, h_pauli):
        n = n_qubits

        def br_cnot(r, a0, a1, ax, t_idx):
            return jnp.matmul(cnot_table[a0 * n + a1], r, precision=HIGHEST)

        def br_rot(r, a0, a1, ax, t_idx):
            theta = jnp.where(
                t_idx >= 0, angles[jnp.maximum(t_idx, 0)], jnp.float32(0.0)
            )
            blocks = rot_blocks_table[ax * n + a0]
            ch = jnp.cos(0.5 * theta)
            sh = jnp.sin(0.5 * theta)
            b0 = jnp.matmul(blocks[0], r, precision=HIGHEST)
            b1 = jnp.matmul(blocks[1], r, precision=HIGHEST)
            b2 = jnp.matmul(blocks[2], r, precision=HIGHEST)
            return (ch * ch) * b0 - (ch * sh) * b1 + (sh * sh) * b2

        def br_noop(r, a0, a1, ax, t_idx):
            return r

        def step(r, slot):
            op_t, a0, a1, ax, t_idx = slot
            r_new = jax.lax.switch(op_t, (br_cnot, br_rot, br_noop), r, a0, a1, ax, t_idx)
            return r_new, None

        r_final, _ = jax.lax.scan(
            step, r0, (op_type, q0, q1, axis, theta_idx), length=scan_len
        )
        d = 2 ** n
        return jnp.dot(h_pauli, r_final, precision=HIGHEST) / d

    return jax.jit(_forward)


def _get_forward(n_qubits: int, scan_len: int):
    key = (n_qubits, scan_len)
    fwd = _FWD_CACHE.get(key)
    if fwd is None:
        fwd = _build_forward(n_qubits, scan_len)
        _FWD_CACHE[key] = fwd
    return fwd


def to_device_slots(slots: PaddedSlots, max_slots: int | None = None) -> tuple:
    _, jnp = _ensure_jax()
    cap = int(max_slots if max_slots is not None else slots.op_type.shape[0])
    L = bucket_len(slots.n_active_slots, cap)
    return (
        jnp.asarray(slots.op_type[:L]),
        jnp.asarray(slots.q0[:L]),
        jnp.asarray(slots.q1[:L]),
        jnp.asarray(slots.axis[:L]),
        jnp.asarray(slots.theta_idx[:L]),
    )


def _nonempty(angles: np.ndarray) -> np.ndarray:
    arr = np.asarray(angles, dtype=np.float32).reshape(-1)
    return np.zeros((1,), dtype=np.float32) if arr.size == 0 else arr


def expectation_dev(
    *, compiled: PaddedCompiled, dev_slots: tuple, angles, energy_shift: float = 0.0
) -> float:
    _, jnp = _ensure_jax()
    fwd = _get_forward(compiled.n_qubits, int(dev_slots[0].shape[0]))
    out = fwd(
        jnp.asarray(_nonempty(angles), dtype=jnp.float32),
        *dev_slots,
        compiled.rot_blocks_table,
        compiled.cnot_table,
        compiled.r0_jnp,
        compiled.h_pauli_jnp,
    )
    out.block_until_ready()
    return float(out) + float(energy_shift)


__all__ = [
    "OP_CNOT", "OP_ROT", "OP_NOOP",
    "PaddedCompiled", "PaddedSlots",
    "compile_padded", "parse_state_tensor", "state_tensor_to_padded_slots",
    "to_device_slots", "expectation_dev",
]
