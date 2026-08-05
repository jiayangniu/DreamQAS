"""JAX padded PTM forward pass — one JIT per (n_qubits,), circuit structure as data.

ADAPTED from NAHyRLQAS ``nahyrlqas/backends/ptm_jax_padded.py`` @ commit bd763a4.

WHY PADDED (upstream's insight, kept verbatim in spirit)
--------------------------------------------------------
RL adds one gate per step, so the circuit's instruction signature changes every step. A
naive per-circuit JIT recompiles on **every step** — ~600k XLA compiles over a full run,
which is what made noisy RL-QAS infeasible. The fix is to pass the circuit structure as
**runtime tensors** through ``lax.scan``, padded to a fixed ``max_slots``. Then there is
exactly one compile per ``n_qubits``, and the circuit is just data.

WHAT DIFFERS FROM UPSTREAM
--------------------------
* **No TIME and no READOUT slots.** This project's spec has no T1/T2, and readout is a
  compile-time Hamiltonian rescale (:mod:`noise_ptm.readout`). Upstream emits one TIME
  slot after every circuit moment plus a final READOUT slot, which for a 40-gate circuit
  is ~30 extra slots — all of them identity here. Dropping them shortens the scan by
  roughly half and removes two of the five candidate branches per slot.
* **Self-contained parser.** Upstream's ``state_tensor_to_padded_slots`` imports the
  parser from ``ptm_jax.py``; ours reads the state tensor directly so this package has
  no dependency on the NAHyRLQAS tree.
* **Explicit ``Precision.HIGHEST`` on every matmul** — see the TF32 note below.

⚠ TF32 TRAP
-----------
On GPU, XLA lowers matmuls to TF32 by default (~10-bit mantissa, ~1e-3 relative error).
Upstream lost a debugging session to exactly this: their tensor-contraction path
silently returned 1e-3 energy errors until they forced full float32. Every matmul below
therefore passes ``precision=jax.lax.Precision.HIGHEST``. ``tests/test_equivalence.py``
is the regression guard — it compares against qulacs at 1e-5 Ha, which TF32 would fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compile import CompiledPTM, compile_ptm
from .ptm_utils import hamiltonian_to_pauli_vector, zero_state_pauli_vector
from .readout import apply_measurement_noise
from .spec import NoiseSpec4q

# Op-type tags. Upstream also has TIME=2 / READOUT=3; we never emit them (see header),
# so the codes are renumbered to keep the candidate stack as small as possible.
OP_CNOT = 0
OP_ROT = 1
OP_NOOP = 2  # padding slot — applies identity

# Dense PTM is 4^n x 4^n float32. n=4 -> 256 (256 KB/matrix, fine).
# n=6 would be 4096^2*4 = 64 MB per matrix x ~100 matrices — hence the cap.
_MAX_DENSE_QUBITS = 5

_JAX = None
_JNP = None


def _ensure_jax():
    """Import JAX lazily so that merely importing this package costs nothing."""
    global _JAX, _JNP
    if _JAX is None:
        import jax
        import jax.numpy as jnp
        _JAX = jax
        _JNP = jnp
    return _JAX, _JNP


# ---------------------------------------------------------------------------
# Compiled bundle (device-resident)
# ---------------------------------------------------------------------------


@dataclass
class PaddedCompiled:
    """PTM tables on the device, indexable by integer."""

    # (3*n, 3, d4, d4) — index = axis*n + q. The per-rotation 1q depolarizing is already
    # multiplied in, so there is no separate noise table.
    rot_blocks_table: object
    cnot_table: object         # (n*n, d4, d4)   — index = c*n + t; diagonal = identity
    h_pauli_jnp: object        # (d4,) — ALREADY rescaled for readout + basis-change
    r0_jnp: object             # (d4,)
    n_qubits: int
    d4: int
    spec_tag: str


def compile_padded(
    spec: NoiseSpec4q, hamiltonian: np.ndarray, n_qubits: int
) -> PaddedCompiled:
    """Build the device-resident PTM tables for one (spec, Hamiltonian) pair.

    All (axis, q) rotation blocks and (c, t) CNOT pairs are pre-built, so no per-circuit
    recompile is ever needed. Readout + basis-change noise are folded into ``h_pauli``
    here, which is why they cost nothing at run time.
    """
    _, jnp = _ensure_jax()
    n = int(n_qubits)
    if not 1 <= n <= _MAX_DENSE_QUBITS:
        raise NotImplementedError(
            f"dense PTM supports n_qubits 1..{_MAX_DENSE_QUBITS}; got {n}. "
            "For 6q+ port NAHyRLQAS's local/sparse backend (ptm_local_jax_padded.py)."
        )
    d4 = 4 ** n

    compiled: CompiledPTM = compile_ptm(spec, n)

    # Fold the per-rotation 1q noise INTO the three rotation blocks at compile time:
    #     noise @ M_R(θ) @ r  ==  c²(noise@B0)r - cs(noise@B1)r + s²(noise@B2)r
    # so the hot loop is three matvecs instead of "build M_R from three scaled (d4,d4)
    # matrices, then two matvecs".
    # MEASURED: a WASH on speed (2.06 -> 2.01 ms/eval at 4q, ~2%), NOT the ~25% a FLOP
    # count suggests — three matvecs cost about what one build plus two matvecs did.
    # Kept because it removes a table from the device and a matmul from the loop, not
    # because it is faster. Do not cite it as an optimization.
    rot_blocks_table = np.zeros((3 * n, 3, d4, d4), dtype=np.float32)
    for (axis, q), blocks in compiled.rotation_blocks.items():
        noise_q = compiled.per_rotation_noise[q]
        rot_blocks_table[axis * n + q] = np.einsum(
            "ij,kjl->kil", noise_q, blocks
        ).astype(np.float32)

    # Diagonal entries (c == t) are never indexed; identity keeps the gather in bounds.
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


# ---------------------------------------------------------------------------
# state tensor -> padded slots
# ---------------------------------------------------------------------------


@dataclass
class PaddedSlots:
    op_type: np.ndarray     # (D,) int32
    q0: np.ndarray          # (D,) int32 — control (CNOT) or rotation qubit (ROT)
    q1: np.ndarray          # (D,) int32 — target (CNOT)
    axis: np.ndarray        # (D,) int32 — rotation axis 0/1/2 (ROT)
    theta_idx: np.ndarray   # (D,) int32 — index into angles[]; -1 for non-ROT
    n_active_slots: int
    n_rotations: int


def parse_state_tensor(state_tensor, n_qubits: int):
    """Walk DreamQAS's state tensor and return (ops, angles).

    Layout, written by ``environment.py:236-247`` and identical to CRLQAS/HyRLQAS:

        rows ``0 : N``        CNOT adjacency,  ``state[layer][target][control] = 1``
        rows ``N : N+3``      rotation axis one-hot, ``state[layer][N+axis-1][q] = 1``
        rows ``N+3 : N+6``    rotation angle,        ``state[layer][N+3+axis-1][q] = θ``

    Within a layer the emission order is: all CNOTs, then rotations in
    (axis, qubit) row-major order. That reproduces the parameter ordering that
    ``environment.py:537-538`` (``rot_pos = (st[:, n:n+3] == 1).nonzero()``) relies on,
    so ``angles[i]`` here indexes the same rotation as COBYLA's ``x[i]``.

    Accepts a torch tensor or anything ``np.asarray`` can digest.
    """
    n = int(n_qubits)
    st = state_tensor
    if hasattr(st, "detach"):        # torch tensor
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

    # Trailing all-zero layers carry no gate; stop at the last populated one. Only rows
    # 0:N+3 count — a nonzero angle row with a zero indicator row is not a gate.
    relevant = np.abs(st[:, : n + 3, :]).sum(axis=(1, 2))
    nz = np.nonzero(relevant > 0)[0]
    if nz.size == 0:
        return ops, np.empty(0, dtype=np.float32)
    max_layer = int(nz[-1]) + 1

    for layer in range(max_layer):
        local = st[layer]
        # CNOTs: rows 0:N, entry [target][control]
        tgts, ctrls = np.nonzero(local[:n] == 1)
        for tgt, ctrl in zip(tgts.tolist(), ctrls.tolist()):
            ops.append((OP_CNOT, int(ctrl), int(tgt), 0))
        # Rotations: rows N:N+3 one-hot, angle in rows N+3:N+6
        axes, qubits = np.nonzero(local[n : n + 3] == 1)
        for axis, q in zip(axes.tolist(), qubits.tolist()):
            ops.append((OP_ROT, int(q), 0, int(axis)))
            angles.append(float(local[n + 3 + int(axis), int(q)]))

    return ops, np.asarray(angles, dtype=np.float32)


def state_tensor_to_padded_slots(
    state_tensor, n_qubits: int, max_slots: int
) -> tuple[PaddedSlots, np.ndarray]:
    """Convert a DreamQAS state tensor into fixed-width padded slot arrays."""
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
        else:                       # OP_ROT
            q0[i] = a
            axis[i] = ax
            theta_idx[i] = rot_count
            rot_count += 1

    slots = PaddedSlots(
        op_type=op_type, q0=q0, q1=q1, axis=axis, theta_idx=theta_idx,
        n_active_slots=n_ops, n_rotations=rot_count,
    )
    return slots, angles


# ---------------------------------------------------------------------------
# JIT'd forward
# ---------------------------------------------------------------------------

_FWD_CACHE: dict = {}

# Scan length is rounded up to a multiple of this, so a growing RL circuit only triggers
# a handful of compiles (num_layers=40 -> lengths 8,16,24,32,40,48) instead of one per
# step, while a 3-gate circuit still does not pay for 48 slots.
SCAN_BUCKET = 8


def bucket_len(n_active: int, max_slots: int) -> int:
    """Round the active slot count up to the next SCAN_BUCKET, capped at max_slots."""
    n = max(1, int(n_active))
    return int(min(max_slots, ((n + SCAN_BUCKET - 1) // SCAN_BUCKET) * SCAN_BUCKET))


def _build_forward(n_qubits: int, scan_len: int):
    """One JIT per (n_qubits, scan_len).

    Two things make this ~7x cheaper than the naive padded form measured at 4q:

    * ``lax.switch`` instead of candidate-stacking. Stacking evaluates BOTH branches at
      every slot; the ROT branch alone builds ``M_R`` from three (d4, d4) blocks, so a
      CNOT slot was paying ~5 matrix ops it never used. Upstream hit the same wall at 6q
      and moved to switch there for the same reason.
    * ``scan_len`` is the bucketed active length, not ``max_slots``. RL circuits start at
      one gate and grow; scanning 48 NOOP slots for a 3-gate circuit was most of the cost
      early in an episode.
    """
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
            blocks = rot_blocks_table[ax * n + a0]          # (3, d4, d4), noise pre-folded
            ch = jnp.cos(0.5 * theta)
            sh = jnp.sin(0.5 * theta)
            # qulacs convention R(θ)=exp(+iθA/2) -> minus on the cross term. Blocks are
            # applied to r individually (see compile_padded) so no (d4,d4) matrix is ever
            # materialised inside the loop.
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
    """Upload the slot arrays ONCE, to be reused across a whole COBYLA run.

    Uploads only the bucketed active prefix, not all ``max_slots`` — the returned length
    is what the scan will iterate over.

    Upstream measured the upload-once part as a **1.21x** end-to-end win at 4q: the
    kernel is tiny, so five redundant host->device transfers per objective evaluation are
    a large fraction of the (dispatch-dominated) cost. After this, only ``angles``
    crosses per eval.
    """
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
    """Zero-parameter circuits still trace the ROT gather; a dummy keeps it in bounds."""
    arr = np.asarray(angles, dtype=np.float32).reshape(-1)
    return np.zeros((1,), dtype=np.float32) if arr.size == 0 else arr


def expectation_dev(
    *, compiled: PaddedCompiled, dev_slots: tuple, angles, energy_shift: float = 0.0
) -> float:
    """One noisy ⟨H⟩. This is the function COBYLA calls thousands of times per step."""
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
