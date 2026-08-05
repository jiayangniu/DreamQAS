"""``NoisyEvaluator`` — the one class DreamQAS touches.

Usage from ``environment.py`` (all guarded so the noiseless path is untouched):

    # once, when the env is built
    env._noisy_eval = NoisyEvaluator(spec, hamiltonian, n_qubits, energy_shift, max_slots)

    # once per RL step, before the optimizer runs
    env._noisy_eval.rebind(env.state)

    # inside COBYLA's objective, thousands of times per step
    e = env._noisy_eval.energy(thetas)

The split matters: ``rebind`` does the state-tensor parse and the host->device upload of
the slot arrays; ``energy`` then ships only the angle vector. Upstream measured that
separation as a 1.21x end-to-end win at 4 qubits.
"""

from __future__ import annotations

import numpy as np

from .forward_4q import (
    PaddedCompiled, compile_padded, expectation_dev, state_tensor_to_padded_slots,
    to_device_slots,
)
from .spec import NoiseSpec4q


class NoisyEvaluator:
    """Exact noisy ⟨H⟩ on a PTM/Pauli-Liouville backend. No shots, deterministic."""

    def __init__(
        self,
        spec: NoiseSpec4q,
        hamiltonian: np.ndarray,
        n_qubits: int,
        energy_shift: float = 0.0,
        max_slots: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        self.spec = spec
        self.n_qubits = int(n_qubits)
        self.energy_shift = float(energy_shift)
        if max_slots is None:
            # One slot per gate (we emit no TIME/READOUT slots). num_layers is
            # DreamQAS's gate budget; pad a little so a full circuit always fits.
            base = int(num_layers) if num_layers else 64
            max_slots = int(base) + 8
        self.max_slots = int(max_slots)

        self.compiled: PaddedCompiled = compile_padded(
            spec, hamiltonian, self.n_qubits
        )
        self._dev_slots = None
        self._n_rotations = 0
        self._n_active = 0
        # Diagnostics: how many objective evaluations this evaluator has served.
        self.n_energy_calls = 0
        self.n_rebinds = 0

    # -- per RL step ------------------------------------------------------

    def rebind(self, state_tensor) -> int:
        """Parse the circuit and upload its slot arrays. Returns the rotation count."""
        slots, _angles = state_tensor_to_padded_slots(
            state_tensor, self.n_qubits, self.max_slots
        )
        self._dev_slots = to_device_slots(slots, self.max_slots)
        self._n_rotations = slots.n_rotations
        self._n_active = slots.n_active_slots
        self.n_rebinds += 1
        return slots.n_rotations

    # -- per objective evaluation ----------------------------------------

    def energy(self, thetas) -> float:
        """Noisy ⟨H⟩ for the currently bound circuit at angles ``thetas``.

        ``thetas`` must be in the same order DreamQAS's optimizer uses — layer-major,
        then (axis, qubit) row-major — which is what ``parse_state_tensor`` produces.
        """
        if self._dev_slots is None:
            raise RuntimeError("NoisyEvaluator.energy() called before rebind()")
        arr = np.asarray(thetas, dtype=np.float32).reshape(-1)
        if arr.size != self._n_rotations:
            raise ValueError(
                f"expected {self._n_rotations} angles for the bound circuit; "
                f"got {arr.size}"
            )
        self.n_energy_calls += 1
        return expectation_dev(
            compiled=self.compiled,
            dev_slots=self._dev_slots,
            angles=arr,
            energy_shift=self.energy_shift,
        )

    # -- convenience: bind and evaluate in one call (offline sweeps, tests) --

    def energy_of(self, state_tensor, thetas) -> float:
        self.rebind(state_tensor)
        return self.energy(thetas)

    def energy_from_state(self, state_tensor) -> float:
        """Noisy ⟨H⟩ using the angles already embedded in the state tensor.

        This is the once-per-step path (``environment.get_energy``), as opposed to the
        thousands-per-step ``energy()``. It re-parses, so do not use it inside an
        optimizer loop.
        """
        from .forward_4q import state_tensor_to_padded_slots  # local: keeps import cheap

        slots, angles = state_tensor_to_padded_slots(
            state_tensor, self.n_qubits, self.max_slots
        )
        self._dev_slots = to_device_slots(slots, self.max_slots)
        self._n_rotations = slots.n_rotations
        self._n_active = slots.n_active_slots
        self.n_rebinds += 1
        return self.energy(angles)

    @property
    def n_rotations(self) -> int:
        return self._n_rotations

    def __repr__(self) -> str:
        return (
            f"NoisyEvaluator(n_qubits={self.n_qubits}, spec={self.spec.to_tag()}, "
            f"max_slots={self.max_slots}, calls={self.n_energy_calls})"
        )


__all__ = ["NoisyEvaluator"]
