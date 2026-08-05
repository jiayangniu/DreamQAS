"""noise_ptm — exact, fast noisy VQE evaluation for DreamQAS.

Kept in its own package, deliberately outside ``code/WM_QAS/``: the noise experiments
are an additive layer, and DreamQAS's noiseless path must stay byte-identical when the
switch is off.

The physics is a PTM / Pauli-Liouville simulation ported from the user's NAHyRLQAS
project (commit ``bd763a4``), where it was built and validated specifically to make
noisy RL-QAS affordable. See ``docs/noise_acceleration.md`` in that repo for the cost
model; the short version is that a run needs ~10^8 noisy ⟨H⟩ evaluations, so per-eval
cost and CPU<->GPU round trips dominate everything.

Noise model (three numbers, no shots, deterministic):

* ``p1q``  depolarizing after each single-qubit rotation
* ``p2q``  depolarizing after each CNOT
* ``p_ro`` symmetric readout assignment error, folded into the Hamiltonian at compile
  time together with the measurement basis-change gates' depolarizing

Typical use::

    from noise_ptm import NoiseSpec4q, NoisyEvaluator

    ev = NoisyEvaluator(NoiseSpec4q(p1q=5e-4, p2q=5e-3, p_ro=1e-2),
                        hamiltonian, n_qubits=4, energy_shift=shift, num_layers=40)
    ev.rebind(state_tensor)      # once per RL step
    e = ev.energy(thetas)        # once per COBYLA objective evaluation
"""

from .evaluator import NoisyEvaluator
from .spec import (BOSTON, COMPOSITE, CRLQAS_GENERIC, DEVICE_TIERS, FEZ, HIGH, TIER_NAMES,
                   LOW, MIAMI, SWEEP_1Q, SWEEP_2Q, SWEEP_RO, NoiseSpec4q)

__all__ = [
    "NoisyEvaluator",
    "NoiseSpec4q",
    "COMPOSITE",
    "LOW",
    "HIGH",
    "BOSTON",
    "FEZ",
    "MIAMI",
    "DEVICE_TIERS",
    "TIER_NAMES",
    "CRLQAS_GENERIC",
    "SWEEP_1Q",
    "SWEEP_2Q",
    "SWEEP_RO",
]
