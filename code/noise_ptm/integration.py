"""The single entry point DreamQAS calls to turn noise on.

Kept here rather than in ``code/WM_QAS/`` so that the wiring on the DreamQAS side stays
three lines per call site, and so this package has no dependency on DreamQAS's ``Config``
dataclass (it only reads plain attributes off the env).

Contract: returns ``None`` when ``mode == "off"``, in which case the caller must leave
``env._noisy_eval`` alone and DreamQAS behaves byte-identically to before.
"""

from __future__ import annotations

import numpy as np

from .evaluator import NoisyEvaluator
from .spec import NoiseSpec4q

_VALID_MODES = ("off", "ptm")


def build_evaluator(
    env,
    mode: str = "off",
    p1q: float = 0.0,
    p2q: float = 0.0,
    p_ro: float = 0.0,
    basis_change: bool = True,
    verbose: bool = True,
):
    """Build a :class:`NoisyEvaluator` for ``env``, or return ``None`` if noise is off.

    Reads ``hamiltonian`` / ``num_qubits`` / ``energy_shift`` / ``num_layers`` off the
    ``CircuitEnv``. Raises rather than silently degrading: a run that *meant* to be noisy
    but quietly wasn't would be indistinguishable from a clean run in the output.
    """
    mode = str(mode or "off").lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"noise_mode must be one of {_VALID_MODES}; got {mode!r}")
    if mode == "off":
        return None

    if p1q == 0.0 and p2q == 0.0 and p_ro == 0.0:
        raise ValueError(
            "noise_mode='ptm' with all three strengths at zero. That is an expensive "
            "way to reproduce the noiseless path — set noise_mode='off' instead, or "
            "give a non-zero p1q / p2q / p_ro."
        )

    n = int(env.num_qubits)
    spec = NoiseSpec4q(
        p1q=float(p1q), p2q=float(p2q), p_ro=float(p_ro),
        model_basis_change=bool(basis_change),
    )
    ev = NoisyEvaluator(
        spec,
        np.asarray(env.hamiltonian),
        n_qubits=n,
        energy_shift=float(env.energy_shift),
        num_layers=int(env.num_layers),
    )
    if verbose:
        print(
            f"[noise_ptm] ON  n_qubits={n}  {spec.to_tag()}  "
            f"max_slots={ev.max_slots}  basis_change={spec.model_basis_change}",
            flush=True,
        )
    return ev


def attach(env, cfg, verbose: bool = True):
    """Read the five ``noise_*`` fields off ``cfg`` and attach the evaluator to ``env``.

    No-op when ``cfg.noise_mode`` is absent or "off", so old checkpoints (whose stored
    cfg predates these fields) keep working unchanged.
    """
    ev = build_evaluator(
        env,
        mode=getattr(cfg, "noise_mode", "off"),
        p1q=getattr(cfg, "noise_p1q", 0.0),
        p2q=getattr(cfg, "noise_p2q", 0.0),
        p_ro=getattr(cfg, "noise_p_ro", 0.0),
        basis_change=getattr(cfg, "noise_basis_change", True),
        verbose=verbose,
    )
    if ev is not None:
        env.attach_noisy_evaluator(ev)
    return ev


__all__ = ["build_evaluator", "attach"]
