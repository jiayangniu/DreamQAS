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
