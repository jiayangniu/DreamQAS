from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseSpec4q:

    p1q: float = 0.0
    p2q: float = 0.0
    p_ro: float = 0.0
    model_basis_change: bool = True

    def __post_init__(self) -> None:
        for name, v in (("p1q", self.p1q), ("p2q", self.p2q), ("p_ro", self.p_ro)):
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float; got {type(v)}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{name} out of [0, 1]: {v}")
        if float(self.p_ro) > 0.5:
            raise ValueError(f"p_ro must be <= 0.5; got {self.p_ro}")

    @property
    def is_noiseless(self) -> bool:
        return self.p1q == 0.0 and self.p2q == 0.0 and self.p_ro == 0.0

    @property
    def has_gate_noise(self) -> bool:
        return self.p1q > 0.0 or self.p2q > 0.0

    def to_tag(self) -> str:
        if self.is_noiseless:
            return "noiseless"
        bc = "" if self.model_basis_change else "_nobc"
        return f"p1q{self.p1q:.3e}_p2q{self.p2q:.3e}_pro{self.p_ro:.3e}{bc}"


BOSTON = NoiseSpec4q(p1q=1.5e-4, p2q=1.1e-3, p_ro=5.4e-3)
FEZ    = NoiseSpec4q(p1q=3.3e-4, p2q=2.9e-3, p_ro=1.5e-2)
MIAMI  = NoiseSpec4q(p1q=7.0e-4, p2q=7.8e-3, p_ro=2.0e-2)

DEVICE_TIERS = {"boston": BOSTON, "fez": FEZ, "miami": MIAMI}

CRLQAS_GENERIC = NoiseSpec4q(p1q=1.0e-3, p2q=5.0e-3, p_ro=2.25e-2)

LOW, HIGH = BOSTON, MIAMI
TIER_NAMES = {"low": "boston", "high": "miami"}
COMPOSITE = MIAMI

SWEEP_1Q = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p1q=1e-4),
    "medium": NoiseSpec4q(p1q=5e-4),
    "strong": NoiseSpec4q(p1q=1e-3),
}
SWEEP_2Q = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p2q=1e-3),
    "medium": NoiseSpec4q(p2q=5e-3),
    "strong": NoiseSpec4q(p2q=1e-2),
}
SWEEP_RO = {
    "clean":  NoiseSpec4q(),
    "mild":   NoiseSpec4q(p_ro=5e-3),
    "medium": NoiseSpec4q(p_ro=1e-2),
    "strong": NoiseSpec4q(p_ro=2e-2),
}

__all__ = ["NoiseSpec4q", "COMPOSITE", "LOW", "HIGH", "BOSTON", "FEZ", "MIAMI",
           "DEVICE_TIERS", "TIER_NAMES", "CRLQAS_GENERIC", "SWEEP_1Q", "SWEEP_2Q", "SWEEP_RO"]
