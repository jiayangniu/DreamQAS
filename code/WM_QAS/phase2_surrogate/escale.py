import math

import numpy as np
import torch

_LN10 = math.log(10.0)


class EnergyScale:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pending_frontier = float("inf")
        self.F_adopted = None
        mode = getattr(cfg, "score_margin_mode", "fixed")
        if mode == "fixed":
            self.margin_mHa = float(cfg.score_margin_mHa)
            self.frozen = True
        else:
            self.margin_mHa = float(cfg.score_margin_init_mHa)
            self.frozen = False
        self.scale_version = 0
        self.below_frontier_events = 0

    def initialized(self):
        return self.F_adopted is not None

    def init_from(self, energies):
        e = float(np.min(energies))
        self.pending_frontier = e
        self.F_adopted = e
        self.scale_version = 1

    def observe(self, E):
        E = float(E)
        if E < self.pending_frontier:
            jump = (self.pending_frontier - E) * 1000.0
            if math.isfinite(jump) and jump > self.cfg.frontier_jump_warn_mHa:
                print(f"[escale] WARNING: frontier jump {jump:.2f} mHa in one observation "
                      f"(possible numeric variational violation)")
            self.pending_frontier = E

    def adopt(self):
        if self.F_adopted is None or self.pending_frontier < self.F_adopted:
            self.F_adopted = self.pending_frontier
            self.scale_version += 1

    def _d_mHa(self, E):
        return (np.asarray(E, np.float64) - self.F_adopted) * 1000.0

    def score_np(self, E, count=True):
        d = self._d_mHa(E)
        m = self.margin_mHa
        neg = d < 0
        if count and np.any(neg):
            self.below_frontier_events += int(np.sum(neg))
        s = np.sign(d) * np.log10(1.0 + np.abs(d) / m)
        return s.astype(np.float32)

    def score_t(self, E: torch.Tensor) -> torch.Tensor:
        d = (E.double() - self.F_adopted) * 1000.0
        m = self.margin_mHa
        s = torch.sign(d) * torch.log10(1.0 + d.abs() / m)
        return s.float()

    def s_mHa_np(self, E):
        return self.margin_mHa * np.power(10.0, self.score_np(E, count=False).astype(np.float64))

    def freeze_margin_from_buffer(self, buf):
        cfg = self.cfg
        d = np.concatenate([(np.asarray(e["E"], np.float64) - self.pending_frontier) * 1000.0
                            for e in buf.eps]) if buf.eps else np.array([])
        pos = d[d > 0]
        if pos.size:
            self.margin_mHa = float(np.clip(np.quantile(pos, cfg.score_margin_q),
                                            cfg.score_margin_min_mHa, cfg.score_margin_max_mHa))
        self.frozen = True
        self.scale_version += 1
        print(f"[escale] margin frozen at {self.margin_mHa:.3f} mHa "
              f"(q{int(cfg.score_margin_q*100)} of {pos.size} warm-up distances)")

    def state_dict(self):
        return {"pending_frontier": self.pending_frontier, "F_adopted": self.F_adopted,
                "margin_mHa": self.margin_mHa, "frozen": self.frozen,
                "scale_version": self.scale_version,
                "below_frontier_events": self.below_frontier_events}

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)


if __name__ == "__main__":
    class _C:
        score_margin_mode = "fixed"; score_margin_mHa = 0.1
        score_margin_init_mHa = 10.0; score_margin_q = 0.10
        score_margin_min_mHa = 1.0; score_margin_max_mHa = 50.0; frontier_jump_warn_mHa = 50.0
    sc = EnergyScale(_C())
    assert sc.frozen and sc.margin_mHa == 0.1
    sc.init_from([-7.70, -7.75])
    assert abs(sc.score_np(np.array([-7.75]), count=False)[0]) < 1e-12, "S(frontier)=0"
    E = np.array([-7.70, -7.75, -7.76, -7.77])
    s = sc.score_np(E)
    order = np.argsort(E)
    assert np.all(np.diff(s[order]) > 0), "S strictly increasing in E"
    assert np.all(np.isfinite(s)) and s[2] != s[3], "new bests must get distinct scores"
    assert s[3] > -math.log10(1 + 20.0 / 0.1) - 1e-5, "d<0 bounded log (no linear blow-up)"
    for e in E: sc.observe(e)
    v0 = sc.scale_version; sc.adopt(); assert sc.scale_version == v0 + 1
    sm = sc.s_mHa_np(np.array([-7.80, -7.77, -7.76]))
    assert np.all(sm > 0) and np.all(np.diff(sm) > 0), "s_mHa positive + monotone"
    d = (np.array([-7.75]) - sc.F_adopted) * 1000.0
    assert abs(sc.s_mHa_np(np.array([-7.75]))[0] - (0.1 + d[0])) < 2e-3
    print("escale smoke OK:", sc.state_dict())
