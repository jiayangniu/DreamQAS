"""Replay buffer for the energy surrogate.

Energy target is STATIONARY (a circuit's true_error never changes) -> keep ALL data,
no staleness. Sensitivity to the RARE good circuits (which appear late + are <1% of data)
comes from two knobs, NOT from dropping old data:
  - elite buffer   : the top-N episodes by best-error-reached, always in the batch
  - prioritized    : p ∝ quality(low err) × recency × wm_error(active learning)
Imagination seeds are drawn from a NARROW recent window (the search frontier).
"""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, cfg, scale=None):
        self.cfg = cfg
        self.scale = scale       # EnergyScale when cfg.oracle_free (None on the legacy path)
        # dict{acts:int[T], err:float[T] (E0 te, NaN when diagnostics off), E:float[T] (raw Ha),
        #      idx, min_err, min_E, wm_err}
        self.eps = []
        self._next = 0

    def add(self, acts, err_mHa, E_Ha):
        self.eps.append({"acts": np.asarray(acts, np.int64),
                         "err": np.asarray(err_mHa, np.float32),
                         "E": np.asarray(E_Ha, np.float32),
                         "idx": self._next, "min_err": float(np.min(err_mHa)),
                         "min_E": float(np.min(E_Ha)),
                         "wm_err": 1.0})   # wm_err starts high (unseen -> prioritize)
        self._next += 1

    def __len__(self):
        return len(self.eps)

    @property
    def n_steps(self):
        return sum(len(e["acts"]) for e in self.eps)

    def _rank_key(self):
        # ordering key: legacy = E0 true error; oracle-free = raw energy (identical ranking by
        # variationality; well-defined when diagnostics are off and err is NaN)
        return "min_E" if self.scale is not None else "min_err"

    def elite_idxs(self):
        k = self._rank_key()
        return sorted(range(len(self.eps)), key=lambda i: self.eps[i][k])[: self.cfg.elite_N]

    def _stratified_idxs(self, n):
        """Sample n episodes evenly across error DECILES (B: keeps coarse cross-scale
        contrast in the batch so the WM doesn't lose ranking resolution when the policy
        converges into a narrow good-circuit band — the 6q resolution-collapse fix)."""
        if n <= 0 or not self.eps:
            return []
        me = np.array([e[self._rank_key()] for e in self.eps])
        order = np.argsort(me)
        bins = [b for b in np.array_split(order, min(10, len(order))) if len(b)]  # deciles
        return [int(np.random.choice(bins[i % len(bins)])) for i in range(n)]

    def _priorities(self):
        idx = np.array([e["idx"] for e in self.eps], float)
        rec = (idx - idx.min() + 1) / (idx.max() - idx.min() + 1)          # newer -> higher
        if self.scale is not None:            # oracle-free: positive frontier score replaces te
            s = self.scale.s_mHa_np(np.array([e["min_E"] for e in self.eps]))
            assert np.all(s > 0), "priorities consumed post-adopt: s >= margin must hold"
            quality = 1.0 / (s + 1.0)
        else:
            quality = 1.0 / (np.array([e["min_err"] for e in self.eps]) + 1.0)  # lower err -> higher
        wmerr = np.array([e["wm_err"] for e in self.eps])                   # active learning
        p = rec * quality * (0.1 + wmerr)
        return p / p.sum()

    def sample_batch(self):
        n = len(self.eps)
        B = min(self.cfg.batch_size, n)
        n_el = int(B * self.cfg.frac_elite)
        n_pr = int(B * self.cfg.frac_prioritized)
        n_st = int(B * getattr(self.cfg, "frac_stratified", 0.0))
        n_rd = max(0, B - n_el - n_pr - n_st)              # guard: never negative if fracs>1
        elite = self.elite_idxs()
        pick = []
        if elite:
            pick += list(np.random.choice(elite, n_el, replace=len(elite) < n_el))
        pick += list(np.random.choice(n, n_pr, p=self._priorities()))
        pick += self._stratified_idxs(n_st)                    # B: even across error deciles
        pick += list(np.random.choice(n, n_rd))
        return self._collate([self.eps[i] for i in pick]), pick

    def update_wm_err(self, idxs, errs):
        for i, e in zip(idxs, errs):
            self.eps[i]["wm_err"] = float(e)

    def _collate(self, batch):
        T = max(len(e["acts"]) for e in batch)
        B = len(batch)
        acts = torch.zeros(B, T, dtype=torch.long)
        err = torch.zeros(B, T)
        E = torch.zeros(B, T)
        mask = torch.zeros(B, T)
        for i, e in enumerate(batch):
            t = len(e["acts"])
            acts[i, :t] = torch.from_numpy(e["acts"])
            err[i, :t] = torch.from_numpy(e["err"])
            E[i, :t] = torch.from_numpy(e["E"])
            mask[i, :t] = 1.0
        return acts, err, E, mask

    def recent_val(self, n):
        """Most-recent n episodes -> (acts,err,E,mask) for the fidelity self-check (held-out)."""
        return self._collate(self.eps[-n:]) if self.eps else None

    def imag_seeds(self, n, window, strategy="frontier"):
        """n prefixes (random cut) to seed imagination. Strategy picks the seed POOL (P3):
          frontier      : most-recent `window` episodes (the search frontier; default)
          elite         : the best-min_err episodes (imagine forward from known-good neighborhoods)
          high_disagree : highest-wm_err episodes (imagine where the WM is most uncertain)
          stratified    : even across error deciles (broad coverage)"""
        if not self.eps:
            return []
        if strategy == "elite":
            pool = [self.eps[i] for i in self.elite_idxs()[: max(window, 1)]]
        elif strategy == "high_disagree":
            order = sorted(range(len(self.eps)), key=lambda i: -self.eps[i]["wm_err"])[: max(window, 1)]
            pool = [self.eps[i] for i in order]
        elif strategy == "stratified":
            idxs = self._stratified_idxs(min(window, len(self.eps)))
            pool = [self.eps[i] for i in idxs] or self.eps
        else:  # frontier
            pool = self.eps[-window:] if len(self.eps) > window else self.eps
        out = []
        for _ in range(n):
            e = pool[np.random.randint(len(pool))]
            cut = np.random.randint(1, len(e["acts"]) + 1)
            out.append(e["acts"][:cut])
        return out


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    buf = ReplayBuffer(cfg)
    rng = np.random.default_rng(0)
    for k in range(300):
        T = rng.integers(10, 30)
        acts = rng.integers(0, 24, T)
        # later episodes reach lower error (simulate improving policy)
        err = rng.uniform(50, 200, T).astype(np.float32); err[-1] = max(0.5, 100 - k * 0.3)
        buf.add(acts, err, -7.8 + err / 1000.0)   # fake raw energies consistent with err
    print(f"buffer: {len(buf)} eps, {buf.n_steps} steps")
    (a, e, E_, m), pick = buf.sample_batch()
    print(f"batch: acts{tuple(a.shape)} mask sum {int(m.sum())} | elite top-3 min_err "
          f"{[round(buf.eps[i]['min_err'],1) for i in buf.elite_idxs()[:3]]}")
    seeds = buf.imag_seeds(cfg.imag_n_seeds, cfg.imag_seed_window)
    print(f"imag seeds: {len(seeds)}, prefix lens {[len(s) for s in seeds[:5]]} (from recent {cfg.imag_seed_window})")
