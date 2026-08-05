"""Energy-surrogate world model.

GRU encoder over the gate sequence -> K-head ENSEMBLE predicting log10(true_error_mHa).
  - ensemble MEAN  = energy prediction (value for imagination)
  - ensemble STD   = epistemic confidence (gates how far imagination may roll)

The GRU gives KNOWN dynamics for free: `step(gate, hstate)` feeds one gate and advances
the hidden state == re-encoding the (known) longer circuit. No learned transition prior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _head(hidden):
    return nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 1))


class SurrogateWM(nn.Module):
    def __init__(self, action_size, cfg):
        super().__init__()
        self.embed = nn.Embedding(action_size, cfg.wm_embed)
        self.gru = nn.GRU(cfg.wm_embed, cfg.wm_hidden, cfg.wm_gru_layers,
                          batch_first=True, dropout=0.1 if cfg.wm_gru_layers > 1 else 0.0)
        self.heads = nn.ModuleList([_head(cfg.wm_hidden) for _ in range(cfg.wm_ensemble_K)])
        self.hidden, self.layers, self.K = cfg.wm_hidden, cfg.wm_gru_layers, cfg.wm_ensemble_K
        self.feature_dim = cfg.wm_hidden

    # ---------- full-sequence (WM training + actor features) ----------
    def encode_seq(self, acts):
        """acts [B,T] int -> h [B,T,hidden] (GRU output at every prefix)."""
        return self.gru(self.embed(acts))[0]

    def head_preds(self, h):
        """h [...,hidden] -> [K, ...]  per-head log10(err_mHa)."""
        return torch.stack([hd(h).squeeze(-1) for hd in self.heads], dim=0)

    def predict(self, acts):
        """-> mean_logerr [B,T], conf_std [B,T], h [B,T,hidden]."""
        h = self.encode_seq(acts)
        p = self.head_preds(h)
        return p.mean(0), p.std(0), h

    def seq_preds(self, acts):
        """[K,B,T] per-head preds for WM training (uniform API across WMs)."""
        return self.head_preds(self.encode_seq(acts))

    # ---------- incremental (imagination rollout = known dynamics) ----------
    def init_state(self, B, device):
        return torch.zeros(self.layers, B, self.hidden, device=device)

    def encode_to_state(self, acts):
        """Encode a real circuit prefix -> (last feat [B,hidden], hstate [layers,B,hidden])
        to seed an imagined rollout from a frontier circuit."""
        out, hstate = self.gru(self.embed(acts))
        return out[:, -1], hstate

    def step(self, action, hstate):
        """Feed ONE gate: action [B], hstate [layers,B,hidden]
        -> feat [B,hidden], new hstate, per-head logerr [K,B]."""
        out, hstate = self.gru(self.embed(action).unsqueeze(1), hstate)
        feat = out.squeeze(1)
        return feat, hstate, self.head_preds(feat)

    def kl_loss(self):
        """GRU encoder has no latent bottleneck -> 0 (keeps the training loop uniform)."""
        return torch.zeros((), device=self.embed.weight.device)

    # PopArt no-ops (legacy shared-trunk WM predicts raw log-error) — uniform runner API.
    def denorm(self, x):
        return x

    def normalize(self, y):
        return y

    def update_popart(self, y):
        pass


class RSSMSurrogateWM(nn.Module):
    """RSSM-flavored encoder for a FULLY-OBSERVED, KNOWN-DYNAMICS discrete circuit.

    Deterministic GRU recurrent core (dynamics stay known: step == re-encode) + a
    DreamerV3-style CATEGORICAL latent bottleneck that regularizes the representation.
    With no partial observability the posterior == prior, so the "KL" reduces to a
    free-bits entropy floor that keeps the latent from collapsing — its job here is
    representation regularization, NOT future-uncertainty modeling (epistemic uncertainty
    is the K-head ensemble's job). Imagination uses the ARGMAX latent (eval mode) so
    step-by-step == eval-mode encode: the known-dynamics guarantee is preserved.

    Interface matches SurrogateWM exactly (encode_seq/head_preds/predict/step/…).
    """
    def __init__(self, action_size, cfg):
        super().__init__()
        self.embed = nn.Embedding(action_size, cfg.wm_embed)
        self.gru = nn.GRU(cfg.wm_embed, cfg.wm_hidden, cfg.wm_gru_layers,
                          batch_first=True, dropout=0.1 if cfg.wm_gru_layers > 1 else 0.0)
        self.G, self.C = cfg.wm_latent_groups, cfg.wm_latent_classes
        self.to_logits = nn.Linear(cfg.wm_hidden, self.G * self.C)
        self.merge = nn.Sequential(nn.Linear(cfg.wm_hidden + self.G * self.C, cfg.wm_hidden),
                                   nn.LayerNorm(cfg.wm_hidden), nn.SiLU())
        self.heads = nn.ModuleList([_head(cfg.wm_hidden) for _ in range(cfg.wm_ensemble_K)])
        self.hidden, self.layers, self.K = cfg.wm_hidden, cfg.wm_gru_layers, cfg.wm_ensemble_K
        self.feature_dim = cfg.wm_hidden
        self.kl_free_bits = cfg.kl_free_bits
        self._last_kl = torch.zeros(())

    def _latent(self, h):
        """h [...,hidden] -> (feature [...,hidden], kl scalar). Straight-through categorical
        sample in train mode, deterministic argmax in eval mode (imagination)."""
        logits = self.to_logits(h).unflatten(-1, (self.G, self.C))       # [...,G,C]
        probs = torch.softmax(logits, -1)
        if self.training:
            idx = torch.distributions.Categorical(probs=probs).sample()
            z = F.one_hot(idx, self.C).float() + probs - probs.detach()  # straight-through
        else:
            z = F.one_hot(probs.argmax(-1), self.C).float()              # deterministic
        feat = self.merge(torch.cat([h, z.flatten(-2)], -1))
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1)             # [...,G]
        kl = (torch.log(torch.tensor(float(self.C))) - ent).clamp_min(self.kl_free_bits).sum(-1)
        return feat, kl.mean()

    def encode_seq(self, acts):
        h = self.gru(self.embed(acts))[0]
        feat, self._last_kl = self._latent(h)
        return feat

    def head_preds(self, feat):
        return torch.stack([hd(feat).squeeze(-1) for hd in self.heads], dim=0)

    def predict(self, acts):
        feat = self.encode_seq(acts)
        p = self.head_preds(feat)
        return p.mean(0), p.std(0), feat

    def seq_preds(self, acts):
        return self.head_preds(self.encode_seq(acts))

    def init_state(self, B, device):
        return torch.zeros(self.layers, B, self.hidden, device=device)

    def encode_to_state(self, acts):
        out, hstate = self.gru(self.embed(acts))
        feat, _ = self._latent(out[:, -1])
        return feat, hstate

    def step(self, action, hstate):
        out, hstate = self.gru(self.embed(action).unsqueeze(1), hstate)
        feat, _ = self._latent(out.squeeze(1))
        return feat, hstate, self.head_preds(feat)

    def kl_loss(self):
        return self._last_kl

    def denorm(self, x):
        return x

    def normalize(self, y):
        return y

    def update_popart(self, y):
        pass


class EnsembleSurrogateWM(nn.Module):
    """Independent-encoder ensemble + Randomized Prior Functions + PopArt target-norm.

    Fixes three root causes exposed by the 50-run campaign:
      (C) shared-trunk uncertainty collapse -> K INDEPENDENT (embed, GRU, head) members;
          the actor feature is the mean of members' GRU outputs. Real epistemic diversity.
      (C) ensembles agree too much OOD -> RPF (Osband'18): head_k = trainable_k(h) +
          beta * prior_k(h), prior_k a FROZEN random net -> members disagree where data is thin.
      (B) 6q resolution-collapse -> PopArt: predict NORMALIZED log-error (y-mu)/sigma with mu,
          sigma tracking the current buffer distribution, so gradient/imag scale stays alive as
          the good-circuit band narrows. head_preds returns normalized; denorm() maps back.

    Interface matches the other WMs; step() returns per-member preds so imagine.py stays
    architecture-agnostic. Known dynamics preserved (each member's GRU is deterministic).
    """
    def __init__(self, action_size, cfg):
        super().__init__()
        K = cfg.wm_ensemble_K
        self.K, self.hidden, self.layers = K, cfg.wm_hidden, cfg.wm_gru_layers
        self.feature_dim = cfg.wm_hidden
        self.rpf_beta, self.popart, self.pa_m = cfg.rpf_beta, cfg.popart, cfg.popart_momentum
        drop = 0.1 if cfg.wm_gru_layers > 1 else 0.0
        self.embeds = nn.ModuleList([nn.Embedding(action_size, cfg.wm_embed) for _ in range(K)])
        self.grus = nn.ModuleList([nn.GRU(cfg.wm_embed, cfg.wm_hidden, cfg.wm_gru_layers,
                                          batch_first=True, dropout=drop) for _ in range(K)])
        self.heads = nn.ModuleList([_head(cfg.wm_hidden) for _ in range(K)])
        self.priors = nn.ModuleList([_head(cfg.wm_hidden) for _ in range(K)])  # RPF: frozen
        for p in self.priors.parameters():
            p.requires_grad_(False)
        self.register_buffer("pa_mean", torch.zeros(()))
        self.register_buffer("pa_std", torch.ones(()))

    # ---------- PopArt target normalization ----------
    def normalize(self, y):
        return (y - self.pa_mean) / self.pa_std.clamp_min(1e-4) if self.popart else y

    def denorm(self, x):
        return x * self.pa_std.clamp_min(1e-4) + self.pa_mean if self.popart else x

    def update_popart(self, y):
        if not self.popart:
            return
        with torch.no_grad():
            self.pa_mean.mul_(self.pa_m).add_((1 - self.pa_m) * y.mean())
            self.pa_std.mul_(self.pa_m).add_((1 - self.pa_m) * y.std().clamp_min(1e-3))

    # ---------- per-member prediction (normalized log-error) ----------
    def _pred_k(self, k, h):
        return (self.heads[k](h) + self.rpf_beta * self.priors[k](h)).squeeze(-1)

    def head_preds(self, h_all):
        """h_all [K,...,hidden] -> [K,...] normalized per-member log-error."""
        return torch.stack([self._pred_k(k, h_all[k]) for k in range(self.K)], 0)

    # ---------- full-sequence ----------
    def encode_all(self, acts):
        """acts [B,T] -> per-member hidden [K,B,T,hidden]."""
        return torch.stack([self.grus[k](self.embeds[k](acts))[0] for k in range(self.K)], 0)

    def encode_seq(self, acts):
        """actor-facing feature = mean over members -> [B,T,hidden]."""
        return self.encode_all(acts).mean(0)

    def predict(self, acts):
        """-> mean_logerr [B,T] (RAW), disagree [B,T] (RAW std over members), feat [B,T,hidden]."""
        h = self.encode_all(acts)
        pd = self.denorm(self.head_preds(h))
        return pd.mean(0), pd.std(0), h.mean(0)

    def seq_preds(self, acts):
        """[K,B,T] per-member NORMALIZED preds for WM training."""
        return self.head_preds(self.encode_all(acts))

    # ---------- incremental (imagination) ----------
    def init_state(self, B, device):
        return torch.zeros(self.K, self.layers, B, self.hidden, device=device)

    def encode_to_state(self, acts):
        outs, hs = [], []
        for k in range(self.K):
            o, h = self.grus[k](self.embeds[k](acts))
            outs.append(o[:, -1]); hs.append(h)
        return torch.stack(outs, 0).mean(0), torch.stack(hs, 0)

    def step(self, action, hstate):
        """action [B], hstate [K,layers,B,hidden] -> feat [B,hidden] (mean), new hstate,
        per-member NORMALIZED logerr [K,B]."""
        outs, new = [], []
        for k in range(self.K):
            o, h = self.grus[k](self.embeds[k](action).unsqueeze(1), hstate[k])
            outs.append(o.squeeze(1)); new.append(h)
        h_all = torch.stack(outs, 0)                       # [K,B,hidden]
        return h_all.mean(0), torch.stack(new, 0), self.head_preds(h_all)

    def kl_loss(self):
        return torch.zeros((), device=self.pa_mean.device)


def build_wm(action_size, cfg):
    """WM factory. independent_ensemble -> EnsembleSurrogateWM (C+B fixes);
    else legacy shared-trunk gru | rssm."""
    if getattr(cfg, "independent_ensemble", False):
        return EnsembleSurrogateWM(action_size, cfg)
    return RSSMSurrogateWM(action_size, cfg) if cfg.encoder == "rssm" else SurrogateWM(action_size, cfg)


if __name__ == "__main__":
    # smoke: shapes + consistency (seq-encode == step-by-step)
    from config import Config
    cfg = Config()
    wm = SurrogateWM(action_size=24, cfg=cfg)
    wm.eval()                                         # imagination runs in eval (no dropout)
    acts = torch.randint(0, 24, (4, 6))
    with torch.no_grad():
        mean, std, h = wm.predict(acts)
    print("predict:", mean.shape, std.shape, h.shape, "| conf(std) range",
          f"{float(std.min()):.3f}..{float(std.max()):.3f}")
    # consistency: encode full vs step (eval mode -> must match: known-dynamics)
    h_full = wm.encode_seq(acts)                      # [4,6,H]
    hstate = wm.init_state(4, acts.device)
    feats = []
    for t in range(6):
        f, hstate, _ = wm.step(acts[:, t], hstate)
        feats.append(f)
    h_step = torch.stack(feats, 1)                    # [4,6,H]
    err = (h_full - h_step).abs().max().item()
    print(f"seq-vs-step max |Δ| = {err:.2e}  (should be ~0 -> known-dynamics consistent)")
    print(f"params: {sum(p.numel() for p in wm.parameters())/1e6:.2f}M, ensemble K={wm.K}")
