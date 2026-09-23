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

    def encode_seq(self, acts):
        return self.gru(self.embed(acts))[0]

    def head_preds(self, h):
        return torch.stack([hd(h).squeeze(-1) for hd in self.heads], dim=0)

    def predict(self, acts):
        h = self.encode_seq(acts)
        p = self.head_preds(h)
        return p.mean(0), p.std(0), h

    def seq_preds(self, acts):
        return self.head_preds(self.encode_seq(acts))

    def init_state(self, B, device):
        return torch.zeros(self.layers, B, self.hidden, device=device)

    def encode_to_state(self, acts):
        out, hstate = self.gru(self.embed(acts))
        return out[:, -1], hstate

    def step(self, action, hstate):
        out, hstate = self.gru(self.embed(action).unsqueeze(1), hstate)
        feat = out.squeeze(1)
        return feat, hstate, self.head_preds(feat)

    def kl_loss(self):
        return torch.zeros((), device=self.embed.weight.device)

    def denorm(self, x):
        return x

    def normalize(self, y):
        return y

    def update_popart(self, y):
        pass


class RSSMSurrogateWM(nn.Module):
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
        logits = self.to_logits(h).unflatten(-1, (self.G, self.C))
        probs = torch.softmax(logits, -1)
        if self.training:
            idx = torch.distributions.Categorical(probs=probs).sample()
            z = F.one_hot(idx, self.C).float() + probs - probs.detach()
        else:
            z = F.one_hot(probs.argmax(-1), self.C).float()
        feat = self.merge(torch.cat([h, z.flatten(-2)], -1))
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
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
        self.priors = nn.ModuleList([_head(cfg.wm_hidden) for _ in range(K)])
        for p in self.priors.parameters():
            p.requires_grad_(False)
        self.register_buffer("pa_mean", torch.zeros(()))
        self.register_buffer("pa_std", torch.ones(()))

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

    def _pred_k(self, k, h):
        return (self.heads[k](h) + self.rpf_beta * self.priors[k](h)).squeeze(-1)

    def head_preds(self, h_all):
        return torch.stack([self._pred_k(k, h_all[k]) for k in range(self.K)], 0)

    def encode_all(self, acts):
        return torch.stack([self.grus[k](self.embeds[k](acts))[0] for k in range(self.K)], 0)

    def encode_seq(self, acts):
        return self.encode_all(acts).mean(0)

    def predict(self, acts):
        h = self.encode_all(acts)
        pd = self.denorm(self.head_preds(h))
        return pd.mean(0), pd.std(0), h.mean(0)

    def seq_preds(self, acts):
        return self.head_preds(self.encode_all(acts))

    def init_state(self, B, device):
        return torch.zeros(self.K, self.layers, B, self.hidden, device=device)

    def encode_to_state(self, acts):
        outs, hs = [], []
        for k in range(self.K):
            o, h = self.grus[k](self.embeds[k](acts))
            outs.append(o[:, -1]); hs.append(h)
        return torch.stack(outs, 0).mean(0), torch.stack(hs, 0)

    def step(self, action, hstate):
        outs, new = [], []
        for k in range(self.K):
            o, h = self.grus[k](self.embeds[k](action).unsqueeze(1), hstate[k])
            outs.append(o.squeeze(1)); new.append(h)
        h_all = torch.stack(outs, 0)
        return h_all.mean(0), torch.stack(new, 0), self.head_preds(h_all)

    def kl_loss(self):
        return torch.zeros((), device=self.pa_mean.device)


def build_wm(action_size, cfg):
    if getattr(cfg, "independent_ensemble", False):
        return EnsembleSurrogateWM(action_size, cfg)
    return RSSMSurrogateWM(action_size, cfg) if cfg.encoder == "rssm" else SurrogateWM(action_size, cfg)


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    wm = SurrogateWM(action_size=24, cfg=cfg)
    wm.eval()
    acts = torch.randint(0, 24, (4, 6))
    with torch.no_grad():
        mean, std, h = wm.predict(acts)
    print("predict:", mean.shape, std.shape, h.shape, "| conf(std) range",
          f"{float(std.min()):.3f}..{float(std.max()):.3f}")
    h_full = wm.encode_seq(acts)
    hstate = wm.init_state(4, acts.device)
    feats = []
    for t in range(6):
        f, hstate, _ = wm.step(acts[:, t], hstate)
        feats.append(f)
    h_step = torch.stack(feats, 1)
    err = (h_full - h_step).abs().max().item()
    print(f"seq-vs-step max |Δ| = {err:.2e}  (should be ~0 -> known-dynamics consistent)")
    print(f"params: {sum(p.numel() for p in wm.parameters())/1e6:.2f}M, ensemble K={wm.K}")
