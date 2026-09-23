from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class MLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        act: str = "silu",
        norm: bool = True,
        out_act: bool = False,
    ):
        super().__init__()
        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "elu": nn.ELU}[act]

        layers = []
        d_in = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        if out_act:
            if norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(act_fn())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockGRU(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, norm: bool = True):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_next = self.gru(x, h)
        h_next = self.norm(h_next)
        return h_next


class CategoricalLatent(nn.Module):

    def __init__(self, num_categories: int = 32, num_classes: int = 32,
                 unimix: float = 0.01):
        super().__init__()
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.flat_dim = num_categories * num_classes
        self.unimix = unimix

    def _dist(self, logits: torch.Tensor) -> OneHotCategorical:
        B = logits.shape[0]
        logits = logits.reshape(B, self.num_categories, self.num_classes)
        if self.unimix > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = (1.0 - self.unimix) * probs + self.unimix / self.num_classes
            return OneHotCategorical(probs=probs)
        return OneHotCategorical(logits=logits)

    def forward(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        B = logits.shape[0]
        dist = self._dist(logits)

        sample = dist.sample()
        z = sample + dist.probs - dist.probs.detach()
        z = z.reshape(B, -1)
        return z, dist

    def kl_divergence(
        self,
        post_logits: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        post = self._dist(post_logits)
        prior = self._dist(prior_logits)

        kl = torch.distributions.kl_divergence(post, prior)
        return kl.sum(-1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        return self._dist(logits).entropy().sum(-1)


class TwohotDist(nn.Module):

    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.num_bins = num_bins
        bins = torch.linspace(low, high, num_bins)
        self.register_buffer("bins", bins)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        mean_symlog = (probs * self.bins).sum(-1)
        return symexp(mean_symlog)

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_symlog = symlog(target)
        below = (self.bins <= target_symlog.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        b_val = self.bins[below]
        a_val = self.bins[above]
        weight = (target_symlog - b_val) / (a_val - b_val + 1e-8)
        weight = weight.clamp(0.0, 1.0)

        target_probs = torch.zeros_like(logits)
        target_probs.scatter_(-1, below.unsqueeze(-1), (1.0 - weight).unsqueeze(-1))
        target_probs.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))

        log_probs = F.log_softmax(logits, dim=-1)
        return (target_probs * log_probs).sum(-1)
