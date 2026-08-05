"""Building blocks for DreamerV3-style RSSM world model.

Components:
    - symlog / symexp: symmetric log/exp transforms (DreamerV3)
    - MLP: configurable multi-layer perceptron with LayerNorm
    - BlockGRU: GRU with optional layer normalization (DreamerV3 uses this)
    - TwohotDist: two-hot encoded scalar distribution for reward/value prediction
    - CategoricalLatent: discrete categorical latent variable (z_t)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


# ── Symmetric log/exp (DreamerV3 §3) ────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symexp(x) = sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ── MLP with LayerNorm ──────────────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activation and LayerNorm (DreamerV3 style)."""

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


# ── BlockGRU (DreamerV3 §A) ──────────────────────────────────────────────────

class BlockGRU(nn.Module):
    """GRU with optional LayerNorm on the hidden state.

    DreamerV3 applies LayerNorm to the GRU output to stabilize training
    across the long imagination horizon.
    """

    def __init__(self, input_dim: int, hidden_dim: int, norm: bool = True):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
            h: [B, hidden_dim]
        Returns:
            h_next: [B, hidden_dim]
        """
        h_next = self.gru(x, h)
        h_next = self.norm(h_next)
        return h_next


# ── Categorical latent variable (z_t) ────────────────────────────────────────

class CategoricalLatent(nn.Module):
    """Discrete categorical latent for RSSM posterior/prior.

    z_t is represented as num_categories independent categorical distributions,
    each with num_classes outcomes. The sample is one-hot per category,
    concatenated to form a flat vector of size num_categories * num_classes.

    Uses straight-through gradient estimator for backpropagation.
    """

    def __init__(self, num_categories: int = 32, num_classes: int = 32,
                 unimix: float = 0.01):
        super().__init__()
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.flat_dim = num_categories * num_classes
        # DreamerV3 unimix: blend 1% uniform into the categorical to (a) bound KL
        # (prevents inf/NaN when a class probability underflows to exactly 0 in
        # float32) and (b) keep straight-through gradients alive (no exact one-hot
        # saturation that would zero the softmax gradient).
        self.unimix = unimix

    def _dist(self, logits: torch.Tensor) -> OneHotCategorical:
        """Build the (unimixed) categorical distribution from flat logits [B, C*K]."""
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
        """
        Args:
            logits: [B, num_categories * num_classes]
        Returns:
            z: [B, num_categories * num_classes]  one-hot (straight-through)
            dist: distribution for KL computation
        """
        B = logits.shape[0]
        dist = self._dist(logits)

        # Straight-through: sample one-hot, but gradient flows through logits
        sample = dist.sample()                    # [B, C, K] one-hot
        z = sample + dist.probs - dist.probs.detach()  # straight-through
        z = z.reshape(B, -1)                      # [B, C*K]
        return z, dist

    def kl_divergence(
        self,
        post_logits: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL(posterior || prior) per batch element.

        Args:
            post_logits:  [B, C*K]
            prior_logits: [B, C*K]
        Returns:
            kl: [B]  sum over categories
        """
        post = self._dist(post_logits)
        prior = self._dist(prior_logits)

        # KL per category, summed over categories
        kl = torch.distributions.kl_divergence(post, prior)  # [B, C]
        return kl.sum(-1)  # [B]

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-datapoint categorical entropy, summed over categories. [B]

        Diagnostic for latent utilization: near 0 = collapsed to one-hot (bad),
        near ln(num_classes)*num_categories = uniform (unused). Reads the (unimixed)
        distribution so it reflects what training/imagination actually sees.
        """
        return self._dist(logits).entropy().sum(-1)


# ── Twohot distribution for reward/value (DreamerV3 §B) ─────────────────────

class TwohotDist(nn.Module):
    """Twohot-encoded scalar distribution.

    Discretizes a continuous scalar into a set of bins and predicts
    softmax probabilities over bins. The predicted value is the expected
    value under this distribution (weighted sum of bin centers).

    Supports symlog/symexp transforms for the bin edges.
    """

    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.num_bins = num_bins
        # Bin edges in symlog space
        bins = torch.linspace(low, high, num_bins)
        self.register_buffer("bins", bins)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Predicted mean (symexp of expected bin value).

        Args:
            logits: [*, num_bins]
        Returns:
            mean: [*]
        """
        probs = F.softmax(logits, dim=-1)
        mean_symlog = (probs * self.bins).sum(-1)
        return symexp(mean_symlog)

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Log probability of target under twohot distribution.

        Args:
            logits: [*, num_bins]
            target: [*]  scalar targets
        Returns:
            log_prob: [*]
        """
        target_symlog = symlog(target)
        # Find the two bins that bracket the target
        below = (self.bins <= target_symlog.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        # Interpolation weight
        b_val = self.bins[below]
        a_val = self.bins[above]
        weight = (target_symlog - b_val) / (a_val - b_val + 1e-8)
        weight = weight.clamp(0.0, 1.0)

        # Twohot target: [0, ..., 1-w, w, ..., 0]
        target_probs = torch.zeros_like(logits)
        target_probs.scatter_(-1, below.unsqueeze(-1), (1.0 - weight).unsqueeze(-1))
        target_probs.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))

        log_probs = F.log_softmax(logits, dim=-1)
        return (target_probs * log_probs).sum(-1)
