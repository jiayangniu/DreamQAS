"""Baseline policy for WM-QAS comparison.

Plain MLP: raw obs → discrete action distribution.
No RSSM, no world model. Used as the no-WM baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from world_model.networks import MLP


class BaselineDiscreteActor(nn.Module):
    """Discrete gate selection policy on raw circuit observations.

    Same MLP architecture as WMDiscreteActor but takes raw obs as input
    instead of RSSM (h, z) features.
    """

    def __init__(
        self,
        obs_dim: int,
        action_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        entropy_coef: float = 1e-3,
    ):
        super().__init__()
        self.action_size = action_size
        self.entropy_coef = entropy_coef
        self.net = MLP(obs_dim, action_size, hidden_dim, num_layers)

    def forward(
        self,
        obs: torch.Tensor,
        ill_mask: torch.Tensor | None = None,
    ) -> Categorical:
        logits = self.net(obs)
        if ill_mask is not None:
            logits = logits.masked_fill(ill_mask, -1e9)
        return Categorical(logits=logits)

    def reinforce_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        ill_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.forward(obs, ill_masks)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return -(log_prob * returns).mean() - self.entropy_coef * entropy.mean()


class BaselinePolicy:
    """Collects real episodes using BaselineDiscreteActor on raw obs."""

    def __init__(
        self,
        actor: BaselineDiscreteActor,
        translate: dict,
        device: torch.device,
    ):
        self.actor = actor
        self.translate = translate
        self.device = device

    def act(
        self,
        obs: torch.Tensor,
        ill_actions: list[int] | None = None,
    ) -> tuple:
        """Select one action from raw obs.

        Returns:
            env_action: list for env.step()
            action_idx: int for REINFORCE
        """
        obs_t = obs.unsqueeze(0).to(self.device)

        ill_mask = None
        if ill_actions:
            ill_mask = torch.zeros(1, self.actor.action_size, dtype=torch.bool, device=self.device)
            ill_mask[0, ill_actions] = True

        with torch.no_grad():
            dist = self.actor(obs_t, ill_mask)
            action_t = dist.sample()  # [1]

        action_idx = int(action_t.item())
        env_action = self.translate[action_idx]
        return env_action, action_idx
