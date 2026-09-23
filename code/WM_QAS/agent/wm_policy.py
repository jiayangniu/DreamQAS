from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from world_model.networks import MLP


class WMDiscreteActor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        action_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        entropy_coef: float = 1e-3,
    ):
        super().__init__()
        self.action_size = action_size
        self.entropy_coef = entropy_coef
        self.net = MLP(feature_dim, action_size, hidden_dim, num_layers)

    def forward(
        self,
        features: torch.Tensor,
        ill_mask: torch.Tensor | None = None,
    ) -> Categorical:
        logits = self.net(features)
        if ill_mask is not None:
            logits = logits.masked_fill(ill_mask, -1e9)
        return Categorical(logits=logits)

    def reinforce_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        ill_masks: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.forward(features, ill_masks)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if weights is not None:
            denom = weights.sum().clamp(min=1e-8)
            pg = -(log_prob * returns * weights).sum() / denom
            ent = (entropy * weights).sum() / denom
            return pg - self.entropy_coef * ent
        return -(log_prob * returns).mean() - self.entropy_coef * entropy.mean()


class WMRealPolicy:

    def __init__(
        self,
        rssm: nn.Module,
        actor: WMDiscreteActor,
        translate: dict,
        device: torch.device,
    ):
        self.rssm = rssm
        self.actor = actor
        self.translate = translate
        self.device = device
        self._state: dict | None = None
        self._prev_action: torch.Tensor | None = None

    def reset(self) -> None:
        self._state = self.rssm.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, dtype=torch.long, device=self.device)

    def act(
        self,
        obs: torch.Tensor,
        ill_actions: list[int] | None = None,
        greedy: bool = False,
    ) -> tuple:
        obs_t = obs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.rssm.observe_step(self._state, self._prev_action, obs_t)

        h = result["h"].detach()
        z = result["z"].detach()
        self._state = {"h": h, "z": z}
        features = self.rssm.get_actor_features(h, z)

        ill_mask = None
        if ill_actions:
            ill_mask = torch.zeros(1, self.actor.action_size, dtype=torch.bool, device=self.device)
            ill_mask[0, ill_actions] = True

        with torch.no_grad():
            dist = self.actor(features, ill_mask)
            action_t = dist.mode if greedy else dist.sample()

        action_idx = int(action_t.item())
        env_action = self.translate[action_idx]
        self._prev_action = action_t

        return env_action, action_idx, h[0], z[0]
