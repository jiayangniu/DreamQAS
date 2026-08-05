"""V2 WM-QAS Policy.

WMDiscreteActor: pure discrete gate selection policy on RSSM features (h, z).
    - Shared between real episode collection and imagination training.
    - Imagination gradients directly update the gate selection used in real episodes.
    - No continuous angle parameter: VQE handles angle optimization independently.

WMRealPolicy: wrapper for collecting real episodes.
    - Maintains RSSM state (h, z) within each episode via observe_step (posterior).
    - Handles illegal action masking.
    - Returns (h, z) states per step for downstream REINFORCE updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from world_model.networks import MLP


class WMDiscreteActor(nn.Module):
    """Discrete gate selection policy on RSSM features.

    Shared between WMRealPolicy (real episodes) and ImagTrainer (imagination).
    Gradients from both paths update the same parameters.
    """

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
        """Return action distribution.

        Args:
            features: [B, feature_dim]
            ill_mask: [B, action_size] bool, True = illegal → masked to -1e9
        """
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
        """REINFORCE loss (used by both real and imagination update paths).

        Args:
            features:  [T, feature_dim]
            actions:   [T] integer actions
            returns:   [T] normalized discounted returns
            ill_masks: [T, action_size] bool, True = illegal (optional)
            weights:   [T] optional per-step loss weights. Imagination passes the
                       DreamerV3 cumulative continuation discount so steps past a
                       predicted termination contribute ~0 gradient. None → mean.
        """
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
    """Collects real episodes using WMDiscreteActor + RSSM posterior.

    Maintains (h, z) RSSM state across episode steps via observe_step.
    The stored (h, z) features are used for real REINFORCE after each episode.

    The WMDiscreteActor is SHARED with ImagTrainer — imagination gradients
    directly update the same network used here for real gate selection.
    """

    def __init__(
        self,
        rssm: nn.Module,
        actor: WMDiscreteActor,
        translate: dict,
        device: torch.device,
    ):
        """
        Args:
            rssm:      RSSM world model (shared with WorldModelTrainer)
            actor:     WMDiscreteActor (shared with ImagTrainer)
            translate: {action_idx: [ctrl, targ_offset, rot_qubit, rot_axis]}
            device:    torch device
        """
        self.rssm = rssm
        self.actor = actor
        self.translate = translate
        self.device = device
        self._state: dict | None = None
        self._prev_action: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset RSSM state at the start of a new episode."""
        self._state = self.rssm.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, dtype=torch.long, device=self.device)

    def act(
        self,
        obs: torch.Tensor,
        ill_actions: list[int] | None = None,
        greedy: bool = False,
    ) -> tuple:
        """Select one action.

        Args:
            obs:         [obs_dim] structure-only circuit tensor (on any device)
            ill_actions: list of illegal action indices from env.illegal_action_new()
            greedy:      if True, take argmax

        Returns:
            env_action:  list [ctrl, targ_offset, rot_qubit, rot_axis] for env.step()
            action_idx:  int for buffer storage and REINFORCE
            h:           [H] detached deterministic state
            z:           [Z] detached stochastic state
        """
        obs_t = obs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.rssm.observe_step(self._state, self._prev_action, obs_t)

        h = result["h"].detach()  # [1, H]
        z = result["z"].detach()  # [1, Z]
        self._state = {"h": h, "z": z}
        # Actor features = [h, z] always, plus symlog-space potential_head([h,z]) when
        # M2 enabled. M2 logic lives in rssm.get_actor_features so it stays consistent
        # across WMRealPolicy / imagination / REINFORCE.
        features = self.rssm.get_actor_features(h, z)  # [1, actor_feature_dim]

        ill_mask = None
        if ill_actions:
            ill_mask = torch.zeros(1, self.actor.action_size, dtype=torch.bool, device=self.device)
            ill_mask[0, ill_actions] = True

        with torch.no_grad():
            dist = self.actor(features, ill_mask)
            action_t = dist.mode if greedy else dist.sample()  # [1]

        action_idx = int(action_t.item())
        env_action = self.translate[action_idx]
        self._prev_action = action_t

        return env_action, action_idx, h[0], z[0]
