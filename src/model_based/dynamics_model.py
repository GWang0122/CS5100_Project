from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsMLP(nn.Module):
    """
    Predicts state delta and reward:
      input  = [s_t, a_t]
      output = [delta_s, r_t]
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = obs_dim + act_dim
        out_dim = obs_dim + 1

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, actions], dim=-1)
        pred = self.net(x)
        delta = pred[..., : self.obs_dim]
        reward = pred[..., self.obs_dim :]
        return delta, reward

    def predict_next_obs_reward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, reward = self.forward(obs, actions)
        next_obs = obs + delta
        return next_obs, reward


def train_dynamics_step(
    model: DynamicsMLP,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> float:
    obs = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], device=device, dtype=torch.float32)
    next_obs = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
    rewards = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32)

    pred_delta, pred_reward = model(obs, actions)
    target_delta = next_obs - obs

    loss_delta = F.mse_loss(pred_delta, target_delta)
    loss_reward = F.mse_loss(pred_reward, rewards)
    loss = loss_delta + loss_reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
