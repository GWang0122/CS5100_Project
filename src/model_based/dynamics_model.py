"""
Learned MLP dynamics model for model-based RL.

The model takes (state, action) pairs and predicts:
  - The *change* in state (delta_s = s_{t+1} - s_t), and
  - The scalar reward r_t.

Predicting deltas rather than absolute next states is standard practice
because consecutive states tend to differ by a small amount, making the
regression target easier to fit and reducing compounding error during
multi-step rollouts (Nagabandi et al., 2018).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsMLP(nn.Module):
    """Two-hidden-layer MLP that predicts (state_delta, reward).

    Architecture:
        [obs_dim + act_dim] -> Linear(hidden_dim) -> ReLU
                            -> Linear(hidden_dim) -> ReLU
                            -> Linear(obs_dim + 1)

    The final output is split into:
        - first obs_dim values  -> predicted state delta
        - last 1 value          -> predicted reward

    Args:
        obs_dim:    Dimensionality of the observation / state vector.
        act_dim:    Dimensionality of the action vector.
        hidden_dim: Width of each hidden layer (default 256).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = obs_dim + act_dim       # input is [s_t ; a_t]
        out_dim = obs_dim + 1            # output is [delta_s ; r_t]

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.obs_dim = obs_dim

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict state delta and reward from (obs, action).

        Args:
            obs:     Batch of observations, shape (B, obs_dim).
            actions: Batch of actions,      shape (B, act_dim).

        Returns:
            delta:  Predicted change in state, shape (B, obs_dim).
            reward: Predicted reward,          shape (B, 1).
        """
        x = torch.cat([obs, actions], dim=-1)
        pred = self.net(x)
        delta = pred[..., : self.obs_dim]
        reward = pred[..., self.obs_dim :]
        return delta, reward

    def predict_next_obs_reward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience method: predict next observation and reward.

        Adds the predicted delta to the current observation to obtain
        the predicted next state.

        Returns:
            next_obs: Predicted next observation, shape (B, obs_dim).
            reward:   Predicted reward,            shape (B, 1).
        """
        delta, reward = self.forward(obs, actions)
        next_obs = obs + delta  # s_{t+1} = s_t + delta
        return next_obs, reward


def train_dynamics_step(
    model: DynamicsMLP,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> float:
    """Perform one gradient step on the dynamics model.

    The loss is the sum of:
      - MSE between predicted and actual state deltas
      - MSE between predicted and actual rewards

    Args:
        model:     The dynamics MLP to update.
        optimizer: PyTorch optimizer (e.g. Adam) attached to model params.
        batch:     Dict with keys "obs", "actions", "next_obs", "rewards",
                   each a NumPy array of shape (batch_size, dim).
        device:    Torch device (cpu or cuda) for tensor placement.

    Returns:
        Scalar loss value (float) for logging.
    """
    # Move batch data to tensors on the target device.
    obs = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], device=device, dtype=torch.float32)
    next_obs = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
    rewards = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32)

    # Forward pass: predict delta and reward.
    pred_delta, pred_reward = model(obs, actions)

    # Regression target for the delta head is the actual state change.
    target_delta = next_obs - obs

    # Combined MSE loss on state deltas and rewards.
    loss_delta = F.mse_loss(pred_delta, target_delta)
    loss_reward = F.mse_loss(pred_reward, rewards)
    loss = loss_delta + loss_reward

    # Standard PyTorch backward pass + optimizer step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
