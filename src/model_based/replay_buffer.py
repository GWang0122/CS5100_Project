"""
Fixed-capacity replay buffer for storing environment transitions.

Stores (s, a, r, s', done) tuples in pre-allocated NumPy arrays for
memory efficiency.  When the buffer is full, the oldest transitions are
overwritten in a circular fashion.  Uniform random sampling is used
when the dynamics model requests a training minibatch.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Ring-buffer that stores transitions for dynamics model training.

    Args:
        obs_dim:  Dimensionality of observation vectors.
        act_dim:  Dimensionality of action vectors.
        capacity: Maximum number of transitions to store.  Once full,
                  new transitions overwrite the oldest ones.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.capacity = capacity

        # Pre-allocate contiguous arrays for fast indexing and sampling.
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0    # Next write position (circular).
        self.size = 0   # Number of valid transitions currently stored.

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single (s, a, r, s', done) transition.

        Args:
            obs:      Current observation.
            action:   Action taken.
            reward:   Reward received.
            next_obs: Next observation after taking the action.
            done:     Whether the episode ended (terminated or truncated).
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        # Advance the write pointer, wrapping around at capacity.
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a uniformly random minibatch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys "obs", "actions", "next_obs", "rewards",
            "dones", each a NumPy array of shape (batch_size, dim).
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "next_obs": self.next_obs[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }
