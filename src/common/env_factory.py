"""
Environment construction helpers.

Provides a single entry point for creating Gymnasium environments with
consistent wrappers and seeding.  Used by both the model-free and
model-based training scripts so that every experiment starts from an
identically configured environment.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create, wrap, and seed a Gymnasium environment.

    Wrappers applied (in order):
      1. RecordEpisodeStatistics -- tracks per-episode return and length
         so that training scripts can log performance without manual
         bookkeeping.
      2. FlattenObservation -- applied only when the observation space is
         a Dict (e.g. FetchReach-v2 returns {"observation", "achieved_goal",
         "desired_goal"}).  Flattening concatenates all values into a
         single 1-D array so that standard MLP policies can consume it.

    Args:
        env_id: Gymnasium environment identifier, e.g. "Ant-v4".
        seed:   Random seed for the environment reset and action space.

    Returns:
        A wrapped, seeded Gymnasium environment ready for interaction.
    """
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)

    # Goal-conditioned envs (e.g. FetchReach) use Dict obs spaces;
    # flatten them to a single vector for MLP compatibility.
    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    # Seed both the environment state and the action sampler so that
    # the initial observation and any random actions are reproducible.
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
