from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
