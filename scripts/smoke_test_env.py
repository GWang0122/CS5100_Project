"""
Quick sanity-check script for the project environment.

Verifies that:
  - Ant-v4 and FetchReach-v2 (or the highest available version) can
    be created, reset, and stepped without errors.
  - PyTorch is installed and reports the expected CUDA availability.

Run this after setting up the Conda environment to confirm everything
is working before launching training experiments.

Example:
    python scripts/smoke_test_env.py
"""

import gymnasium as gym
import torch

# Import gymnasium-robotics so that FetchReach-v2 etc. are registered.
import gymnasium_robotics  # noqa: F401


def run_env_once(env_id: str, max_steps: int = 5) -> None:
    """Create an environment, reset it, and run a few random steps.

    Args:
        env_id:    Gymnasium environment ID to test.
        max_steps: Number of random-action steps to execute.
    """
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    print(f"[OK] reset {env_id}: obs type={type(obs).__name__}")

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"[OK] stepped {env_id} for {max_steps} steps")


def resolve_fetch_reach_id() -> str:
    """Find the highest available FetchReach version in the registry.

    Different gymnasium-robotics releases may register v1, v2, or v3.
    We try in descending order and return the first match.

    Returns:
        The environment ID string (e.g. "FetchReach-v2").

    Raises:
        RuntimeError: If no FetchReach environment is found.
    """
    candidates = ["FetchReach-v3", "FetchReach-v2", "FetchReach-v1"]
    available = {spec.id for spec in gym.envs.registry.values()}
    for env_id in candidates:
        if env_id in available:
            return env_id
    raise RuntimeError(
        "No FetchReach environment found. Installed env ids did not include "
        "FetchReach-v3/v2/v1."
    )


if __name__ == "__main__":
    print("Running environment smoke tests...")

    # Report PyTorch and CUDA info for hardware verification.
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device[0]: {device_name}")

    # Test each target environment.
    run_env_once("Ant-v4")
    fetch_id = resolve_fetch_reach_id()
    run_env_once(fetch_id)
    print("All smoke tests passed.")
