import gymnasium as gym
import torch

# Needed to register robotics environments
import gymnasium_robotics  # noqa: F401


def run_env_once(env_id: str, max_steps: int = 5) -> None:
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
    # Gymnasium-Robotics versions can differ on the FetchReach suffix.
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
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device[0]: {device_name}")

    run_env_once("Ant-v4")
    fetch_id = resolve_fetch_reach_id()
    run_env_once(fetch_id)
    print("All smoke tests passed.")
