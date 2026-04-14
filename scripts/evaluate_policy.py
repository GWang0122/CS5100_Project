"""
Evaluate a trained Stable-Baselines3 policy on an environment.

Loads a saved .zip checkpoint (SAC or PPO) and runs it for a specified
number of episodes, printing per-episode returns and lengths, followed
by aggregated statistics.

Example:
    python scripts/evaluate_policy.py --algo sac \
        --model runs/model_free/sac_Ant-v4_seed1/policy.zip \
        --env-id Ant-v4 --n-episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register robotics environments if available.
try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    pass

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from src.common.env_factory import make_env
from src.common.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse arguments for model path, environment, and eval settings."""
    p = argparse.ArgumentParser(description="Evaluate SB3 policy checkpoint.")
    p.add_argument("--model", type=str, required=True,
                   help="Path to the saved policy.zip file")
    p.add_argument("--env-id", type=str, default="Ant-v4",
                   help="Gymnasium environment ID for evaluation")
    p.add_argument("--algo", type=str, choices=["sac", "ppo"], required=True,
                   help="Algorithm used to train the policy (needed for loading)")
    p.add_argument("--n-episodes", type=int, default=10,
                   help="Number of evaluation episodes to run")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed (each episode uses seed + ep_idx)")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic (greedy) action selection")
    return p.parse_args()


def load_model(path: str, algo: str, env) -> BaseAlgorithm:
    """Load a Stable-Baselines3 checkpoint from disk.

    Args:
        path: Filesystem path to the .zip file.
        algo: "sac" or "ppo" -- determines which SB3 class to use.
        env:  A Gymnasium environment (attached for shape validation).

    Returns:
        The loaded SB3 model ready for .predict().
    """
    if algo == "sac":
        return SAC.load(path, env=env)
    return PPO.load(path, env=env)


def main() -> None:
    """Load the policy and run evaluation episodes."""
    args = parse_args()
    set_global_seed(args.seed)

    env = make_env(args.env_id, args.seed)
    model = load_model(args.model, args.algo, env)

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(args.n_episodes):
        # Use a different seed per episode so each rollout starts from
        # a distinct initial state.
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            steps += 1
            done = term or trunc

        returns.append(total)
        lengths.append(steps)
        print(f"Episode {ep + 1}: return={total:.3f} length={steps}")

    env.close()

    # Print aggregate evaluation statistics.
    mean_r = sum(returns) / len(returns)
    mean_l = sum(lengths) / len(lengths)
    print(
        f"\nMean return: {mean_r:.3f}  Mean length: {mean_l:.1f}  "
        f"over {args.n_episodes} episodes"
    )


if __name__ == "__main__":
    main()
