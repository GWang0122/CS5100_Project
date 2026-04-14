"""
Train a model-free reinforcement learning agent (PPO or SAC).

Uses Stable-Baselines3 to train an MLP-policy agent on a Gymnasium
continuous-control environment.  The trained policy is saved as a
.zip checkpoint and training statistics (episode returns, lengths)
are logged via the SB3 Monitor wrapper.

Example usage:
    python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 300000 --seed 1

Output:
    runs/model_free/<algo>_<env>_seed<s>/policy.zip   (trained model)
    runs/model_free/<algo>_<env>_seed<s>/monitor.csv   (episode log)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

# Register Gymnasium-Robotics environments (e.g. FetchReach-v2) if the
# package is installed.  This import adds the envs to the Gymnasium
# registry so that gym.make("FetchReach-v2") works.
try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    gymnasium_robotics = None

# Ensure the project root is on sys.path so that `src.*` imports work
# when running this script directly (i.e. `python scripts/...`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.env_factory import make_env
from src.common.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for environment, algorithm, and training."""
    parser = argparse.ArgumentParser(description="Train model-free baseline (PPO/SAC).")
    parser.add_argument("--env-id", type=str, default="Ant-v4",
                        help="Gymnasium environment ID (default: Ant-v4)")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="sac",
                        help="RL algorithm to use (default: sac)")
    parser.add_argument("--total-timesteps", type=int, default=200_000,
                        help="Total environment interaction steps for training")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-dir", type=str, default="runs/model_free",
                        help="Root directory for run outputs")
    return parser.parse_args()


def build_model(algo: str, env, seed: int, log_dir: str):
    """Instantiate a Stable-Baselines3 model with tuned hyperparameters.

    Both PPO and SAC use MlpPolicy (two hidden layers of 256 units by
    default in SB3) with a shared learning rate and discount factor.

    Args:
        algo:    "ppo" or "sac".
        env:     A Gymnasium environment (wrapped with Monitor).
        seed:    Random seed for SB3 internal RNGs.
        log_dir: Directory for TensorBoard logs.

    Returns:
        A Stable-Baselines3 model ready for .learn().
    """
    common_kwargs = {
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": log_dir,
    }
    if algo == "ppo":
        # PPO with on-policy rollouts of 2048 steps per update.
        # Batch size 256 for mini-batch SGD within each epoch.
        return PPO(
            "MlpPolicy",
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            **common_kwargs,
        )
    # SAC (off-policy) — replay buffer allows sample reuse, making it
    # much more sample-efficient than PPO on continuous control tasks.
    return SAC(
        "MlpPolicy",
        buffer_size=300_000,          # Replay buffer capacity.
        batch_size=256,               # Minibatch size per gradient step.
        learning_rate=3e-4,           # Adam learning rate.
        learning_starts=5_000,        # Random exploration before training.
        train_freq=1,                 # One gradient step per env step.
        gradient_steps=1,             # Number of gradient updates per train_freq.
        gamma=0.99,                   # Discount factor.
        **common_kwargs,
    )


def main() -> None:
    """Entry point: parse args, create env, train model, save checkpoint."""
    args = parse_args()

    # Fix all random seeds for reproducibility.
    set_global_seed(args.seed)

    # Create a descriptive run name (e.g. "sac_Ant-v4_seed1") and ensure
    # the output directory exists.
    run_name = f"{args.algo}_{args.env_id}_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the environment with standard wrappers, then add SB3 Monitor
    # to log per-episode statistics to monitor.csv.
    env = make_env(args.env_id, args.seed)
    env = Monitor(env, str(run_dir))

    # Construct and train the agent.
    model = build_model(args.algo, env, args.seed, str(run_dir))
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name)

    # Persist the trained policy for later evaluation / visualization.
    model_path = run_dir / "policy.zip"
    model.save(str(model_path))
    print(f"Saved model to: {model_path}")

    env.close()


if __name__ == "__main__":
    main()
