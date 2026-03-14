from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

# Registers Fetch/robotics envs when available.
try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    gymnasium_robotics = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.env_factory import make_env
from src.common.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model-free baseline (PPO/SAC).")
    parser.add_argument("--env-id", type=str, default="Ant-v4")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="sac")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs/model_free")
    return parser.parse_args()


def build_model(algo: str, env, seed: int, log_dir: str):
    common_kwargs = {
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": log_dir,
    }
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            **common_kwargs,
        )
    return SAC(
        "MlpPolicy",
        buffer_size=300_000,
        batch_size=256,
        learning_rate=3e-4,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        **common_kwargs,
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_name = f"{args.algo}_{args.env_id}_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, args.seed)
    env = Monitor(env, str(run_dir))

    model = build_model(args.algo, env, args.seed, str(run_dir))
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name)

    model_path = run_dir / "policy.zip"
    model.save(str(model_path))
    print(f"Saved model to: {model_path}")

    env.close()


if __name__ == "__main__":
    main()
