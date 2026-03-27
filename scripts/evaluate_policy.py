"""
Evaluate a trained Stable-Baselines3 policy (.zip) on an environment.

Example:
  python scripts/evaluate_policy.py --algo sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4 --n-episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    pass

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from src.common.env_factory import make_env
from src.common.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SB3 policy checkpoint.")
    p.add_argument("--model", type=str, required=True, help="Path to policy.zip")
    p.add_argument("--env-id", type=str, default="Ant-v4")
    p.add_argument("--algo", type=str, choices=["sac", "ppo"], required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def load_model(path: str, algo: str, env) -> BaseAlgorithm:
    if algo == "sac":
        return SAC.load(path, env=env)
    return PPO.load(path, env=env)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env = make_env(args.env_id, args.seed)
    model = load_model(args.model, args.algo, env)

    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(args.n_episodes):
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
    mean_r = sum(returns) / len(returns)
    mean_l = sum(lengths) / len(lengths)
    print(f"\nMean return: {mean_r:.3f}  Mean length: {mean_l:.1f}  over {args.n_episodes} episodes")


if __name__ == "__main__":
    main()
