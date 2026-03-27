"""
Run SAC, PPO, and/or model-based MPC with the SAME environment-interaction budget
for fair comparison. Each (method, seed) is one subprocess.

Example:
  python scripts/run_matched_experiments.py --env-id Ant-v4 --budget 100000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-budget multi-seed experiment runner.")
    p.add_argument("--env-id", type=str, default="Ant-v4")
    p.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Total env steps: SB3 total_timesteps and MPC total_env_steps.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["sac", "ppo", "mpc"],
        default=["sac", "ppo", "mpc"],
    )
    p.add_argument("--log-dir-mf", type=str, default="runs/model_free")
    p.add_argument("--log-dir-mpc", type=str, default="runs/model_based_mpc")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scripts = {
        "sac": ("train_model_free.py", ["--algo", "sac", "--total-timesteps", str(args.budget)]),
        "ppo": ("train_model_free.py", ["--algo", "ppo", "--total-timesteps", str(args.budget)]),
        "mpc": ("train_model_based_mpc.py", ["--total-env-steps", str(args.budget)]),
    }

    for method in args.methods:
        script_name, extra = scripts[method]
        script_path = PROJECT_ROOT / "scripts" / script_name
        for seed in args.seeds:
            cmd = [
                sys.executable,
                str(script_path),
                "--env-id",
                args.env_id,
                "--seed",
                str(seed),
            ]
            if method in ("sac", "ppo"):
                cmd += extra
                cmd += ["--log-dir", args.log_dir_mf]
            else:
                cmd += extra
                cmd += ["--log-dir", args.log_dir_mpc]

            print("\n" + "=" * 72)
            print("RUN:", " ".join(cmd))
            print("=" * 72 + "\n")
            if args.dry_run:
                continue
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if r.returncode != 0:
                print(f"FAILED (exit {r.returncode}): {method} seed={seed}", file=sys.stderr)
                sys.exit(r.returncode)

    print("\nAll requested runs finished OK.")
    print("Next: python scripts/summarize_runs.py")
    print("      python scripts/plot_learning_curves.py --env-id", args.env_id)


if __name__ == "__main__":
    main()
