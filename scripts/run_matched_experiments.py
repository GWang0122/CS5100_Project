"""
Run matched-budget, multi-seed experiments for fair comparison.

Launches training for SAC, PPO, and/or MPC on a given environment,
ensuring every method receives the exact same number of environment
interactions (the --budget flag).  Each (method, seed) combination
runs as a subprocess so that crashes in one run do not prevent
subsequent runs from starting.

The script calls:
  - scripts/train_model_free.py      for SAC and PPO
  - scripts/train_model_based_mpc.py for MPC

Examples:
    python scripts/run_matched_experiments.py --env-id Ant-v4 --budget 300000 --seeds 1 2
    python scripts/run_matched_experiments.py --env-id FetchReach-v2 --budget 150000 --seeds 1 2 --methods sac mpc
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse experiment configuration: env, budget, seeds, and methods."""
    p = argparse.ArgumentParser(
        description="Matched-budget multi-seed experiment runner."
    )
    p.add_argument("--env-id", type=str, default="Ant-v4",
                   help="Gymnasium environment ID")
    p.add_argument("--budget", type=int, default=100_000,
                   help="Total env steps shared by all methods "
                        "(maps to --total-timesteps for SB3, "
                        "--total-env-steps for MPC)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="List of random seeds to run")
    p.add_argument("--methods", type=str, nargs="+",
                   choices=["sac", "ppo", "mpc"],
                   default=["sac", "ppo", "mpc"],
                   help="Which methods to include in the sweep")
    p.add_argument("--log-dir-mf", type=str, default="runs/model_free",
                   help="Output dir for model-free runs")
    p.add_argument("--log-dir-mpc", type=str, default="runs/model_based_mpc",
                   help="Output dir for model-based MPC runs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    return p.parse_args()


def main() -> None:
    """Build and execute the subprocess commands for each (method, seed)."""
    args = parse_args()

    # Map each method name to its training script and method-specific args.
    scripts = {
        "sac": (
            "train_model_free.py",
            ["--algo", "sac", "--total-timesteps", str(args.budget)],
        ),
        "ppo": (
            "train_model_free.py",
            ["--algo", "ppo", "--total-timesteps", str(args.budget)],
        ),
        "mpc": (
            "train_model_based_mpc.py",
            ["--total-env-steps", str(args.budget)],
        ),
    }

    for method in args.methods:
        script_name, extra = scripts[method]
        script_path = PROJECT_ROOT / "scripts" / script_name

        for seed in args.seeds:
            # Construct the full command to invoke as a subprocess.
            cmd = [
                sys.executable,
                str(script_path),
                "--env-id", args.env_id,
                "--seed", str(seed),
            ]
            # Append method-specific flags and the appropriate log dir.
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

            # Execute the training subprocess; abort the sweep on failure.
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if r.returncode != 0:
                print(
                    f"FAILED (exit {r.returncode}): {method} seed={seed}",
                    file=sys.stderr,
                )
                sys.exit(r.returncode)

    print("\nAll requested runs finished OK.")
    print("Next: python scripts/summarize_runs.py")
    print("      python scripts/plot_learning_curves.py --env-id", args.env_id)


if __name__ == "__main__":
    main()
