"""
Summarize training run statistics across all experiments.

Scans the runs/ directory tree for completed training logs and prints a
summary table of per-run episode statistics, including total episodes,
mean / median / best episode return, and the mean return over the last
K episodes (default K=10).

Reads:
  - runs/model_free/<algo>_<env>_seedN/monitor.csv     (SB3 Monitor)
  - runs/model_based_mpc/mpc_<env>_seedN/episode_returns.csv

Example:
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --env-id Ant-v4
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse optional filters for the summary scan."""
    p = argparse.ArgumentParser(description="Summarize logged training runs.")
    p.add_argument("--runs-root", type=str, default="runs",
                   help="Root directory containing run folders")
    p.add_argument("--env-id", type=str, default=None,
                   help="If set, only show runs whose name contains this env ID")
    return p.parse_args()


def load_monitor_returns(path: Path) -> list[float]:
    """Extract episode returns from an SB3 Monitor CSV.

    The first two lines are a JSON comment and a header row; data lines
    start at index 2.  Column 0 is the episode reward.
    """
    rows: list[float] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append(float(parts[0]))
    return rows


def load_mpc_returns(path: Path) -> list[float]:
    """Extract episode returns from an MPC episode_returns.csv."""
    out: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(float(row["episode_return"]))
    return out


def summarize_returns(name: str, returns: list[float], last_k: int = 10) -> None:
    """Print key statistics for one training run.

    Args:
        name:    Descriptive run name (used as label in output).
        returns: List of per-episode returns.
        last_k:  Number of most-recent episodes to average for the
                 "last-K mean" metric (indicates final performance).
    """
    if not returns:
        print(f"  {name}: (no data)")
        return
    last = returns[-last_k:] if len(returns) >= last_k else returns
    print(f"  {name}")
    print(
        f"    episodes: {len(returns)}  mean: {statistics.mean(returns):.2f}  "
        f"median: {statistics.median(returns):.2f}  best: {max(returns):.2f}"
    )
    print(f"    last-{len(last)} mean: {statistics.mean(last):.2f}")


def main() -> None:
    """Discover all training runs and print their summaries."""
    args = parse_args()
    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"No directory: {root}")
        return

    mf = root / "model_free"
    mpc_root = root / "model_based_mpc"

    # Model-free runs
    print("=== Model-free (monitor.csv) ===\n")
    if mf.is_dir():
        for d in sorted(mf.iterdir()):
            if not d.is_dir():
                continue
            mon = d / "monitor.csv"
            if not mon.is_file():
                continue
            # Apply optional environment filter.
            if args.env_id and args.env_id not in d.name:
                continue
            rets = load_monitor_returns(mon)
            summarize_returns(d.name, rets)

    # Model-based MPC runs
    print("\n=== Model-based MPC (episode_returns.csv) ===\n")
    if mpc_root.is_dir():
        for d in sorted(mpc_root.iterdir()):
            if not d.is_dir():
                continue
            csvp = d / "episode_returns.csv"
            if not csvp.is_file():
                continue
            if args.env_id and args.env_id not in d.name:
                continue
            rets = load_mpc_returns(csvp)
            summarize_returns(d.name, rets)


if __name__ == "__main__":
    main()
