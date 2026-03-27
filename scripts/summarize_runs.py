"""
Scan runs/ and print mean/median/best episode return per run (for progress tables).

Reads:
  - runs/model_free/<algo>_<env>_seedN/monitor.csv  (SB3 Monitor)
  - runs/model_based_mpc/mpc_<env>_seedN/episode_returns.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize logged training runs.")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--env-id", type=str, default=None, help="Filter e.g. Ant-v4")
    return p.parse_args()


def load_monitor_returns(path: Path) -> list[float]:
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
    out: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(float(row["episode_return"]))
    return out


def summarize_returns(name: str, returns: list[float], last_k: int = 10) -> None:
    if not returns:
        print(f"  {name}: (no data)")
        return
    last = returns[-last_k:] if len(returns) >= last_k else returns
    print(f"  {name}")
    print(f"    episodes: {len(returns)}  mean: {statistics.mean(returns):.2f}  median: {statistics.median(returns):.2f}  best: {max(returns):.2f}")
    print(f"    last-{len(last)} mean: {statistics.mean(last):.2f}")


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"No directory: {root}")
        return

    mf = root / "model_free"
    mpc_root = root / "model_based_mpc"

    print("=== Model-free (monitor.csv) ===\n")
    if mf.is_dir():
        for d in sorted(mf.iterdir()):
            if not d.is_dir():
                continue
            mon = d / "monitor.csv"
            if not mon.is_file():
                continue
            if args.env_id and args.env_id not in d.name:
                continue
            rets = load_monitor_returns(mon)
            summarize_returns(d.name, rets)

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
