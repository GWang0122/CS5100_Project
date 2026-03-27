"""
Plot learning curves from SB3 monitor.csv and MPC episode_returns.csv.

Example:
  python scripts/plot_learning_curves.py --env-id Ant-v4 --out figures/ant_learning.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot episode returns vs training progress.")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--env-id", type=str, required=True, help="e.g. Ant-v4")
    p.add_argument("--out", type=str, default=None, help="PNG path (default: figures/<env>_learning.png)")
    p.add_argument("--rolling", type=int, default=5, help="Rolling mean window for SB3 episodes.")
    return p.parse_args()


def load_monitor(path: Path) -> tuple[list[int], list[float]]:
    """Returns cumulative timesteps at episode end, episode returns."""
    ts: list[int] = []
    rs: list[float] = []
    cum = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            r = float(parts[0])
            length = int(float(parts[1]))
            cum += length
            ts.append(cum)
            rs.append(r)
    return ts, rs


def load_mpc_csv(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    rets: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["env_step"]))
            rets.append(float(row["episode_return"]))
    return steps, rets


def rolling_mean(xs: list[float], w: int) -> list[float]:
    if w <= 1 or len(xs) < w:
        return xs
    out = []
    for i in range(len(xs)):
        start = max(0, i - w + 1)
        chunk = xs[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: conda install matplotlib -y", file=sys.stderr)
        sys.exit(1)

    root = Path(args.runs_root)
    env_tag = args.env_id.replace("/", "_")
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "figures" / f"{env_tag}_learning.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    mf = root / "model_free"
    if mf.is_dir():
        for d in sorted(mf.iterdir()):
            if args.env_id not in d.name:
                continue
            mon = d / "monitor.csv"
            if not mon.is_file():
                continue
            ts, rs = load_monitor(mon)
            rs_smooth = rolling_mean(rs, args.rolling)
            label = d.name.replace(f"_{args.env_id}", "").replace("_", " ")
            ax.plot(ts, rs_smooth, alpha=0.85, label=f"MF {label}")

    mpc_root = root / "model_based_mpc"
    if mpc_root.is_dir():
        for d in sorted(mpc_root.iterdir()):
            if args.env_id not in d.name:
                continue
            csvp = d / "episode_returns.csv"
            if not csvp.is_file():
                continue
            steps, rets = load_mpc_csv(csvp)
            rets_s = rolling_mean(rets, args.rolling)
            ax.plot(steps, rets_s, alpha=0.85, linestyle="--", label=f"MPC {d.name}")

    ax.set_xlabel("Environment steps (cumulative)")
    ax.set_ylabel(f"Episode return (rolling-{args.rolling} mean)")
    ax.set_title(f"Learning curves — {args.env_id}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
