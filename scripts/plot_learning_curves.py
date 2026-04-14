"""
Plot learning curves from training logs.

Reads episode return data from:
  - SB3 monitor.csv        (model-free runs: SAC, PPO)
  - MPC episode_returns.csv (model-based runs)

Generates a PNG figure showing rolling-mean episode return vs.
cumulative environment steps.  Supports filtering by seed, method,
and environment so that per-condition or combined plots can be
produced easily.

Examples:
    python scripts/plot_learning_curves.py --env-id Ant-v4 --seed 1
    python scripts/plot_learning_curves.py --env-id Ant-v4 --seed 1 --method sac --out figures/Ant-v4_seed1_sac.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    p = argparse.ArgumentParser(description="Plot episode returns vs training progress.")
    p.add_argument("--runs-root", type=str, default="runs",
                   help="Root folder containing model_free/ and model_based_mpc/")
    p.add_argument("--env-id", type=str, required=True,
                   help="Gymnasium env ID to filter on (e.g. Ant-v4)")
    p.add_argument("--seed", type=int, default=None,
                   help="If set, only include runs whose folder name ends "
                        "with _seed<N>")
    p.add_argument("--method", type=str, choices=["all", "sac", "ppo", "mpc"],
                   default="all",
                   help="Plot only a specific method or all matching runs")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path (auto-generated if omitted)")
    p.add_argument("--rolling", type=int, default=5,
                   help="Window size for rolling-mean smoothing")
    return p.parse_args()


def matches_seed(run_name: str, seed: int | None) -> bool:
    """Return True if `run_name` matches the requested seed filter."""
    if seed is None:
        return True
    suffix = f"_seed{seed}"
    return run_name.endswith(suffix)


def load_monitor(path: Path) -> tuple[list[int], list[float]]:
    """Load SB3 Monitor CSV and return (cumulative_steps, episode_returns).

    The SB3 Monitor CSV has a 2-line header (comment + column names).
    Each data row is: reward, episode_length, wall_time.
    We reconstruct cumulative timesteps by summing episode lengths.
    """
    ts: list[int] = []
    rs: list[float] = []
    cum = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip the 2-line SB3 monitor header.
            if i < 2:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            r = float(parts[0])              # Episode return
            length = int(float(parts[1]))    # Episode length (steps)
            cum += length
            ts.append(cum)
            rs.append(r)
    return ts, rs


def load_mpc_csv(path: Path) -> tuple[list[int], list[float]]:
    """Load MPC episode_returns.csv with columns: episode, env_step,
    episode_return, episode_length.

    Returns (env_steps, episode_returns).
    """
    steps: list[int] = []
    rets: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["env_step"]))
            rets.append(float(row["episode_return"]))
    return steps, rets


def rolling_mean(xs: list[float], w: int) -> list[float]:
    """Compute a causal (trailing) rolling mean with window size `w`.

    For the first `w-1` elements the window is shorter (expanding).
    """
    if w <= 1 or len(xs) < w:
        return xs
    out = []
    for i in range(len(xs)):
        start = max(0, i - w + 1)
        chunk = xs[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    """Discover matching runs, plot learning curves, and save figure."""
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: conda install matplotlib -y", file=sys.stderr)
        sys.exit(1)

    root = Path(args.runs_root)

    # Determine output path
    env_tag = args.env_id.replace("/", "_")
    if args.out:
        out_path = Path(args.out)
    else:
        parts = [env_tag, "learning"]
        if args.seed is not None:
            parts.insert(1, f"seed{args.seed}")
        if args.method != "all":
            parts.insert(-1, args.method)
        out_path = PROJECT_ROOT / "figures" / ("_".join(parts) + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Decide which method categories to include based on --method flag.
    plot_mf = args.method in ("all", "sac", "ppo")
    plot_mpc = args.method in ("all", "mpc")

    # Model-free runs (SB3 monitor.csv)
    mf = root / "model_free"
    if mf.is_dir() and plot_mf:
        for d in sorted(mf.iterdir()):
            if args.env_id not in d.name:
                continue
            if not matches_seed(d.name, args.seed):
                continue
            # Respect --method filter for SAC vs PPO.
            if args.method == "sac" and not d.name.startswith("sac_"):
                continue
            if args.method == "ppo" and not d.name.startswith("ppo_"):
                continue
            mon = d / "monitor.csv"
            if not mon.is_file():
                continue
            ts, rs = load_monitor(mon)
            rs_smooth = rolling_mean(rs, args.rolling)
            label = d.name.replace(f"_{args.env_id}", "").replace("_", " ")
            ax.plot(ts, rs_smooth, alpha=0.85, label=f"MF {label}")

    # Model-based MPC runs (episode_returns.csv)
    mpc_root = root / "model_based_mpc"
    if mpc_root.is_dir() and plot_mpc:
        for d in sorted(mpc_root.iterdir()):
            if args.env_id not in d.name:
                continue
            if not matches_seed(d.name, args.seed):
                continue
            csvp = d / "episode_returns.csv"
            if not csvp.is_file():
                continue
            steps, rets = load_mpc_csv(csvp)
            rets_s = rolling_mean(rets, args.rolling)
            ax.plot(steps, rets_s, alpha=0.85, linestyle="--", label=f"MPC {d.name}")

    # Formatting and save
    ax.set_xlabel("Environment steps (cumulative)")
    ax.set_ylabel(f"Episode return (rolling-{args.rolling} mean)")
    title_bits = [args.env_id]
    if args.seed is not None:
        title_bits.append(f"seed {args.seed}")
    if args.method != "all":
        title_bits.append(args.method.upper())
    ax.set_title("Learning curves — " + ", ".join(title_bits))

    if not ax.get_lines():
        print(
            "No runs matched (check --env-id, --seed, --method and that "
            "runs/ contains data).",
            file=sys.stderr,
        )
        sys.exit(1)

    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
