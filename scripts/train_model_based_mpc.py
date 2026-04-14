"""
Train a model-based agent using random-shooting MPC.

Pipeline overview:
  1. Collect initial transitions using random actions to fill the buffer.
  2. At each step after the warm-up phase, select actions via MPC:
     - Sample N candidate action sequences of length H (horizon).
     - Evaluate each sequence by rolling out the learned dynamics model
       to estimate the discounted cumulative reward.
     - Execute the first action of the best-scoring sequence in the
       real environment and store the resulting transition.
  3. Periodically retrain the dynamics model on a minibatch from the
     replay buffer to improve prediction accuracy.
  4. Repeat until the total environment-step budget is exhausted.

Example usage:
    python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 300000 --seed 1

Output:
    runs/model_based_mpc/mpc_<env>_seed<s>/dynamics_model.pt
    runs/model_based_mpc/mpc_<env>_seed<s>/episode_returns.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# Register Gymnasium-Robotics environments if installed.
try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    gymnasium_robotics = None

# Ensure project root is on sys.path for `src.*` imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.env_factory import make_env
from src.common.seed import set_global_seed
from src.model_based.dynamics_model import DynamicsMLP, train_dynamics_step
from src.model_based.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MPC training loop."""
    parser = argparse.ArgumentParser(description="Train lightweight model-based MPC agent.")
    parser.add_argument("--env-id", type=str, default="Ant-v4",
                        help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--total-env-steps", type=int, default=60_000,
                        help="Total real-environment interactions")
    parser.add_argument("--init-random-steps", type=int, default=5_000,
                        help="Number of initial random-action steps for warm-up")
    parser.add_argument("--buffer-capacity", type=int, default=300_000,
                        help="Maximum transitions stored in replay buffer")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size for dynamics model training")
    parser.add_argument("--train-every", type=int, default=250,
                        help="Retrain the dynamics model every N env steps")
    parser.add_argument("--grad-steps", type=int, default=200,
                        help="Gradient updates per training phase")
    parser.add_argument("--model-hidden-dim", type=int, default=256,
                        help="Hidden layer width for the dynamics MLP")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate for dynamics model")
    parser.add_argument("--horizon", type=int, default=15,
                        help="MPC planning horizon (steps)")
    parser.add_argument("--candidates", type=int, default=256,
                        help="Number of random action sequences to evaluate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for cumulative reward")
    parser.add_argument("--log-dir", type=str, default="runs/model_based_mpc",
                        help="Root directory for run outputs")
    return parser.parse_args()


def mpc_random_shooting(
    model: DynamicsMLP,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    horizon: int,
    candidates: int,
    gamma: float,
    device: torch.device,
) -> np.ndarray:
    """Select the best first action via random-shooting MPC.

    Algorithm (random-shooting / zeroth-order trajectory optimization):
      1. Sample `candidates` random action sequences, each of length
         `horizon`, uniformly within the action bounds.
      2. For each candidate, simulate a trajectory through the learned
         dynamics model and accumulate the discounted predicted reward.
      3. Return the first action of the sequence with the highest
         predicted cumulative reward.

    This is a simple but effective planning strategy; its main
    weakness is poor coverage in high-dimensional action spaces.

    Args:
        model:       Trained DynamicsMLP used for forward predictions.
        obs:         Current real observation, shape (obs_dim,).
        action_low:  Lower bounds of the action space.
        action_high: Upper bounds of the action space.
        horizon:     Number of time-steps to look ahead.
        candidates:  Number of random sequences to evaluate.
        gamma:       Discount factor for reward accumulation.
        device:      Torch device for model inference.

    Returns:
        Best first action as a NumPy array of shape (act_dim,).
    """
    act_dim = action_low.shape[0]

    # Sample candidate action sequences uniformly within bounds.
    # Shape: (candidates, horizon, act_dim)
    action_seqs = np.random.uniform(
        low=action_low,
        high=action_high,
        size=(candidates, horizon, act_dim),
    ).astype(np.float32)

    with torch.no_grad():
        # Replicate the current observation for all candidate sequences.
        obs_batch = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        obs_batch = obs_batch.repeat(candidates, 1)  # (candidates, obs_dim)

        returns = torch.zeros(candidates, device=device)
        discount = 1.0

        # Imagined rollout: step through the learned dynamics model.
        for t in range(horizon):
            a_t = torch.as_tensor(action_seqs[:, t, :], device=device, dtype=torch.float32)
            obs_batch, r_t = model.predict_next_obs_reward(obs_batch, a_t)
            returns = returns + discount * r_t.squeeze(-1)
            discount *= gamma

    # Pick the sequence with the highest predicted return and execute
    # only its first action (receding-horizon control).
    best_idx = int(torch.argmax(returns).item())
    return action_seqs[best_idx, 0]


def main() -> None:
    """Run the full MPC training pipeline."""
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env = make_env(args.env_id, args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    # Random-shooting MPC requires bounded action spaces.
    assert np.all(np.isfinite(env.action_space.low)), "Action low bound must be finite."
    assert np.all(np.isfinite(env.action_space.high)), "Action high bound must be finite."

    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    # Replay buffer and dynamics model
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=args.buffer_capacity)
    model = DynamicsMLP(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.model_hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging setup
    run_name = f"mpc_{args.env_id}_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    returns_csv = run_dir / "episode_returns.csv"

    # Episode tracking variables
    episode_return = 0.0
    episode_length = 0
    episode_idx = 0
    obs, _ = env.reset(seed=args.seed)

    # Open CSV for streaming episode statistics as they complete.
    with returns_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "env_step", "episode_return", "episode_length"])

        for step in range(1, args.total_env_steps + 1):
            # Action selection
            # During the warm-up phase, act randomly to seed the
            # replay buffer with diverse transitions.  After warm-up,
            # use the MPC planner with the current dynamics model.
            if step <= args.init_random_steps:
                action = env.action_space.sample()
            else:
                action = mpc_random_shooting(
                    model=model,
                    obs=obs,
                    action_low=action_low,
                    action_high=action_high,
                    horizon=args.horizon,
                    candidates=args.candidates,
                    gamma=args.gamma,
                    device=device,
                )

            # Environment step and buffer storage
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_return += reward
            episode_length += 1

            # Periodic dynamics model training
            # Every `train_every` steps, perform `grad_steps` gradient
            # updates on the dynamics model using random minibatches
            # from the replay buffer.
            if buffer.size >= args.batch_size and step % args.train_every == 0:
                losses = []
                for _ in range(args.grad_steps):
                    batch = buffer.sample(args.batch_size)
                    loss = train_dynamics_step(model, optimizer, batch, device)
                    losses.append(loss)
                mean_loss = float(np.mean(losses))
                print(f"[step={step}] dynamics_loss={mean_loss:.6f} buffer={buffer.size}")

            # Episode bookkeeping
            if done:
                episode_idx += 1
                writer.writerow([episode_idx, step, episode_return, episode_length])
                f.flush()
                print(
                    f"[episode={episode_idx}] step={step} return={episode_return:.2f} "
                    f"length={episode_length}"
                )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

    # Save trained dynamics model
    model_path = run_dir / "dynamics_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved dynamics model to: {model_path}")
    print(f"Saved training curve to: {returns_csv}")
    env.close()


if __name__ == "__main__":
    main()
