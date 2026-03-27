"""
Watch or record rollouts for SAC/PPO (policy.zip) or MPC (dynamics_model.pt).

Examples:
  # On-screen window (close window to continue / Ctrl+C to stop)
  python scripts/visualize_rollout.py --mode sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4

  python scripts/visualize_rollout.py --mode mpc --checkpoint runs/model_based_mpc/mpc_Ant-v4_seed0/dynamics_model.pt --env-id Ant-v4

  # Save MP4 (pip install imageio-ffmpeg)
  python scripts/visualize_rollout.py --mode sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4 --record-out videos/sac.mp4 --max-steps 1000 --seed 0 --n-episodes 1
  python scripts/visualize_rollout.py --mode mpc --checkpoint runs/model_based_mpc/mpc_Ant-v4_seed0/dynamics_model.pt --env-id Ant-v4 --record-out videos/mpc.mp4 --max-steps 1000 --seed 0 --n-episodes 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gymnasium_robotics  # noqa: F401
except ImportError:
    pass

from stable_baselines3 import PPO, SAC

from src.common.seed import set_global_seed
from src.model_based.dynamics_model import DynamicsMLP


def make_render_env(env_id: str, seed: int, render_mode: str) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


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
    act_dim = action_low.shape[0]
    action_seqs = np.random.uniform(
        low=action_low,
        high=action_high,
        size=(candidates, horizon, act_dim),
    ).astype(np.float32)
    with torch.no_grad():
        obs_batch = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        obs_batch = obs_batch.repeat(candidates, 1)
        returns = torch.zeros(candidates, device=device)
        discount = 1.0
        for t in range(horizon):
            a_t = torch.as_tensor(action_seqs[:, t, :], device=device, dtype=torch.float32)
            obs_batch, r_t = model.predict_next_obs_reward(obs_batch, a_t)
            returns = returns + discount * r_t.squeeze(-1)
            discount *= gamma
    best_idx = int(torch.argmax(returns).item())
    return action_seqs[best_idx, 0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAC/PPO or MPC rollouts.")
    p.add_argument("--mode", choices=["sac", "ppo", "mpc"], required=True)
    p.add_argument("--env-id", type=str, default="Ant-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default=None, help="Path to policy.zip (sac/ppo).")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to dynamics_model.pt (mpc).")
    p.add_argument("--render", choices=["human", "rgb_array"], default="human")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    p.add_argument("--record-out", type=str, default=None, help="If set, save video to this path (use .mp4).")
    # MPC hyperparams (match training)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--model-hidden-dim", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    render_mode = "rgb_array" if args.record_out else args.render
    env = make_render_env(args.env_id, args.seed, render_mode=render_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("sac", "ppo"):
        if not args.model:
            print("--model required for sac/ppo", file=sys.stderr)
            sys.exit(1)
        if args.mode == "sac":
            model = SAC.load(args.model, env=env)
        else:
            model = PPO.load(args.model, env=env)

        def get_action(obs):
            a, _ = model.predict(obs, deterministic=True)
            return np.asarray(a, dtype=np.float32)

    else:
        if not args.checkpoint:
            print("--checkpoint required for mpc", file=sys.stderr)
            sys.exit(1)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        dyn = DynamicsMLP(obs_dim, act_dim, hidden_dim=args.model_hidden_dim).to(device)
        # state_dict only (saved via torch.save(model.state_dict(), ...))
        state = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
        dyn.load_state_dict(state)
        dyn.eval()

        def get_action(obs):
            return mpc_random_shooting(
                dyn, obs, action_low, action_high,
                args.horizon, args.candidates, args.gamma, device,
            )

    frames: list[np.ndarray] = []
    if args.record_out:
        try:
            import imageio.v2 as imageio  # noqa: F401
        except ImportError:
            print("pip install imageio imageio-ffmpeg", file=sys.stderr)
            sys.exit(1)
        Path(args.record_out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        term = trunc = False
        for step in range(args.max_steps):
            action = get_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            if args.record_out:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif args.render == "human":
                env.render()
            if term or trunc:
                break
        # Ant-v4: terminated = usually fell / unhealthy; truncated = hit time limit (e.g. 1000)
        if term:
            why = "terminated (e.g. fell / unhealthy)"
        elif trunc:
            why = "truncated (time limit)"
        else:
            why = f"stopped at --max-steps={args.max_steps}"
        print(
            f"Episode {ep + 1}: return={total_r:.2f} steps={step + 1} — {why}"
        )

    if args.record_out and frames:
        import imageio.v2 as imageio

        fps = 30
        imageio.mimsave(args.record_out, frames, fps=fps)
        print(f"Saved video: {args.record_out} ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
