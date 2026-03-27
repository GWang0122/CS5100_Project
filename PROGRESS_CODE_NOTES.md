# Implemented Progress Artifacts

This repository now contains executable code for both project directions in the proposal.

## 1) Environment and validation
- `environment.yml`: GPU-enabled PyTorch + Gymnasium + Gymnasium-Robotics + MuJoCo setup.
- `scripts/smoke_test_env.py`: verifies `Ant-v4` and available `FetchReach-*` environment IDs.

## 2) Model-free baseline
- `scripts/train_model_free.py`: trains PPO or SAC from environment interaction using Stable-Baselines3.
- Uses shared wrappers/utilities in `src/common/` for seeding and observation flattening.
- Saves trained policy checkpoints and logs to `runs/model_free/`.

## 3) Model-based prototype
- `src/model_based/dynamics_model.py`: lightweight MLP dynamics model predicting state deltas and rewards.
- `src/model_based/replay_buffer.py`: transition replay buffer for training the world model.
- `scripts/train_model_based_mpc.py`: data collection + periodic model updates + random-shooting MPC control.
- Saves learned dynamics model and episode return CSV in `runs/model_based_mpc/`.

## Suggested demo commands
```powershell
python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 100000
python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 30000
```

## 4) Matched experiments and reporting
- `scripts/run_matched_experiments.py`: same `--budget` for SAC/PPO (timesteps) and MPC (env steps), multiple seeds.
- `scripts/summarize_runs.py`: print per-run stats from logs.
- `scripts/plot_learning_curves.py`: one figure from all matching runs.
- `scripts/evaluate_policy.py`: evaluate a saved `policy.zip` (SAC/PPO).
- See **`WORKFLOW.md`** for the full pipeline.
