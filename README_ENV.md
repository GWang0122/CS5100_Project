# CS5100 Final Project — Model-Free vs. Model-Based RL

Comparing SAC, PPO, and MLP-based MPC on continuous-control tasks
(Ant-v4 and FetchReach-v2) under matched interaction budgets.

## Project Structure

```
src/
  common/
    seed.py              # Global seed utility for reproducibility
    env_factory.py       # Gymnasium env creation with standard wrappers
  model_based/
    dynamics_model.py    # MLP dynamics model (predicts state delta + reward)
    replay_buffer.py     # Fixed-capacity circular replay buffer

scripts/
    train_model_free.py        # Train SAC or PPO via Stable-Baselines3
    train_model_based_mpc.py   # Train MPC with learned dynamics model
    evaluate_policy.py         # Evaluate a saved SB3 policy for N episodes
    visualize_rollout.py       # On-screen or MP4 rollout visualization
    plot_learning_curves.py    # Generate learning curve figures
    summarize_runs.py          # Print per-run episode return statistics
    run_matched_experiments.py # Launch multi-seed matched-budget sweeps
    smoke_test_env.py          # Verify environment + CUDA setup

run_experiments.sh    # All commands used to produce the paper's results
figures/              # Generated learning curve plots
runs/                 # Training outputs (checkpoints, logs) — git-ignored
```

## Setup

```powershell
conda env create -f environment.yml
conda activate CS5100_Project
python scripts/smoke_test_env.py
```

Verify GPU access:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Reproducing Results

Run all experiments from the paper:
```bash
bash run_experiments.sh
```

Or run individually:
```powershell
# SAC on Ant-v4 (300k steps, seed 1)
python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 300000 --seed 1

# PPO on Ant-v4 (300k steps, seed 1)
python scripts/train_model_free.py --env-id Ant-v4 --algo ppo --total-timesteps 300000 --seed 1

# MPC on Ant-v4 (300k steps, seed 1)
python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 300000 --seed 1
```

## Analyzing Results

```powershell
# Summary statistics for all runs
python scripts/summarize_runs.py

# Learning curve plots
python scripts/plot_learning_curves.py --env-id Ant-v4 --seed 1
python scripts/plot_learning_curves.py --env-id FetchReach-v2 --seed 1

# Evaluate a trained policy
python scripts/evaluate_policy.py --algo sac --model runs/model_free/sac_Ant-v4_seed1/policy.zip --env-id Ant-v4 --n-episodes 10
```

## Visualization

```powershell
# Watch SAC rollout on-screen
python scripts/visualize_rollout.py --mode sac --model runs/model_free/sac_Ant-v4_seed1/policy.zip --env-id Ant-v4

# Watch MPC rollout on-screen
python scripts/visualize_rollout.py --mode mpc --checkpoint runs/model_based_mpc/mpc_Ant-v4_seed1/dynamics_model.pt --env-id Ant-v4
```

## Notes

- Robotics tasks require MuJoCo; `environment.yml` pins `mujoco==2.3.7` for compatibility with `gymnasium-robotics==1.2.4`.
- Training outputs are saved under `runs/` (git-ignored). Checkpoints are `.zip` (SB3) and `.pt` (dynamics model).

## TLDR
Core algorithm - MLP dynamics model: `src/model_based/dynamics_model.py` 
Core algorithm - MPC training loop: `scripts/train_model_based_mpc.py` 
Model-free baseline (SAC + PPO): `scripts/train_model_free.py` 
Exact commands used to run all experiments: `run_experiments.sh` 
Pre-generated learning curve figures (Figures 1 & 2 in paper): `figures/` 
Raw training logs backing Table 2: `runs/`