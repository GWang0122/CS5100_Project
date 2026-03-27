# Experiment workflow (matched budget + multi-seed)

All commands assume: `conda activate CS5100_Project` and project root as cwd.

## 1) Full comparison (same step budget for SAC, PPO, MPC)

```powershell
# Preview commands without running
python scripts/run_matched_experiments.py --dry-run --env-id Ant-v4 --budget 100000 --seeds 0 1 2

# Run (can take hours on Ant-v4)
python scripts/run_matched_experiments.py --env-id Ant-v4 --budget 100000 --seeds 0 1 2
```

Use `--methods sac mpc` to skip PPO, or `--methods sac` for SAC only.

## 2) Summarize logged runs (table for report)

```powershell
python scripts/summarize_runs.py --env-id Ant-v4
```

## 3) Plot learning curves

```powershell
python scripts/plot_learning_curves.py --env-id Ant-v4 --out figures/Ant-v4_learning.png
```

## 4) Evaluate a saved SB3 policy

```powershell
python scripts/evaluate_policy.py --algo sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4 --n-episodes 10
```

## 5) Record MP4 rollouts (fixed paths for `Ant-v4`, seed `0`)

```powershell
python scripts/visualize_rollout.py --mode sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4 --record-out videos/sac.mp4 --max-steps 1000 --seed 0 --n-episodes 1
python scripts/visualize_rollout.py --mode mpc --checkpoint runs/model_based_mpc/mpc_Ant-v4_seed0/dynamics_model.pt --env-id Ant-v4 --record-out videos/mpc.mp4 --max-steps 1000 --seed 0 --n-episodes 1
```

(`pip install imageio-ffmpeg` if saving video fails.)

## Notes

- **Budget:** `--budget` sets SB3 `total_timesteps` and MPC `total_env_steps` to the same number for fair comparison.
- **Git:** `runs/` and `*.zip` policies are gitignored; keep plots/small CSV summaries for submission if needed.
