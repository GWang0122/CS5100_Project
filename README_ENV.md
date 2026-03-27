# RL Environment Setup (Conda)

This project uses a Conda environment with GPU-enabled PyTorch, Gymnasium, MuJoCo, and Gymnasium-Robotics for continuous-control experiments.

## 1) Create and activate the environment

```powershell
conda env create -f environment.yml
conda activate CS5100_Project
```

## 2) Run a smoke test

```powershell
python scripts/smoke_test_env.py
```

You should see successful resets/steps for:
- `Ant-v4`
- `FetchReach` (version available in your installed robotics package)

## 3) Verify GPU is visible to PyTorch

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

## 4) Run baseline/model-based scripts

```powershell
# Model-free baseline (SAC)
python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 200000

# Model-based prototype (MLP dynamics + random-shooting MPC)
python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 60000
```

## 5) Optional: Jupyter support

```powershell
conda install ipykernel -y
python -m ipykernel install --user --name CS5100_Project --display-name "Python (CS5100_Project)"
```

## 6) Matched experiments, plots, evaluation

See **`WORKFLOW.md`** for multi-seed runs with the same step budget, summaries, and plotting.

Quick start:

```powershell
python scripts/run_matched_experiments.py --env-id Ant-v4 --budget 100000 --seeds 0 1 2 --methods sac mpc
python scripts/summarize_runs.py --env-id Ant-v4
python scripts/plot_learning_curves.py --env-id Ant-v4 --out figures/Ant-v4_learning.png
```

## 7) Record rollout videos (paths after a default train run)

From project root, with checkpoints under `runs/` (adjust `seed0` if you used another seed):

```powershell
# SAC — requires runs/model_free/sac_Ant-v4_seed0/policy.zip
python scripts/visualize_rollout.py --mode sac --model runs/model_free/sac_Ant-v4_seed0/policy.zip --env-id Ant-v4 --record-out videos/sac.mp4 --max-steps 1000 --seed 0 --n-episodes 1

# MPC — uses runs/model_based_mpc/mpc_Ant-v4_seed0/dynamics_model.pt
python scripts/visualize_rollout.py --mode mpc --checkpoint runs/model_based_mpc/mpc_Ant-v4_seed0/dynamics_model.pt --env-id Ant-v4 --record-out videos/mpc.mp4 --max-steps 1000 --seed 0 --n-episodes 1
```

Install encoder if needed: `pip install imageio-ffmpeg`

## Notes

- Robotics tasks require MuJoCo; this environment pins a version compatible with `gymnasium-robotics==1.2.4`.
