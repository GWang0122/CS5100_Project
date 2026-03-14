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

## Notes

- Robotics tasks require MuJoCo; this environment pins a version compatible with `gymnasium-robotics==1.2.4`.
