#!/usr/bin/env bash
#
# Commands used to produce all results in the paper.

set -euo pipefail

# Ant-v4
python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 300000 --seed 1
python scripts/train_model_free.py --env-id Ant-v4 --algo sac --total-timesteps 300000 --seed 2
python scripts/train_model_free.py --env-id Ant-v4 --algo ppo --total-timesteps 300000 --seed 1
python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 300000 --seed 1
python scripts/train_model_based_mpc.py --env-id Ant-v4 --total-env-steps 300000 --seed 2

# FetchReach-v2
python scripts/train_model_free.py --env-id FetchReach-v2 --algo sac --total-timesteps 150000 --seed 1
python scripts/train_model_free.py --env-id FetchReach-v2 --algo sac --total-timesteps 150000 --seed 2
python scripts/train_model_free.py --env-id FetchReach-v2 --algo ppo --total-timesteps 150000 --seed 1
python scripts/train_model_based_mpc.py --env-id FetchReach-v2 --total-env-steps 150000 --seed 1
python scripts/train_model_based_mpc.py --env-id FetchReach-v2 --total-env-steps 150000 --seed 2
