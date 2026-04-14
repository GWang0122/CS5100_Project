"""
Reproducibility utilities.

Sets global random seeds across all libraries used in the project
(Python stdlib, NumPy, PyTorch CPU, and PyTorch CUDA) so that
experiments are deterministic given the same seed value.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy, and PyTorch (CPU + GPU).

    This should be called once at the start of every training or
    evaluation script before any stochastic operations occur.

    Args:
        seed: Integer seed shared across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all visible CUDA devices for reproducible GPU operations.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
