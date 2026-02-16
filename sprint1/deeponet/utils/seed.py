"""

This file centralizes random seed control for the entire
project to ensure reproducible experiments.

Why this matters:
-----------------
- SciML experiments are sensitive to initialization
- Operator learning can be unstable without fixed seeds
- Reproducibility is critical for research and debugging

This utility:
-------------
- Sets seeds for Python, NumPy, and PyTorch
- Optionally enforces deterministic CUDA behavior
"""

import random
import numpy as np
import torch


# ----------------------------------------------------------
# Set Global Seed
# ----------------------------------------------------------
def set_seed(seed=42, deterministic=True):
    """
    Sets random seeds across all relevant libraries.

    Parameters
    ----------
    seed : int
        Random seed value

    deterministic : bool
        If True, enforces deterministic CUDA behavior
        (may slightly reduce performance)
    """

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
