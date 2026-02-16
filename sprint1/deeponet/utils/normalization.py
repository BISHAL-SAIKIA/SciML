"""

This file contains helper functions for computing and
applying normalization to inputs used in Neural Operators.

Why normalization matters:
--------------------------
- Neural Operators see HIGH-dimensional inputs (functions)
- Poor scaling leads to unstable training
- Consistent normalization improves convergence and accuracy

Design principles:
------------------
- Normalization logic should NOT live inside training loops
- Statistics should be computed ONCE and reused
- Supports both NumPy and PyTorch tensors
"""

import torch
import numpy as np


# ----------------------------------------------------------
# Compute Mean and Variance
# ----------------------------------------------------------
def compute_mean_variance(forcing_samples, time_samples):
    """
    Computes mean and variance for normalization.

    Parameters
    ----------
    forcing_samples : torch.Tensor or np.ndarray
        Shape [num_samples, forcing_dim]

    time_samples : torch.Tensor or np.ndarray
        Shape [num_samples, coord_dim]

    Returns
    -------
    mean : dict
        {
            "forcing": mean_forcing,
            "time": mean_time
        }

    var : dict
        {
            "forcing": var_forcing,
            "time": var_time
        }

    Notes
    -----
    - Statistics are computed across the dataset
    - Used by DeepONet for internal normalization
    """

    if isinstance(forcing_samples, np.ndarray):
        forcing_samples = torch.from_numpy(forcing_samples)
    if isinstance(time_samples, np.ndarray):
        time_samples = torch.from_numpy(time_samples)

    mean_forcing = torch.mean(forcing_samples, dim=0)
    var_forcing = torch.var(forcing_samples, dim=0, unbiased=False)

    mean_time = torch.mean(time_samples, dim=0)
    var_time = torch.var(time_samples, dim=0, unbiased=False)

    mean = {
        "forcing": mean_forcing,
        "time": mean_time
    }

    var = {
        "forcing": var_forcing,
        "time": var_time
    }

    return mean, var


# ----------------------------------------------------------
# Apply Normalization
# ----------------------------------------------------------
def normalize(x, mean, var):
    """
    Applies standard normalization.

    Formula:
        x_norm = (x - mean) / sqrt(var)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    mean : torch.Tensor
    var  : torch.Tensor

    Returns
    -------
    x_norm : torch.Tensor
        Normalized tensor
    """
    return (x - mean) / torch.sqrt(var + 1e-6)
