"""
===========================================================
Data Generation Utilities (data/generate_data.py)
===========================================================

This file is responsible for generating synthetic training
data for Physics-Informed DeepONet / PINO.

Philosophy:
-----------
- Data generation is SEPARATE from models and training
- This file describes the "physics scenarios" we sample
- Neural Operators need MANY input functions, not labels

In this project, we generate:
-----------------------------
1. Random forcing functions f(t)
2. Corresponding time samples t
3. f(t) evaluated at sampled times (used in ODE residual)

Target operator:
----------------
    G: f(t)  --->  u(t)
where:
    du/dt = f(t),  u(0) = 0
"""

import numpy as np
import torch

# ----------------------------------------------------------
# Generate Random Forcing Functions
# ----------------------------------------------------------
def generate_forcing_functions(num_samples, forcing_dim, t_min=0.0, t_max=1.0):
    """
    Generates random forcing functions sampled on a grid.

    Parameters
    ----------
    num_samples : int
        Number of forcing functions to generate

    forcing_dim : int
        Number of discretization points per function

    t_min, t_max : float
        Time domain bounds

    Returns
    -------
    forcing_samples : torch.Tensor
        Shape [num_samples, forcing_dim]

    time_grid : torch.Tensor
        Shape [forcing_dim, 1]

    Intuition
    ---------
    Each row represents ONE function f(t)
    sampled on a fixed grid.
    """

    time_grid = torch.linspace(t_min, t_max, forcing_dim).unsqueeze(1)

    # Random smooth forcing (Gaussian)
    forcing_samples = torch.randn(num_samples, forcing_dim)

    return forcing_samples, time_grid


# ----------------------------------------------------------
# Sample Training Points for ODE Residual
# ----------------------------------------------------------
def generate_ode_training_data(forcing_samples, time_grid, num_points_per_function):
    """
    Generates training points used to evaluate ODE residuals.

    For each forcing function:
    - Randomly sample time points
    - Extract corresponding f(t)

    Parameters
    ----------
    forcing_samples : torch.Tensor
        Shape [num_functions, forcing_dim]

    time_grid : torch.Tensor
        Shape [forcing_dim, 1]

    num_points_per_function : int
        Number of collocation points per function

    Returns
    -------
    X : torch.Tensor
        Shape [N, 1 + forcing_dim + 1]

        Column layout:
            [:, :1]   -> t
            [:, 1:-1] -> forcing samples
            [:, -1:]  -> f(t)
    """

    num_functions, forcing_dim = forcing_samples.shape

    X_list = []

    for i in range(num_functions):
        f = forcing_samples[i]

        # Random indices on the time grid
        idx = torch.randint(low=0, high=forcing_dim, size=(num_points_per_function,))

        t_samples = time_grid[idx]
        f_t = f[idx].unsqueeze(1)

        # Repeat full forcing vector
        f_repeated = f.unsqueeze(0).repeat(num_points_per_function, 1)

        X_i = torch.cat([t_samples, f_repeated, f_t], dim=1)
        X_list.append(X_i)

    X = torch.cat(X_list, dim=0)
    return X


# ----------------------------------------------------------
# Initial Condition Dataset
# ----------------------------------------------------------
def generate_initial_condition_data(forcing_samples):
    """
    Generates data for enforcing initial conditions.

    For all forcing functions:
        t = 0

    Parameters
    ----------
    forcing_samples : torch.Tensor
        Shape [num_functions, forcing_dim]

    Returns
    -------
    X_init : torch.Tensor
        Shape [num_functions, 1 + forcing_dim + 1]

    Notes
    -----
    The last column is a placeholder to keep
    dataset format consistent.
    """

    num_functions, forcing_dim = forcing_samples.shape

    t0 = torch.zeros(num_functions, 1)
    dummy = torch.zeros(num_functions, 1)

    X_init = torch.cat([t0, forcing_samples, dummy], dim=1)
    return X_init
