"""
===========================================================
Initial Condition Definitions (physics/initial_conditions.py)
===========================================================

This file contains utilities for enforcing initial conditions
(ICs) in Physics-Informed Neural Operators (PINO).

Design principles:
------------------
- Initial conditions are NOT part of the model architecture
- They are enforced via additional loss terms
- This keeps physics modular and extensible

In this project:
----------------
We consider problems where the solution must satisfy:
    u(t = 0) = u_0

This file provides:
------------------
1. A generic initial condition loss function
2. A clean interface usable by the training loop
"""

import torch

def initial_condition_loss(model, X_init):
    """
    Computes the initial condition (IC) loss.

    The IC enforced here is:
        u(t = 0) = 0

    Parameters
    ----------
    model : torch.nn.Module
        DeepONet model

    X_init : torch.Tensor
        Initial condition dataset with shape:
            [batch_size, 1 + forcing_dim + 1]

        Column layout:
            X_init[:, :1]   -> time (t = 0)
            X_init[:, 1:-1] -> forcing samples
            X_init[:, -1:]  -> (unused or placeholder)

    Returns
    -------
    IC_loss : torch.Tensor
        Mean squared error enforcing initial condition

    Intuition
    ---------
    - The DeepONet predicts u(t)
    - At t = 0, physics tells us the exact value
    - We penalize deviation from this known value
    """

    # Extract time and forcing
    t_init = X_init[:, :1]
    forcing_init = X_init[:, 1:-1]

    # Predict solution at initial time
    u_pred_init = model(forcing_init, t_init)

    # Enforce u(0) = 0
    IC_loss = torch.mean(u_pred_init ** 2)

    return IC_loss
