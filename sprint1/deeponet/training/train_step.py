"""

This file defines ONE optimization step for training a
Physics-Informed Neural Operator (PINO).

What this file does:
--------------------
- Computes initial condition loss
- Computes physics (ODE) residual loss
- Combines losses with user-defined weights
- Performs backpropagation and optimizer step

What this file does NOT do:
---------------------------
- No epochs
- No data loading
- No experiment configuration

This separation keeps training logic modular and reusable.
"""

import torch

from ..physics.residuals import ODE_residual_calculator
from ..physics.initial_conditions import initial_condition_loss



# ----------------------------------------------------------
# Training Step
# ----------------------------------------------------------
def train_step(X, X_init, model, optimizer, IC_weight=1.0, ODE_weight=1.0):
    """
    Performs ONE training step for PINO.

    Parameters
    ----------
    X : torch.Tensor
        ODE residual dataset
        Shape: [N, 1 + forcing_dim + 1]
        Columns:
            X[:, :1]   -> time t
            X[:, 1:-1] -> forcing samples
            X[:, -1:]  -> f(t)

    X_init : torch.Tensor
        Initial condition dataset
        Shape: [N, 1 + forcing_dim + 1]

    model : torch.nn.Module
        DeepONet model

    optimizer : torch.optim.Optimizer
        Optimizer (Adam, etc.)

    IC_weight : float
        Weight for initial condition loss

    ODE_weight : float
        Weight for ODE residual loss

    Returns
    -------
    ODE_loss : float
    IC_loss  : float
    total_loss : float
    """

    # Reset gradients
    optimizer.zero_grad()

    # --------------------------------------------------
    # Initial Condition Loss
    # --------------------------------------------------
    IC_loss = initial_condition_loss(model, X_init)

    # --------------------------------------------------
    # ODE Residual Loss
    # --------------------------------------------------
    t = X[:, :1]
    forcing = X[:, 1:-1]
    u_t = X[:, -1:]

    residual = ODE_residual_calculator(t=t, u=forcing, u_t=u_t, model=model
    )

    ODE_loss = torch.mean(residual ** 2)

    # --------------------------------------------------
    # Total Loss
    # --------------------------------------------------
    total_loss = IC_weight * IC_loss + ODE_weight * ODE_loss

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    return ODE_loss.item(), IC_loss.item(), total_loss.item()
