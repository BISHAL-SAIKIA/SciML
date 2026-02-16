"""
This file defines the high-level training loop for a
Physics-Informed Neural Operator (PINO).

Responsibilities:
-----------------
- Manage epochs
- Call the single training step
- Log losses
- Keep training logic clean and readable

What this file does NOT do:
---------------------------
- No model architecture definitions
- No physics equations
- No data generation

This separation allows:
----------------------
- Easy experimentation
- Clear debugging
- Reusability across different operators
"""

import torch
from .train_step import train_step




# ----------------------------------------------------------
# Trainer
# ----------------------------------------------------------
def train(model, optimizer, X, X_init, epochs=1000, IC_weight=1.0, ODE_weight=1.0, verbose=True):
    """
    Trains a Physics-Informed Neural Operator.

    Parameters
    ----------
    model : torch.nn.Module
        DeepONet model

    optimizer : torch.optim.Optimizer
        Optimizer instance

    X : torch.Tensor
        ODE residual dataset

    X_init : torch.Tensor
        Initial condition dataset

    epochs : int
        Number of training epochs

    IC_weight : float
        Weight for initial condition loss

    ODE_weight : float
        Weight for ODE residual loss

    verbose : bool
        Whether to print training progress

    Returns
    -------
    history : dict
        Training loss history
    """

    history = {
        "ODE_loss": [],
        "IC_loss": [],
        "total_loss": []
    }

    for epoch in range(epochs):

        ODE_loss, IC_loss, total_loss = train_step(X=X, X_init=X_init, model=model, optimizer=optimizer, IC_weight=IC_weight, ODE_weight=ODE_weight)

        history["ODE_loss"].append(ODE_loss)
        history["IC_loss"].append(IC_loss)
        history["total_loss"].append(total_loss)

        if verbose and epoch % 100 == 0:
            print(
                f"Epoch {epoch:05d} | "
                f"ODE Loss: {ODE_loss:.6e} | "
                f"IC Loss: {IC_loss:.6e} | "
                f"Total Loss: {total_loss:.6e}"
            )

    return history
