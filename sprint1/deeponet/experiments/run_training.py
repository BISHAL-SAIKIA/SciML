"""
This file is the ENTRY POINT of the project.

What this script does:
---------------------
1. Sets random seeds (reproducibility)
2. Generates synthetic physics data
3. Computes normalization statistics
4. Initializes the DeepONet model
5. Trains a Physics-Informed Neural Operator (PINO)
6. Prints training progress

This file:
----------
- DOES glue everything together
- DOES NOT define models, physics, or training logic
- Mimics how real SciML experiments are run

Run using:
----------
python experiments/run_training.py
"""

import torch
import torch.optim as optim

# -----------------------------
# Utilities
# -----------------------------
from deeponet.utils.seed import set_seed
from deeponet.utils.normalization import compute_mean_variance

# -----------------------------
# Data
# -----------------------------
from deeponet.data.generate_data import (
    generate_forcing_functions,
    generate_ode_training_data,
    generate_initial_condition_data
)
# -----------------------------
# Model & Training
# -----------------------------
from deeponet.models.deeponet import DeepONet
from deeponet.training.trainer import train

# ----------------------------------------------------------
# Main Experiment
# ----------------------------------------------------------
def main():

    # ======================================================
    # 1. Reproducibility
    # ======================================================
    set_seed(42)

    # ======================================================
    # 2. Problem Setup
    # ======================================================
    num_functions = 100        # Number of forcing functions
    forcing_dim = 100          # Discretization of each function
    num_points_per_function = 20

    epochs = 1000
    learning_rate = 1e-3

    IC_weight = 1.0
    ODE_weight = 1.0

    # ======================================================
    # 3. Generate Data
    # ======================================================
    forcing_samples, time_grid = generate_forcing_functions(
        num_samples=num_functions,
        forcing_dim=forcing_dim
    )

    X = generate_ode_training_data(
        forcing_samples=forcing_samples,
        time_grid=time_grid,
        num_points_per_function=num_points_per_function
    )

    X_init = generate_initial_condition_data(
        forcing_samples=forcing_samples
    )

    # ======================================================
    # 4. Normalization Statistics
    # ======================================================
    mean, var = compute_mean_variance(
        forcing_samples=forcing_samples,
        time_samples=time_grid
    )

    # ======================================================
    # 5. Model Initialization
    # ======================================================
    model = DeepONet(
        mean=mean,
        var=var,
        hidden_dim=50,
        depth=3
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ======================================================
    # 6. Training
    # ======================================================
    history = train(
        model=model,
        optimizer=optimizer,
        X=X,
        X_init=X_init,
        epochs=epochs,
        IC_weight=IC_weight,
        ODE_weight=ODE_weight,
        verbose=True
    )

    # ======================================================
    # 7. Final Output
    # ======================================================
    print("\nTraining completed successfully.")
    print(f"Final Total Loss: {history['total_loss'][-1]:.6e}")


# ----------------------------------------------------------
# Script Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
