import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dotenv import load_dotenv
import wandb

from data_gen import BurgersDataset
from model import PINN

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
load_dotenv()

nu = 0.01 / np.pi  # viscosity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Weights & Biases init
# ------------------------------------------------------------------
wandb.init(
    project="Burgers-eq",
    config={
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": None
    }
)

# ------------------------------------------------------------------
# Burgers PDE residual
# ------------------------------------------------------------------
def burgers_residual(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True
    )[0]

    return u_t + u * u_x - nu * u_xx

# ------------------------------------------------------------------
# Model, optimizer, loss
# ------------------------------------------------------------------
model = PINN().to(device)
optimizer = optim.LBFGS(model.parameters(), lr=wandb.config.lr)
mse = nn.MSELoss()

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
dataset = BurgersDataset()
loader = DataLoader(dataset, batch_size=None)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
epochs = 500

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch in loader:
        (
            x, t,
            x_ic, t_ic, u_ic,
            x_bc_left, x_bc_right, t_bc
        ) = batch

        optimizer.zero_grad()

        # PDE loss
        f_res = burgers_residual(model, x, t)
        loss_pde = mse(f_res, torch.zeros_like(f_res))

        # Initial condition loss
        u_ic_pred = model(x_ic, t_ic)
        loss_ic = mse(u_ic_pred, u_ic)

        # Boundary condition loss
        u_left = model(x_bc_left, t_bc)
        u_right = model(x_bc_right, t_bc)
        loss_bc = (
            mse(u_left, torch.zeros_like(u_left)) +
            mse(u_right, torch.zeros_like(u_right))
        )

        # Total loss
        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Logging
    wandb.log({
        "epoch": epoch + 1,
        "total_loss": epoch_loss / len(loader)
    })

    if (epoch+1) % 100 == 0:
        print(f"\nEpoch {epoch+1} | Loss: {loss.item():.6f}")

# ------------------------------------------------------------------
# Finish
# ------------------------------------------------------------------
wandb.finish()
