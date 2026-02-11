import torch
from torch.utils.data import Dataset, DataLoader
from model import HeatDataset, PINN, heat_residual
from dotenv import load_dotenv
import wandb
load_dotenv()
#WANDB Initialization
wandb.init(
    project="Heat-eq",
    config={
        "lr": 0.001,
        "epochs": 4000,
        "batch_size": 64
    }
)


# =====================================================
# Training Setup
# =====================================================

# Load dataset
dataset = HeatDataset("heat_data.csv")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Collocation points (physics)
N_f = 2000
x_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1)

# Initial condition
x_ic = torch.rand(200, 1)
t_ic = torch.zeros_like(x_ic)
u_ic = torch.sin(torch.pi * x_ic)

# Boundary condition
t_bc = torch.rand(200, 1)
x_bc0 = torch.zeros_like(t_bc)
x_bc1 = torch.ones_like(t_bc)

# =====================================================
# Training Loop
# =====================================================

epochs = 4000

for epoch in range(epochs):
    epoch_loss = 0.0

    for x, t, u in loader:
        optimizer.zero_grad()

        # Data loss
        u_pred = model(x, t)
        loss_data = torch.mean((u_pred - u) ** 2)

        # PDE loss
        res = heat_residual(model, x_f, t_f)
        loss_pde = torch.mean(res ** 2)

        # Initial condition loss
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)

        # Boundary condition loss
        loss_bc = (
            torch.mean(model(x_bc0, t_bc) ** 2) +
            torch.mean(model(x_bc1, t_bc) ** 2)
        )

        # Total loss
        loss = loss_data + loss_pde + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        wandb.log({
            "epoch":epoch+1,
            "total_loss":epoch_loss/len(loader)

        })

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {epoch_loss:.6f}")

wandb.finish()