import torch
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn


class HeatDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.x = torch.tensor(data["x"].values, dtype=torch.float32).view(-1, 1)
        self.t = torch.tensor(data["t"].values, dtype=torch.float32).view(-1, 1)
        self.u = torch.tensor(data["u"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.u[idx]

# =====================================================
# PINN Model
# =====================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.Tanh()

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        h = self.act(self.fc1(xt))
        h = self.act(self.fc2(h))
        return self.fc3(h)

# =====================================================
# PDE Residual (Heat Equation)
# =====================================================

def heat_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    return u_t - u_xx
