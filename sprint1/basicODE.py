import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Analytic solution of dy/dx + 2x y = 0, y(0)=1
def analytic_solution(x):
    return torch.exp(-x**2)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def pinn_loss(model, x_interior, ic_weight):
    # Prediction
    y_pred = model(x_interior)

    # dy/dx via autograd
    dy_dx = torch.autograd.grad(
        outputs=y_pred,
        inputs=x_interior,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # ODE residual: dy/dx + 2x y = 0
    residual = dy_dx + 2 * x_interior * y_pred
    ode_loss = torch.mean(residual**2)

    # Initial condition y(0)=1
    x0 = torch.zeros((1, 1), requires_grad=True)
    y0_pred = model(x0)
    ic_loss = (y0_pred - 1.0) ** 2

    return ode_loss + ic_weight * ic_loss


# ============================
# Training data (collocation points)
# ============================
x_train = torch.linspace(0, 2, 2000).reshape(-1, 1)
x_train.requires_grad_(True)

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

ic_weight = 10.0

# ============================
# Training loop
# ============================
for epoch in range(5000):
    optimizer.zero_grad()
    loss = pinn_loss(model, x_train, ic_weight)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6e}")


# ============================
# Testing & visualization
# ============================
x_test = torch.linspace(0, 2, 200).reshape(-1, 1)
y_exact = analytic_solution(x_test).detach().numpy()
y_pred = model(x_test).detach().numpy()

plt.figure(figsize=(6,4))
plt.plot(x_test.numpy(), y_exact, label="Exact")
plt.plot(x_test.numpy(), y_pred, "--", label="PINN")
plt.legend()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("PINN: dy/dx + 2x y = 0")
plt.grid(True)
plt.savefig("dy_dx_2x_y.png", dpi=300, bbox_inches="tight")
