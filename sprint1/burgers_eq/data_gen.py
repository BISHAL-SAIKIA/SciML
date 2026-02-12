import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 


nu = 0.01 / np.pi #nu stands for viscosity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data Generation (collocation + BC + IC)

def generate_training_points(N_f = 10000, N_bc=200, N_ic=200):

    #collocation points (x,t)
    x = torch.rand(N_f, 1) * 2 -1 #[-1,1]
    t= torch.rand(N_f, 1)        #[0,1]

    #Initial condition
    x_ic = torch.rand(N_ic, 1) * 2 -1
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic)

    #Boundary condition (x = -1, 1)
    t_bc = torch.rand(N_bc, 1)
    x_bc_left = - torch.ones_like(t_bc)
    x_bc_right = torch.ones_like(t_bc)

    return x, t, x_ic, t_ic, u_ic, x_bc_left, x_bc_right, t_bc

class BurgersDataset(Dataset):
    def __init__(self):
        data = generate_training_points()
        self.data = [d.float().to(device) for d in data]

    def __len__(self):
        return 1  # PINNs don’t iterate like normal datasets, we train on all physics points at once

    def __getitem__(self, idx):
        return self.data
