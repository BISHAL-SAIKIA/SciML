import torch
import torch.nn as nn


# ----------------------------------------------------------
# Branch Network
# ----------------------------------------------------------

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, depth=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

# ----------------------------------------------------------
# Trunk Network
# ----------------------------------------------------------

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim = 50, depth = 3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------
# Bias Layer
# ----------------------------------------------------------

class BiasLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return x + self.bias

# ----------------------------------------------------------
# DeepONet
# ----------------------------------------------------------

class DeepONet(nn.Module):
    def __init__(self, mean, var, hidden_dim =50, depth = 3):
        super().__init__()

        #storing normalization statistics
        self.register_buffer("mean_forcing", mean["forcing"].clone().detach())
        self.register_buffer("var_forcing",  var["forcing"].clone().detach())
        self.register_buffer("mean_coord",   mean["time"].clone().detach())
        self.register_buffer("var_coord",    var["time"].clone().detach())



        #Networks
        self.branch = BranchNet(
            input_dim = len(mean["forcing"]),
            hidden_dim = hidden_dim,
            depth = depth
        )

        self.trunk = TrunkNet(
            input_dim = len(mean["time"]),
            hidden_dim= hidden_dim,
            depth = depth
        )

        self.bias = BiasLayer()
    
    def normalize(self, x, mean, var):
        """
        Standard normalization:
            (x - mean) / sqrt(var)
        """

        return (x - mean)/ torch.sqrt(var + 1e-6) 

    def forward(self, forcing, coords):
        """
        Forward pass of DeepONet.

        Steps:
        ------
        1. Normalize inputs
        2. Encode forcing via branch net
        3. Encode coordinates via trunk net
        4. Compute dot product
        5. Add bias
        """

        forcing = self.normalize(forcing, self.mean_forcing, self.var_forcing)
        coords = self.normalize(coords, self.mean_coord, self.var_coord)

        branch_output = self.branch(forcing)
        trunk_output = self.trunk(coords)

        # Operator evaluation (dot product)
        output = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)

        return self.bias(output)