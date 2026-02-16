"""
===========================================================
Custom Neural Network Layers (models/layers.py)
===========================================================
-----------------
1. BiasLayer:
   - Adds a learnable scalar bias
   - Used by DeepONet to improve expressiveness

This file is intentionally minimal.
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------
# Bias Layer
# ----------------------------------------------------------
class BiasLayer(nn.Module):
    """
    BiasLayer
    ---------
    Adds a trainable scalar bias to the network output.

    Mathematical form:
        y = x + b

    Why this matters:
    -----------------
    - Helps match constant offsets in solutions
    - Part of the original DeepONet formulation
    - Improves convergence in operator learning
    """
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.bias
