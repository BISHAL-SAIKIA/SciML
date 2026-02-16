"""
===========================================================
Dataset Definitions (data/dataset.py)
===========================================================

This file defines PyTorch Dataset classes used for training
Physics-Informed Neural Operators (PINO).

Purpose:
--------
- Wrap raw tensors into PyTorch Dataset objects
- Enable batching, shuffling, and DataLoader usage
- Keep data-handling logic separate from training logic

Design choice:
--------------
We use a SINGLE dataset format for both:
- ODE residual training data
- Initial condition data

This keeps the training loop clean and consistent.
"""

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------
# PINO Dataset
# ----------------------------------------------------------
class PINODataset(Dataset):
    """
    Physics-Informed Neural Operator Dataset

    Each sample has the following structure:
        [ t | forcing_samples | target ]

    Where:
    -------
    t               : time coordinate
    forcing_samples : discretized forcing function
    target          : f(t) (for ODE residual) or dummy (for IC)

    This dataset is intentionally generic so it can be reused
    for ODEs, PDEs, ICs, and BCs.
    """

    def __init__(self, X):
        """
        Parameters
        ----------
        X : torch.Tensor
            Shape [N, 1 + forcing_dim + 1]

            Column layout:
                X[:, :1]   -> time t
                X[:, 1:-1] -> forcing samples
                X[:, -1:]  -> target value
        """
        super().__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns a single training sample.

        Output:
        -------
        t        : Tensor [1]
        forcing  : Tensor [forcing_dim]
        target   : Tensor [1]
        """
        sample = self.X[idx]

        t = sample[:1]
        forcing = sample[1:-1]
        target = sample[-1:]

        return t, forcing, target
