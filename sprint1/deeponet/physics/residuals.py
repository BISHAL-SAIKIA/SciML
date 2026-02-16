"""
Current implementation:
-----------------------
1. ODE residual for:
       du/dt = f(t)

This residual can be extended to PDEs later.
"""
import torch

def ODE_residual_calculator(t, u, u_t, model):
    """
    Computes the residual of the governing ODE:

        du/dt - u_t = 0

    This function is the physics constraint used
    during training.
    Parameters
    ----------
    t : torch.Tensor
        Temporal coordinate, shape [batch_size, 1]
        Must have requires_grad=True for autograd

    u : torch.Tensor
        Forcing function samples (branch input),
        shape [batch_size, forcing_dim]

    u_t : torch.Tensor
        True forcing evaluated at time t,
        shape [batch_size, 1]

    model : torch.nn.Module
        DeepONet model

    Returns
    -------
    residual : torch.Tensor
        Physics residual, shape [batch_size, 1]

    Intuition
    ---------
    - The DeepONet predicts u(t)
    - Autograd computes du/dt
    - Residual penalizes violation of physics
    """

    # ensure gradient wrt time
    t.requires_grad_(True)

    #forward pass through deeponet
    s = model(u, t)

    #compute time derivative using autograd
    ds_dt = torch.autograd.grad(
        outputs=s,
        inputs=t,
        grad_outputs = torch.ones_like(s),
        create_graph= True
    )[0]

    #ODE residual
    residual = ds_dt - u_t 

    return residual
