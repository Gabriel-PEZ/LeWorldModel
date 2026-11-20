# %%

import torch
from leworldmodel.planning.cost import latent_cost

from leworldmodel.rollout import rollout

# %%

@torch.no_grad()
def mppi(model, Z, A_past, Z_goal, H=5, n_samples=300, n_iters=30, temperature=10.0):
    """
    Optimize a sequence of H action blocks using MPPI.

    Inputs:
        Z : (1, L, d), initial latent history
        A_past : (1, L-1, K*action_dim), past actions
        Z_goal : (d,), target latent representation

        H : planning horizon in action blocks
        n_samples : number of candidate trajectories per iteration
        n_iters : number of MPPI iterations
        temperature : controls the weighting of trajectories

    Output:
        U : (H, K*action_dim), optimized action sequence

    Example:
        L = 3, H = 5, d = 192, K = 5, action_dim = 2

        Z = (z_0, z_1, z_2)              -> (1, 3, 192)
        A_past = (A_0, A_1)             -> (1, 2, 10)
        Z_goal = z_goal                 -> (192,)

        At each iteration:
        1. Sample 300 action sequences    -> (300, 5, 10)
        2. Predict 300 latent trajectories -> (300, 5, 192)
        3. Compute 300 terminal costs      -> (300,)
        4. Compute normalized weights      -> (300,)
        5. Update mu using weighted perturbations

        Repeat for 30 iterations.

        Return the final mean:
        U = mu                           -> (5, 10)
    """

    action_dim = A_past.shape[-1]

    mu = torch.zeros(H, action_dim, device=Z.device)
    sigma = torch.ones(H, action_dim, device=Z.device)

    for _ in range(n_iters):

        noise = sigma * torch.randn(n_samples, H, action_dim, device=Z.device)
        A_future = mu + noise

        Z_pred = rollout(
            model,
            Z.expand(n_samples, -1, -1),
            A_past.expand(n_samples, -1, -1),
            A_future,
        )

        costs = latent_cost(Z_pred, Z_goal)
        weights = torch.softmax(-costs / temperature, dim=0)

        mu = mu + (weights[:, None, None] * noise).sum(dim=0)

    return mu

# %%