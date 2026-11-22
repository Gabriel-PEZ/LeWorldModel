# %%

import torch

from leworldmodel.planning.cost import latent_cost
from leworldmodel.rollout import rollout

# %%

@torch.no_grad()
def cem(model, Z, A_past, Z_goal, H=5, n_samples=300, n_elites=30, n_iters=30, action_low=None, action_high=None):
    """
    Optimize a sequence of H action blocks using CEM.

    Inputs:
        Z : (1, L, d), initial latent history
        A_past : (1, L-1, K*action_dim), past actions
        Z_goal : (d,), target latent representation

        H : planning horizon in action blocks
        n_samples : number of candidate trajectories per iteration
        n_elites : number of best trajectories retained
        n_iters : number of CEM iterations
        action_low : (K*action_dim,), lower bounds in normalized action space
        action_high : (K*action_dim,), upper bounds in normalized action space

    Output:
        U : (H, K*action_dim), optimized action sequence

    Example:
        L = 3, H = 5, d = 192, K = 5, action_dim = 2

        Z = (z_0, z_1, z_2)              -> (1, 3, 192)
        A_past = (A_0, A_1)             -> (1, 2, 10)
        Z_goal = z_goal                 -> (192,)

        Initialize Gaussian distribution:
        mu = 0, sigma = 1               -> (5, 10)

        At each iteration:
        1. Sample 300 action sequences    -> (300, 5, 10)
        2. Predict 300 latent trajectories -> (300, 5, 192)
        3. Compute 300 terminal costs      -> (300,)
        4. Select the 30 best trajectories -> (30, 5, 10)
        5. Update mu and sigma from the 30 elite trajectories

        Repeat for 30 iterations.

        Return the final mean:
        U = mu                           -> (5, 10)

        Each action block contains K=5 elementary actions.
        The optimized sequence therefore represents 25 environment steps.
    """

    action_dim = A_past.shape[-1]

    mu = torch.zeros(H, action_dim, device=Z.device)
    sigma = torch.ones(H, action_dim, device=Z.device)

    for _ in range(n_iters):

        A_future = mu + sigma * torch.randn(n_samples, H, action_dim, device=Z.device)

        if action_low is not None:
            A_future = torch.maximum(A_future, action_low)

        if action_high is not None:
            A_future = torch.minimum(A_future, action_high)

        Z_pred = rollout(
            model,
            Z.expand(n_samples, -1, -1),
            A_past.expand(n_samples, -1, -1),
            A_future,
        )

        costs = latent_cost(Z_pred, Z_goal)

        elite_indices = torch.topk(costs, n_elites, largest=False).indices
        elite_actions = A_future[elite_indices]

        mu = elite_actions.mean(dim=0)
        sigma = elite_actions.std(dim=0, unbiased=False).clamp_min(1e-3)

    return mu

# %%