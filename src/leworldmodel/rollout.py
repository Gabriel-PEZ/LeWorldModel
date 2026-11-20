# %%

import torch

# %%

@torch.no_grad()
def rollout(model, Z, A_past, A_future):
    """
    Predict H future latent representations autoregressively.

    Inputs:
        Z : (B, L, d), history of L latent representations
        A_past : (B, L-1, K*action_dim), actions between the L past states
        A_future : (B, H, K*action_dim), H future action blocks to evaluate

    Output:
        Z_pred : (B, H, d), predicted latent trajectory

    Example:
        B = 16, L = 3, H = 5, d = 192, K = 5, action_dim = 2

        Z = (z_0, z_1, z_2)                  -> (16, 3, 192)
        A_past = (A_0, A_1)                 -> (16, 2, 10)
        A_future = (A_2, A_3, ..., A_6)     -> (16, 5, 10)

        Each A_i contains K=5 elementary actions of dimension 2.

        The rollout predicts:
        Z_pred = (z_hat_3, z_hat_4, ..., z_hat_7) -> (16, 5, 192)

        At each step, the predicted embedding is reused to predict
        the next one, keeping at most N=3 latent states in memory.

        For planning, Z_pred[:, -1, :] gives the final predicted
        embedding for each trajectory, of shape (B, d).
    """

    N = model.predictor.E_pos.shape[1]

    assert 1 <= Z.shape[1] <= N
    assert A_past.shape[1] == Z.shape[1] - 1
    assert A_future.shape[1] > 0

    predictions = []

    for t in range(A_future.shape[1]):

        A = torch.cat([A_past, A_future[:, t:t+1, :]], dim=1)

        Z_next = model.predictor(Z, A)[:, -1:, :]
        predictions.append(Z_next)

        Z = torch.cat([Z, Z_next], dim=1)[:, -N:, :]
        A_past = A[:, -(Z.shape[1]-1):, :]

    return torch.cat(predictions, dim=1)

# %%