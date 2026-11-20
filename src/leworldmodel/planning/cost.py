# %%

def latent_cost(Z_pred, Z_goal):
    """
    Z_pred : (B, H, d), predicted latent trajectories
    Z_goal : (d,), target latent representation

    Output : (B,), one cost per trajectory
    ie J(U) = ||hat(Z_{t+H})(U) - Z_goal||_2^2
    """

    Z_final = Z_pred[:, -1, :]
    return ((Z_final - Z_goal) ** 2).sum(dim=-1)

# %%