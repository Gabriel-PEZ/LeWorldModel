# %%
import torch
from torch import nn
from torch.nn import functional  as F # pour le dot product de l'attention


def sigreg(z, num_proj = 1024, K = 17, t_max = 3.0):

    # directions. on en tire 1024 de dimensions 192
    u = F.normalize(torch.randn(z.shape[-1], num_proj, device = z.device), dim = 0)

    # (B, T )
    proj = z @ u
    t = torch.linspace(0.0, t_max, K, device = z.device)
    phi_theoric = torch.exp(-(t**2)/2)

    # étendre proj à [B, T, 1024,  1]
    # ty est [B, T, 2024, 17]
    ty = proj.unsqueeze(-1) * t

    # partie cos lié à la var
    C_n = ty.cos().mean(dim=0)

    # partie sin lié à la moyenne
    S_n = ty.sin().mean(dim=0)

    # phi empirique - phi theoric
    err_sq = (C_n - phi_theoric) ** 2 + S_n ** 2
    # (T, 1024, 17)
    f = phi_theoric * err_sq
    # (T, 1024) car l'intégrale est paire
    # on intègre f sur les t via la méthode trapezoides
    T_stat = 2.0 * torch.trapezoid(f, t, dim=-1)
    T_mean = T_stat.mean() # moyenne sur T et sur 1024 
    return z.shape[0] * T_stat.mean()  # pour la même échelle que le repo officiel

def lewm_loss(z, z_pred, lam):
    mse = F.mse_loss(z_pred, z[:, 1:, :])
    sig = sigreg(z)
    return mse + lam * sig, mse, sig
    