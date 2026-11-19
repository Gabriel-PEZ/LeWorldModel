# %%

import torch
import torch.nn.functional as F

# %%

def lewm_loss(Z, Z_pred, lamb_reg=0.1):
    loss_mse = F.mse_loss(Z_pred, Z[:, 1:, :])
    loss_sigreg = SIGReg(Z, M=1024, a=0.0, b=3.0, J=17, lamb=1.0)

    loss = loss_mse + lamb_reg * loss_sigreg

    return loss

# %%

def SIGReg(Z, M, a, b, J, lamb):
    """
    Z : latent representations
    M : number of unit vectors for approximation of the Cramer Wold theorem
    a : lower bound of the integral
    b : upper bound of the integral 
    J : number of quadrature points 
    lamb : lambda in the weight function w(t) = exp(-t^2 / (2 * lambda^2))

    Returns the mean Epps-Pulley statistic over time steps and random projection 
    directions
    """

    U = torch.randn((Z.shape[-1], M), device=Z.device)
    U = U / torch.linalg.norm(U, dim=0).unsqueeze(0)
    H = Z @ U # (B, T, M) (=128, 4, 1024)

    quadrature_pts = torch.linspace(a, b, J, device=H.device)
  
    X = H.unsqueeze(-1) * quadrature_pts # (B, T, M, J) (=128, 4, 1024, 17)
    X = torch.exp(1j * X)  

    phi_hat = X.mean(dim=0) # mean on the batch dimension  (=4, 1024, 17)
    phi = torch.exp(-quadrature_pts**2 / 2)

    w = torch.exp(-quadrature_pts**2 / (2 * lamb**2)) # weights
    integrand = w * (phi_hat - phi).abs().square() # w(s) |\hat{phi}(s) - phi(s)|^2
    dt = (b - a) / (J - 1)

    # trapezoidal rule
    integral = 2 * dt * (
        (integrand[:, :, 0] + integrand[:, :, -1]) / 2
        + integrand[:, :, 1:-1].sum(dim=2)
    ) # (T, M) = (4, 1024)

    sigreg = Z.shape[0] * integral.mean()
    return sigreg