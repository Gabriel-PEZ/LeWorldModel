# %%

import torch
from torch import nn

# %%

class AdaLNZero(nn.Module):
    """
    Take as input the embedded action and returns 
    (gamma_1, beta_1, g_1, gamma_2, beta_2, g_2) 
    used in the transformer block : 
    Z = X + g_1 o MHA((1 + gamma_1) o LN(X) + beta_1)
    H = Z + g_2 o MLP((1 + gamma_2) o LN(Z) + beta_2)
    where LN does not learn gamma and beta 

    Weight and bias of the last linear layer are set to 0
    """

    def __init__(self, embedding_dim):

        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 6 * embedding_dim)
        )
        # one column parameter (gamma_1, beta_1, g_1, gamma_2, beta_2, g_2)

        # W and b set to 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, a):

        params = torch.chunk(self.mlp(a), chunks=6, dim=-1)
        return params
