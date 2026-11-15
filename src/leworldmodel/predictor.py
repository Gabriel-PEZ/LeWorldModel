# %%

import torch
from torch import nn

from .action_embedder import ActionEmbedder
from .projector import Projector
from .transformer import TransformerBlockPredictor

# %%

class Predictor(nn.Module):

    def __init__(
            self, 
            num_layers,
            num_heads,
            embedding_dim,
            QK_dim,
            V_dim,
            hidden_dim_mlp,
            p_dropout,
            frame_skip,
            action_dim,
            hidden_dim_proj,
            N
        ):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockPredictor(
                num_heads=num_heads,
                embedding_dim=embedding_dim, 
                QK_dim=QK_dim, 
                V_dim=V_dim, 
                hidden_dim_mlp=hidden_dim_mlp,
                p_dropout=p_dropout
            ) for _ in range(num_layers)
        ])

        self.action_embedder = ActionEmbedder(
            frame_skip=frame_skip,
            action_dim=action_dim
        )
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.proj = Projector(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim_proj
        )

        self.num_layers = num_layers
        # positional embedding like in the vision transformer
        # but for the vit is used for the spatial position while 
        # here it encodes the temporal position
        self.E_pos = nn.Parameter(torch.randn(1, N, embedding_dim))

    def forward(self, Z, A, mask=True):
        """
        Z : (B, L, d)
        A : (B, L, K * action_dim)

        B : batch size
        L : input sequence length (L <= N) during training = T-1
        d : embedding dimension
        N : maximum history length
        K : frame skip

        For example if we have a sequence of observations o1,...,o20, with a frame skip 
        K = 5, we have z0 = Embedding(o0), z1 = Embedding(o5), z2 = Embedding(o10), 
        z3 = Embedding(o15), z4 = Embedding(o20). If T = 4, N = 3, X = (z1, z2, z3, z4), 
        hat(z2) = Predictor(z1, A1) where A1 = (a5, a6, a7, a8, a9)
        hat(z3) = Predictor(z1, z2, A1, A2) where A2 = (a10, a11, a12, a13, a14)
        hat(z4) = Predictor(z1, z2, z3, A1, A2, A3) where A3 = (a15, a16, a17, a18, a19)
        ie A_j = (a_Kj, a_{Kj+1}, ..., a_{Kj+K-1})
        """
        A = self.action_embedder(A)
        Z = Z + self.E_pos[:, :Z.shape[1], :] # in case where dim 2 of X is < N
        for layer in range(self.num_layers):
            Z = self.transformer_blocks[layer](Z, A, mask)
        return self.proj(self.ln(Z))