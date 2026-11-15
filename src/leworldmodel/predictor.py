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

    def forward(self, X, a, mask=True):
        """
        X : (B, N, d) B = batch_size, N = history length, d = embedding_dim
        """
        a = self.action_embedder(a)
        X = X + self.E_pos[:, :X.sahpe[1], :] # in case where dim 2 of X is < N
        for layer in range(self.num_layers):
            X = self.transformer_blocks[layer](X, a, mask)
        return self.proj(self.ln(X))