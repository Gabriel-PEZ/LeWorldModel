# %%

import torch.nn as nn

from .attention import MultiHeadAttention

# %%

class TransformerBlock(nn.Module):

    def __init__(
            self, 
            num_heads, 
            embedding_dim, 
            QK_dim, 
            V_dim, 
            hidden_dim_mlp,
        ):

        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            QK_dim=QK_dim,
            V_dim=V_dim,
        )

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),
            nn.GELU(),
            nn.Linear(hidden_dim_mlp, embedding_dim)
        )

    def forward(self, X, mask=False):

        Z = X + self.mha(self.ln1(X), mask)
        return Z + self.mlp(self.ln2(Z))