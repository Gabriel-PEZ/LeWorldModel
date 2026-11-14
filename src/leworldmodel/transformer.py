# %%

from torch import nn

from .adaln import AdaLNZero
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


# %%

class TransformerBlockPredictor(nn.Module):

    def __init__(
            self, 
            num_heads, 
            embedding_dim, 
            QK_dim, 
            V_dim, 
            hidden_dim_mlp,
            p_dropout
        ):

        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            QK_dim=QK_dim,
            V_dim=V_dim,
            p_dropout=p_dropout
        )

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(hidden_dim_mlp, embedding_dim),
            nn.Dropout(p=p_dropout)
        )

        self.adaln_zero = AdaLNZero(embedding_dim=embedding_dim)

    def forward(self, X, a, mask=True):

        gamma_1, beta_1, g_1, gamma_2, beta_2, g_2 = self.adaln_zero(a)
        Z = X + g_1 * self.mha((1 + gamma_1) * self.ln1(X) + beta_1, mask)
        return Z + g_2 * self.mlp((1 + gamma_2) * self.ln2(Z) + beta_2)