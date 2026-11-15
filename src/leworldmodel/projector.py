# %%

from torch import nn 

# %%

class Projector(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim
        ):

        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, X):
        """
        X is the output of the Encoder (B * embedding_dim) 
        or the Predictor (B * N * embedding_dim)
        Need to reshape for the Predictor output because its 
        size is (B, N, embedding_dim) while BatchNorm1d requires 
        either (B, embedding_dim) or (B, embedding_dim, N)
        """
        X_r = X.reshape(-1, X.shape[-1])
        return self.proj(X_r).reshape(X.shape)