# %%

from torch import nn

# %%

class ActionEmbedder(nn.Module):
    """
    Architecture from the official repo of LeWM
    """

    def __init__(
            self, 
            frame_skip, 
            action_dim,
            input_dim=10, 
            hidden_dim=768, # = hidden dim mlp of encoder
            output_dim=192 # = embedding dim 
        ):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frame_skip * action_dim, input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, A):
        """
        Size of A is batch_size * N * (K * action_dim)
        where K is the frame skip, action_dim the dimension of 
        the action (example dim = 2 for a 2D displacement) and 
        N the history length (number of past latent states)
        """
        return self.mlp(A)