# %%

from torch import nn

from .encoder import Encoder
from .predictor import Predictor

# %%

class LeWorldModel(nn.Module):
    """
    LeWorldModel
    params_encoder : dict 
    params_predictor : dict 
    """

    def __init__(
            self,
            params_encoder,
            params_predictor
    ):

        super().__init__()
        self.encoder = Encoder(
            **params_encoder
        )

        self.predictor = Predictor(
            **params_predictor
        )

    def forward(self, X, A):
        """
        X : (B, T, 3, image_size, image_size) -> encoder -> Z : (B, T, d=embedding_dim=192)
        A : (B, T-1, K * action_dim), K = frame skip
        Warning : in the code it is supposed that T = N + 1 (it works due to the causal mask)
        (because in the paper the authors use T = 4 et N = 3). If T > N+1
        then, the code must be adapted in order to take account only to the last N 
        observations. For example, if N = 3, T = 10, then
        hat(z_10) = f(z_1, ..., z_9) instead of hat(z_10) = f(z_7, z_8, z_9)
        If T = 4, the causal mask insures that hat(z_10) = f(z_7, z_8, z_9)
        since the sequence is (z_7, z_8, z_9, z_10)

        Returns : 
            - Z : (B, T, embedding_dim) the learned representations from the encoder 
            - Z_pred : (B, T-1, embedding_dim) the predictions 
            hat(Z_{t+1}) = Pred(Z_t, Z_{t-1}, Z_{t-3}, A_t, A_{t-1}, A_{t-3}) 
        """

        # encoder accepts only (B, 3, image_size, image_size)
        # so (B, T, 3, image_size, image_size) reshaped as
        # (BT, 3, image_size, image_size) -> encoder -> (BT, 192)
        # -> reshape -> (B, T, 192)
        Z = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Z = self.encoder(Z).reshape(X.shape[0], X.shape[1], -1)

        Z_pred = self.predictor(Z[:, :-1, :], A)
        return Z, Z_pred


# %%