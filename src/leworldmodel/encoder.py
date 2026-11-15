# %%

from torch import nn

from .projector import Projector
from .vision_transformer import VisionTransformer

# %%

class Encoder(nn.Module):
    """
    Encoder of LeWM. Applies the projection to the output 
    of VIT (CLS representation)
    """

    def __init__(
            self, 
            num_layers,
            patch_size,
            image_size,
            num_heads, 
            embedding_dim, 
            QK_dim, 
            V_dim, 
            hidden_dim_mlp,
            hidden_dim_proj
        ):

        super().__init__()
        self.vit = VisionTransformer(
            num_layers=num_layers,
            patch_size=patch_size,
            image_size=image_size,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            QK_dim=QK_dim,
            V_dim=V_dim,
            hidden_dim_mlp=hidden_dim_mlp
        )

        self.proj = Projector(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim_proj
        )

    def forward(self, X, mask=False):
        """
        X : (B, 3, 224, 224)
        Returns latent representations Z : B * embedding_dim (192)
        """
        X_cls = self.vit(X, mask)
        return self.proj(X_cls)