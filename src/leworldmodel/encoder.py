# %%

import torch.nn as nn

from .vision_transformer import VisionTransformer

# %%

class Encoder(nn.Module):

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

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, X, mask=False):
        X_cls = self.vit(X, mask)
        return self.bn(self.mlp(X_cls))