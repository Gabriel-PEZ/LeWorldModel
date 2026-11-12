# %%

import torch
import torch.nn as nn

from .transformer import TransformerBlock

# %%

class VisionTransformer(nn.Module):

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
        self.num_layers = num_layers
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_size
        )

        self.linear = nn.Linear(
            3 * patch_size**2,
            embedding_dim
        )

        num_patches = (image_size // patch_size) ** 2 # 16*16 pour patch_size = 14

        # pas de détail dans LeWM mais dans VIT cls init à 0 et E_pos à N(0,0.02^2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.E_pos = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_heads, embedding_dim, QK_dim, V_dim, hidden_dim_mlp) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embedding_dim)


    def forward(self, X, mask=False):

        patches = self.linear(self.unfold(X).transpose(1,2))
        batch_cls_token = self.cls_token.repeat(X.shape[0],1,1)
        X = torch.cat([batch_cls_token, patches], dim=1)
        X = X + self.E_pos # pos embedding

        for layer in range(self.num_layers):
            X = self.transformer_blocks[layer](X, mask)
        X = self.ln(X) # normalisation finale
        return X[:,0,:] # return uniquement token_cls