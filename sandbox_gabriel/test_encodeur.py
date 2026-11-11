# %%

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

# %%

torch.manual_seed(10)

# %%

B = 128
images = torch.randn(B, 3, 224, 224)

# %%

image_0 = images[0]
print(image_0.shape)

# %%

# image 3 * 224 * 224 ---> 16 * 16 * 3 * 14 * 14 ie on découpe l'image 
# en carrés de 3 * patch_size * patch_size, donc un objet de dim 16 * 16 * patches
# ensuite on concatène et on trouve 256 * 588, nn.Unfold le fait automatiquement pour 
# des donnés (B, C, H, W)

patch_size = 14
unfold = nn.Unfold(
    kernel_size=patch_size,
    stride=patch_size
)
patches = unfold(images).transpose(1,2)
print(patches.shape[-1])
# %%

embedding_dim = 192 # embedding_dim
linear = nn.Linear(
    patches.shape[-1],
    embedding_dim
)

patches = linear(patches)
print(patches.shape)

# cls token
cls_token = torch.randn((1,1,192))
# dans la classe il faudra utiliser nn.Parameter(data=cls_token)
batch_cls_token = cls_token.repeat(B,1,1)

# positional embedding matrix 
# à utiliser avec nn.Parameter quand on utilisera le réseau
E = torch.randn(1, 257, 192).repeat(B, 1, 1)

# %%

X = torch.cat([batch_cls_token, patches], dim=1)
X_0 = X + E

# %%

class AttentionHead(nn.Module):

    def __init__(
            self,
            embedding_dim,
            QK_dim,
            V_dim
    ):
        super().__init__()
        self.QK_dim = QK_dim
        self.W_Q = nn.Linear(embedding_dim, QK_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, QK_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, V_dim, bias=False)

    def forward(self, X, mask=False):

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        A = Q @ K.transpose(1,2)/np.sqrt(self.QK_dim)

        if mask:   
            causal_mask = torch.triu(torch.ones(X.shape[1], X.shape[1], dtype=torch.bool, device=X.device), diagonal=1)
            A = A.masked_fill(causal_mask, float("-inf"))
            
        S = softmax(A, dim=-1)
        
        return S @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, embedding_dim, QK_dim, V_dim):
        super().__init__()

        self.multi_head = nn.ModuleList(
            [AttentionHead(embedding_dim, QK_dim, V_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(
            num_heads * V_dim,
            embedding_dim
        )
        self.num_heads = num_heads

    def forward(self, X, mask=False):

        H = []
        for head in range(self.num_heads):
            H.append(self.multi_head[head](X, mask))

        return self.linear(torch.concat(H, dim=-1))

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

# %%

torch.randn(B, 257, 192) + torch.randn(1, 257, 192)
# %%
