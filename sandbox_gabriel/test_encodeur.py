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
            causal_mask = torch.triu(torch.ones(X.shape[1], X.shape[1], dtype=torch.bool), diagonal=1, device=X.device)
            A = A.masked_fill(causal_mask, float("-inf"))
            
        S = softmax(A, dim=-1)
        
        return S @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, embedding_dim, QK_dim, V_dim, output_dim):
        super().__init__()

        self.multi_head = nn.ModuleList(
            [AttentionHead(embedding_dim, QK_dim, V_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(
            num_heads * V_dim,
            output_dim
        )
        self.num_heads = num_heads

    def forward(self, X, mask=False):

        H = []
        for head in range(self.num_heads):
            H.append(self.multi_head[head](X, mask))

        return self.linear(torch.concat(H, dim=-1))


# %%

