# %%

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

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