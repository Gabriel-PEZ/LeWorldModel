#%%
import torch
from torch import nn
from torch.nn import functional  as F # pour le dot product de l'attention

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, causal):
        super().__init__()
        self.heads = heads # nb de têtes
        self.dim_head = dim_head # dim par tête = dim/heads
        self.dim_qkv = self.heads * self.dim_head # dim interne
        self.causal = causal # pour le projector
        # projection globale pour toutes les têtes puis on sépare par tête
        self.key = nn.Linear(dim, self.dim_qkv)
        self.query = nn.Linear(dim, self.dim_qkv)
        self.value = nn.Linear(dim, self.dim_qkv)

        # donc chaque tête = self.dim_qkv/self.heads

        # à la fin on concat les têtes et on projette back à la dim des tokens
        self.out_layer = nn.Linear(self.dim_qkv, dim)

    def forward(self, x):
        # x sont les séquences de tokens (patches)
        # forme (B, L, dim)
        B, L, dim = x.shape

        # (B, L, dim_qkv)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # on sépare par têtes pour faire le calcul de similarité
        # on découpe puis change l'ordre pour le dot product
        # il faut tokens,dim en derniers axes
        q_head = q.view(B, L, self.heads, self.dim_head).transpose(1,2)
        k_head = k.view(B, L, self.heads, self.dim_head).transpose(1,2)
        v_head = v.view(B, L, self.heads, self.dim_head).transpose(1,2)

        # utilise l'implémentation pytorch
        # fais le softmax normalisé puis le dot product avec values
        # sort (B, heads, L, dim_head)

        x_attention_pytorch = F.scaled_dot_product_attention(q_head,k_head,v_head,is_causal = self.causal)

        # reshape après transpose car sinon non contigu
        merged_heads = x_attention_pytorch.transpose(1,2).reshape(B,L,self.dim_qkv)

        y = self.out_layer(merged_heads)
        return y

attention_layer = Attention(16,4,4,False)

