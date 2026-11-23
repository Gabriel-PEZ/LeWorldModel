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

# %%
x = torch.randn(5, 120, 16)
y = attention_layer(x)
y

# %%
# class pour la fin de l'encodeur (post CLS) et pour le predictor, post AdaLN sur les actions

class Projector(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_in)
        )

    def forward(self,x):

        # le problème : 
        # la fin de l'encodeur est de forme (B,dim) -> ok pour batchnorm
        # l'input du predictor est de forme (B, L, dim) avec L le nombre de latents (transfomés par conditonnement) traités
        # genre (Z0, Z5, Z10)
        # du coup on passe en (BxL, dim)
        d = x.shape[-1]
        x_standard = x.reshape(-1, d)
        y_standard = self.MLP(x_standard)
        y = y_standard.reshape_as(x)

        return y

x_3 = torch.randn(5,3,120)
projector_layer = Projector(120,240)
print(projector_layer(x_3))

x_2 = torch.randn(5,120)
projector_layer = Projector(120,240)
print(projector_layer(x_2))
