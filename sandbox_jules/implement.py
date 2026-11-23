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

#%%
# un block transformer
# LN(X) -> MHA(LN(X)) -> X1 = X + MHA(LN(X)) ->  X1 + MLP(LN(X1))
class BlockVIT(nn.Module):
    # pas besoin de causal pour attention car c'est l'encoder
    def __init__(self, dim_in, hidden_dim, nb_heads, dim_head):
        super().__init__()

        self.LN1 = nn.LayerNorm(dim_in)
        self.attention = Attention(dim_in, nb_heads, dim_head, causal = False)
        self.LN2 = nn.LayerNorm(dim_in)
        self.MLP = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_in),
        )
    def forward(self, x):
        # x = (B, L, d)
        B, L, d = x.shape
        x_ln = self.LN1(x)
        x_att = self.attention(x_ln)
        x1 = x + x_att
        x1_ln = self.LN2(x1)
        y = x1 + self.MLP(x1_ln)
        return y

#%%
# part des images (3, H, W)
# transforme via les patchs
# ajout CLS
# embedding + embedding pos
# BlockVIT
# Extraction CLS
# LN
# Projector

class Embedding(nn.Module):
    def __init__(self, size_img, dim_token, size_patch):
        super().__init__()
        # size_img = scalaire = hauteur x largeur car image carée
        # dim_token = dim pour embedding
        # size_patch = scalaire =  hauteur x largeur du patch car patch carré

        # pour le kernel on choisit size_patch car on veut résumer un patch en une valeur
        # valeur de dimension dim_token
        # on a Lout = (Lin - k / s) + 1 sans padding
        # ça donne (224-14)/s + 1 = size_imag/size_patch donc s = size_patch
        self.conv_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=dim_token,
            kernel_size=size_patch,
            stride = size_patch,
        )

        # -> part de (B, 3, 224, 224) et donne (B, 192, 16, 16)
        # puis on flatten (B, 256, 192)

        # initialization CLS
        self.CLS = nn.Parameter(torch.zeros(1,1,dim_token))

        # initialization pos
        self.nb_patch = (size_img // size_patch)**2 + 1
        self.pos = nn.Parameter(torch.randn(1, self.nb_patch, dim_token) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape

        x_conv = self.conv_embedding(x)

        # B, dim, patchs
        x_flat = x_conv.flatten(2,3)

        # il faut transpose pour avoir patchs, dim
        x_tokens = x_flat.transpose(2, 1)
        # B, L, dim

        # ajout CLS au début
        # faut la même taille pour les autres axes
        # repeat répète CLS B fois et conserve 1 et dim token sur le reste
        x_tokens = torch.cat([self.CLS.repeat(B, 1, 1), x_tokens], dim = 1)

        # ajout pos
        x_tokens = x_tokens + self.pos

        return x_tokens

class Encoder(nn.Module):
    def __init__(self, size_img, dim_token, size_patch, nb_blocks, hidden_dim_transformer, nb_heads, dim_head, hidden_dim_proj):
        super().__init__()
        self.patchembedding = Embedding(size_img, dim_token, size_patch)
        # crée nb_blocks blocks de transformers différents pour apprendre des transformations différentes
        self.blocks = nn.ModuleList(BlockVIT(dim_token, hidden_dim_transformer, nb_heads, dim_head) for i in range(nb_blocks))
        self.LN = nn.LayerNorm(dim_token)
        self.projector = Projector(dim_token, hidden_dim_proj)
    def forward(self, x):
        x = self.patchembedding(x)
        for block_layer in self.blocks:
            x = block_layer(x)
        y = self.LN(x)
        y_CLS = y[:,0,:]
        z = self.projector(y_CLS)
        return z

        
x_image = torch.randn(4, 3, 224, 224)
encoder = Encoder(
    size_img=224,
    dim_token=192,
    size_patch=14,
    nb_blocks=12,
    hidden_dim_transformer=768,
    nb_heads=3,
    dim_head=64,
    hidden_dim_proj=2048,
)
z = encoder(x_image)
z.shape
