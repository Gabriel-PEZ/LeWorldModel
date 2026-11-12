# %%

import torch

from leworldmodel.encoder import Encoder

# %%

torch.manual_seed(10)
B = 128
images = torch.randn(B, 3, 224, 224)

# %%

encoder = Encoder(
    num_layers=12,
    patch_size=14,
    image_size=224,
    num_heads=3, 
    embedding_dim=192, 
    QK_dim=64, 
    V_dim=64, 
    hidden_dim_mlp=768,
)

encoder(images)

# %%

