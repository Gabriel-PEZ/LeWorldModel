# %%

import torch

from leworldmodel.model import LeWorldModel

# %%

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

# %%

torch.manual_seed(10)
B = 128
T = 4
K = 5

images = torch.randn(B, T, 3, 224, 224).to(device)
actions = torch.randn(B, T-1, K * 2).to(device)

# %%

params_encoder = {
    "num_layers":12,
    "patch_size":14,
    "image_size":224,
    "num_heads":3, 
    "embedding_dim":192, 
    "QK_dim":64, 
    "V_dim":64, 
    "hidden_dim_mlp":768,
    "hidden_dim_proj":2048
}

params_predictor = {
    "num_layers":6,
    "num_heads":16,
    "embedding_dim":192,
    "QK_dim":64,
    "V_dim":64,
    "hidden_dim_mlp":2048,
    "p_dropout":0.1,
    "frame_skip":5,
    "action_dim":2,
    "hidden_dim_proj":2048,
    "N":3
}

# %%

model = LeWorldModel(
    params_encoder=params_encoder,
    params_predictor=params_predictor
).to(device)

print(next(model.parameters()).device)

Z, Z_pred = model(images, actions)

# %%