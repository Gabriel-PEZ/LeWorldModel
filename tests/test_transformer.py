# %%

import torch

from leworldmodel.transformer import TransformerBlockPredictor

# %%

def make_transformer():
    return TransformerBlockPredictor(
        num_heads=16,
        embedding_dim=192,
        QK_dim=64,
        V_dim=64,
        hidden_dim_mlp=2048,
        p_dropout=0.1
    )

def test_identity_at_initialization():
    """
    at init, transformer = identity due to zero init
    """
    torch.manual_seed(10)

    # exemple where B = 2, N = 3, z in R^192
    X = torch.randn(2, 3, 192)
    a = torch.randn(2, 3, 192)

    transformer = make_transformer()
    H = transformer(X, a)

    assert torch.allclose(H, X)