import torch
from lie_transformer_pytorch import LieTransformer

def test_transformer():
    model = LieTransformer(
        dim = 512,
        depth = 1
    )

    feats = torch.randn(1, 64, 512)
    coors = torch.randn(1, 64, 3)
    mask = torch.ones(1, 64).bool()

    out = model(feats, coors, mask = mask)
    assert out.shape == (1, 256, 512), 'transformer runs'
