import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

class LieTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
