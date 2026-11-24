from typing import Any

import torch
import torch.nn as nn

class ChannelEmbedding(nn.Module):
    """Camgoz et al. (2020) eq. 3 without PosEnc: Softsign(BatchNorm(xW + b))"""
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.Softsign()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.activation(x)

class MultiStreamEmbedding(nn.Module):
    def __init__(self, in_dim: dict, embed_dim: int=128):
        super().__init__()
        self.streams = nn.ModuleDict({k: ChannelEmbedding(d, embed_dim) for k, d in in_dim.items()})

    def forward(self, inputs: dict) -> dict[str, Any]:
        outputs = {}
        for name in self.streams:
            layer = self.streams[name]
            outputs[name] = layer(inputs[name])
        return outputs