"""Independent small MLPs for objective regression, denoising, and amortization."""
import torch
from torch import nn


def mlp(inputs, outputs, width, depth):
    layers = []
    for _ in range(depth - 1):
        layers.extend((nn.Linear(inputs, width), nn.SiLU()))
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class Ensemble(nn.Module):
    def __init__(self, d, m, cfg, size):
        super().__init__()
        self.members = nn.ModuleList([mlp(d, m, cfg.hidden_dim, cfg.depth) for _ in range(size)])

    def forward(self, x):
        values = torch.stack([member(x) for member in self.members])
        return values.mean(0), values.std(0, unbiased=False)


class Denoiser(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.net = mlp(d + 1, d, cfg.hidden_dim, cfg.depth)

    def forward(self, noisy_x, sigma):
        return self.net(torch.cat((noisy_x, sigma.log()), dim=-1))


class ParetoMap(nn.Module):
    def __init__(self, d, m, cfg, lower, upper):
        super().__init__()
        self.net = mlp(m, d, cfg.hidden_dim, cfg.depth)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.bounded = cfg.bounded_map

    def forward(self, weights):
        x = self.net(weights)
        return self.lower + (self.upper - self.lower) * x.sigmoid() if self.bounded else x


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module
