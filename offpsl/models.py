from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be at least 1.")
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class OffPSLGenerator(nn.Module):
    def __init__(
        self,
        preference_dim: int,
        design_dim: int,
        hidden_dim: int,
        depth: int,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = make_mlp(preference_dim, design_dim, hidden_dim, depth)
        self.register_buffer("lower_bound", lower_bound.reshape(1, -1).float())
        self.register_buffer("upper_bound", upper_bound.reshape(1, -1).float())

    def forward(self, preferences: torch.Tensor) -> torch.Tensor:
        raw = self.net(preferences)
        span = self.upper_bound - self.lower_bound
        return self.lower_bound + span * torch.sigmoid(raw)


class SurrogateEnsemble(nn.Module):
    def __init__(
        self,
        design_dim: int,
        objective_dim: int,
        ensemble_size: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                make_mlp(design_dim, objective_dim, hidden_dim, depth, dropout=0.05)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(x) for member in self.members], dim=0)

    def mean_and_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = self.forward(x)
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
