"""PolicyValueModel and PolicyOutput — PyTorch neural network for Orbit Wars."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyOutput:
    action_type_logits: torch.Tensor  # (B, 2)
    source_logits: torch.Tensor       # (B, MAX_PLANETS)
    target_logits: torch.Tensor       # (B, MAX_PLANETS)
    amount_logits: torch.Tensor       # (B, N_BINS)
    value: torch.Tensor               # (B, 1)


@dataclass
class PolicyValueConfig:
    input_dim: int
    hidden_dims: list

    max_planets: int = 50
    n_amount_bins: int = 5
    dropout: float = 0.1


class PolicyValueModel(nn.Module):
    def __init__(self, config: PolicyValueConfig) -> None:
        super().__init__()
        self.config = config

        # Build MLP encoder as a public nn.Sequential (swappable in future)
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        repr_dim = config.hidden_dims[-1]
        self.action_type_head = nn.Linear(repr_dim, 2)
        self.source_head = nn.Linear(repr_dim, config.max_planets)
        self.target_head = nn.Linear(repr_dim, config.max_planets)
        self.amount_head = nn.Linear(repr_dim, config.n_amount_bins)
        self.value_head = nn.Linear(repr_dim, 1)

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        shared = self.encoder(state)
        return PolicyOutput(
            action_type_logits=self.action_type_head(shared),
            source_logits=self.source_head(shared),
            target_logits=self.target_head(shared),
            amount_logits=self.amount_head(shared),
            value=torch.tanh(self.value_head(shared)),
        )
