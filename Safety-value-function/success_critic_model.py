"""Action-conditioned rollout-success critic used by action-expert guidance."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, frequencies: int = 16) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(-1, 2, frequencies))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * value[..., None] * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class SuccessCritic(nn.Module):
    """Q_phi(hidden_t, action_t, t), with an explicit differentiable action path."""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        action_dim: int = 32,
        horizon: int = 10,
        width: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        time_frequencies: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.width = width
        self.hidden_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_embedding = FourierTimeEmbedding(time_frequencies)
        self.time_projection = nn.Sequential(
            nn.Linear(2 * time_frequencies, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.position = nn.Parameter(torch.empty(1, horizon, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.pool_query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.pool_query, std=width**-0.5)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        denoising_time: torch.Tensor,
    ) -> torch.Tensor:
        if denoising_time.ndim > 1:
            denoising_time = denoising_time.reshape(denoising_time.shape[0])
        token = self.hidden_projection(hidden_states) + self.action_projection(
            noisy_actions
        )
        token = (
            token
            + self.time_projection(self.time_embedding(denoising_time))[:, None, :]
        )
        token = token + self.position
        token = self.final_norm(self.transformer(token))
        weights = torch.softmax(
            torch.einsum("btd,d->bt", token, self.pool_query) / math.sqrt(self.width),
            dim=1,
        )
        pooled = torch.einsum("bt,btd->bd", weights, token)
        return self.head(pooled).squeeze(-1)
