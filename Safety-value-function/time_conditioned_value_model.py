"""Compact token model for time-conditioned pi0.5 safety value estimation."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, frequencies: int = 16):
        super().__init__()
        values = torch.logspace(-1, 2, frequencies)
        self.register_buffer("frequencies", values, persistent=True)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * time[..., None] * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimeConditionedSafetyValue(nn.Module):
    """Predict collision safety and signed clearance from denoising tokens."""

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
        bilinear_action_head: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.width = width
        self.bilinear_action_head = bilinear_action_head
        self.hidden_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_embedding = FourierTimeEmbedding(time_frequencies)
        self.time_projection = nn.Sequential(
            nn.Linear(2 * time_frequencies, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.position = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.pool_query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.pool_query, std=width**-0.5)
        self.safety_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.clearance_head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        if bilinear_action_head:
            self.action_condition_norm = nn.LayerNorm(width)
            self.safety_action_head = nn.Linear(width, action_dim)
            self.clearance_action_head = nn.Linear(width, action_dim)
            nn.init.normal_(self.safety_action_head.weight, std=0.01)
            nn.init.zeros_(self.safety_action_head.bias)
            nn.init.normal_(self.clearance_action_head.weight, std=0.01)
            nn.init.zeros_(self.clearance_action_head.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        denoising_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if denoising_time.ndim > 1:
            denoising_time = denoising_time.reshape(denoising_time.shape[0])
        context_token = self.hidden_projection(hidden_states)
        context_token = context_token + self.time_projection(
            self.time_embedding(denoising_time)
        )[:, None, :]
        context_token = context_token + self.position
        token = context_token
        if not self.bilinear_action_head:
            token = token + self.action_projection(noisy_actions)
        token = self.final_norm(self.transformer(token))
        weights = torch.softmax(
            torch.einsum("btd,d->bt", token, self.pool_query) / math.sqrt(self.width),
            dim=1,
        )
        pooled = torch.einsum("bt,btd->bd", weights, token)
        safety = self.safety_head(pooled).squeeze(-1)
        clearance = self.clearance_head(pooled).squeeze(-1)
        if self.bilinear_action_head:
            action_condition = self.action_condition_norm(context_token)
            normalizer = math.sqrt(self.horizon * self.action_dim)
            safety = safety + torch.sum(
                self.safety_action_head(action_condition) * noisy_actions,
                dim=(1, 2),
            ) / normalizer
            clearance = clearance + torch.sum(
                self.clearance_action_head(action_condition) * noisy_actions,
                dim=(1, 2),
            ) / normalizer
        return safety, clearance
