"""Continuous safety-score helpers for pi0.5 flow-matching guidance."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class SafetyValueMLP(nn.Module):
    def __init__(self, input_dim: int = 4096) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def safety_score(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(features))


def hidden_feature(hidden: np.ndarray) -> np.ndarray:
    """Summarize one action chunk shaped (tokens, hidden_dim)."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError(f"expected (tokens, hidden_dim), received {hidden.shape}")
    return np.concatenate(
        (hidden.mean(0), hidden.std(0), hidden.max(0), hidden[-1]),
        axis=0,
    ).astype(np.float32)


@dataclass
class ContinuousSafetyScorer:
    """Loads the trained continuous safety model and scores hidden states."""

    model: SafetyValueMLP
    mean: np.ndarray
    std: np.ndarray
    device: torch.device

    @classmethod
    def load(cls, run_dir: pathlib.Path, device: str = "cpu") -> "ContinuousSafetyScorer":
        device_obj = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device_obj)
        normalizer = np.load(run_dir / "normalizer.npz")
        model = SafetyValueMLP(int(checkpoint["input_dim"])).to(device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        mean = normalizer["mean"].astype(np.float32)
        std = normalizer["std"].astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(model=model, mean=mean, std=std, device=device_obj)

    def score_hidden_state(self, hidden_state: np.ndarray) -> float:
        """Return the predicted safety probability for a single chunk hidden state."""
        return float(self.score_hidden_states(np.asarray(hidden_state)[None, ...])[0])

    def score_hidden_states(self, hidden_states: np.ndarray) -> np.ndarray:
        """Score a batch of chunk hidden states shaped (batch, tokens, hidden_dim)."""
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        features = np.stack([hidden_feature(hidden) for hidden in hidden_states])
        features = (features - self.mean) / self.std
        with torch.no_grad():
            tensor = torch.from_numpy(features).float().to(self.device)
            scores = self.model.safety_score(tensor).detach().cpu().numpy()
        return scores.astype(np.float32)

    def score_response(self, response: dict) -> float:
        hidden_state = np.asarray(response["last_layer_hidden_state"], dtype=np.float32)
        return self.score_hidden_state(hidden_state)
