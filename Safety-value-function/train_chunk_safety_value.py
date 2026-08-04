#!/usr/bin/env python3
"""Train a continuous [0, 1] safety value from pi0.5 action-chunk hidden states."""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        """Return logits for stable BCE training."""
        return self.net(features).squeeze(-1)

    def safety_score(self, features: torch.Tensor) -> torch.Tensor:
        """Return a continuous safety value: 0=unsafe, 1=safe."""
        return torch.sigmoid(self(features))


class SafetyValueLoss(nn.Module):
    """Asymmetric focal + calibration + safe/unsafe ranking objective."""

    def __init__(self, gamma: float = 2.0, unsafe_weight: float = 0.75, margin: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.unsafe_weight = unsafe_weight
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        target_probability = targets * scores + (1.0 - targets) * (1.0 - scores)
        class_weight = targets * (1.0 - self.unsafe_weight) + (1.0 - targets) * self.unsafe_weight
        focal = (
            -class_weight
            * (1.0 - target_probability).pow(self.gamma)
            * target_probability.log()
        ).mean()
        brier = (scores - targets).pow(2).mean()
        safe_scores = scores[targets > 0.5]
        unsafe_scores = scores[targets <= 0.5]
        if len(safe_scores) and len(unsafe_scores):
            ranking = torch.relu(
                self.margin - safe_scores[:, None] + unsafe_scores[None, :]
            ).mean()
        else:
            ranking = logits.new_zeros(())
        return focal + 0.25 * brier + 0.25 * ranking


def hidden_feature(hidden: np.ndarray) -> np.ndarray:
    """Summarize one pi0.5 action chunk shaped (tokens, hidden_dim)."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError(f"expected (tokens, hidden_dim), received {hidden.shape}")
    return np.concatenate(
        (hidden.mean(0), hidden.std(0), hidden.max(0), hidden[-1]), axis=0
    ).astype(np.float32)


def load_pairs(data_root: pathlib.Path):
    features, scores, groups, records = [], [], [], []
    for path in sorted(data_root.rglob("*_last_layer_hidden_states.npz")):
        with np.load(path) as data:
            hidden = data["last_layer_hidden_states"]
            actions = data["action_chunks"]
            safety = data["chunk_safety_scores"].astype(np.float32)
            starts = data["chunk_start_steps"]
        if not (len(hidden) == len(actions) == len(safety) == len(starts)):
            raise ValueError(f"unaligned chunk arrays: {path}")
        group = str(path.relative_to(data_root))
        for chunk_index in range(len(hidden)):
            features.append(hidden_feature(hidden[chunk_index]))
            scores.append(float(safety[chunk_index]))
            groups.append(group)
            records.append(
                {
                    "rollout": group,
                    "chunk_index": chunk_index,
                    "start_step": int(starts[chunk_index]),
                    "safety_score": float(safety[chunk_index]),
                    "action_chunk": actions[chunk_index].tolist(),
                }
            )
    if not features:
        raise FileNotFoundError(f"no chunk pairs found under {data_root}")
    return np.stack(features), np.asarray(scores, np.float32), np.asarray(groups), records


def split_groups(groups: np.ndarray, seed: int):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError("at least five completed rollouts are required for grouped splitting")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    test_count = max(1, round(0.2 * len(unique_groups)))
    val_count = max(1, round(0.2 * len(unique_groups)))
    test_groups = set(unique_groups[:test_count])
    val_groups = set(unique_groups[test_count : test_count + val_count])
    test = np.flatnonzero(np.asarray([group in test_groups for group in groups]))
    val = np.flatnonzero(np.asarray([group in val_groups for group in groups]))
    train = np.flatnonzero(
        np.asarray([group not in test_groups and group not in val_groups for group in groups])
    )
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT / "training_dataset/pi05_hidden_chunks")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "Safety-value-function/chunk_safety_value_run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    x, y, groups, records = load_pairs(args.data_root)
    train_idx, val_idx, test_idx = split_groups(groups, args.seed)
    mean, std = x[train_idx].mean(0), x[train_idx].std(0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    train_y = y[train_idx].astype(np.int64)
    counts = np.bincount(train_y, minlength=2)
    weights = np.asarray([1.0 / max(counts[label], 1) for label in train_y])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]).float(), torch.from_numpy(y[train_idx]).float()), batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[val_idx]).float(), torch.from_numpy(y[val_idx]).float()), batch_size=args.batch_size)
    model = SafetyValueMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = SafetyValueLoss()
    best_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward(); optimizer.step(); train_losses.append(float(loss))
        model.eval(); val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                val_losses.append(float(loss_fn(model(bx.to(device)), by.to(device))))
        val_loss = float(np.mean(val_losses))
        print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "input_dim": x.shape[1], "target": "continuous safety score: 0=unsafe, 1=safe"}, args.output_dir / "best_model.pt")

    np.savez_compressed(args.output_dir / "normalizer.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    split = np.full(len(records), "train", dtype=object); split[val_idx] = "val"; split[test_idx] = "test"
    for index, record in enumerate(records): record["split"] = str(split[index])
    (args.output_dir / "chunk_pairs.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"pairs={len(y)} safe={int(y.sum())} unsafe={int((y == 0).sum())} best_val_loss={best_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
