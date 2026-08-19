#!/usr/bin/env python3
"""Train a continuous [0, 1] safety value from pi0.5 action-chunk hidden states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

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


class SoftCalibrationLoss(nn.Module):
    """Proper scoring rule for continuous future-safety probabilities."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        brier = (torch.sigmoid(logits) - targets).pow(2).mean()
        return bce + 0.25 * brier


def hidden_feature(hidden: np.ndarray) -> np.ndarray:
    """Summarize one pi0.5 action chunk shaped (tokens, hidden_dim)."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError(f"expected (tokens, hidden_dim), received {hidden.shape}")
    return np.concatenate(
        (hidden.mean(0), hidden.std(0), hidden.max(0), hidden[-1]), axis=0
    ).astype(np.float32)


def load_pairs(
    data_root: pathlib.Path,
    *,
    target: str = "chunk",
    hazard_lookahead: int = 20,
    hazard_tau: float = 10.0,
):
    features, scores, groups, records = [], [], [], []
    for path in sorted(data_root.rglob("*_last_layer_hidden_states.npz")):
        with np.load(path) as data:
            hidden = data["last_layer_hidden_states"]
            actions = data["action_chunks"]
            safety = data["chunk_safety_scores"].astype(np.float32)
            starts = data["chunk_start_steps"]
            executed_steps = data["executed_action_steps"]
            collision_flags = data["per_action_collision_flags"]
        if not (len(hidden) == len(actions) == len(safety) == len(starts)):
            raise ValueError(f"unaligned chunk arrays: {path}")
        group = str(path.relative_to(data_root))
        collision_by_step = dict(
            zip(executed_steps.tolist(), collision_flags.tolist())
        )
        for chunk_index in range(len(hidden)):
            chunk_target = float(safety[chunk_index])
            if target == "future_hazard":
                start = int(starts[chunk_index])
                discounted_hazards = [
                    np.exp(-offset / hazard_tau)
                    for offset in range(hazard_lookahead)
                    if collision_by_step.get(start + offset, False)
                ]
                chunk_target = 1.0 - max(discounted_hazards, default=0.0)
            features.append(hidden_feature(hidden[chunk_index]))
            scores.append(chunk_target)
            groups.append(group)
            records.append(
                {
                    "rollout": group,
                    "chunk_index": chunk_index,
                    "start_step": int(starts[chunk_index]),
                    "safety_score": chunk_target,
                    "original_chunk_safety_score": float(safety[chunk_index]),
                    "action_chunk": actions[chunk_index].tolist(),
                }
            )
    if not features:
        raise FileNotFoundError(f"no chunk pairs found under {data_root}")
    return np.stack(features), np.asarray(scores, np.float32), np.asarray(groups), records


def _group_stratum(group: str) -> str:
    """Keep every task/level represented in both train and validation."""
    parts = pathlib.Path(group).parts
    if len(parts) < 2:
        raise ValueError(f"cannot infer task/level stratum from rollout path: {group}")
    return f"{parts[0]}/{parts[1]}"


def split_train_val_groups(
    groups: np.ndarray,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError("at least five completed rollouts are required for grouped splitting")
    strata: dict[str, list[str]] = {}
    for group in unique_groups:
        strata.setdefault(_group_stratum(str(group)), []).append(str(group))
    rng = np.random.RandomState(seed)
    validation_groups: set[str] = set()
    for stratum, stratum_groups in sorted(strata.items()):
        if len(stratum_groups) < 2:
            raise ValueError(
                f"stratum {stratum!r} has only {len(stratum_groups)} rollout; "
                "cannot make disjoint train/validation splits"
            )
        shuffled = np.asarray(sorted(stratum_groups), dtype=object)
        rng.shuffle(shuffled)
        validation_count = min(
            len(shuffled) - 1,
            max(1, round(validation_fraction * len(shuffled))),
        )
        validation_groups.update(str(group) for group in shuffled[:validation_count])
    val = np.flatnonzero(
        np.asarray([str(group) in validation_groups for group in groups])
    )
    train = np.flatnonzero(
        np.asarray([str(group) not in validation_groups for group in groups])
    )
    return train, val


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


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata(root: pathlib.Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        status = run("git", "status", "--porcelain")
        diff = subprocess.check_output(
            ("git", "diff", "--binary", "HEAD"), cwd=root
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "changed_paths": [line[3:] for line in status.splitlines()],
    }


def _dataset_fingerprint(data_root: pathlib.Path) -> dict[str, object]:
    files = sorted(data_root.rglob("*_last_layer_hidden_states.npz"))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = str(path.relative_to(data_root))
        file_size = path.stat().st_size
        file_digest = _sha256(path)
        digest.update(f"{relative}\0{file_size}\0{file_digest}\n".encode())
        total_bytes += file_size
    return {
        "root": str(data_root.resolve()),
        "rollout_files": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _runtime_metadata(device: torch.device) -> dict[str, object]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    return {
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=ROOT / "training_dataset/pi05_hidden_chunks")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "Safety-value-function/chunk_safety_value_run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--target", default="chunk", choices=("chunk", "future_hazard"))
    parser.add_argument("--hazard-lookahead", type=int, default=20)
    parser.add_argument("--hazard-tau", type=float, default=10.0)
    parser.add_argument("--loss", default="asymmetric_focal", choices=("asymmetric_focal", "soft_bce"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--external-test", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    x, y, groups, records = load_pairs(
        args.data_root,
        target=args.target,
        hazard_lookahead=args.hazard_lookahead,
        hazard_tau=args.hazard_tau,
    )
    if args.external_test:
        train_idx, val_idx = split_train_val_groups(
            groups,
            args.split_seed,
            args.validation_fraction,
        )
        test_idx = np.empty((0,), dtype=np.int64)
    else:
        train_idx, val_idx, test_idx = split_groups(groups, args.split_seed)
    train_groups = sorted(set(groups[train_idx].tolist()))
    val_groups = sorted(set(groups[val_idx].tolist()))
    test_groups = sorted(set(groups[test_idx].tolist()))
    if set(train_groups) & set(val_groups):
        raise RuntimeError("rollout leakage between training and validation")
    print(
        f"split train_rollouts={len(train_groups)} val_rollouts={len(val_groups)} "
        f"internal_test_rollouts={len(test_groups)} train_chunks={len(train_idx)} "
        f"val_chunks={len(val_idx)} external_test={args.external_test}",
        flush=True,
    )
    mean, std = x[train_idx].mean(0), x[train_idx].std(0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    train_dataset = TensorDataset(
        torch.from_numpy(x[train_idx]).float(), torch.from_numpy(y[train_idx]).float()
    )
    if args.loss == "asymmetric_focal":
        train_y = (y[train_idx] >= 0.5).astype(np.int64)
        counts = np.bincount(train_y, minlength=2)
        weights = np.asarray([1.0 / max(counts[label], 1) for label in train_y])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[val_idx]).float(), torch.from_numpy(y[val_idx]).float()), batch_size=args.batch_size)
    model = SafetyValueMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = SafetyValueLoss() if args.loss == "asymmetric_focal" else SoftCalibrationLoss()
    best_loss = float("inf")
    stale_epochs = 0
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
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "target": args.target,
                    "hazard_lookahead": args.hazard_lookahead,
                    "hazard_tau": args.hazard_tau,
                    "split_seed": args.split_seed,
                    "external_test": args.external_test,
                },
                args.output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
                print(f"early_stop epoch={epoch:03d}", flush=True)
                break

    np.savez_compressed(args.output_dir / "normalizer.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    split = np.full(len(records), "train", dtype=object); split[val_idx] = "val"; split[test_idx] = "test"
    for index, record in enumerate(records): record["split"] = str(split[index])
    (args.output_dir / "chunk_pairs.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "trained_validation_selected_external_test_pending",
        "model": {
            "architecture": "final_hidden_pool_mlp_4096_1024_256_1",
            "input": "final pi0.5 action-token hidden state (10, 1024)",
            "time_conditioned": False,
            "limitation": (
                "The source corpus does not contain intermediate noisy actions or "
                "denoising times; this checkpoint is not V(h_t, a_t, t)."
            ),
        },
        "target": {
            "name": args.target,
            "hazard_lookahead": args.hazard_lookahead,
            "hazard_tau": args.hazard_tau,
            "minimum_clearance_available": False,
        },
        "optimization": {
            "loss": args.loss,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "early_stop_patience": args.early_stop_patience,
            "seed": args.seed,
        },
        "split": {
            "policy": "rollout_grouped_stratified_train_val" if args.external_test else "rollout_grouped_train_val_test",
            "split_seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "train_rollouts": train_groups,
            "validation_rollouts": val_groups,
            "internal_test_rollouts": test_groups,
            "external_test": (
                "fresh original SafeLIBERO simulator rollouts"
                if args.external_test
                else None
            ),
        },
        "data": _dataset_fingerprint(args.data_root),
        "git": _git_metadata(ROOT),
        "runtime": _runtime_metadata(device),
        "artifacts": {
            name: {"sha256": _sha256(args.output_dir / name)}
            for name in ("best_model.pt", "normalizer.npz", "chunk_pairs.json")
        },
        "best_validation_loss": best_loss,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"pairs={len(y)} target_mean={y.mean():.4f} "
        f"fully_safe={int((y == 1).sum())} immediate_hazard={int((y == 0).sum())} "
        f"best_val_loss={best_loss:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
