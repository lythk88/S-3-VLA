#!/usr/bin/env python3
"""Train an MLP safety classifier from pi0.5 hidden-state rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


DEFAULT_DATA_ROOT = pathlib.Path(
    "/home/namn1/vlsa-aegis/results/pi05_no_safety_with_hidden_full"
)
DEFAULT_OUTPUT_BASE = pathlib.Path("/home/namn1/vlsa-aegis/MLP-classifier/runs")
FILENAME_RE = re.compile(
    r"^(?P<episode>\d+)_(?P<outcome>success|failure)_(?P<label>safe|unsafe)_last_layer_hidden_states\.npz$"
)


@dataclass
class Example:
    path: str
    rel_path: str
    task: str
    run: str
    episode: int
    outcome: str
    label: str
    target_unsafe: int
    success: Optional[bool]
    collision: Optional[bool]
    safe_success: Optional[bool]
    shape: str
    mtime: float


class HiddenStateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dims", default="1024,256")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    return parser.parse_args()


def bool_from_npz(value: Optional[np.ndarray]) -> Optional[bool]:
    if value is None:
        return None
    return bool(np.asarray(value).item())


def feature_from_hidden_states(hidden_states: np.ndarray) -> np.ndarray:
    """Convert variable-length hidden states to one fixed MLP feature vector."""
    if hidden_states.ndim != 3:
        raise ValueError(f"expected hidden states with 3 dims, got {hidden_states.shape}")
    if hidden_states.shape[0] == 0 or hidden_states.shape[-1] == 0:
        raise ValueError(f"empty hidden states: {hidden_states.shape}")

    hidden_states = hidden_states.astype(np.float32, copy=False)
    token_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    chunk_mean = hidden_states.mean(axis=1)

    pieces = [
        token_flat.mean(axis=0),
        token_flat.std(axis=0),
        token_flat.max(axis=0),
        chunk_mean[-1],
    ]
    feature = np.concatenate(pieces, axis=0).astype(np.float32)
    return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)


def load_examples(data_root: pathlib.Path) -> Tuple[List[Example], np.ndarray, np.ndarray]:
    examples: List[Example] = []
    features: List[np.ndarray] = []
    targets: List[int] = []

    paths = sorted(data_root.rglob("*_last_layer_hidden_states.npz"))
    if not paths:
        raise FileNotFoundError(f"no hidden-state .npz files found under {data_root}")

    for path in paths:
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"[skip] unexpected filename: {path}")
            continue

        label_from_name = match.group("label")
        target_from_name = 1 if label_from_name == "unsafe" else 0

        try:
            
            with np.load(path) as data:
                hidden_states = data["last_layer_hidden_states"]
                feature = feature_from_hidden_states(hidden_states)
                success = bool_from_npz(data.get("success"))
                collision = bool_from_npz(data.get("collision"))
                safe_success = bool_from_npz(data.get("safe_success"))
        except Exception as exc:
            print(f"[skip] failed to load {path}: {type(exc).__name__}: {exc}")
            continue

        if collision is not None:
            target_from_npz = 1 if collision else 0
            if target_from_npz != target_from_name:
                print(f"[warn] filename/npz label mismatch; using npz collision: {path}")
            target = target_from_npz
        else:
            target = target_from_name

        rel_path = path.relative_to(data_root)
        examples.append(
            Example(
                path=str(path),
                rel_path=str(rel_path),
                task=path.parent.parent.name,
                run=path.parent.name,
                episode=int(match.group("episode")),
                outcome=match.group("outcome"),
                label="unsafe" if target else "safe",
                target_unsafe=target,
                success=success,
                collision=collision,
                safe_success=safe_success,
                shape="x".join(str(x) for x in hidden_states.shape),
                mtime=path.stat().st_mtime,
            )
        )
        features.append(feature)
        targets.append(target)

    if not examples:
        raise RuntimeError(f"found files under {data_root}, but none were usable")

    return examples, np.stack(features, axis=0), np.asarray(targets, dtype=np.int64)


def split_indices(targets: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    indices = np.arange(len(targets))
    test_fraction = args.test_fraction
    val_fraction = args.val_fraction

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=args.seed,
        stratify=targets,
    )
    relative_val_fraction = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_fraction,
        random_state=args.seed,
        stratify=targets[train_val_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    train: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    if not train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class_counts = np.bincount(y.astype(np.int64), minlength=2)
    weights = np.asarray([1.0 / max(class_counts[int(label)], 1) for label in y], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits.append(model(batch_x).detach().cpu().numpy())
            labels.append(batch_y.numpy())

    y_true = np.concatenate(labels).astype(np.int64)
    y_score = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    y_pred = (y_score >= threshold).astype(np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    metrics: Dict[str, object] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "safe_f1": float(f1[0]),
        "safe_support": int(support[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
        "unsafe_support": int(support[1]),
        "confusion_matrix_safe_unsafe": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision_unsafe"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision_unsafe"] = None
    return metrics, y_score, y_pred


def save_manifest(path: pathlib.Path, examples: List[Example], split_by_index: Dict[int, str]) -> None:
    rows = []
    for idx, example in enumerate(examples):
        row = asdict(example)
        row["split"] = split_by_index[idx]
        rows.append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    path: pathlib.Path,
    examples: List[Example],
    splits: Dict[str, np.ndarray],
    scores_by_index: Dict[int, float],
    preds_by_index: Dict[int, int],
) -> None:
    fieldnames = [
        "split",
        "rel_path",
        "task",
        "run",
        "episode",
        "target_label",
        "pred_label",
        "unsafe_prob",
        "success",
        "collision",
        "safe_success",
    ]
    split_by_index = {int(idx): split for split, idxs in splits.items() for idx in idxs}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, example in enumerate(examples):
            if idx not in scores_by_index:
                continue
            pred = preds_by_index[idx]
            writer.writerow(
                {
                    "split": split_by_index[idx],
                    "rel_path": example.rel_path,
                    "task": example.task,
                    "run": example.run,
                    "episode": example.episode,
                    "target_label": example.label,
                    "pred_label": "unsafe" if pred else "safe",
                    "unsafe_prob": f"{scores_by_index[idx]:.8f}",
                    "success": example.success,
                    "collision": example.collision,
                    "safe_success": example.safe_success,
                }
            )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir or (DEFAULT_OUTPUT_BASE / f"pi05_hidden_state_mlp_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading snapshot from {args.data_root}", flush=True)
    examples, features, targets = load_examples(args.data_root)
    class_counts = np.bincount(targets, minlength=2)
    print(
        f"[data] usable={len(examples)} safe={class_counts[0]} unsafe={class_counts[1]} "
        f"input_dim={features.shape[1]}",
        flush=True,
    )

    splits = split_indices(targets, args)
    split_by_index = {int(idx): split for split, idxs in splits.items() for idx in idxs}
    save_manifest(output_dir / "manifest.csv", examples, split_by_index)

    train_features = features[splits["train"]]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    features = (features - mean) / std
    np.savez_compressed(output_dir / "normalizer.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    loaders = {
        split: make_loader(
            features[idxs],
            targets[idxs],
            batch_size=args.batch_size,
            train=(split == "train"),
            seed=args.seed,
        )
        for split, idxs in splits.items()
    }

    model = HiddenStateMLP(features.shape[1], hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -math.inf
    best_epoch = 0
    bad_epochs = 0
    train_log: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics, _, _ = evaluate(model, loaders["train"], device, args.threshold)
        val_metrics, _, _ = evaluate(model, loaders["val"], device, args.threshold)
        val_score = float(val_metrics["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_safe_recall": float(val_metrics["safe_recall"]),
            "val_unsafe_recall": float(val_metrics["unsafe_recall"]),
        }
        train_log.append(row)
        print(
            "[epoch {epoch:03d}] loss={loss:.4f} train_bal_acc={train_balanced_accuracy:.3f} "
            "val_bal_acc={val_balanced_accuracy:.3f} val_safe_rec={val_safe_recall:.3f} "
            "val_unsafe_rec={val_unsafe_recall:.3f}".format(**row),
            flush=True,
        )

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": features.shape[1],
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                    "feature": "concat(mean_all_tokens,std_all_tokens,max_all_tokens,last_chunk_mean)",
                    "target": "unsafe=1,safe=0",
                    "threshold": args.threshold,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early-stop] no validation improvement for {args.patience} epochs", flush=True)
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    args_dict = vars(args).copy()
    args_dict.update({"output_dir": str(output_dir), "data_root": str(args.data_root), "device": str(device)})

    all_metrics: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val,
        "class_counts": {"safe": int(class_counts[0]), "unsafe": int(class_counts[1])},
        "split_counts": {
            split: {
                "total": int(len(idxs)),
                "safe": int(np.sum(targets[idxs] == 0)),
                "unsafe": int(np.sum(targets[idxs] == 1)),
            }
            for split, idxs in splits.items()
        },
        "args": args_dict,
    }

    scores_by_index: Dict[int, float] = {}
    preds_by_index: Dict[int, int] = {}
    for split, loader in loaders.items():
        metrics, scores, preds = evaluate(model, loader, device, args.threshold)
        all_metrics[split] = metrics
        for idx, score, pred in zip(splits[split], scores, preds):
            scores_by_index[int(idx)] = float(score)
            preds_by_index[int(idx)] = int(pred)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    with (output_dir / "training_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
        writer.writeheader()
        writer.writerows(train_log)

    save_predictions(output_dir / "predictions.csv", examples, splits, scores_by_index, preds_by_index)
    print(f"[done] saved artifacts to {output_dir}", flush=True)
    print(json.dumps({"test": all_metrics["test"], "best_epoch": best_epoch}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
