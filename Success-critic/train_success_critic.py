"""Train Q_phi from rollout-level SafeLIBERO success/failure labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = pathlib.Path(__file__).with_name("success_critic_model.py")
spec = importlib.util.spec_from_file_location("success_critic_model", MODEL_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
SuccessCritic = module.SuccessCritic


def _group(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    return f"{relative.parts[-3]}/{relative.parts[-2]}/{path.name.split('_', 1)[0]}"


def _split(groups: list[str], seed: int, validation_fraction: float):
    strata: dict[str, list[str]] = {}
    for group in groups:
        stratum = group.rsplit("/", 1)[0]
        strata.setdefault(stratum, []).append(group)
    train, validation = set(), set()
    for stratum, values in strata.items():
        values = sorted(
            values, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest()
        )
        count = (
            max(1, round(len(values) * validation_fraction)) if len(values) > 1 else 0
        )
        validation.update(values[:count])
        train.update(values[count:])
    if not train or not validation:
        raise RuntimeError("need at least two rollout groups")
    return train, validation


def _load(root: pathlib.Path, allowed: set[str]):
    hidden, action, time, target, weight = [], [], [], [], []
    sample_groups = []
    rollout_counts = {0: 0, 1: 0}
    for path in sorted(root.rglob("*_success_trace.npz")):
        group = _group(path, root)
        if group not in allowed:
            continue
        with np.load(path, allow_pickle=False) as archive:
            success = int(bool(archive["success"]))
            rollout_counts[success] += 1
            states = np.asarray(archive["denoising_hidden_states"], dtype=np.float16)
            actions = np.asarray(archive["denoising_noisy_actions"], dtype=np.float32)
            times = np.asarray(archive["denoising_times"], dtype=np.float32)
            starts = np.asarray(archive["chunk_start_steps"], dtype=np.float32)
            steps = max(float(archive["episode_steps"]), 1.0)
            progress = np.repeat(starts[:, None] / steps, states.shape[1], axis=1)
            sample_weight = np.ones_like(progress, dtype=np.float32)
            if not success:
                sample_weight = 0.25 + 0.75 * progress
            hidden.append(states.reshape(-1, *states.shape[-2:]))
            action.append(actions.reshape(-1, *actions.shape[-2:]))
            time.append(times.reshape(-1))
            target.append(np.full(times.size, success, dtype=np.float32))
            weight.append(sample_weight.reshape(-1))
            sample_groups.extend([group] * times.size)
    if not hidden:
        raise RuntimeError(f"no success traces found in {root}")
    values = tuple(
        np.concatenate(value) for value in (hidden, action, time, target, weight)
    )
    return values, rollout_counts, np.asarray(sample_groups)


def _metrics(model, values, device, sample_groups):
    hidden, action, time, target, _ = values
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(target), 256):
            logits = model(
                torch.from_numpy(hidden[start : start + 256]).to(
                    device, dtype=torch.float32
                ),
                torch.from_numpy(action[start : start + 256]).to(
                    device, dtype=torch.float32
                ),
                torch.from_numpy(time[start : start + 256]).to(device),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(probabilities)
    metrics = {
        "samples": len(target),
        "success_rate": float(np.mean(target)),
        "auc": float(roc_auc_score(target, probability))
        if len(np.unique(target)) == 2
        else None,
        "brier": float(brier_score_loss(target, probability)),
        "accuracy_at_0.5": float(accuracy_score(target, probability >= 0.5)),
    }
    unique_groups = np.unique(sample_groups)
    rollout_target = np.asarray(
        [target[sample_groups == group][0] for group in unique_groups]
    )
    rollout_probability = np.asarray(
        [np.mean(probability[sample_groups == group]) for group in unique_groups]
    )
    metrics.update(
        {
            "rollouts": len(unique_groups),
            "rollout_auc": float(roc_auc_score(rollout_target, rollout_probability))
            if len(np.unique(rollout_target)) == 2
            else None,
            "rollout_brier": float(
                brier_score_loss(rollout_target, rollout_probability)
            ),
            "rollout_accuracy_at_0.5": float(
                accuracy_score(rollout_target, rollout_probability >= 0.5)
            ),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = sorted(args.trace_root.rglob("*_success_trace.npz"))
    groups = [_group(path, args.trace_root) for path in paths]
    train_groups, validation_groups = _split(
        groups, args.seed, args.validation_fraction
    )
    train, train_rollouts, _ = _load(args.trace_root, train_groups)
    validation, validation_rollouts, validation_sample_groups = _load(
        args.trace_root, validation_groups
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_config = {"width": 256, "layers": 2, "heads": 4, "dropout": 0.1}
    model = SuccessCritic(**model_config).to(device)
    positives = float(np.sum(train[3]))
    negatives = float(len(train[3]) - positives)
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(negatives / max(positives, 1), device=device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    dataset = TensorDataset(*(torch.from_numpy(value) for value in train))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_brier, best_epoch = float("inf"), -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for hidden, action, time, target, sample_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                hidden.to(device, dtype=torch.float32),
                action.to(device, dtype=torch.float32),
                time.to(device),
            )
            loss = (
                criterion(logits, target.to(device)) * sample_weight.to(device)
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _metrics(model, validation, device, validation_sample_groups)
        metrics.update({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        if metrics["brier"] < best_brier:
            best_brier, best_epoch = metrics["brier"], epoch + 1
            torch.save(
                {
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                },
                args.output_dir / "best_model.pt",
            )
    checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    hidden, action, time, _, _ = validation
    action_tensor = (
        torch.from_numpy(action[: min(len(action), 256)])
        .to(device)
        .requires_grad_(True)
    )
    logits = model(
        torch.from_numpy(hidden[: len(action_tensor)]).to(device, dtype=torch.float32),
        action_tensor,
        torch.from_numpy(time[: len(action_tensor)]).to(device),
    )
    gradient = torch.autograd.grad(
        torch.nn.functional.logsigmoid(logits).sum(), action_tensor
    )[0]
    gradient_rms = float(
        torch.sqrt(torch.mean(gradient[:, :, :3].square())).detach().cpu()
    )
    manifest = {
        "schema_version": 1,
        "label": "rollout task success repeated over each selected denoising state",
        "failure_weighting": "0.25 + 0.75 * normalized chunk progress",
        "split_unit": "rollout",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_rollouts": train_rollouts,
        "validation_rollouts": validation_rollouts,
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "validation": _metrics(model, validation, device, validation_sample_groups),
        "translation_gradient_rms": gradient_rms,
        "model_config": model_config,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(manifest["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
