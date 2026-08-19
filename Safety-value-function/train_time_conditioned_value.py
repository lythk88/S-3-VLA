"""Train V(hidden_t, noisy_action_t, t) from trace and branch rollouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import subprocess
import sys
import platform
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = pathlib.Path(__file__).with_name("time_conditioned_value_model.py")
SOURCE_PATHS = (
    "Safety-value-function/PHASE1_UPDATED_PHASE2_BACKUP39396_METHOD.md",
    "Safety-value-function/time_conditioned_value_model.py",
    "Safety-value-function/train_time_conditioned_value.py",
    "Safety-value-function/evaluate_time_conditioned_gradient.py",
    "Safety-value-function/evaluate_live_clearance_gradient.py",
    "Safety-value-function/audit_denoising_value_dataset.py",
    "scripts/run_time_conditioned_value_training_v1.sh",
    "scripts/train_time_conditioned_value_updated_backup39396.sh",
    "Safety-value-function/time_conditioned_10way_variants.py",
    "scripts/train_time_conditioned_10way_variant.sh",
    "scripts/run_time_conditioned_10way_training_array.sh",
)
spec = importlib.util.spec_from_file_location("time_conditioned_value_model", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_module)
TimeConditionedSafetyValue = model_module.TimeConditionedSafetyValue


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Config:
    bootstrap_root: str
    trace_root: str
    output_dir: str
    split_seed: int = 7
    validation_fraction: float = 0.2
    future_lookahead: int = 20
    future_tau: float = 10.0
    trace_times: tuple[float, ...] = (0.1, 0.3, 0.5)
    width: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    bilinear_action_head: bool = False
    batch_size: int = 256
    loader_workers: int = 0
    learning_rate: float = 1e-4
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    pretrain_epochs: int = 10
    finetune_epochs: int = 50
    early_stop_patience: int = 8
    brier_weight: float = 0.25
    phase2_safety_weight: float = 1.0
    clearance_weight: float = 1.0
    pair_weight: float = 1.0
    pair_rank_weight: float = 1.0
    pair_difference_weight: float = 1.0
    informative_pair_sampling_fraction: float = 0.5
    pair_local_fraction: float = 1.0
    pair_rank_temperature: float = 1.0
    clearance_score_weight: float = 0.5
    minimum_pair_clearance_difference: float = 0.0005
    clearance_cap_m: float = 0.03
    seed: int = 7
    device: str = "cuda"


def _group_from_trace(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected Phase-2 trace path: {relative}")
    level = relative.parts[-2].rsplit("_", 1)[-1]
    episode = path.name.split("_", 1)[0]
    return f"{relative.parts[-3]}/{level}/{episode}"


def _group_from_bootstrap(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected Phase-1 bootstrap path: {relative}")
    level = relative.parts[-2].rsplit("_", 1)[-1]
    episode = path.name.split("_", 1)[0]
    return f"{relative.parts[-3]}/{level}/{episode}"


def _split_groups(groups: list[str], seed: int, fraction: float):
    strata: dict[str, list[str]] = {}
    for group in groups:
        task, level, _ = group.rsplit("/", 2)
        strata.setdefault(f"{task}/{level}", []).append(group)
    train, validation = set(), set()
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"{seed}:{stratum}:{value}".encode()
            ).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * fraction))) if len(ordered) > 1 else 0
        validation.update(ordered[:count])
        train.update(ordered[count:])
    if not train or not validation:
        raise RuntimeError("Trace corpus needs at least two rollout groups for train/validation")
    return train, validation


def _split_optional_groups(groups: set[str], seed: int, fraction: float):
    """Split groups that exist in only one phase without requiring both outputs."""
    if not groups:
        return set(), set()
    strata: dict[str, list[str]] = {}
    for group in groups:
        task, level, _ = group.rsplit("/", 2)
        strata.setdefault(f"{task}/{level}", []).append(group)
    train, validation = set(), set()
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"{seed}:{stratum}:{value}".encode()
            ).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * fraction))) if len(ordered) > 1 else 0
        validation.update(ordered[:count])
        train.update(ordered[count:])
    return train, validation


def _future_target(flags: np.ndarray, start: int, lookahead: int, tau: float) -> float:
    future = np.flatnonzero(flags[start : start + lookahead])
    if not len(future):
        return 1.0
    return float(1.0 - math.exp(-float(future[0]) / tau))


def _load_bootstrap(
    root: pathlib.Path,
    allowed_groups: set[str],
    config: Config,
):
    hidden, safety = [], []
    for path in sorted(root.rglob("*.npz")):
        group = _group_from_bootstrap(path, root)
        if group not in allowed_groups:
            continue
        with np.load(path, allow_pickle=False) as archive:
            states = np.asarray(archive["last_layer_hidden_states"], dtype=np.float16)
            starts = np.asarray(archive["chunk_start_steps"], dtype=np.int32)
            flags = np.asarray(archive["per_action_collision_flags"], dtype=np.bool_)
        hidden.append(states)
        safety.append(
            np.asarray(
                [
                    _future_target(
                        flags, int(start), config.future_lookahead, config.future_tau
                    )
                    for start in starts
                ],
                dtype=np.float32,
            )
        )
    hidden_array = np.concatenate(hidden)
    return {
        "hidden": hidden_array,
        "action": np.zeros((len(hidden_array), 10, 32), dtype=np.float32),
        "time": np.zeros(len(hidden_array), dtype=np.float32),
        "safety": np.concatenate(safety),
    }


def _load_trace(root: pathlib.Path, allowed_groups: set[str], config: Config):
    hidden, action, times, safety, clearance = [], [], [], [], []
    pair_keys: list[str | None] = []
    pair_signs: list[int] = []
    for path in sorted(root.rglob("*_denoising_value.npz")):
        group = _group_from_trace(path, root)
        if group not in allowed_groups:
            continue
        with np.load(path, allow_pickle=False) as archive:
            starts = np.asarray(archive["chunk_start_steps"], dtype=np.int32)
            flags = np.asarray(archive["nominal_collision"], dtype=np.bool_)
            distances = np.asarray(archive["nominal_clearance"], dtype=np.float32)
            trace_times = np.asarray(archive["denoising_times"], dtype=np.float32)
            # NpzFile.__getitem__ decompresses an array on every access. Cache each
            # large tensor once: indexing archive[...] inside the nested loops kept
            # one full decompressed backing array alive per selected sample and
            # exhausted 128 GiB before the first epoch on the full corpus.
            denoising_hidden = np.asarray(
                archive["denoising_hidden_states"], dtype=np.float16
            )
            denoising_actions = np.asarray(
                archive["denoising_noisy_actions"], dtype=np.float32
            )
            branch_hidden = np.asarray(archive["branch_hidden_states"], dtype=np.float16)
            branch_actions = np.asarray(archive["branch_noisy_actions"], dtype=np.float32)
            branch_times = np.asarray(archive["branch_time"], dtype=np.float32)
            branch_collisions = np.asarray(archive["branch_collision"], dtype=np.bool_)
            branch_clearances = np.asarray(archive["branch_clearance"], dtype=np.float32)
            branch_chunk_ids = np.asarray(archive["branch_chunk_id"], dtype=np.int32)
            branch_direction_ids = np.asarray(
                archive["branch_direction_id"], dtype=np.int32
            )
            branch_signs = np.asarray(archive["branch_sign"], dtype=np.int8)
            time_indices = sorted(
                {
                    int(np.argmin(np.abs(trace_times[0] - requested)))
                    for requested in config.trace_times
                }
            )
            for chunk_id, start in enumerate(starts):
                target = _future_target(
                    flags, int(start), config.future_lookahead, config.future_tau
                )
                end = min(int(start) + config.future_lookahead, len(distances))
                clearance_target = float(np.min(distances[int(start) : end]))
                for time_id in time_indices:
                    hidden.append(denoising_hidden[chunk_id, time_id])
                    action.append(denoising_actions[chunk_id, time_id])
                    times.append(float(trace_times[chunk_id, time_id]))
                    safety.append(target)
                    clearance.append(clearance_target)
                    pair_keys.append(None)
                    pair_signs.append(0)

            branch_count = len(branch_times)
            for branch_id in range(branch_count):
                hidden.append(branch_hidden[branch_id])
                action.append(branch_actions[branch_id])
                times.append(float(branch_times[branch_id]))
                safety.append(0.0 if bool(branch_collisions[branch_id]) else 1.0)
                clearance.append(float(np.min(branch_clearances[branch_id])))
                pair_keys.append(
                    f"{group}:{int(branch_chunk_ids[branch_id])}:"
                    f"{float(branch_times[branch_id]):.5f}:"
                    f"{int(branch_direction_ids[branch_id])}"
                )
                pair_signs.append(int(branch_signs[branch_id]))

    values = {
        "hidden": np.asarray(hidden, dtype=np.float16),
        "action": np.asarray(action, dtype=np.float32),
        "time": np.asarray(times, dtype=np.float32),
        "safety": np.asarray(safety, dtype=np.float32),
        "clearance": np.asarray(clearance, dtype=np.float32),
    }
    pairs: dict[str, dict[int, int]] = {}
    for index, (key, sign) in enumerate(zip(pair_keys, pair_signs)):
        if key is not None:
            pairs.setdefault(key, {})[sign] = index
    pair_indices = np.asarray(
        [(value[-1], value[1]) for value in pairs.values() if -1 in value and 1 in value],
        dtype=np.int64,
    )
    return values, pair_indices


def _loader(values, batch_size: int, shuffle: bool, workers: int):
    dataset = TensorDataset(
        torch.from_numpy(values["hidden"]),
        torch.from_numpy(values["action"]),
        torch.from_numpy(values["time"]),
        torch.from_numpy(values["safety"]),
        *(
            [torch.from_numpy(values["clearance"])]
            if "clearance" in values
            else []
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _base_loss(logit, safety, brier_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logit, safety)
    brier = torch.mean(torch.square(torch.sigmoid(logit) - safety))
    return bce + brier_weight * brier, bce, brier


def _informative_pairs(values, pairs, minimum_difference: float):
    if pairs is None or not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    delta = values["clearance"][pairs[:, 1]] - values["clearance"][pairs[:, 0]]
    return pairs[np.abs(delta) >= minimum_difference]


def _pair_loss(model, values, pairs, device, config):
    """Pair loss matching the fixed-hidden online action-gradient convention."""
    negative, positive = pairs[:, 0], pairs[:, 1]
    center_hidden = (
        values["hidden"][negative].astype(np.float32)
        + values["hidden"][positive].astype(np.float32)
    ) / 2.0
    pair_hidden = torch.from_numpy(
        np.concatenate((center_hidden, center_hidden), axis=0)
    ).to(device)
    negative_action = values["action"][negative]
    positive_action = values["action"][positive]
    center_action = (negative_action + positive_action) / 2.0
    half_direction = (positive_action - negative_action) / 2.0
    pair_action = torch.from_numpy(
        np.concatenate(
            (
                center_action - config.pair_local_fraction * half_direction,
                center_action + config.pair_local_fraction * half_direction,
            ),
            axis=0,
        )
    ).to(device)
    center_time = (
        values["time"][negative] + values["time"][positive]
    ) / 2.0
    pair_time = torch.from_numpy(np.concatenate((center_time, center_time))).to(device)
    pair_logit, pair_clearance = model(pair_hidden, pair_action, pair_time)
    half = len(pairs)
    true_delta = torch.from_numpy(
        (
            values["clearance"][positive]
            - values["clearance"][negative]
        )
        / config.clearance_cap_m
    ).to(device)
    objective = nn.functional.logsigmoid(pair_logit) + (
        config.clearance_score_weight * pair_clearance
    )
    objective_delta = objective[half:] - objective[:half]
    predicted_clearance_delta = pair_clearance[half:] - pair_clearance[:half]
    directional_objective_delta = objective_delta / config.pair_local_fraction
    directional_clearance_delta = (
        predicted_clearance_delta / config.pair_local_fraction
    )
    direction = torch.sign(true_delta)
    rank_mask = torch.abs(true_delta) >= (
        config.minimum_pair_clearance_difference / config.clearance_cap_m
    )
    rank_values = nn.functional.softplus(
        -direction
        * directional_objective_delta
        / config.pair_rank_temperature
    )
    rank = (
        torch.mean(rank_values[rank_mask])
        if torch.any(rank_mask)
        else torch.zeros((), device=device)
    )
    difference = nn.functional.smooth_l1_loss(
        directional_clearance_delta,
        true_delta,
    )
    accuracy = (
        torch.mean(
            (objective_delta[rank_mask] * true_delta[rank_mask] > 0.0).float()
        )
        if torch.any(rank_mask)
        else torch.zeros((), device=device)
    )
    return (
        config.pair_rank_weight * rank
        + config.pair_difference_weight * difference,
        rank,
        difference,
        accuracy,
    )


@torch.no_grad()
def _evaluate(
    model,
    loader,
    device,
    config,
    with_clearance: bool,
    validation_values=None,
    validation_pairs=None,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "brier": 0.0,
        "clearance_mae_m": 0.0,
        "pair_loss": 0.0,
        "pair_direction_accuracy": 0.0,
        "pairs_informative": 0,
        "n": 0,
    }
    for batch in loader:
        batch = [value.to(device, non_blocking=True) for value in batch]
        hidden, action, time, target = batch[:4]
        logit, clearance_prediction = model(hidden.float(), action, time)
        loss, _, brier = _base_loss(logit, target, config.brier_weight)
        if with_clearance:
            loss = config.phase2_safety_weight * loss
            clearance_target = batch[4] / config.clearance_cap_m
            loss = loss + config.clearance_weight * nn.functional.smooth_l1_loss(
                clearance_prediction, clearance_target
            )
            totals["clearance_mae_m"] += float(
                torch.sum(
                    torch.abs(clearance_prediction - clearance_target)
                    * config.clearance_cap_m
                )
            )
        size = len(target)
        totals["loss"] += float(loss) * size
        totals["brier"] += float(brier) * size
        totals["n"] += size
    count = max(totals.pop("n"), 1)
    metrics = {
        "loss": totals["loss"] / count,
        "brier": totals["brier"] / count,
        "clearance_mae_m": totals["clearance_mae_m"] / count,
        "pair_loss": 0.0,
        "pair_direction_accuracy": 0.0,
        "pairs_informative": 0,
    }
    if with_clearance and validation_values is not None:
        informative = _informative_pairs(
            validation_values,
            validation_pairs,
            config.minimum_pair_clearance_difference,
        )
        pair_loss_sum = 0.0
        pair_accuracy_sum = 0.0
        for start in range(0, len(informative), config.batch_size):
            batch_pairs = informative[start : start + config.batch_size]
            loss, _, _, accuracy = _pair_loss(
                model, validation_values, batch_pairs, device, config
            )
            pair_loss_sum += float(loss) * len(batch_pairs)
            pair_accuracy_sum += float(accuracy) * len(batch_pairs)
        if len(informative):
            metrics["pair_loss"] = pair_loss_sum / len(informative)
            metrics["pair_direction_accuracy"] = (
                pair_accuracy_sum / len(informative)
            )
            metrics["pairs_informative"] = len(informative)
            metrics["loss"] += config.pair_weight * metrics["pair_loss"]
    return metrics


def _train_stage(
    model,
    train_values,
    validation_values,
    config,
    device,
    optimizer,
    epochs,
    with_clearance,
    train_pairs=None,
    validation_pairs=None,
):
    train_loader = _loader(
        train_values, config.batch_size, True, config.loader_workers
    )
    validation_loader = _loader(
        validation_values, config.batch_size, False, config.loader_workers
    )
    best_state, best_metrics, best_epoch, stale = None, None, None, 0
    history = []
    rng = np.random.default_rng(config.seed)
    pair_training_pool = (
        np.asarray(train_pairs, dtype=np.int64)
        if with_clearance and train_pairs is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    informative_pair_pool = (
        _informative_pairs(
            train_values,
            pair_training_pool,
            config.minimum_pair_clearance_difference,
        )
        if with_clearance
        else np.empty((0, 2), dtype=np.int64)
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = [value.to(device, non_blocking=True) for value in batch]
            hidden, action, time, target = batch[:4]
            logit, clearance_prediction = model(hidden.float(), action, time)
            loss, _, _ = _base_loss(logit, target, config.brier_weight)
            if with_clearance:
                loss = config.phase2_safety_weight * loss
                clearance_target = batch[4] / config.clearance_cap_m
                loss = loss + config.clearance_weight * nn.functional.smooth_l1_loss(
                    clearance_prediction, clearance_target
                )
                if len(pair_training_pool):
                    pair_batch_size = min(64, len(pair_training_pool))
                    informative_count = (
                        min(
                            pair_batch_size,
                            int(
                                round(
                                    pair_batch_size
                                    * config.informative_pair_sampling_fraction
                                )
                            ),
                        )
                        if len(informative_pair_pool)
                        else 0
                    )
                    general_count = pair_batch_size - informative_count
                    chosen_parts = []
                    if informative_count:
                        chosen_parts.append(
                            informative_pair_pool[
                                rng.integers(
                                    0,
                                    len(informative_pair_pool),
                                    size=informative_count,
                                )
                            ]
                        )
                    if general_count:
                        chosen_parts.append(
                            pair_training_pool[
                                rng.integers(
                                    0,
                                    len(pair_training_pool),
                                    size=general_count,
                                )
                            ]
                        )
                    chosen = np.concatenate(chosen_parts, axis=0)
                    pair_loss, _, _, _ = _pair_loss(
                        model, train_values, chosen, device, config
                    )
                    loss = loss + config.pair_weight * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = _evaluate(
            model,
            validation_loader,
            device,
            config,
            with_clearance,
            validation_values,
            validation_pairs,
        )
        history.append({"epoch": epoch, **metrics})
        print(f"epoch={epoch} validation={json.dumps(metrics, sort_keys=True)}", flush=True)
        if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stop_patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation": best_metrics,
        "history": history,
    }


def train(config: Config) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    trace_root = pathlib.Path(config.trace_root)
    bootstrap_root = pathlib.Path(config.bootstrap_root)
    audit_path = trace_root / "audit_report.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing required data audit: {audit_path}")
    audit_report = json.loads(audit_path.read_text())
    if audit_report.get("status") != "passed":
        raise RuntimeError(f"Dataset audit did not pass: {audit_path}")
    if pathlib.Path(audit_report["trace_root"]) != trace_root.resolve():
        raise RuntimeError("Audit trace root does not match the configured Phase-2 root")
    if pathlib.Path(audit_report["source_root"]) != bootstrap_root.resolve():
        raise RuntimeError("Audit source root does not match the configured Phase-1 root")

    output = pathlib.Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    trace_files = sorted(trace_root.rglob("*_denoising_value.npz"))
    trace_groups = [_group_from_trace(path, trace_root) for path in trace_files]
    trace_train_groups, trace_validation_groups = _split_groups(
        trace_groups, config.split_seed, config.validation_fraction
    )
    bootstrap_files = sorted(bootstrap_root.rglob("*.npz"))
    bootstrap_groups = {
        _group_from_bootstrap(path, bootstrap_root) for path in bootstrap_files
    }
    trace_group_set = set(trace_groups)
    trace_only_groups = trace_group_set - bootstrap_groups
    if trace_only_groups:
        raise RuntimeError(
            "Phase-2 traces have no matching Phase-1 source groups: "
            f"{sorted(trace_only_groups)}"
        )
    bootstrap_only_groups = bootstrap_groups - trace_group_set
    bootstrap_only_train, bootstrap_only_validation = _split_optional_groups(
        bootstrap_only_groups, config.split_seed, config.validation_fraction
    )
    bootstrap_train_groups = trace_train_groups | bootstrap_only_train
    bootstrap_validation_groups = (
        trace_validation_groups | bootstrap_only_validation
    )

    bootstrap_train = _load_bootstrap(
        bootstrap_root, bootstrap_train_groups, config
    )
    bootstrap_validation = _load_bootstrap(
        bootstrap_root, bootstrap_validation_groups, config
    )
    trace_train, train_pairs = _load_trace(
        trace_root, trace_train_groups, config
    )
    trace_validation, validation_pairs = _load_trace(
        trace_root, trace_validation_groups, config
    )
    print(
        json.dumps(
            {
                "bootstrap_train": len(bootstrap_train["safety"]),
                "bootstrap_validation": len(bootstrap_validation["safety"]),
                "trace_train": len(trace_train["safety"]),
                "trace_validation": len(trace_validation["safety"]),
                "train_pairs": len(train_pairs),
                "validation_pairs": len(validation_pairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is false"
        )
    model = TimeConditionedSafetyValue(
        width=config.width,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        bilinear_action_head=config.bilinear_action_head,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    pretrain_metrics = _train_stage(
        model,
        bootstrap_train,
        bootstrap_validation,
        config,
        device,
        optimizer,
        config.pretrain_epochs,
        False,
    )
    # Reset optimizer moments at the objective boundary. Reusing the pretraining
    # Adam state with the original learning rate caused immediate held-out
    # degradation during clearance/pair fine-tuning.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
    )
    finetune_metrics = _train_stage(
        model,
        trace_train,
        trace_validation,
        config,
        device,
        optimizer,
        config.finetune_epochs,
        True,
        train_pairs,
        validation_pairs,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "width": config.width,
            "layers": config.layers,
            "heads": config.heads,
            "dropout": config.dropout,
            "bilinear_action_head": config.bilinear_action_head,
        },
        "training_config": asdict(config),
    }
    torch.save(checkpoint, output / "best_model.pt")
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        # Hash only the implementation sources. The worktree may contain large,
        # unrelated model artifacts; materializing their binary patch made this
        # metadata step slower than training and could consume gigabytes of RAM.
        diff = subprocess.check_output(
            ("git", "diff", "--no-ext-diff", "HEAD", "--", *SOURCE_PATHS),
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        commit, diff = None, b""
    manifest = {
        "schema_version": 2,
        "status": "trained_gradient_gate_pending",
        "architecture": "token_transformer_V(hidden_t,noisy_action_t,t)",
        "configuration": asdict(config),
        "split": {
            # Keep these aliases for the gradient evaluator. They are the
            # Phase-2 groups on which the final checkpoint is selected.
            "train_groups": sorted(trace_train_groups),
            "validation_groups": sorted(trace_validation_groups),
            "trace_train_groups": sorted(trace_train_groups),
            "trace_validation_groups": sorted(trace_validation_groups),
            "bootstrap_train_groups": sorted(bootstrap_train_groups),
            "bootstrap_validation_groups": sorted(bootstrap_validation_groups),
            "bootstrap_only_train_groups": sorted(bootstrap_only_train),
            "bootstrap_only_validation_groups": sorted(
                bootstrap_only_validation
            ),
        },
        "counts": {
            "bootstrap_groups": len(bootstrap_groups),
            "trace_groups": len(trace_group_set),
            "bootstrap_only_groups": len(bootstrap_only_groups),
            "bootstrap_train": len(bootstrap_train["safety"]),
            "bootstrap_validation": len(bootstrap_validation["safety"]),
            "trace_train": len(trace_train["safety"]),
            "trace_validation": len(trace_validation["safety"]),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "train_pairs_informative": len(
                _informative_pairs(
                    trace_train,
                    train_pairs,
                    config.minimum_pair_clearance_difference,
                )
            ),
            "validation_pairs_informative": len(
                _informative_pairs(
                    trace_validation,
                    validation_pairs,
                    config.minimum_pair_clearance_difference,
                )
            ),
        },
        "metrics": {
            "pretrain_validation": pretrain_metrics,
            "finetune_validation": finetune_metrics,
        },
        "training_method": {
            "phase_1": {
                "purpose": "initialize the hidden-state representation from final pi0.5 action-token states",
                "inputs": "hidden[10,1024], with noisy_action=zeros[10,32] and denoising_time=0",
                "target": (
                    "discounted future safety y=1 when no collision occurs in the next "
                    "20 actions; otherwise y=1-exp(-d/10), where d is the first "
                    "collision offset"
                ),
                "loss": "BCEWithLogits(safety_logit,y) + 0.25*Brier(sigmoid(safety_logit),y)",
            },
            "phase_2": {
                "purpose": "fine-tune V(hidden_t,noisy_action_t,t) on nominal denoising states and same-state perturbation branches",
                "safety_loss": "BCEWithLogits(safety_logit,y) + 0.25*Brier(sigmoid(safety_logit),y)",
                "clearance_loss": (
                    "SmoothL1(predicted_clearance, measured_min_clearance/0.03)"
                ),
                "pair_objective": (
                    "logsigmoid(safety_logit) + "
                    f"{config.clearance_score_weight}*predicted_normalized_clearance"
                ),
                "pair_loss": (
                    f"{config.pair_rank_weight}*softplus(-sign(delta_clearance)*"
                    f"delta_pair_objective/{config.pair_rank_temperature}) + "
                    f"{config.pair_difference_weight}*SmoothL1(predicted_delta_clearance, "
                    f"measured_delta_clearance/{config.clearance_cap_m})"
                ),
                "total_loss": (
                    f"{config.phase2_safety_weight}*safety_loss + "
                    f"{config.clearance_weight}*clearance_loss + "
                    f"{config.pair_weight}*pair_loss"
                ),
                "configured_loss_weights": {
                    "phase2_safety": config.phase2_safety_weight,
                    "clearance": config.clearance_weight,
                    "pair": config.pair_weight,
                    "pair_rank_within_pair": config.pair_rank_weight,
                    "pair_difference_within_pair": config.pair_difference_weight,
                },
                "pair_sampling": (
                    f"{config.informative_pair_sampling_fraction:.3f} of each pair minibatch "
                    f"is sampled from pairs with at least {config.minimum_pair_clearance_difference} m absolute measured "
                    "clearance difference; the remainder is sampled from all pairs"
                ),
            },
            "optimizer": (
                "AdamW is reset between phases; global gradient norm is clipped to 1.0"
            ),
            "checkpoint_selection": (
                "minimum rollout-grouped Phase-2 validation total loss with early stopping"
            ),
        },
        "git_commit": commit,
        "git": {
            "commit": commit,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "diff_scope": list(SOURCE_PATHS),
        },
        "source_sha256": {
            relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "guidance_convention": {
            "objective": (
                "log(sigmoid(safety_logit)) + 0.5 * normalized_clearance"
            ),
            "derivative": (
                "partial derivative with respect to normalized padded noisy_action_t; "
                "hidden_t is held fixed"
            ),
            "mask": "XYZ action channels 0:3 across all ten tokens",
            "normalization": "unit RMS over the active 10x3 XYZ entries",
            "online_update": "a_t <- a_t + 0.05 * normalized_gradient at t=0.3",
        },
        "runtime": {
            "host": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "data_provenance": {
            "bootstrap_root": str(bootstrap_root.resolve()),
            "trace_root": str(trace_root.resolve()),
            "collection_manifest_sha256": _sha256(trace_root / "collection_manifest.json"),
            "audit_report_sha256": _sha256(trace_root / "audit_report.json"),
            "bootstrap_dataset_sha256": audit_report["source_dataset_sha256"],
            "trace_dataset_sha256": audit_report["dataset_sha256"],
            "source_only_groups_allowed": audit_report.get(
                "source_only_groups_allowed", False
            ),
            "source_only_groups": audit_report.get("missing_groups", []),
        },
        "artifacts": {
            "best_model.pt": {
                "bytes": (output / "best_model.pt").stat().st_size,
                "sha256": _sha256(output / "best_model.pt"),
            }
        },
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-root", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--future-lookahead", type=int, default=20)
    parser.add_argument("--future-tau", type=float, default=10.0)
    parser.add_argument(
        "--trace-times",
        default="0.1,0.3,0.5",
        help="Comma-separated denoising times sampled during Phase 2",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--brier-weight", type=float, default=0.25)
    parser.add_argument("--phase2-safety-weight", type=float, default=1.0)
    parser.add_argument("--clearance-weight", type=float, default=1.0)
    parser.add_argument(
        "--bilinear-action-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--pair-rank-weight", type=float, default=1.0)
    parser.add_argument("--pair-difference-weight", type=float, default=1.0)
    parser.add_argument(
        "--informative-pair-sampling-fraction", type=float, default=0.5
    )
    parser.add_argument("--pair-local-fraction", type=float, default=1.0)
    parser.add_argument("--pair-rank-temperature", type=float, default=1.0)
    parser.add_argument("--clearance-score-weight", type=float, default=0.5)
    parser.add_argument(
        "--minimum-pair-clearance-difference", type=float, default=0.0005
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    values = parser.parse_args()
    try:
        values.trace_times = tuple(
            float(value) for value in values.trace_times.split(",") if value.strip()
        )
    except ValueError as error:
        parser.error(f"--trace-times must contain comma-separated floats: {error}")
    if not values.trace_times:
        parser.error("--trace-times must contain at least one time")
    if any(not 0.0 < value <= 1.0 for value in values.trace_times):
        parser.error("--trace-times values must be in (0,1]")
    if not 0.0 <= values.informative_pair_sampling_fraction <= 1.0:
        parser.error("--informative-pair-sampling-fraction must be in [0,1]")
    for name in (
        "phase2_safety_weight",
        "clearance_weight",
        "pair_weight",
        "pair_rank_weight",
        "pair_difference_weight",
    ):
        if getattr(values, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return Config(**vars(values))


if __name__ == "__main__":
    train(parse_args())
