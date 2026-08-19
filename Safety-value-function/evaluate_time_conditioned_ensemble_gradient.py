"""Read-only held-out gradient diagnostics for value-model ensembles."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pathlib
import sys

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "time_conditioned_value_model", HERE / "time_conditioned_value_model.py"
)
model_module = importlib.util.module_from_spec(spec)
sys.modules["time_conditioned_value_model"] = model_module
assert spec.loader is not None
spec.loader.exec_module(model_module)
TimeConditionedSafetyValue = model_module.TimeConditionedSafetyValue

gate_spec = importlib.util.spec_from_file_location(
    "time_conditioned_gradient_gate",
    HERE / "evaluate_time_conditioned_gradient.py",
)
gate_module = importlib.util.module_from_spec(gate_spec)
sys.modules["time_conditioned_gradient_gate"] = gate_module
assert gate_spec.loader is not None
gate_spec.loader.exec_module(gate_module)


def _load_model(run_dir: pathlib.Path, device: torch.device):
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model = TimeConditionedSafetyValue(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _metrics(predicted_delta, actual_delta, clusters, minimum_difference, seed):
    informative = np.abs(actual_delta) >= minimum_difference
    predicted_delta = predicted_delta[informative]
    actual_delta = actual_delta[informative]
    clusters = clusters[informative]
    aligned = (predicted_delta * actual_delta > 0.0).astype(np.float64)
    selected_positive = predicted_delta >= 0.0
    gains = np.where(selected_positive, actual_delta, -actual_delta)
    accuracy_ci = gate_module._cluster_bootstrap(aligned, clusters, seed)
    clearance_ci = gate_module._cluster_bootstrap(gains, clusters, seed)
    return {
        "pairs_informative": int(len(aligned)),
        "rollout_clusters": int(len(np.unique(clusters))),
        "gradient_direction_accuracy": float(np.mean(aligned)),
        "gradient_direction_accuracy_cluster_ci95": accuracy_ci,
        "selected_minus_rejected_clearance_m": float(np.mean(gains)),
        "selected_minus_rejected_clearance_cluster_ci95": clearance_ci,
        "pass": bool(accuracy_ci[0] > 0.5 and clearance_ci[0] > 0.0),
    }


def evaluate(args):
    run_dirs = [pathlib.Path(value) for value in args.run_dir]
    manifests = [
        json.loads((run_dir / "training_manifest.json").read_text())
        for run_dir in run_dirs
    ]
    splits = [manifest["split"]["validation_groups"] for manifest in manifests]
    if any(split != splits[0] for split in splits[1:]):
        raise RuntimeError("Ensemble members do not use the same validation split")
    device = torch.device(args.device)
    models = [_load_model(run_dir, device) for run_dir in run_dirs]
    records = gate_module._load_pairs(pathlib.Path(args.trace_root), set(splits[0]))
    gradients = [[] for _ in models]
    actual_delta = []
    clusters = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        hidden = torch.from_numpy(
            np.stack([record["hidden"] for record in batch])
        ).to(device)
        action = torch.from_numpy(
            np.stack([record["center"] for record in batch])
        ).to(device)
        action.requires_grad_(True)
        time = torch.tensor([record["time"] for record in batch], device=device)
        for model_id, model in enumerate(models):
            logit, clearance = model(hidden, action, time)
            objective = torch.nn.functional.logsigmoid(logit) + (
                args.clearance_score_weight * clearance
            )
            gradient = torch.autograd.grad(
                objective.sum(), action, retain_graph=model_id < len(models) - 1
            )[0]
            gradients[model_id].append(gradient.detach().cpu().numpy())
        actual_delta.extend(
            record["clearance_positive"] - record["clearance_negative"]
            for record in batch
        )
        clusters.extend(record["rollout"] for record in batch)
    gradient_arrays = [np.concatenate(values) for values in gradients]
    directions = np.stack([record["direction"] for record in records])
    actual_delta = np.asarray(actual_delta, dtype=np.float64)
    clusters = np.asarray(clusters)
    results = {}
    minimum_size = 1 if args.include_individual else 2
    for size in range(minimum_size, len(models) + 1):
        for member_ids in itertools.combinations(range(len(models)), size):
            ensemble_gradient = np.mean(
                np.stack([gradient_arrays[index] for index in member_ids]), axis=0
            )
            predicted_delta = np.sum(ensemble_gradient * directions, axis=(1, 2))
            key = "+".join(run_dirs[index].name for index in member_ids)
            results[key] = _metrics(
                predicted_delta,
                actual_delta,
                clusters,
                args.minimum_clearance_difference,
                args.seed,
            )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clearance-score-weight", type=float, default=0.5)
    parser.add_argument("--minimum-clearance-difference", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-individual", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
