"""Held-out finite-difference gate for time-conditioned value gradients."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import hashlib

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


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected trace path: {relative}")
    level = relative.parts[-2].rsplit("_", 1)[-1]
    episode = path.name.split("_", 1)[0]
    return f"{relative.parts[-3]}/{level}/{episode}"


def _load_pairs(root: pathlib.Path, allowed: set[str]):
    records = []
    for path in sorted(root.rglob("*_denoising_value.npz")):
        rollout = _group(path, root)
        if rollout not in allowed:
            continue
        with np.load(path, allow_pickle=False) as archive:
            grouped: dict[tuple[int, float, int], dict[int, int]] = {}
            for index in range(len(archive["branch_time"])):
                key = (
                    int(archive["branch_chunk_id"][index]),
                    round(float(archive["branch_time"][index]), 5),
                    int(archive["branch_direction_id"][index]),
                )
                grouped.setdefault(key, {})[
                    int(archive["branch_sign"][index])
                ] = index
            for key, indices in grouped.items():
                if -1 not in indices or 1 not in indices:
                    continue
                negative, positive = indices[-1], indices[1]
                records.append(
                    {
                        "rollout": rollout,
                        "hidden": (
                            np.asarray(archive["branch_hidden_states"][negative], dtype=np.float32)
                            + np.asarray(archive["branch_hidden_states"][positive], dtype=np.float32)
                        )
                        / 2.0,
                        "center": (
                            np.asarray(archive["branch_noisy_actions"][negative], dtype=np.float32)
                            + np.asarray(archive["branch_noisy_actions"][positive], dtype=np.float32)
                        )
                        / 2.0,
                        "direction": (
                            np.asarray(archive["branch_noisy_actions"][positive], dtype=np.float32)
                            - np.asarray(archive["branch_noisy_actions"][negative], dtype=np.float32)
                        ),
                        "time": key[1],
                        "clearance_negative": float(
                            np.min(archive["branch_clearance"][negative])
                        ),
                        "clearance_positive": float(
                            np.min(archive["branch_clearance"][positive])
                        ),
                    }
                )
    return records


def _cluster_bootstrap(values, clusters, seed=7, replicates=2000):
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    means = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(clusters == group) for group in sampled])
        means.append(float(np.mean(values[selected])))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(args) -> dict:
    run_dir = pathlib.Path(args.run_dir)
    trace_root = pathlib.Path(args.trace_root)
    manifest = json.loads((run_dir / "training_manifest.json").read_text())
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = TimeConditionedSafetyValue(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    records = _load_pairs(
        trace_root, set(manifest["split"]["validation_groups"])
    )
    if not records:
        raise RuntimeError("No validation perturbation pairs found")

    aligned, gains, directional = [], [], []
    clusters = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        hidden = torch.from_numpy(np.stack([item["hidden"] for item in batch])).to(device)
        action = torch.from_numpy(np.stack([item["center"] for item in batch])).to(device)
        action.requires_grad_(True)
        time = torch.tensor([item["time"] for item in batch], device=device)
        logit, clearance = model(hidden, action, time)
        objective = torch.nn.functional.logsigmoid(logit) + args.clearance_score_weight * clearance
        gradient = torch.autograd.grad(objective.sum(), action)[0].detach().cpu().numpy()
        for item, grad in zip(batch, gradient):
            predicted_delta = float(np.sum(grad * item["direction"]))
            actual_delta = item["clearance_positive"] - item["clearance_negative"]
            informative = abs(actual_delta) >= args.minimum_clearance_difference
            if informative:
                aligned.append(float(predicted_delta * actual_delta > 0.0))
                selected_positive = predicted_delta >= 0.0
                selected = (
                    item["clearance_positive"]
                    if selected_positive
                    else item["clearance_negative"]
                )
                rejected = (
                    item["clearance_negative"]
                    if selected_positive
                    else item["clearance_positive"]
                )
                gains.append(selected - rejected)
                directional.append(predicted_delta * np.sign(actual_delta))
                clusters.append(item["rollout"])

    aligned = np.asarray(aligned, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    directional = np.asarray(directional, dtype=np.float64)
    clusters = np.asarray(clusters)
    if not len(aligned):
        raise RuntimeError("No perturbation pairs exceeded the clearance threshold")
    metrics = {
        "pairs_total": len(records),
        "pairs_informative": len(aligned),
        "rollout_clusters": len(np.unique(clusters)),
        "gradient_direction_accuracy": float(np.mean(aligned)),
        "gradient_direction_accuracy_cluster_ci95": _cluster_bootstrap(
            aligned, clusters, args.seed
        ),
        "selected_minus_rejected_clearance_m": float(np.mean(gains)),
        "selected_minus_rejected_clearance_cluster_ci95": _cluster_bootstrap(
            gains, clusters, args.seed
        ),
        "mean_signed_directional_derivative": float(np.mean(directional)),
    }
    accuracy_lower = metrics["gradient_direction_accuracy_cluster_ci95"][0]
    clearance_lower = metrics["selected_minus_rejected_clearance_cluster_ci95"][0]
    metrics["pass"] = bool(accuracy_lower > 0.5 and clearance_lower > 0.0)
    output = run_dir / "gradient_gate_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest["status"] = (
        "offline_gradient_gate_passed_live_gate_pending"
        if metrics["pass"]
        else "offline_gradient_gate_failed"
    )
    manifest.setdefault("artifacts", {})["gradient_gate_metrics.json"] = {
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clearance-score-weight", type=float, default=0.5)
    parser.add_argument("--minimum-clearance-difference", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)
