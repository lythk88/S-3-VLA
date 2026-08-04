#!/usr/bin/env python3
"""Compare paired pi0.5 baseline and residual-flow SafeLIBERO rollouts."""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

from continuous_score_guidance import ContinuousSafetyScorer


VIDEO_PATTERN = re.compile(r"^(\d+)_(success|failure)_(safe|unsafe)\.mp4$")


def collect(root: pathlib.Path, run_prefix: str) -> dict[tuple[str, str, int], dict]:
    records = {}
    for video in root.glob(f"*/{run_prefix}_*/*.mp4"):
        match = VIDEO_PATTERN.match(video.name)
        if not match:
            continue
        run_name = video.parent.name
        level = run_name.rsplit("_", 1)[-1]
        key = (video.parent.parent.name, level, int(match.group(1)))
        records[key] = {
            "success": match.group(2) == "success",
            "safe": match.group(3) == "safe",
            "npz": video.with_name(f"{video.stem}_last_layer_hidden_states.npz"),
        }
    return records


def summarize(records: list[dict]) -> tuple[float, float, float]:
    n = max(len(records), 1)
    success = sum(item["success"] for item in records) / n
    collision = sum(not item["safe"] for item in records) / n
    safe_success = sum(item["success"] and item["safe"] for item in records) / n
    return success, collision, safe_success


def mean_chunk_score(record: dict, scorer: ContinuousSafetyScorer) -> float:
    with np.load(record["npz"], allow_pickle=True) as data:
        hidden = data["last_layer_hidden_states"]
    return float(np.mean(scorer.score_hidden_states(hidden)))


def load_rollout_arrays(record: dict) -> dict[str, np.ndarray]:
    with np.load(record["npz"], allow_pickle=True) as data:
        return {
            key: np.asarray(data[key])
            for key in (
                "last_layer_hidden_states",
                "per_action_collision_flags",
                "chunk_infer_ms",
                "flow_residual_ratios",
            )
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--score-run-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    baseline = collect(args.results_root, "pi05_no_safety")
    guided = collect(args.results_root, "pi05_flow_guided")
    paired_keys = sorted(baseline.keys() & guided.keys())
    scorer = ContinuousSafetyScorer.load(args.score_run_dir, device="cpu")
    baseline_records = [baseline[key] for key in paired_keys]
    guided_records = [guided[key] for key in paired_keys]
    baseline_metrics = summarize(baseline_records)
    guided_metrics = summarize(guided_records)
    baseline_scores = [mean_chunk_score(item, scorer) for item in baseline_records]
    guided_scores = [mean_chunk_score(item, scorer) for item in guided_records]
    baseline_arrays = [load_rollout_arrays(item) for item in baseline_records]
    guided_arrays = [load_rollout_arrays(item) for item in guided_records]

    baseline_chunk_scores = np.concatenate(
        [
            scorer.score_hidden_states(arrays["last_layer_hidden_states"])
            for arrays in baseline_arrays
        ]
    )
    guided_chunk_scores = np.concatenate(
        [
            scorer.score_hidden_states(arrays["last_layer_hidden_states"])
            for arrays in guided_arrays
        ]
    )
    baseline_contacts = np.concatenate([arrays["per_action_collision_flags"] for arrays in baseline_arrays])
    guided_contacts = np.concatenate([arrays["per_action_collision_flags"] for arrays in guided_arrays])
    baseline_times = np.concatenate([arrays["chunk_infer_ms"] for arrays in baseline_arrays])
    guided_times = np.concatenate([arrays["chunk_infer_ms"] for arrays in guided_arrays])
    first_score_delta = np.asarray(
        [
            scorer.score_hidden_state(guided_array["last_layer_hidden_states"][0])
            - scorer.score_hidden_state(baseline_array["last_layer_hidden_states"][0])
            for baseline_array, guided_array in zip(baseline_arrays, guided_arrays)
        ]
    )
    residual_ratios = np.concatenate(
        [arrays["flow_residual_ratios"] for arrays in guided_arrays]
    )

    print(f"paired_rollouts: {len(paired_keys)}")
    print("method,success_rate,collision_rate,safe_success_rate,mean_predicted_safety")
    print(
        f"baseline,{baseline_metrics[0]:.4f},{baseline_metrics[1]:.4f},"
        f"{baseline_metrics[2]:.4f},{np.mean(baseline_scores):.4f}"
    )
    print(
        f"guided,{guided_metrics[0]:.4f},{guided_metrics[1]:.4f},"
        f"{guided_metrics[2]:.4f},{np.mean(guided_scores):.4f}"
    )
    print(
        "delta_guided_minus_baseline,"
        f"{guided_metrics[0] - baseline_metrics[0]:.4f},"
        f"{guided_metrics[1] - baseline_metrics[1]:.4f},"
        f"{guided_metrics[2] - baseline_metrics[2]:.4f},"
        f"{np.mean(guided_scores) - np.mean(baseline_scores):.4f}"
    )
    print(f"weighted_chunk_safety: baseline={baseline_chunk_scores.mean():.4f} guided={guided_chunk_scores.mean():.4f}")
    print(
        f"matched_first_chunk_score_delta: mean={first_score_delta.mean():.4f} "
        f"min={first_score_delta.min():.4f} max={first_score_delta.max():.4f}"
    )
    print(
        f"collision_action_fraction: baseline={baseline_contacts.mean():.4f} "
        f"guided={guided_contacts.mean():.4f} "
        f"counts={int(baseline_contacts.sum())}/{len(baseline_contacts)}->"
        f"{int(guided_contacts.sum())}/{len(guided_contacts)}"
    )
    print(
        f"steady_infer_ms_without_compile_max: "
        f"baseline={np.delete(baseline_times, np.argmax(baseline_times)).mean():.2f} "
        f"guided={np.delete(guided_times, np.argmax(guided_times)).mean():.2f}"
    )
    print(
        f"guided_residual_ratio: mean={residual_ratios.mean():.6f} "
        f"finite={bool(np.isfinite(residual_ratios).all())}"
    )
    for key in paired_keys:
        print(
            f"pair={key} baseline=({baseline[key]['success']},{baseline[key]['safe']}) "
            f"guided=({guided[key]['success']},{guided[key]['safe']})"
        )


if __name__ == "__main__":
    main()
