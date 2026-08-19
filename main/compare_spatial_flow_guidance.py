#!/usr/bin/env python3
"""Generate a three-way report for paired Spatial SafeLIBERO guidance runs."""

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
        if match is None:
            continue
        level = video.parent.name.rsplit("_", 1)[-1]
        key = (video.parent.parent.name, level, int(match.group(1)))
        records[key] = {
            "success": match.group(2) == "success",
            "safe": match.group(3) == "safe",
            "npz": video.with_name(f"{video.stem}_last_layer_hidden_states.npz"),
        }
    return records


def load_arrays(record: dict) -> dict[str, np.ndarray]:
    with np.load(record["npz"], allow_pickle=True) as data:
        return {
            "hidden": np.asarray(data["last_layer_hidden_states"]),
            "contacts": np.asarray(data["per_action_collision_flags"]),
            "times": np.asarray(data["chunk_infer_ms"]),
            "ratios": np.asarray(data["flow_residual_ratios"]),
        }


def summarize(records: list[dict], scorer: ContinuousSafetyScorer) -> dict[str, float | int]:
    arrays = [load_arrays(record) for record in records]
    hidden_scores = [scorer.score_hidden_states(item["hidden"]) for item in arrays]
    contacts = np.concatenate([item["contacts"] for item in arrays])
    times = np.concatenate([item["times"] for item in arrays])
    ratios = np.concatenate([item["ratios"] for item in arrays])
    steady_times = np.delete(times, np.argmax(times)) if len(times) > 1 else times
    return {
        "n": len(records),
        "success": np.mean([record["success"] for record in records]),
        "collision": np.mean([not record["safe"] for record in records]),
        "safe_success": np.mean(
            [record["success"] and record["safe"] for record in records]
        ),
        "predicted_safety": np.mean([scores.mean() for scores in hidden_scores]),
        "chunk_safety": np.concatenate(hidden_scores).mean(),
        "contact_fraction": contacts.mean(),
        "contact_count": int(contacts.sum()),
        "action_count": len(contacts),
        "infer_ms": steady_times.mean(),
        "residual_ratio": ratios.mean() if len(ratios) else 0.0,
    }


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=pathlib.Path, required=True)
    parser.add_argument("--legacy-root", type=pathlib.Path, required=True)
    parser.add_argument("--strong-root", type=pathlib.Path, required=True)
    parser.add_argument("--score-run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    methods = {
        "pi0.5 baseline": collect(args.baseline_root, "pi05_no_safety"),
        "old guidance (scale 0.25)": collect(args.legacy_root, "pi05_flow_guided"),
        "increased guidance (scale 1.0)": collect(args.strong_root, "pi05_flow_guided"),
    }
    paired_keys = set.intersection(*(set(records) for records in methods.values()))
    if not paired_keys:
        raise RuntimeError("No rollouts are paired across all three methods")
    ordered_keys = sorted(paired_keys)
    scorer = ContinuousSafetyScorer.load(args.score_run_dir, device="cpu")
    metrics = {
        name: summarize([records[key] for key in ordered_keys], scorer)
        for name, records in methods.items()
    }

    lines = [
        "# Increased Flow Guidance: Spatial SafeLIBERO Report",
        "",
        "## Protocol",
        "",
        f"- Paired rollouts: {len(ordered_keys)} (identical task, safety level, episode, and deterministic flow-noise protocol).",
        "- Policy: pi0.5 LIBERO; safety layer disabled.",
        "- Guidance schedule: quadratic late guidance for `t <= 0.5`.",
        "- Compared scales: baseline 0, old guidance 0.25, increased guidance 1.0.",
        "- This is a one-episode-per-task/level pilot, not a statistically sufficient benchmark.",
        "",
        "## Results",
        "",
        "| Method | Success | Collision | Safe success | Predicted safety | Contact actions | Residual/task RMS | Steady inference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        lines.append(
            f"| {name} | {percentage(item['success'])} | {percentage(item['collision'])} | "
            f"{percentage(item['safe_success'])} | {item['predicted_safety']:.4f} | "
            f"{item['contact_count']}/{item['action_count']} ({percentage(item['contact_fraction'])}) | "
            f"{item['residual_ratio']:.4f} | {item['infer_ms']:.1f} ms |"
        )

    baseline = metrics["pi0.5 baseline"]
    old = metrics["old guidance (scale 0.25)"]
    strong = metrics["increased guidance (scale 1.0)"]
    lines += [
        "",
        "## Deltas",
        "",
        "| Comparison | Success | Collision | Safe success | Predicted safety | Contact occupancy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, reference in (("increased - baseline", baseline), ("increased - old", old)):
        lines.append(
            f"| {label} | {100 * (strong['success'] - reference['success']):+.1f} pp | "
            f"{100 * (strong['collision'] - reference['collision']):+.1f} pp | "
            f"{100 * (strong['safe_success'] - reference['safe_success']):+.1f} pp | "
            f"{strong['predicted_safety'] - reference['predicted_safety']:+.4f} | "
            f"{100 * (strong['contact_fraction'] - reference['contact_fraction']):+.2f} pp |"
        )

    lines += ["", "## Paired outcomes", ""]
    for key in ordered_keys:
        outcome = []
        for name, records in methods.items():
            record = records[key]
            outcome.append(
                f"{name}={'success' if record['success'] else 'failure'}/"
                f"{'safe' if record['safe'] else 'unsafe'}"
            )
        lines.append(f"- `{key}`: " + "; ".join(outcome))

    report = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
