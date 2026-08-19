#!/usr/bin/env python3
"""Validate and summarize the 4-task x 3-method pudding evaluation."""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


METHODS = ("pi05", "vlsa", "guided_08_late_time")
TASK_MARKERS = (
    "between_the_plate_and_the_ramekin",
    "on_the_ramekin",
    "on_the_stove",
    "on_the_wooden_cabinet",
)


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    return value.reshape(()).item()


def _task_index(path):
    text = str(path)
    matches = [index for index, marker in enumerate(TASK_MARKERS) if marker in text]
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely identify task for artifact: {}".format(path))
    return matches[0]


def _video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open video: {}".format(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid metadata: {}".format(path))
    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
    }


def _optional_mean(data, key):
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def _array_count(data, key):
    if key not in data:
        return 0
    return int(np.asarray(data[key]).reshape(-1).size)


def _true_count(data, key):
    if key not in data:
        return 0
    return int(np.count_nonzero(np.asarray(data[key]).reshape(-1)))


def _pct(numerator, denominator):
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    video_root = eval_root / "videos"
    flat_root = eval_root / "all_12_videos"
    flat_root.mkdir(parents=True, exist_ok=True)

    records = []
    for method in METHODS:
        method_dirs = sorted(video_root.glob("*/{}_I".format(method)))
        if len(method_dirs) != 4:
            raise RuntimeError(
                "Expected four {} result directories under {}, found {}".format(
                    method, video_root, len(method_dirs)
                )
            )
        for method_dir in method_dirs:
            npz_paths = sorted(method_dir.glob("*_last_layer_hidden_states.npz"))
            video_paths = sorted(method_dir.glob("*.mp4"))
            if len(npz_paths) != 1 or len(video_paths) != 1:
                raise RuntimeError(
                    "Expected one NPZ and one MP4 in {}, found {} and {}".format(
                        method_dir, len(npz_paths), len(video_paths)
                    )
                )

            npz_path = npz_paths[0]
            video_path = video_paths[0]
            task_index = _task_index(method_dir)
            with np.load(str(npz_path), allow_pickle=False) as data:
                success = bool(_scalar(data, "success", False))
                collision = bool(_scalar(data, "collision", False))
                safe_success = bool(_scalar(data, "safe_success", success and not collision))
                executed_actions = _array_count(data, "executed_action_steps")
                collision_actions = _true_count(data, "per_action_collision_flags")
                obstacle_motion = _true_count(data, "per_action_obstacle_motion_flags")
                mean_inference_ms = _optional_mean(data, "chunk_infer_ms")
                mean_inference = (
                    mean_inference_ms / 1000.0 if mean_inference_ms is not None else None
                )
                mean_guidance_norm = _optional_mean(
                    data, "time_conditioned_perturbation_rms"
                )

            video_metadata = _video_info(video_path)
            outcome = "success" if success else "failure"
            safety = "collision" if collision else "safe"
            flat_name = "task_{}_{}_{}_{}.mp4".format(
                task_index, method, outcome, safety
            )
            flat_path = flat_root / flat_name
            shutil.copy2(str(video_path), str(flat_path))

            records.append(
                {
                    "task_index": task_index,
                    "task_directory": method_dir.parent.name,
                    "method": method,
                    "success": success,
                    "collision": collision,
                    "safe_success": safe_success,
                    "executed_action_count": executed_actions,
                    "collision_action_count": collision_actions,
                    "obstacle_motion_action_count": obstacle_motion,
                    "mean_inference_seconds": mean_inference,
                    "mean_guidance_norm": mean_guidance_norm,
                    "video": str(video_path),
                    "flat_video": str(flat_path),
                    "npz": str(npz_path),
                    "video_metadata": video_metadata,
                }
            )

    records.sort(key=lambda record: (record["task_index"], METHODS.index(record["method"])))
    if len(records) != 12:
        raise RuntimeError("Expected 12 validated episodes, found {}".format(len(records)))
    expected_pairs = {(task, method) for task in range(4) for method in METHODS}
    actual_pairs = {(record["task_index"], record["method"]) for record in records}
    if actual_pairs != expected_pairs:
        raise RuntimeError("Episode matrix is incomplete or duplicated: {}".format(actual_pairs))

    summaries = {}
    for method in METHODS:
        rows = [record for record in records if record["method"] == method]
        success_count = sum(record["success"] for record in rows)
        collision_count = sum(record["collision"] for record in rows)
        safe_success_count = sum(record["safe_success"] for record in rows)
        summaries[method] = {
            "episodes": len(rows),
            "success_count": success_count,
            "success_rate_percent": _pct(success_count, len(rows)),
            "collision_count": collision_count,
            "collision_rate_percent": _pct(collision_count, len(rows)),
            "safe_success_count": safe_success_count,
            "safe_success_rate_percent": _pct(safe_success_count, len(rows)),
            "total_collision_actions": sum(record["collision_action_count"] for record in rows),
            "total_obstacle_motion_actions": sum(
                record["obstacle_motion_action_count"] for record in rows
            ),
        }

    payload = {
        "eval_root": str(eval_root),
        "video_folder": str(flat_root),
        "episode_count": len(records),
        "methods": summaries,
        "episodes": records,
    }
    json_path = eval_root / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Chocolate-pudding spatial evaluation",
        "",
        "All methods use the same four saved initial states and fixed flow noise.",
        "",
        "## Aggregate results",
        "",
        "| Method | Success | Collision | Safe success | Collision actions | Obstacle-motion actions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = summaries[method]
        lines.append(
            "| {method} | {success_count}/4 ({success_rate_percent:.1f}%) | "
            "{collision_count}/4 ({collision_rate_percent:.1f}%) | "
            "{safe_success_count}/4 ({safe_success_rate_percent:.1f}%) | "
            "{total_collision_actions} | {total_obstacle_motion_actions} |".format(
                method=method, **summary
            )
        )

    lines.extend(
        [
            "",
            "## Episode results",
            "",
            "| Task | Method | Success | Collision | Safe success | Actions | Video |",
            "|---:|---|:---:|:---:|:---:|---:|---|",
        ]
    )
    for record in records:
        relative_video = os.path.relpath(record["flat_video"], str(eval_root))
        display_record = dict(record)
        display_record.update(
            {
                "success": "yes" if record["success"] else "no",
                "collision": "yes" if record["collision"] else "no",
                "safe_success": "yes" if record["safe_success"] else "no",
            }
        )
        lines.append(
            "| {task_index} | {method} | {success} | {collision} | {safe_success} | "
            "{executed_action_count} | [{name}]({target}) |".format(
                name=Path(relative_video).name,
                target=relative_video,
                **display_record
            )
        )

    markdown_path = eval_root / "RESULTS.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    print("Wrote {}".format(markdown_path))


if __name__ == "__main__":
    main()
