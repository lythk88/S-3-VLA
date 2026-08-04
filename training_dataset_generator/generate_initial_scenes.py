#!/usr/bin/env python3
"""Generate a CPU-only SafeLIBERO initial-scene training dataset."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


SUITES = ("safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, default=ROOT / "training_dataset")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=179)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--suite", action="append", choices=SUITES)
    parser.add_argument("--level", action="append", choices=("I", "II"))
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def position_observations(obs: dict) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    object_names: list[str] = []
    object_positions: list[np.ndarray] = []
    obstacle_names: list[str] = []
    obstacle_positions: list[np.ndarray] = []
    for key, value in sorted(obs.items()):
        if not key.endswith("_pos"):
            continue
        position = np.asarray(value)
        if position.shape != (3,) or key.startswith("robot"):
            continue
        name = key[:-4]
        if "obstacle" in name:
            obstacle_names.append(name)
            obstacle_positions.append(position)
        else:
            object_names.append(name)
            object_positions.append(position)
    empty = np.empty((0, 3), dtype=np.float32)
    return (
        object_names,
        np.asarray(object_positions, dtype=np.float32) if object_positions else empty,
        obstacle_names,
        np.asarray(obstacle_positions, dtype=np.float32) if obstacle_positions else empty,
    )


def main() -> int:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suites = args.suite or list(SUITES)
    levels = args.level or ["I", "II"]
    task_indices = args.task_index or [0, 1, 2, 3]
    benchmark_dict = benchmark.get_benchmark_dict()
    dummy_action = [0.0] * 6 + [-1.0]
    records: list[dict] = []

    for suite in suites:
        for level in levels:
            task_suite = benchmark_dict[suite](safety_level=level)
            for task_index in task_indices:
                task = task_suite.get_task(task_index)
                bddl_path = (
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                    camera_depths=True,
                )
                try:
                    for seed in range(args.seed_start, args.seed_end + 1):
                        relative = pathlib.Path(suite) / f"level_{level}" / f"task_{task_index}" / f"seed_{seed}.npz"
                        destination = output_root / relative
                        if destination.exists() and not args.overwrite:
                            print(f"[skip] {relative}", flush=True)
                            continue
                        env.seed(seed)
                        obs = env.reset()
                        for _ in range(args.settle_steps):
                            obs, _, _, _ = env.step(dummy_action)
                        object_names, object_positions, obstacle_names, obstacle_positions = (
                            position_observations(obs)
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            destination,
                            seed=np.asarray(seed, dtype=np.int32),
                            suite=np.asarray(suite),
                            safety_level=np.asarray(level),
                            task_index=np.asarray(task_index, dtype=np.int32),
                            task_description=np.asarray(task.language),
                            bddl_file=np.asarray(str(bddl_path)),
                            object_names=np.asarray(object_names),
                            object_positions=object_positions,
                            obstacle_names=np.asarray(obstacle_names),
                            obstacle_positions=obstacle_positions,
                            robot0_eef_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                            robot0_eef_quat=np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
                            robot0_gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                            agentview_image=np.asarray(obs["agentview_image"]),
                            wrist_image=np.asarray(obs["robot0_eye_in_hand_image"]),
                        )
                        records.append(
                            {
                                "path": str(relative),
                                "seed": seed,
                                "suite": suite,
                                "safety_level": level,
                                "task_index": task_index,
                                "task_description": task.language,
                                "bddl_file": str(bddl_path),
                                "objects": len(object_names),
                                "obstacles": len(obstacle_names),
                            }
                        )
                        print(f"[save] {relative}", flush=True)
                finally:
                    env.close()

    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"[done] new_scenes={len(records)} output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    raise SystemExit(main())
