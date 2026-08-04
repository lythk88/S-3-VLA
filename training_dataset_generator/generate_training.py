#!/usr/bin/env python3
"""Collect pi0.5 hidden/action chunks from saved SafeLIBERO training states."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero import benchmark  # noqa: E402


SUITES = ("safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long")


def completed_episodes(out_dir: pathlib.Path) -> set[int]:
    completed: set[int] = set()
    for path in out_dir.glob("*_last_layer_hidden_states.npz"):
        match = re.match(r"^(\d+)_(?:success|failure)_(?:safe|unsafe)_", path.name)
        if match:
            completed.add(int(match.group(1)))
    for path in out_dir.glob("*_skipped_no_active_obstacle.txt"):
        match = re.match(r"^(\d+)_skipped_no_active_obstacle\.txt$", path.name)
        if match:
            completed.add(int(match.group(1)))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=ROOT / "training_dataset/pi05_hidden_chunks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--suite", action="append", choices=SUITES)
    parser.add_argument("--level", action="append", choices=("I", "II"))
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--episode-index", type=int, action="append")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    suites = args.suite or list(SUITES)
    levels = args.level or ["I", "II"]
    tasks = args.task_index or [0, 1, 2, 3]
    episodes = args.episode_index or list(range(8))
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    env["PYTHONPATH"] = str(root / "safelibero") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["LIBERO_CONFIG_PATH"] = str(root / "training_dataset_generator/libero_training_config")

    benchmark_dict = benchmark.get_benchmark_dict()
    launched = 0
    for suite in suites:
        for level in levels:
            task_suite = benchmark_dict[suite](safety_level=level)
            for task_index in tasks:
                task = task_suite.get_task(task_index)
                init_name = pathlib.Path(task.init_states_file).name.replace(
                    ".pruned_init", f"_level_{level}.pruned_init"
                )
                init_path = (
                    root
                    / "safelibero/libero/libero/init_files_training"
                    / task.problem_folder
                    / init_name
                )
                if not init_path.exists():
                    print(
                        f"[skip unavailable] {suite} level={level} task={task_index} "
                        f"init={init_path}",
                        flush=True,
                    )
                    continue
                task_dir = output_root / task.language.replace(" ", "_") / f"pi05_no_safety_{level}"
                missing = sorted(set(episodes) - completed_episodes(task_dir))
                if not missing:
                    print(f"[skip] {suite} level={level} task={task_index}", flush=True)
                    continue
                print(
                    f"[collect] {suite} level={level} task={task_index} episodes={missing}",
                    flush=True,
                )
                cmd = [
                    str(root / "main/.venv/bin/python"),
                    str(root / "main/main_aegis.py"),
                    "--host", args.host,
                    "--port", str(args.port),
                    "--task-suite-name", suite,
                    "--safety-level", level,
                    "--task-index", str(task_index),
                    "--episode-index", *map(str, missing),
                    "--disable-safety-layer",
                    "--video-out-path", str(output_root),
                ]
                subprocess.run(cmd, cwd=root / "main", env=env, check=True)
                launched += len(missing)
    print(f"[done] collected={launched} output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
