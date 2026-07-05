#!/usr/bin/env python3
"""Resume pi0.5 no-safety SafeLIBERO evals without overwriting outputs."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

DEFAULT_ROOT = pathlib.Path(os.environ.get("ROOT", "/home/namn1/vlsa-aegis")).resolve()
sys.path.insert(0, str(DEFAULT_ROOT / "safelibero"))

from libero.libero import benchmark


DEFAULT_SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)
DEFAULT_LEVELS = ("I", "II")
DEFAULT_TASKS = tuple(range(4))
DEFAULT_EPISODES = tuple(range(50))


def task_segment(task_description: str) -> str:
    return task_description.replace(" ", "_")


def completed_episodes(out_dir: pathlib.Path) -> set[int]:
    done: set[int] = set()
    if not out_dir.exists():
        return done

    for video_path in out_dir.glob("*.mp4"):
        match = re.match(r"^(\d+)_(?:success|failure)_(?:safe|unsafe)\.mp4$", video_path.name)
        if not match:
            continue
        episode_idx = int(match.group(1))
        hidden_path = video_path.with_name(f"{video_path.stem}_last_layer_hidden_states.npz")
        if hidden_path.exists():
            done.add(episode_idx)
    return done


def parse_ints(values: list[str] | None, default: tuple[int, ...]) -> list[int]:
    if not values:
        return list(default)
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output-root", default="/home/namn1/vlsa-aegis/results/pi05_no_safety_with_hidden_full")
    parser.add_argument("--suite", action="append", choices=DEFAULT_SUITES)
    parser.add_argument("--level", action="append", choices=DEFAULT_LEVELS)
    parser.add_argument("--task-index", action="append")
    parser.add_argument("--episode-index", action="append")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    suites = tuple(args.suite or DEFAULT_SUITES)
    levels = tuple(args.level or DEFAULT_LEVELS)
    tasks = parse_ints(args.task_index, DEFAULT_TASKS)
    episodes = parse_ints(args.episode_index, DEFAULT_EPISODES)

    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "osmesa")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "osmesa")
    env["PYTHONPATH"] = f"{root / 'safelibero'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(root / "safelibero")

    benchmark_dict = benchmark.get_benchmark_dict()
    total_missing = 0

    for suite in suites:
        for level in levels:
            task_suite = benchmark_dict[suite](safety_level=level)
            for task_id in tasks:
                task = task_suite.get_task(task_id)
                result_dir = output_root / task_segment(task.language) / f"pi05_no_safety_{level}"
                done = completed_episodes(result_dir)
                missing = [episode for episode in episodes if episode not in done]
                if not missing:
                    print(f"[skip] suite={suite} level={level} task={task_id} completed={len(done)}")
                    continue

                total_missing += len(missing)
                print(
                    f"[run] suite={suite} level={level} task={task_id} "
                    f"missing={len(missing)} episodes={missing}",
                    flush=True,
                )
                cmd = [
                    str(root / "main/.venv/bin/python"),
                    str(root / "main/main_aegis.py"),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--task-suite-name",
                    suite,
                    "--safety-level",
                    level,
                    "--task-index",
                    str(task_id),
                    "--episode-index",
                    *map(str, missing),
                    "--disable-safety-layer",
                    "--video-out-path",
                    str(output_root),
                ]
                subprocess.run(cmd, cwd=root / "main", env=env, check=True)

    print(f"[done] missing episodes launched/completed: {total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
