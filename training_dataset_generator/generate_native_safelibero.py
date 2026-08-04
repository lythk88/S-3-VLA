#!/usr/bin/env python3
"""Generate new SafeLIBERO .pruned_init data using the existing BDDL files."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402


SUITES = ("safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=ROOT / "safelibero/libero/libero/init_files_training",
    )
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=179)
    parser.add_argument("--suite", action="append", choices=SUITES)
    parser.add_argument("--level", action="append", choices=("I", "II"))
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    init_root = output_root
    suites = args.suite or list(SUITES)
    levels = args.level or ["I", "II"]
    task_indices = args.task_index or [0, 1, 2, 3]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    benchmark_dict = benchmark.get_benchmark_dict()
    manifest = {
        "format": "SafeLIBERO native pruned_init",
        "generation_method": (
            "seeded BDDL reset with validated safety-level obstacle coordinates"
        ),
        "bddl_root": str(pathlib.Path(get_libero_path("bddl_files")).resolve()),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "states_per_task_level": len(seeds),
        "entries": [],
    }

    for suite in suites:
        for level in levels:
            task_suite = benchmark_dict[suite](safety_level=level)
            for task_index in task_indices:
                task = task_suite.get_task(task_index)
                source_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                init_name = pathlib.Path(task.init_states_file).name.replace(
                    ".pruned_init", f"_level_{level}.pruned_init"
                )
                destination_init = init_root / task.problem_folder / init_name
                destination_init.parent.mkdir(parents=True, exist_ok=True)
                if destination_init.exists() and not args.overwrite:
                    states = torch.load(destination_init, map_location="cpu")
                    print(f"[skip] {destination_init.relative_to(output_root)} shape={states.shape}", flush=True)
                else:
                    evaluation_init = (
                        ROOT
                        / "safelibero/libero/libero/init_files"
                        / task.problem_folder
                        / init_name
                    )
                    obstacle_templates = np.asarray(
                        torch.load(evaluation_init, map_location="cpu")
                    )
                    env = ControlEnv(
                        bddl_file_name=source_bddl,
                        use_camera_obs=False,
                        has_renderer=False,
                        has_offscreen_renderer=False,
                        camera_depths=False,
                    )
                    states = []
                    try:
                        model = env.env.sim.model
                        obstacle_joints = [
                            name
                            for name in model.joint_names
                            if "obstacle" in name or "box_base" in name
                        ]
                        qpos_addresses = [
                            model.get_joint_qpos_addr(name)[0] for name in obstacle_joints
                        ]
                        qvel_addresses = [
                            model.get_joint_qvel_addr(name)[0] for name in obstacle_joints
                        ]
                        nq = model.nq
                        for seed in seeds:
                            env.seed(seed)
                            env.reset()
                            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                            template = obstacle_templates[seed % len(obstacle_templates)]
                            for qpos_address, qvel_address in zip(
                                qpos_addresses, qvel_addresses
                            ):
                                state[
                                    1 + qpos_address : 1 + qpos_address + 7
                                ] = template[
                                    1 + qpos_address : 1 + qpos_address + 7
                                ]
                                state[
                                    1 + nq + qvel_address : 1 + nq + qvel_address + 6
                                ] = 0.0
                            env.set_state(state)
                            env.env.sim.forward()
                            states.append(
                                np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                            )
                    finally:
                        env.close()
                    states = np.stack(states)
                    torch.save(states, destination_init)
                    print(
                        f"[save] {destination_init.relative_to(output_root)} shape={states.shape}",
                        flush=True,
                    )

                manifest["entries"].append(
                    {
                        "suite": suite,
                        "safety_level": level,
                        "task_index": task_index,
                        "task_description": task.language,
                        "bddl_file": str(source_bddl.resolve()),
                        "init_states_file": str(destination_init.relative_to(output_root)),
                        "shape": list(states.shape),
                    }
                )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    raise SystemExit(main())
