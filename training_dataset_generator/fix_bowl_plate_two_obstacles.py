#!/usr/bin/env python3
"""Create a stable two-obstacle corridor for the close bowl-to-plate task."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "safelibero/libero/libero"
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402


TASK_STEM = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate"
)


def is_active(positions: np.ndarray) -> np.ndarray:
    return (
        (positions[:, 2] > 0)
        & (np.abs(positions[:, 0]) < 0.5)
        & (np.abs(positions[:, 1]) < 0.5)
    )


def main() -> int:
    relative = pathlib.Path("safelibero_spatial") / f"{TASK_STEM}_level_I.pruned_init"
    original_path = LIBERO_ROOT / "init_files" / relative
    output_path = LIBERO_ROOT / "init_files_training" / relative
    bddl_path = LIBERO_ROOT / "bddl_files/safelibero_spatial" / f"{TASK_STEM}.bddl"

    env = ControlEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
    )
    env.reset()
    model = env.sim.model
    joint_names = [name for name in model.joint_names if "obstacle" in name]
    qpos_addresses = [model.get_joint_qpos_addr(name)[0] for name in joint_names]
    qvel_addresses = [model.get_joint_qvel_addr(name)[0] for name in joint_names]
    nq = model.nq
    env.close()

    original = np.asarray(torch.load(original_path, map_location="cpu"))
    output = np.asarray(torch.load(output_path, map_location="cpu")).copy()
    original_positions = np.stack(
        [original[:, 1 + address : 1 + address + 3] for address in qpos_addresses], axis=1
    )
    source_rows = [
        np.flatnonzero(is_active(original_positions[:, index]))
        for index in range(len(joint_names))
    ]
    selected = (
        (joint_names.index("wine_bottle_obstacle_1_joint0"), -0.10),
        (joint_names.index("red_coffee_mug_obstacle_1_joint0"), 0.08),
    )

    for row_index in range(len(output)):
        base_row = (100 + row_index) % len(original)
        active_original = np.flatnonzero(is_active(original_positions[base_row]))
        corridor_x = float(original_positions[base_row, int(active_original[0]), 0])

        # Begin with the validated one-active/five-parked state.
        for qpos_address, qvel_address in zip(qpos_addresses, qvel_addresses):
            output[row_index, 1 + qpos_address : 1 + qpos_address + 7] = original[
                base_row, 1 + qpos_address : 1 + qpos_address + 7
            ]
            output[row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6] = original[
                base_row, 1 + nq + qvel_address : 1 + nq + qvel_address + 6
            ]

        # Park the originally active obstacle, unless it is one of the selected pair.
        for obstacle_index in active_original:
            if int(obstacle_index) not in {item[0] for item in selected}:
                address = qpos_addresses[int(obstacle_index)]
                output[row_index, 1 + address : 1 + address + 3] = (0.0, -10.0, 0.0)

        for obstacle_index, y_position in selected:
            source_row = int(source_rows[obstacle_index][row_index % len(source_rows[obstacle_index])])
            qpos_address = qpos_addresses[obstacle_index]
            qvel_address = qvel_addresses[obstacle_index]
            pose = original[source_row, 1 + qpos_address : 1 + qpos_address + 7].copy()
            pose[:2] = (corridor_x, y_position)
            output[row_index, 1 + qpos_address : 1 + qpos_address + 7] = pose
            output[row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6] = 0.0

    torch.save(output, output_path)
    print(f"saved={output_path} rows={len(output)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    raise SystemExit(main())
