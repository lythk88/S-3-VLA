#!/usr/bin/env python3
"""Replace six-obstacle raw resets with two upright SafeLIBERO obstacle poses."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "safelibero/libero/libero"
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402


def active_mask(positions: np.ndarray) -> np.ndarray:
    return (
        (positions[:, 2] > 0)
        & (np.abs(positions[:, 0]) < 0.5)
        & (np.abs(positions[:, 1]) < 0.5)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=pathlib.Path,
        default=LIBERO_ROOT / "init_files_training_six_obstacles_backup",
    )
    parser.add_argument("--original-root", type=pathlib.Path, default=LIBERO_ROOT / "init_files")
    parser.add_argument("--output-root", type=pathlib.Path, default=LIBERO_ROOT / "init_files_training")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist in the output root.",
    )
    args = parser.parse_args()
    if not args.raw_root.exists():
        if args.output_root.exists():
            print(
                f"[fallback] raw root {args.raw_root} is absent; "
                f"using existing states in {args.output_root}",
                flush=True,
            )
            args.raw_root = args.output_root
        else:
            parser.error(f"raw root does not exist: {args.raw_root}")

    records = []
    def find_upright_pose(joint_name: str) -> np.ndarray:
        raise RuntimeError(
            f"no validated upright source row found for {joint_name} in the task's original states"
        )

    for raw_path in sorted(args.raw_root.rglob("*.pruned_init")):
        relative = raw_path.relative_to(args.raw_root)
        output_path = args.output_root / relative
        if output_path.exists() and not args.overwrite:
            existing = np.asarray(torch.load(output_path, map_location="cpu"))
            records.append({"file": str(relative), "shape": list(existing.shape), "active_obstacles": 2})
            print(f"[skip] {relative} shape={existing.shape}", flush=True)
            continue
        original_path = args.original_root / relative
        bddl_name = relative.name.rsplit("_level_", 1)[0] + ".bddl"
        bddl_path = LIBERO_ROOT / "bddl_files" / relative.parent / bddl_name
        raw_states = np.asarray(torch.load(raw_path, map_location="cpu")).copy()
        raw_source_states = raw_states.copy()
        original_states = np.asarray(torch.load(original_path, map_location="cpu"))

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
        box_base_joints = [name for name in model.joint_names if "box_base" in name]
        box_base_qpos_addresses = [model.get_joint_qpos_addr(name)[0] for name in box_base_joints]
        box_base_qvel_addresses = [model.get_joint_qvel_addr(name)[0] for name in box_base_joints]
        nq = model.nq
        env.close()

        original_positions = np.stack(
            [original_states[:, 1 + address : 1 + address + 3] for address in qpos_addresses],
            axis=1,
        )
        raw_positions = np.stack(
            [raw_states[:, 1 + address : 1 + address + 3] for address in qpos_addresses],
            axis=1,
        )
        original_active = np.stack(
            [active_mask(row) for row in original_positions],
            axis=0,
        )
        source_rows_by_obstacle = [
            np.flatnonzero(original_active[:, obstacle_index])
            for obstacle_index in range(len(joint_names))
        ]
        raw_source_rows_by_obstacle = [
            np.flatnonzero(is_active)
            for is_active in np.stack(
                [active_mask(raw_positions[:, index]) for index in range(len(joint_names))]
            )
        ]
        for row_index in range(len(raw_states)):
            seed = 100 + row_index
            rng = np.random.RandomState(seed)
            base_original_index = seed % len(original_states)

            # Restore the original one-active/five-parked obstacle configuration.
            for qpos_address, qvel_address in zip(qpos_addresses, qvel_addresses):
                raw_states[row_index, 1 + qpos_address : 1 + qpos_address + 7] = original_states[
                    base_original_index, 1 + qpos_address : 1 + qpos_address + 7
                ]
                raw_states[row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6] = (
                    original_states[
                        base_original_index,
                        1 + nq + qvel_address : 1 + nq + qvel_address + 6,
                    ]
                )

            # The box base is not part of the task or obstacle set. Park it
            # outside the workspace so it cannot clutter or contact the scene.
            for qpos_address, qvel_address in zip(
                box_base_qpos_addresses, box_base_qvel_addresses
            ):
                raw_states[row_index, 1 + qpos_address : 1 + qpos_address + 3] = (
                    0.0,
                    -30.0,
                    0.0,
                )
                raw_states[
                    row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6
                ] = 0.0

            current_positions = np.stack(
                [
                    raw_states[row_index, 1 + address : 1 + address + 3]
                    for address in qpos_addresses
                ]
            )
            task_name = relative.name.rsplit("_level_", 1)[0]
            if task_name == (
                "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
                "and_place_it_on_the_plate"
            ):
                # This task has only ~11 cm between the bowl and plate, so two
                # obstacles cannot physically fit on the transfer segment. Put
                # two compact obstacles on separate points of the preceding
                # approach segment. Preserve each object's validated z and
                # quaternion so both settle upright on the tabletop.
                active_indices = np.flatnonzero(active_mask(current_positions))
                corridor_x = float(current_positions[int(active_indices[0]), 0])
                diverse_pairs = (
                    ("milk_obstacle_1_joint0", "wine_bottle_obstacle_1_joint0"),
                    ("milk_obstacle_1_joint0", "red_coffee_mug_obstacle_1_joint0"),
                    ("wine_bottle_obstacle_1_joint0", "red_coffee_mug_obstacle_1_joint0"),
                )
                selected_names = diverse_pairs[row_index % len(diverse_pairs)]
                corridor_offsets = ((-0.20, 0.02), (0.0, 0.02))
                selected_indices = {joint_names.index(name) for name in selected_names}
                for obstacle_index, (qpos_address, qvel_address) in enumerate(
                    zip(qpos_addresses, qvel_addresses)
                ):
                    if obstacle_index in selected_indices:
                        continue
                    inactive_rows = np.flatnonzero(~original_active[:, obstacle_index])
                    if not len(inactive_rows):
                        raise RuntimeError(
                            f"no parked pose available for {joint_names[obstacle_index]}"
                        )
                    parked_row = int(inactive_rows[seed % len(inactive_rows)])
                    raw_states[row_index, 1 + qpos_address : 1 + qpos_address + 7] = (
                        original_states[
                            parked_row, 1 + qpos_address : 1 + qpos_address + 7
                        ]
                    )
                    raw_states[
                        row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6
                    ] = 0.0
                for joint_name, (x_offset, y_position) in zip(
                    selected_names, corridor_offsets
                ):
                    obstacle_index = joint_names.index(joint_name)
                    qpos_address = qpos_addresses[obstacle_index]
                    qvel_address = qvel_addresses[obstacle_index]
                    source_rows = source_rows_by_obstacle[obstacle_index]
                    if len(source_rows):
                        source_row = int(source_rows[seed % len(source_rows)])
                        pose = original_states[
                            source_row, 1 + qpos_address : 1 + qpos_address + 7
                        ].copy()
                    else:
                        raw_source_rows = raw_source_rows_by_obstacle[obstacle_index]
                        if not len(raw_source_rows):
                            pose = find_upright_pose(joint_name)
                        else:
                            source_row = int(raw_source_rows[seed % len(raw_source_rows)])
                            pose = raw_source_states[
                                source_row, 1 + qpos_address : 1 + qpos_address + 7
                            ].copy()
                    pose[:2] = (corridor_x + x_offset, y_position)
                    raw_states[row_index, 1 + qpos_address : 1 + qpos_address + 7] = pose
                    raw_states[
                        row_index, 1 + nq + qvel_address : 1 + nq + qvel_address + 6
                    ] = 0.0
                continue

            first_index = int(np.flatnonzero(active_mask(current_positions))[0])
            choices = [
                index
                for index in range(len(joint_names))
                if index != first_index
                and (
                    len(source_rows_by_obstacle[index])
                    or len(raw_source_rows_by_obstacle[index])
                )
            ]
            if not choices:
                raise RuntimeError(f"no second validated obstacle pose available for {relative}")
            second_index = choices[seed % len(choices)]
            source_rows = source_rows_by_obstacle[second_index]
            second_qpos_address = qpos_addresses[second_index]
            second_qvel_address = qvel_addresses[second_index]

            # Copy the object's validated upright quaternion and zero velocity.
            if len(source_rows):
                source_row = int(source_rows[seed % len(source_rows)])
                second_pose = original_states[
                    source_row, 1 + second_qpos_address : 1 + second_qpos_address + 7
                ]
            else:
                raw_source_rows = raw_source_rows_by_obstacle[second_index]
                if not len(raw_source_rows):
                    second_pose = find_upright_pose(joint_names[second_index])
                else:
                    source_row = int(raw_source_rows[seed % len(raw_source_rows)])
                    second_pose = raw_source_states[
                        source_row, 1 + second_qpos_address : 1 + second_qpos_address + 7
                    ]
            raw_states[row_index, 1 + second_qpos_address : 1 + second_qpos_address + 7] = second_pose
            raw_states[
                row_index, 1 + nq + second_qvel_address : 1 + nq + second_qvel_address + 6
            ] = 0.0

            # Keep the two obstacles physically separate while preserving the
            # validated height. Prefer a displacement along the table's y axis:
            # the original obstacle is already sampled in the robot-to-task
            # corridor, so this makes the two obstacles affect distinct portions
            # of that corridor instead of widening it sideways. 0.26 m clears
            # even the storage-box / moka-pot pair (including the pot handle).
            first_position = raw_states[
                row_index, 1 + qpos_addresses[first_index] : 1 + qpos_addresses[first_index] + 3
            ]
            y_sign = 1.0 if rng.randint(2) else -1.0
            offsets = np.asarray(
                (
                    (0.0, 0.26 * y_sign),
                    (0.0, -0.26 * y_sign),
                    (0.26, 0.0),
                    (-0.26, 0.0),
                )
            )
            for offset in offsets:
                candidate = first_position[:2] + offset
                if np.all(np.abs(candidate) < 0.3):
                    raw_states[row_index, 1 + second_qpos_address : 1 + second_qpos_address + 2] = candidate
                    break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw_states, output_path)
        records.append({"file": str(relative), "shape": list(raw_states.shape), "active_obstacles": 2})
        print(f"[save] {relative} shape={raw_states.shape}", flush=True)

    (args.output_root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"[done] files={len(records)} output={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    raise SystemExit(main())
