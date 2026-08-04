#!/usr/bin/env python3
"""Render and validate a settled SafeLIBERO initialization state."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import imageio
import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safelibero"))

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def object_name(model, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    while body_id and model.body_parentid[body_id] != 0:
        body_id = int(model.body_parentid[body_id])
    return model.body_id2name(body_id) or f"body_{body_id}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", type=pathlib.Path, required=True)
    parser.add_argument("--init", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--video-steps", type=int, default=150)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    states = np.asarray(torch.load(args.init, map_location="cpu"))
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.reset()
    obs = env.set_init_state(states[args.episode])

    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    model, data = env.sim.model, env.sim.data
    obstacle_names = [
        name.replace("_joint0", "")
        for name in model.joint_names
        if "obstacle" in name
    ]
    active = []
    for name in obstacle_names:
        position = np.asarray(obs[f"{name}_pos"])
        if position[2] > -0.05 and np.all(np.abs(position[:2]) < 0.5):
            active.append((name, position))

    active_names = {name for name, _ in active}

    def is_active_obstacle(body_name: str) -> bool:
        return any(body_name.startswith(name) for name in active_names)

    object_contacts = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        left = object_name(model, int(contact.geom1))
        right = object_name(model, int(contact.geom2))
        if left != right and (is_active_obstacle(left) or is_active_obstacle(right)):
            other = right if is_active_obstacle(left) else left
            if other not in {"main_table", "table", "floor", "world"}:
                object_contacts.add(tuple(sorted((left, right))))

    frames = []
    for _ in range(args.video_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.output, frames, fps=30)
    print(f"active_obstacles={len(active)}")
    for name, position in active:
        print(f"{name}: {position.tolist()}")
    print(f"obstacle_object_contacts={sorted(object_contacts)}")
    print(f"video={args.output}")
    status = 1 if len(active) < 2 or object_contacts else 0
    # Some legacy MuJoCo builds can terminate while destroying an offscreen
    # context. Flush the useful result before cleanup and let process teardown
    # reclaim the context.
    sys.stdout.flush()
    return status


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    raise SystemExit(main())
