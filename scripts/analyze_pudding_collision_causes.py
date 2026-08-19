#!/usr/bin/env python3
"""Replay saved rollouts offline and attribute every obstacle contact.

main_aegis.py records the executed action chunks, so a rollout can be replayed
in the simulator without a policy server. This replays each one, and for every
step where a robot body touches the pudding it records which link made contact
and how far the end effector was at that moment.

The end-effector distance matters because time-conditioned guidance perturbs
end-effector translation only: contacts made by the forearm or elbow while the
gripper is far away cannot be avoided by that control channel.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "results/chocolate_pudding_all_suites"
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(SCENE_ROOT / "scenes/config"))
sys.path.insert(0, str(ROOT / "safelibero"))

PUDDING = "chocolate_pudding_obstacle_fixed_1"
DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 20
SEED = 7  # main_aegis.py Args.seed default


def link_group(body_name: str) -> str:
    """Coarse label: which part of the arm is this body?"""
    if body_name.startswith("gripper0_"):
        return "gripper"
    tail = body_name.replace("robot0_", "")
    if tail in {"right_hand", "link7", "link6"}:
        return "wrist"
    if tail in {"link5", "link4"}:
        return "forearm"
    if tail in {"link3", "link2", "link1", "link0", "base"}:
        return "upper_arm/base"
    return tail


def replay(npz_path: pathlib.Path, record: dict) -> dict | None:
    from libero.libero.envs import OffScreenRenderEnv
    import torch

    data = np.load(npz_path, allow_pickle=True)
    chunks = data["action_chunks"]
    if chunks.size == 0:
        return None
    actions = chunks.reshape(-1, chunks.shape[-1])

    env = OffScreenRenderEnv(
        bddl_file_name=record["bddl"],
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
    )
    env.seed(SEED)
    env.reset()
    init_states = torch.load(record["init_state"])
    obs = env.set_init_state(np.asarray(init_states)[0])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    model, sim_data = env.sim.model, env.sim.data
    robot_bodies = {
        i: name
        for i, name in enumerate(model.body_names)
        if name and (name.startswith("robot0_") or name.startswith("gripper0_"))
    }
    obstacle_bodies = {
        i for i, name in enumerate(model.body_names) if name and PUDDING in name
    }

    start_obstacle = np.asarray(obs[f"{PUDDING}_pos"]).copy()
    contacts_by_group: collections.Counter = collections.Counter()
    first_contact = None
    eef_distances = []
    min_eef_distance = float("inf")

    for step, action in enumerate(actions):
        obs, _, _, _ = env.step(np.asarray(action, dtype=np.float64).tolist())
        eef = np.asarray(obs["robot0_eef_pos"])
        obstacle = np.asarray(obs[f"{PUDDING}_pos"])
        distance = float(np.linalg.norm(eef - obstacle))
        min_eef_distance = min(min_eef_distance, distance)

        hit_groups = set()
        for index in range(sim_data.ncon):
            contact = sim_data.contact[index]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            if b1 in robot_bodies and b2 in obstacle_bodies:
                hit_groups.add(link_group(robot_bodies[b1]))
            elif b2 in robot_bodies and b1 in obstacle_bodies:
                hit_groups.add(link_group(robot_bodies[b2]))
        if hit_groups:
            for group in hit_groups:
                contacts_by_group[group] += 1
            eef_distances.append(distance)
            if first_contact is None:
                first_contact = {
                    "step": step,
                    "groups": sorted(hit_groups),
                    "eef_to_obstacle_m": distance,
                }

    displacement = float(
        np.linalg.norm(np.asarray(obs[f"{PUDDING}_pos"]) - start_obstacle)
    )
    env.close()
    return {
        **{k: record[k] for k in ("suite", "task_index", "method")},
        "steps": int(len(actions)),
        "contact_steps": int(sum(contacts_by_group.values())),
        "contacts_by_group": dict(contacts_by_group),
        "first_contact": first_contact,
        "eef_distance_at_contact_mean": (
            float(np.mean(eef_distances)) if eef_distances else None
        ),
        "min_eef_to_obstacle_m": (
            None if min_eef_distance == float("inf") else min_eef_distance
        ),
        "obstacle_displacement_m": displacement,
        "recorded_collision": bool(data["collision"]),
        "recorded_success": bool(data["success"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=pathlib.Path,
        default=ROOT / "results/chocolate_pudding_all_suites_eval",
    )
    parser.add_argument("--methods", nargs="+", default=["pi05", "guided_riskgate_sm"])
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    manifest = json.loads((SCENE_ROOT / "scene_manifest.json").read_text())
    scene_of = {(r["suite"], r["task_index"]): r for r in manifest["records"]}
    index = json.loads((args.eval_root / "video_index.json").read_text())["rollouts"]

    results = []
    for rollout in index:
        if rollout["method"] not in args.methods:
            continue
        scene = scene_of.get((rollout["suite"], rollout["task_index"]))
        if scene is None:
            continue
        npz = pathlib.Path(rollout["source"]).with_suffix("")
        npz = npz.parent / f"{npz.name}_last_layer_hidden_states.npz"
        if not npz.is_file():
            continue
        record = {
            "suite": rollout["suite"],
            "task_index": rollout["task_index"],
            "method": rollout["method"],
            "bddl": scene["bddl"],
            "init_state": scene["init_state"],
        }
        outcome = replay(npz, record)
        if outcome is None:
            continue
        results.append(outcome)
        print(
            f"[{len(results):2d}] {outcome['suite'].replace('safelibero_',''):8s} "
            f"task{outcome['task_index']} {outcome['method']:20s} "
            f"contacts={outcome['contact_steps']:3d} "
            f"{outcome['contacts_by_group']} "
            f"min_eef_dist={outcome['min_eef_to_obstacle_m']:.3f}",
            flush=True,
        )

    out = args.out or args.eval_root / "collision_attribution.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    print("\n=== contact steps attributed by arm segment ===")
    for method in args.methods:
        totals: collections.Counter = collections.Counter()
        for r in results:
            if r["method"] == method:
                totals.update(r["contacts_by_group"])
        total = sum(totals.values())
        print(f"\n{method}  ({total} contact-steps)")
        for group, count in totals.most_common():
            print(f"   {group:16s} {count:5d}  {count/max(total,1):6.1%}")

    print("\n=== end-effector distance when contact occurs ===")
    for method in args.methods:
        distances = [
            r["eef_distance_at_contact_mean"]
            for r in results
            if r["method"] == method and r["eef_distance_at_contact_mean"] is not None
        ]
        if distances:
            print(
                f"{method:20s} mean {np.mean(distances):.3f} m over "
                f"{len(distances)} rollouts with contact"
            )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
