"""Select 20 common, obstacle-active, training-disjoint Spatial episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from libero.libero import benchmark

from collect_denoising_value_data import DUMMY_ACTION
from collect_denoising_value_data import _active_obstacle_names
from collect_denoising_value_data import _get_env


def _training_episodes(root: pathlib.Path, task: str, level: str) -> set[int]:
    directory = root / task / f"pi05_no_safety_{level}"
    return {
        int(path.name.split("_", 1)[0]) for path in directory.glob("*.npz")
    }


def select(args) -> dict:
    source_root = pathlib.Path(args.source_root)
    suite_factory = benchmark.get_benchmark_dict()["safelibero_spatial"]
    strata = {}
    common = set(range(args.first_episode, args.last_episode + 1))
    for level in ("I", "II"):
        suite = suite_factory(safety_level=level)
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            task_segment = task.language.replace(" ", "_")
            initial_states = suite.get_task_init_states(task_id)
            # Activation depends only on simulator positions. Low-resolution
            # cameras avoid expensive 1024x1024 rendering during this audit.
            env = _get_env(task, args.camera_resolution, args.seed)
            active = {}
            try:
                for episode in range(args.first_episode, args.last_episode + 1):
                    env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    names = _active_obstacle_names(env, obs)
                    active[str(episode)] = names
            finally:
                env.close()
            valid = {int(episode) for episode, names in active.items() if names}
            training = _training_episodes(source_root, task_segment, level)
            eligible = valid - training
            common &= eligible
            strata[f"{task_segment}/{level}"] = {
                "task_id": task_id,
                "training_episodes": sorted(training),
                "active_obstacles": active,
                "eligible_episodes": sorted(eligible),
            }
    selected = sorted(common)[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(
            f"Only {len(selected)} common valid held-out episodes found; need {args.count}"
        )
    payload = {
        "schema_version": 1,
        "suite": "safelibero_spatial",
        "selection_rule": (
            "first 20 episode IDs in [first_episode,last_episode] that have at least "
            "one active obstacle in every task/level stratum after stabilization and "
            "do not occur in pi05_hidden_chunks for the corresponding stratum"
        ),
        "configuration": vars(args),
        "selected_episodes": selected,
        "strata": strata,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    payload["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="/home/lythk/safe-flow-matching/training_dataset/pi05_hidden_chunks",
    )
    parser.add_argument(
        "--output",
        default="/home/lythk/safe-flow-matching/results/spatial_heldout20_split.json",
    )
    parser.add_argument("--first-episode", type=int, default=8)
    parser.add_argument("--last-episode", type=int, default=49)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--num-steps-wait", type=int, default=20)
    parser.add_argument("--camera-resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = select(parse_args())
    print(json.dumps({"selected_episodes": result["selected_episodes"], "manifest_sha256": result["manifest_sha256"]}, indent=2))
