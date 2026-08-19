"""Fresh-simulator gate for direct noisy-action safety-value gradients."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import pathlib

from libero.libero import benchmark
import numpy as np
from openpi_client import websocket_client_policy

from collect_denoising_value_data import Args as CollectionArgs
from collect_denoising_value_data import RESOLUTION
from collect_denoising_value_data import _active_obstacle_names
from collect_denoising_value_data import _body_sets
from collect_denoising_value_data import _clearance_and_collision
from collect_denoising_value_data import _enable_inactive_clearance_contacts
from collect_denoising_value_data import _execute_branch
from collect_denoising_value_data import _fixed_noise
from collect_denoising_value_data import _get_env
from collect_denoising_value_data import _policy_input


ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_mapping() -> dict[tuple[str, str], tuple[str, int]]:
    result = {}
    for suite_name in (
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    ):
        for level in ("I", "II"):
            suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level)
            for task_id in range(suite.n_tasks):
                segment = suite.get_task(task_id).language.replace(" ", "_")
                result[(segment, level)] = (suite_name, task_id)
    return result


def _trace_path(trace_root: pathlib.Path, rollout: str) -> pathlib.Path:
    task, level, episode = rollout.rsplit("/", 2)
    path = trace_root / task / f"pi05_trace_{level}" / f"{episode}_denoising_value.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _target_steps(path: pathlib.Path, max_contexts: int) -> list[int]:
    with np.load(path, allow_pickle=False) as archive:
        steps = np.asarray(archive["branch_chunk_step"], dtype=np.int32)
        reasons = np.asarray(archive["branch_selection_reason"]).astype(str)
    hazardous = sorted(set(int(step) for step in steps[reasons == "hazardous_preview"]))
    far = sorted(set(int(step) for step in steps[reasons == "far_context"]))
    return (hazardous + [step for step in far if step not in hazardous])[:max_contexts]


def _cluster_ci(values: np.ndarray, clusters: np.ndarray, seed: int, replicates: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    means = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(clusters == cluster) for cluster in sampled]
        )
        means.append(float(np.mean(values[indices])))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _guidance_element(
    obs: dict,
    prompt: str,
    args,
    guidance: dict,
    direction_sign: int,
) -> dict:
    element = _policy_input(obs, prompt, args.resize_size)
    element.pop("__debug_denoising_trace__")
    element["__time_conditioned_guidance__"] = {
        "time": guidance["time"],
        "scale": guidance["scale"],
        "direction_sign": direction_sign,
        "clearance_score_weight": guidance["clearance_score_weight"],
        "translation_only": guidance["translation_only"],
        "denoising_steps": 10,
        "geometry": guidance["geometry"],
        "normalization": guidance["normalization"],
        "integration": guidance["integration"],
        "value_backtracking": guidance["value_backtracking"],
    }
    return element


def _run_rollout(
    client,
    trace_root: pathlib.Path,
    rollout: str,
    suite_name: str,
    task_id: int,
    args,
    guidance: dict,
) -> tuple[list[dict], list[int]]:
    task_segment, level, episode_text = rollout.rsplit("/", 2)
    episode = int(episode_text)
    trace_path = _trace_path(trace_root, rollout)
    targets = _target_steps(trace_path, args.max_contexts_per_rollout)
    if not targets:
        return [], []
    suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level)
    task = suite.get_task(task_id)
    if task.language.replace(" ", "_") != task_segment:
        raise RuntimeError(f"Task mapping mismatch for {rollout}")
    initial_states = suite.get_task_init_states(task_id)
    env = _get_env(task, RESOLUTION, args.seed)
    records = []
    reached = []
    collection_args = CollectionArgs(safety_level=level, seed=args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[episode])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        active_obstacles = _active_obstacle_names(env, obs)
        robot_bodies, obstacle_bodies = _body_sets(env.sim.model, active_obstacles)
        _enable_inactive_clearance_contacts(
            env, robot_bodies, obstacle_bodies, args.clearance_cap_m
        )
        action_plan = deque()
        step = 0
        done = False
        while step <= max(targets) and not done:
            if not action_plan:
                noise = _fixed_noise(collection_args, task_id, episode, step)
                nominal_element = _policy_input(obs, task.language, args.resize_size)
                nominal_element.pop("__debug_denoising_trace__")
                nominal_response = client.infer(nominal_element, noise=noise)
                nominal_actions = np.asarray(nominal_response["actions"], dtype=np.float32)
                if step in targets:
                    base_state = env.get_sim_state().copy()
                    branch_results = {}
                    telemetry = {}
                    nominal_clearance, nominal_collision = _execute_branch(
                        env,
                        base_state,
                        nominal_actions,
                        robot_bodies,
                        obstacle_bodies,
                        args.clearance_cap_m,
                        args.replan_steps,
                    )
                    branch_results[0] = (
                        float(np.min(nominal_clearance)),
                        bool(nominal_collision),
                    )
                    for sign in (-1, 1):
                        response = client.infer(
                            _guidance_element(
                                obs, task.language, args, guidance, sign
                            ),
                            noise=noise,
                        )
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        clearance, collision = _execute_branch(
                            env,
                            base_state,
                            actions,
                            robot_bodies,
                            obstacle_bodies,
                            args.clearance_cap_m,
                            args.replan_steps,
                        )
                        branch_results[sign] = (
                            float(np.min(clearance)),
                            bool(collision),
                        )
                        telemetry[str(sign)] = {
                            "score_before": float(response["time_conditioned_score_before"]),
                            "clearance_prediction_before": float(
                                response["time_conditioned_clearance_before"]
                            ),
                            "gradient_rms": float(
                                response["time_conditioned_gradient_rms"]
                            ),
                            "perturbation_rms": float(
                                response["time_conditioned_perturbation_rms"]
                            ),
                            "actual_time": float(
                                response["time_conditioned_guidance_time"]
                            ),
                        }
                    obs = env.regenerate_obs_from_state(base_state)
                    records.append(
                        {
                            "rollout": rollout,
                            "suite": suite_name,
                            "task_id": task_id,
                            "chunk_step": step,
                            "clearance_minus_m": branch_results[-1][0],
                            "clearance_nominal_m": branch_results[0][0],
                            "clearance_plus_m": branch_results[1][0],
                            "collision_minus": branch_results[-1][1],
                            "collision_nominal": branch_results[0][1],
                            "collision_plus": branch_results[1][1],
                            "telemetry": telemetry,
                        }
                    )
                    reached.append(step)
                action_plan.extend(nominal_actions[: args.replan_steps])
            action = action_plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            _clearance_and_collision(
                env, robot_bodies, obstacle_bodies, args.clearance_cap_m
            )
            step += 1
    finally:
        env.close()
    return records, sorted(set(targets) - set(reached))


def evaluate(args) -> dict:
    run_dir = pathlib.Path(args.run_dir)
    trace_root = pathlib.Path(args.trace_root)
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "offline_gradient_gate_passed_live_gate_pending":
        raise RuntimeError(
            "Live gate requires status offline_gradient_gate_passed_live_gate_pending"
        )
    approach_path = run_dir / "approach.json"
    approach = json.loads(approach_path.read_text())
    approach_guidance = approach["guidance"]
    times = [float(value) for value in approach_guidance["times"]]
    if len(times) != 1:
        raise RuntimeError(
            "The symmetric live gate requires exactly one configured guidance time"
        )
    guidance = {
        "time": times[0],
        "scale": float(approach_guidance["scale"]),
        "clearance_score_weight": float(
            approach_guidance["clearance_score_weight"]
        ),
        "translation_only": bool(approach_guidance["translation_only"]),
        "geometry": str(approach_guidance["geometry"]),
        "normalization": str(approach_guidance["normalization"]),
        "integration": str(approach_guidance["integration"]),
        "value_backtracking": bool(approach_guidance["value_backtracking"]),
    }
    mapping = _task_mapping()
    rollouts = []
    for rollout in manifest["split"]["validation_groups"]:
        task, level, _ = rollout.rsplit("/", 2)
        suite_name, task_id = mapping[(task, level)]
        if args.suite and suite_name != args.suite:
            continue
        rollouts.append((rollout, suite_name, task_id))
    if args.max_rollouts:
        rollouts = rollouts[: args.max_rollouts]
    if not rollouts:
        raise RuntimeError("No validation rollouts match the live-gate filter")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    records, missing = [], {}
    for index, (rollout, suite_name, task_id) in enumerate(rollouts, 1):
        print(f"live gate rollout {index}/{len(rollouts)}: {rollout}", flush=True)
        rollout_records, missing_steps = _run_rollout(
            client, trace_root, rollout, suite_name, task_id, args, guidance
        )
        records.extend(rollout_records)
        if missing_steps:
            missing[rollout] = missing_steps
    if not records:
        raise RuntimeError("The live gate did not reach any probe contexts")

    plus = np.asarray([record["clearance_plus_m"] for record in records])
    minus = np.asarray([record["clearance_minus_m"] for record in records])
    nominal = np.asarray([record["clearance_nominal_m"] for record in records])
    clusters = np.asarray([record["rollout"] for record in records])
    central_gain = plus - minus
    informative = np.abs(central_gain) >= args.minimum_clearance_difference
    if not np.any(informative):
        raise RuntimeError("No live probes exceeded the clearance threshold")
    accuracy = (central_gain[informative] > 0.0).astype(np.float64)
    central_informative = central_gain[informative]
    cluster_informative = clusters[informative]
    plus_nominal = plus - nominal
    metrics = {
        "schema_version": 1,
        "configuration": vars(args),
        "approach_path": str(approach_path.resolve()),
        "approach_sha256": _sha256(approach_path),
        "deployed_guidance": guidance,
        "rollouts_requested": len(rollouts),
        "rollout_clusters_reached": len(np.unique(clusters)),
        "probe_contexts": len(records),
        "informative_probe_contexts": int(np.sum(informative)),
        "missing_target_steps": missing,
        "gradient_direction_accuracy": float(np.mean(accuracy)),
        "gradient_direction_accuracy_cluster_ci95": _cluster_ci(
            accuracy, cluster_informative, args.seed, args.bootstrap_replicates
        ),
        "plus_minus_clearance_gain_m": float(np.mean(central_informative)),
        "plus_minus_clearance_gain_cluster_ci95": _cluster_ci(
            central_informative,
            cluster_informative,
            args.seed,
            args.bootstrap_replicates,
        ),
        "plus_nominal_clearance_gain_m": float(np.mean(plus_nominal)),
        "plus_nominal_clearance_gain_cluster_ci95": _cluster_ci(
            plus_nominal, clusters, args.seed, args.bootstrap_replicates
        ),
        "collision_fraction": {
            "minus": float(np.mean([record["collision_minus"] for record in records])),
            "nominal": float(np.mean([record["collision_nominal"] for record in records])),
            "plus": float(np.mean([record["collision_plus"] for record in records])),
        },
        "records": records,
    }
    metrics["pass"] = bool(
        metrics["gradient_direction_accuracy_cluster_ci95"][0] > 0.5
        and metrics["plus_minus_clearance_gain_cluster_ci95"][0] > 0.0
    )
    output_path = run_dir / "live_gradient_gate_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest["status"] = (
        "live_gradient_gate_passed" if metrics["pass"] else "live_gradient_gate_failed"
    )
    manifest.setdefault("artifacts", {})[output_path.name] = {
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--suite", default="safelibero_spatial")
    parser.add_argument("--clearance-cap-m", type=float, default=0.03)
    parser.add_argument("--minimum-clearance-difference", type=float, default=0.0005)
    parser.add_argument("--max-contexts-per-rollout", type=int, default=3)
    parser.add_argument("--max-rollouts", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)
