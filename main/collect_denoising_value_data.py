"""Collect pi0.5 denoising traces and same-state clearance perturbations."""

from __future__ import annotations

import dataclasses
import hashlib
import imageio.v2 as imageio
import json
import logging
import math
import os
import pathlib
import subprocess
from typing import List

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro


ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESOLUTION = 1024
TRACE_SOURCE_PATHS = (
    "main/collect_denoising_value_data.py",
    "openpi/src/openpi/models/pi0_denoising_trace.py",
    "openpi/src/openpi/policies/denoising_trace_policy.py",
    "scripts/serve_denoising_trace_policy.py",
    "main/run_denoising_collection_matrix.py",
    "scripts/run_denoising_value_collection_full.sh",
)
POLICY_CHECKPOINT = pathlib.Path(
    "/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
)
COMPATIBLE_BRANCH_CLOCK_SOURCE_SHA256 = {
    # Original producer used by the first 137 groups.
    "7dff6a6ccb2e5d81dab4e08fddcfcd9638ee03dcf3a6eda67008376c4a106a9d",
    # First clock-restoration revision, before invocation provenance keys were
    # made manifest-aware.
    "0dd2b98a34217e04befe6d74bd1d16fdb9979b40aba9c1be43431a24117f13ee",
    # Manifest-aware invocation provenance revision.
    "6b411cb4e0c82b4e22e0cc1e66afcfd37f3514bb2ef95f330c4e510137ce1c03",
}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8006
    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "I"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    output_root: str = str(ROOT / "training_dataset/pi05_denoising_value_v1")
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 20
    max_steps: int = 0
    seed: int = 7
    denoising_steps: int = 10
    branch_times: List[float] = dataclasses.field(
        default_factory=lambda: [0.1, 0.3, 0.5]
    )
    perturbations_per_time: int = 4
    perturbation_scale: float = 0.05
    max_branch_chunks_per_episode: int = 4
    far_branch_chunks_per_episode: int = 1
    clearance_cap_m: float = 0.03
    video_fps: int = 30
    resume_existing: bool = False


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_collection_manifest(output_root: pathlib.Path, args: Args) -> None:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        diff = subprocess.check_output(
            ("git", "diff", "--binary", "HEAD", "--", *TRACE_SOURCE_PATHS),
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        commit, diff = None, b""
    checkpoint_entries = []
    for checkpoint_path in sorted(POLICY_CHECKPOINT.rglob("*")):
        if checkpoint_path.is_file():
            stat = checkpoint_path.stat()
            checkpoint_entries.append(
                (
                    str(checkpoint_path.relative_to(POLICY_CHECKPOINT)),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
    initial_state_root = pathlib.Path(get_libero_path("init_states")).resolve()
    initial_state_entries = []
    for initial_state_path in sorted(initial_state_root.rglob("*.pruned_init")):
        if initial_state_path.is_file():
            initial_state_entries.append(
                (
                    str(initial_state_path.relative_to(initial_state_root)),
                    initial_state_path.stat().st_size,
                    _sha256(initial_state_path),
                )
            )
    if not initial_state_entries:
        raise RuntimeError(
            f"No .pruned_init files found in initial-state root {initial_state_root}"
        )
    payload = {
        "schema_version": 1,
        "action_convention": (
            "denoising_noisy_actions and denoising_task_flows use pi0.5's "
            "internal normalized padded (10,32) convention; actions use the "
            "unnormalized physical (10,7) LIBERO convention"
        ),
        "clearance_convention": {
            "description": (
                "minimum MuJoCo narrow-phase robot/active-obstacle contact "
                "distance, clipped above at clearance_cap_m"
            ),
            "inactive_contact_sensor": (
                "relevant geom margin and gap are both set to clearance_cap_m, "
                "so near contacts are generated but constraints activate only at distance < 0"
            ),
        },
        "video_convention": {
            "description": (
                "one 1024x1024 agent-view frame immediately before each "
                "nominal action; counterfactual branch rollouts are not rendered"
            ),
            "fps": args.video_fps,
            "filename": "<episode>_denoising_value.mp4",
        },
        "git": {
            "commit": commit,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "source_sha256": {
            relative: _sha256(ROOT / relative) for relative in TRACE_SOURCE_PATHS
        },
        "policy_checkpoint": {
            "directory": str(POLICY_CHECKPOINT),
            "files": len(checkpoint_entries),
            "bytes": sum(entry[1] for entry in checkpoint_entries),
            "inventory_sha256": hashlib.sha256(
                json.dumps(checkpoint_entries, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "initial_state_corpus": {
            "directory": str(initial_state_root),
            "files": len(initial_state_entries),
            "bytes": sum(entry[1] for entry in initial_state_entries),
            "inventory_sha256": hashlib.sha256(
                json.dumps(initial_state_entries, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "collection_manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        # The launch-time dirty-worktree hash is provenance, not a live lock on
        # unrelated files. Across task-stratum invocations, enforce only the
        # data-producing source, checkpoint, conventions, and commit. This
        # keeps a long collection immutable while reports or training code are
        # edited elsewhere in the worktree.
        existing_encoded = path.read_text()
        existing = json.loads(existing_encoded)
        invariant_keys = (
            "schema_version",
            "action_convention",
            "clearance_convention",
            "video_convention",
            "source_sha256",
            "policy_checkpoint",
            "initial_state_corpus",
        )
        changed = [key for key in invariant_keys if existing.get(key) != payload.get(key)]
        if existing.get("git", {}).get("commit") != payload.get("git", {}).get("commit"):
            changed.append("git.commit")
        compatible_branch_clock_migration = (
            args.resume_existing
            and changed == ["source_sha256"]
            and existing.get("source_sha256", {}).get(
                "main/collect_denoising_value_data.py"
            )
            in COMPATIBLE_BRANCH_CLOCK_SOURCE_SHA256
            and all(
                existing.get("source_sha256", {}).get(source) == digest
                for source, digest in payload["source_sha256"].items()
                if source != "main/collect_denoising_value_data.py"
            )
        )
        if compatible_branch_clock_migration:
            payload["producer_history"] = [
                *existing.get("producer_history", []),
                {
                    "source_sha256": existing["source_sha256"],
                    "git": existing.get("git"),
                    "superseded_reason": (
                        "preview branches consumed robosuite episode-clock state; "
                        "completed groups remain valid and collection resumes with "
                        "episode-clock restoration"
                    ),
                },
            ]
            encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            path.write_text(encoded)
            changed = []
        if changed:
            raise RuntimeError(
                f"collection manifest producer invariants differ at {path}: {changed}"
            )
        elif not compatible_branch_clock_migration:
            encoded = existing_encoded
    else:
        path.write_text(encoded)
    invocation_dir = output_root / "collection_invocations"
    invocation_dir.mkdir(exist_ok=True)
    configuration = dataclasses.asdict(args)
    invocation_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()[:16]
    invocation_path = invocation_dir / f"{invocation_hash}.json"
    invocation_payload = {
        "schema_version": 1,
        "configuration": configuration,
        "collection_manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }
    invocation_encoded = json.dumps(invocation_payload, indent=2, sort_keys=True) + "\n"
    if invocation_path.exists() and invocation_path.read_text() != invocation_encoded:
        existing_invocation = json.loads(invocation_path.read_text())
        if existing_invocation.get("configuration") != configuration:
            raise RuntimeError(f"collection invocation differs: {invocation_path}")
        # The same stratum can be resumed by a compatible producer revision.
        # Include its manifest identity in the filename while retaining the
        # original invocation record unchanged.
        invocation_hash = hashlib.sha256(
            json.dumps(
                {
                    "configuration": configuration,
                    "collection_manifest_sha256": invocation_payload[
                        "collection_manifest_sha256"
                    ],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        invocation_path = invocation_dir / f"{invocation_hash}.json"
        if invocation_path.exists() and invocation_path.read_text() != invocation_encoded:
            raise RuntimeError(f"collection invocation differs: {invocation_path}")
    invocation_path.write_text(invocation_encoded)


def _get_env(task, resolution: int, seed: int):
    task_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_file,
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=True,
    )
    env.seed(seed)
    return env


def _default_max_steps(task_suite_name: str) -> int:
    return {
        "safelibero_spatial": 220,
        "safelibero_long": 550,
    }.get(task_suite_name, 300)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _fixed_noise(args: Args, task_id: int, episode_id: int, step: int) -> np.ndarray:
    level_id = 1 if args.safety_level == "I" else 2
    seed = np.random.SeedSequence([args.seed, level_id, task_id, episode_id, step])
    return np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)


def _policy_input(obs: dict, prompt: str, resize_size: int) -> dict:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, resize_size, resize_size)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, resize_size, resize_size)
        ),
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(prompt),
        "__debug_denoising_trace__": {"start_time": 1.0, "num_steps": 10},
    }


def _active_obstacle_names(env, obs: dict) -> list[str]:
    names = [
        name.replace("_joint0", "")
        for name in env.sim.model.joint_names
        if "obstacle" in name
    ]
    return [
        name
        for name in names
        if obs[f"{name}_pos"][2] > -0.05
        and -0.5 < obs[f"{name}_pos"][0] < 0.5
        and -0.5 < obs[f"{name}_pos"][1] < 0.5
    ]


def _body_sets(model, active_obstacles: list[str]) -> tuple[set[int], set[int]]:
    robot = {
        body_id
        for body_id, name in enumerate(model.body_names)
        if name and (name.startswith("robot0_") or name.startswith("gripper0_"))
    }
    obstacle = {
        body_id
        for body_id, name in enumerate(model.body_names)
        if name and any(obstacle_name in name for obstacle_name in active_obstacles)
    }
    if not robot or not obstacle:
        raise RuntimeError("Could not resolve robot and active-obstacle bodies")
    return robot, obstacle


def _enable_inactive_clearance_contacts(
    env, robot_bodies: set[int], obstacle_bodies: set[int], cap: float
) -> None:
    del robot_bodies
    model = env.sim.model
    relevant = np.asarray(
        [
            geom_id
            for geom_id, body_id in enumerate(model.geom_bodyid)
            if int(body_id) in obstacle_bodies
        ],
        dtype=np.int32,
    )
    model.geom_margin[relevant] = np.maximum(model.geom_margin[relevant], cap)
    model.geom_gap[relevant] = model.geom_margin[relevant]
    env.sim.forward()


def _clearance_and_collision(
    env, robot_bodies: set[int], obstacle_bodies: set[int], cap: float
) -> tuple[float, bool]:
    model, data = env.sim.model, env.sim.data
    minimum = float(cap)
    collision = False
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        relevant = (
            body_1 in robot_bodies and body_2 in obstacle_bodies
        ) or (
            body_2 in robot_bodies and body_1 in obstacle_bodies
        )
        if relevant:
            distance = float(contact.dist)
            minimum = min(minimum, distance)
            collision = collision or distance <= 0.0
    return minimum, collision


def _restore(env, state: np.ndarray):
    return env.regenerate_obs_from_state(state)


def _execute_branch(
    env,
    state: np.ndarray,
    actions: np.ndarray,
    robot_bodies: set[int],
    obstacle_bodies: set[int],
    clearance_cap: float,
    steps: int,
) -> tuple[np.ndarray, bool]:
    _restore(env, state)
    clearances = []
    collision = False
    # MuJoCo state does not include robosuite's episode clock. Branch rollouts
    # must not consume the nominal episode's horizon: doing so eventually sets
    # ``done`` on the shared environment and makes a later preview branch fail
    # with "executing action in terminated episode".
    base_env = env.env
    episode_clock = (base_env.timestep, base_env.cur_time, base_env.done)
    try:
        for action in actions[:steps]:
            env.step(np.asarray(action).tolist())
            distance, contact = _clearance_and_collision(
                env, robot_bodies, obstacle_bodies, clearance_cap
            )
            clearances.append(distance)
            collision = collision or contact
    finally:
        base_env.timestep, base_env.cur_time, base_env.done = episode_clock
    return np.asarray(clearances, dtype=np.float32), collision


def _branch_directions(args: Args, task_id: int, episode_id: int, chunk_id: int):
    seed = np.random.SeedSequence(
        [args.seed, 1000 + task_id, episode_id, chunk_id]
    )
    rng = np.random.default_rng(seed)
    pair_count = max(args.perturbations_per_time // 2, 1)
    for direction_id in range(pair_count):
        direction = np.zeros((10, 32), dtype=np.float32)
        direction[:, :3] = rng.standard_normal((10, 3))
        rms = float(np.sqrt(np.mean(np.square(direction[:, :3]))))
        direction /= max(rms, 1e-6)
        for sign in (-1, 1):
            yield direction_id, sign, direction


def collect(args: Args) -> None:
    output_root = pathlib.Path(args.output_root)
    _write_collection_manifest(output_root, args)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name](
        safety_level=args.safety_level
    )
    max_steps = args.max_steps or _default_max_steps(args.task_suite_name)

    for task_id in args.task_index:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_segment = task.language.replace(" ", "_")
        out_dir = output_root / task_segment / f"pi05_trace_{args.safety_level}"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = _get_env(task, RESOLUTION, args.seed)
        try:
            for episode_id in args.episode_index:
                output_path = out_dir / f"{episode_id}_denoising_value.npz"
                video_path = output_path.with_suffix(".mp4")
                existing_outputs = [
                    path for path in (output_path, video_path) if path.exists()
                ]
                if existing_outputs:
                    if args.resume_existing:
                        if len(existing_outputs) == 2:
                            logging.info(
                                "Skipping completed output pair %s / %s",
                                output_path,
                                video_path,
                            )
                            continue
                        raise RuntimeError(
                            "Refusing to treat an incomplete output pair as complete: "
                            f"{existing_outputs}"
                        )
                    raise FileExistsError(
                        f"refusing to overwrite existing outputs {existing_outputs}"
                    )
                temporary_output_path = output_path.with_name(
                    f".{output_path.stem}.tmp.npz"
                )
                temporary_video_path = video_path.with_name(
                    f".{video_path.stem}.tmp.mp4"
                )
                for temporary_path in (temporary_output_path, temporary_video_path):
                    if temporary_path.exists():
                        temporary_path.unlink()
                env.reset()
                obs = env.set_init_state(initial_states[episode_id])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                active_obstacles = _active_obstacle_names(env, obs)
                if not active_obstacles:
                    raise RuntimeError("No active obstacle in workspace")
                robot_bodies, obstacle_bodies = _body_sets(
                    env.sim.model, active_obstacles
                )
                _enable_inactive_clearance_contacts(
                    env,
                    robot_bodies,
                    obstacle_bodies,
                    args.clearance_cap_m,
                )

                trace_hidden, trace_noisy, trace_times = [], [], []
                trace_flows, trace_active, trace_actions = [], [], []
                chunk_steps = []
                nominal_clearance, nominal_collision = [], []
                nominal_preview_min_clearance = []
                branch_records = []
                action_plan: list[np.ndarray] = []
                step = 0
                chunk_id = 0
                branch_chunk_count = 0
                far_branch_chunk_count = 0
                done = False
                video_writer = imageio.get_writer(
                    temporary_video_path, fps=args.video_fps
                )

                try:
                    while step < max_steps and not done:
                        if action_plan:
                            video_writer.append_data(
                                np.ascontiguousarray(
                                    obs["agentview_image"][::-1, ::-1]
                                )
                            )
                            action = action_plan.pop(0)
                            obs, _, done, _ = env.step(np.asarray(action).tolist())
                            distance, collided = _clearance_and_collision(
                                env,
                                robot_bodies,
                                obstacle_bodies,
                                args.clearance_cap_m,
                            )
                            nominal_clearance.append(distance)
                            nominal_collision.append(collided)
                            step += 1
                            continue
                        element = _policy_input(obs, task.language, args.resize_size)
                        noise = _fixed_noise(args, task_id, episode_id, step)
                        response = client.infer(element, noise=noise)
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        trace_hidden.append(
                            np.asarray(response["denoising_hidden_states"], dtype=np.float16)
                        )
                        trace_noisy.append(
                            np.asarray(response["denoising_noisy_actions"], dtype=np.float32)
                        )
                        trace_times.append(
                            np.asarray(response["denoising_times"], dtype=np.float32)
                        )
                        trace_flows.append(
                            np.asarray(response["denoising_task_flows"], dtype=np.float32)
                        )
                        trace_active.append(
                            np.asarray(response["denoising_active_mask"], dtype=np.bool_)
                        )
                        trace_actions.append(actions)
                        chunk_steps.append(step)

                        base_state = env.get_sim_state().copy()
                        preview_clearance, preview_collision = _execute_branch(
                            env,
                            base_state,
                            actions,
                            robot_bodies,
                            obstacle_bodies,
                            args.clearance_cap_m,
                            args.replan_steps,
                        )
                        obs = _restore(env, base_state)
                        preview_minimum = float(np.min(preview_clearance))
                        nominal_preview_min_clearance.append(preview_minimum)
                        hazardous_preview = bool(
                            preview_collision
                            or preview_minimum < args.clearance_cap_m - 1e-6
                        )
                        take_far_context = bool(
                            not hazardous_preview
                            and far_branch_chunk_count
                            < args.far_branch_chunks_per_episode
                        )
                        should_branch = bool(
                            branch_chunk_count < args.max_branch_chunks_per_episode
                            and (hazardous_preview or take_far_context)
                        )
                        if should_branch:
                            selection_reason = (
                                "hazardous_preview" if hazardous_preview else "far_context"
                            )
                            nominal_noisy = np.asarray(
                                response["denoising_noisy_actions"], dtype=np.float32
                            )
                            nominal_times = np.asarray(
                                response["denoising_times"], dtype=np.float32
                            )
                            for requested_time in args.branch_times:
                                time_index = int(
                                    np.argmin(np.abs(nominal_times - requested_time))
                                )
                                actual_time = float(nominal_times[time_index])
                                base_noisy = nominal_noisy[time_index]
                                for direction_id, sign, direction in _branch_directions(
                                    args, task_id, episode_id, chunk_id
                                ):
                                    perturbed = base_noisy + (
                                        sign * args.perturbation_scale * direction
                                    )
                                    branch_element = _policy_input(
                                        obs, task.language, args.resize_size
                                    )
                                    branch_element["__debug_denoising_trace__"] = {
                                        "start_time": actual_time,
                                        "num_steps": args.denoising_steps,
                                        "start_noisy_action": perturbed,
                                    }
                                    branch_response = client.infer(branch_element)
                                    branch_actions = np.asarray(
                                        branch_response["actions"], dtype=np.float32
                                    )
                                    clearances, collided = _execute_branch(
                                        env,
                                        base_state,
                                        branch_actions,
                                        robot_bodies,
                                        obstacle_bodies,
                                        args.clearance_cap_m,
                                        args.replan_steps,
                                    )
                                    active = np.asarray(
                                        branch_response["denoising_active_mask"],
                                        dtype=np.bool_,
                                    )
                                    first_active = int(np.flatnonzero(active)[0])
                                    branch_records.append(
                                        {
                                            "chunk_id": chunk_id,
                                            "chunk_step": step,
                                            "time": actual_time,
                                            "direction_id": direction_id,
                                            "sign": sign,
                                            "noisy_action": perturbed,
                                            "hidden_state": np.asarray(
                                                branch_response[
                                                    "denoising_hidden_states"
                                                ][first_active],
                                                dtype=np.float16,
                                            ),
                                            "task_flow": np.asarray(
                                                branch_response[
                                                    "denoising_task_flows"
                                                ][first_active],
                                                dtype=np.float32,
                                            ),
                                            "normalized_action": np.asarray(
                                                branch_response["normalized_actions"],
                                                dtype=np.float32,
                                            ),
                                            "physical_action": branch_actions,
                                            "clearance": clearances,
                                            "collision": collided,
                                            "action_deviation": float(
                                                np.sqrt(
                                                    np.mean(
                                                        np.square(
                                                            branch_actions[
                                                                : args.replan_steps
                                                            ]
                                                            - actions[
                                                                : args.replan_steps
                                                            ]
                                                        )
                                                    )
                                                )
                                            ),
                                            "selection_reason": selection_reason,
                                            "nominal_preview_min_clearance": preview_minimum,
                                        }
                                    )
                            obs = _restore(env, base_state)
                            branch_chunk_count += 1
                            if take_far_context:
                                far_branch_chunk_count += 1
                            action_plan.extend(actions[: args.replan_steps])
                            chunk_id += 1

                        if not action_plan:
                            action_plan.extend(actions[: args.replan_steps])
                            chunk_id += 1

                        video_writer.append_data(
                            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        )
                        action = action_plan.pop(0)
                        obs, _, done, _ = env.step(np.asarray(action).tolist())
                        distance, collided = _clearance_and_collision(
                            env, robot_bodies, obstacle_bodies, args.clearance_cap_m
                        )
                        nominal_clearance.append(distance)
                        nominal_collision.append(collided)
                        step += 1
                except BaseException:
                    video_writer.close()
                    for temporary_path in (
                        temporary_output_path,
                        temporary_video_path,
                    ):
                        if temporary_path.exists():
                            temporary_path.unlink()
                    raise
                else:
                    video_writer.close()

                branch = lambda key, dtype: np.asarray(  # noqa: E731
                    [record[key] for record in branch_records], dtype=dtype
                )
                np.savez_compressed(
                    temporary_output_path,
                    task_id=np.asarray(task_id, dtype=np.int32),
                    episode_id=np.asarray(episode_id, dtype=np.int32),
                    safety_level=np.asarray(args.safety_level),
                    task_description=np.asarray(task.language),
                    active_obstacles=np.asarray(active_obstacles),
                    chunk_start_steps=np.asarray(chunk_steps, dtype=np.int32),
                    denoising_hidden_states=np.stack(trace_hidden),
                    denoising_noisy_actions=np.stack(trace_noisy),
                    denoising_times=np.stack(trace_times),
                    denoising_task_flows=np.stack(trace_flows),
                    denoising_active_mask=np.stack(trace_active),
                    physical_action_chunks=np.stack(trace_actions),
                    nominal_clearance=np.asarray(nominal_clearance, dtype=np.float32),
                    nominal_collision=np.asarray(nominal_collision, dtype=np.bool_),
                    nominal_preview_min_clearance=np.asarray(
                        nominal_preview_min_clearance, dtype=np.float32
                    ),
                    success=np.asarray(done),
                    branch_chunk_id=branch("chunk_id", np.int32),
                    branch_chunk_step=branch("chunk_step", np.int32),
                    branch_time=branch("time", np.float32),
                    branch_direction_id=branch("direction_id", np.int32),
                    branch_sign=branch("sign", np.int8),
                    branch_noisy_actions=branch("noisy_action", np.float32),
                    branch_hidden_states=branch("hidden_state", np.float16),
                    branch_task_flows=branch("task_flow", np.float32),
                    branch_normalized_actions=branch("normalized_action", np.float32),
                    branch_physical_actions=branch("physical_action", np.float32),
                    branch_clearance=branch("clearance", np.float32),
                    branch_collision=branch("collision", np.bool_),
                    branch_action_deviation=branch("action_deviation", np.float32),
                    branch_selection_reason=branch("selection_reason", str),
                    branch_nominal_preview_min_clearance=branch(
                        "nominal_preview_min_clearance", np.float32
                    ),
                    clearance_cap_m=np.asarray(args.clearance_cap_m, dtype=np.float32),
                    perturbation_scale=np.asarray(args.perturbation_scale, dtype=np.float32),
                )
                os.replace(temporary_video_path, video_path)
                os.replace(temporary_output_path, output_path)
                logging.info(
                    "Saved %s and %s: %d chunks, %d branches, %d actions",
                    output_path,
                    video_path,
                    len(chunk_steps),
                    len(branch_records),
                    step,
                )
        finally:
            env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    collect(tyro.cli(Args))
