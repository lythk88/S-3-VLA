import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.policies.action_expert_qp import Ellipsoid as ActionExpertEllipsoid
from openpi.policies.action_expert_qp import ControllerRollout
from openpi.policies.action_expert_qp import ObstaclePrimitive as ActionExpertObstacle
from openpi.policies.action_expert_qp import ellipsoid_obstacle_gap
from openpi.policies.action_expert_qp import ellipsoid_obstacle_gap_details
from openpi.policies.action_expert_qp import primitive_obstacle_gap
from openpi.policies.action_expert_qp import primitive_obstacle_gap_details
from openpi.policies.action_expert_qp import project_action_chunk_with_qp
from openpi.policies.action_expert_qp import project_action_chunk_with_adaptive_radius
from openpi.policies.action_expert_qp import rollout_eef_trajectory
from openpi.policies.action_expert_qp import trajectory_barriers as action_expert_trajectory_barriers
import tqdm
import tyro
import mujoco
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
from utils import rot3, quat_R, quat_euler, vector_hat, project_matrix, \
    compute_h_ij, compute_h_coeffs_3d, get_point_cloud, filtering_points, fit_ellipse, plot_points_ellipse, \
    obstacle_detection, overlay_gripper_ellipsoid_on_rgb
import warnings
warnings.filterwarnings("ignore")
from typing import List

from continuous_score_guidance import ContinuousSafetyScorer
from grasp_rgbd import (
    gripper_width,
    rgbd_to_world_point_cloud,
    select_grasped_object_points,
)
from primitive_fitting import (
    PrimitiveFit,
    fit_best_primitive,
    fit_primitive_candidates,
    plot_primitive_fit,
    primitive_bounding_box_half_extents,
)
from run_manifest import write_run_manifest
from obstacle_selection import select_obstacle_name, simulator_obstacle_prompt

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 1024  # resolution used to render training data
OBSTACLE_POS = np.array([-0.15, 0.03, 1.17])
OBSTACLE_RADIUS = 0.06
ALPHA = 1.0                 # CBF gain
MAX_VEL = 1.0               # Maximum end-effector velocity
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _geom_world_aabb_half_extents(model, data, geom_id: int) -> np.ndarray:
    """Return world-axis AABB half extents for a MuJoCo collision geom."""
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, size[0], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return size[0] + size[1] * np.abs(rotation[:, 2])
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.sqrt(np.square(rotation) @ np.square(size[:3]))
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        return size[1] * np.abs(axis) + size[0] * np.sqrt(
            np.maximum(1.0 - np.square(axis), 0.0)
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.abs(rotation) @ size[:3]
    return np.zeros(3, dtype=np.float64)


def _contact_diagnostic_geom_ids(model, obstacle_names: list[str]):
    """Resolve physical gripper and active-obstacle collision geoms."""
    gripper, obstacle = set(), set()
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 0:
            continue
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("gripper0_"):
            gripper.add(geom_id)
        if any(name in body_name for name in obstacle_names):
            obstacle.add(geom_id)
    return gripper, obstacle


def _primitive_world_z_bounds(
    kind, center, rotation, size, top_padding: float = 0.0
) -> tuple[float, float]:
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    if kind in {"obb", "aabb"}:
        half_z = float(np.abs(rotation[2]) @ size[:3])
    elif kind == "cylinder":
        axis_z = abs(float(rotation[2, 2]))
        half_z = float(size[1] * axis_z + size[0] * math.sqrt(max(1.0 - axis_z**2, 0.0)))
    elif kind == "capsule":
        half_z = float(size[0] + size[1] * abs(float(rotation[2, 2])))
    elif kind == "sphere":
        half_z = float(size[0])
    else:
        return float("nan"), float("nan")
    return float(center[2] - half_z), float(center[2] + half_z + top_padding)


def _load_qwen_vlm_classifier(args):
    """Load Qwen3-VL and return a top-candidate crop classifier.

    The callable accepts a list of crop paths and returns the zero-based
    candidate index.  Keeping this behind the compound-object flag avoids
    loading an additional multimodal model for ordinary obstacle detection.
    """
    model_id = args.action_expert_qwen_vlm_model or os.environ.get(
        "QWEN3_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"
    )
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, local_files_only=args.action_expert_qwen_vlm_local_files_only
    )
    qwen_device = args.action_expert_qwen_vlm_device or os.environ.get(
        "QWEN3_VL_DEVICE", "cpu"
    )
    load_kwargs = {
        "local_files_only": args.action_expert_qwen_vlm_local_files_only,
        "torch_dtype": torch.bfloat16,
    }
    if qwen_device == "cpu":
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).to("cpu")
    else:
        load_kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model = model.eval()
    logging.info("Loaded Qwen VLM classifier: %s on %s", model_id, qwen_device)

    def classify(candidate_paths, target):
        import re
        from PIL import Image, ImageDraw

        task = getattr(args, "task_description", None) or f"put the {target} in the basket"
        tile = 256
        cols = 3
        rows = int(math.ceil(len(candidate_paths) / cols))
        contact = Image.new("RGB", (cols * tile, rows * tile), "white")
        draw = ImageDraw.Draw(contact)
        for idx, path in enumerate(candidate_paths):
            crop = Image.open(path).convert("RGB")
            crop.thumbnail((tile - 12, tile - 36))
            x, y = (idx % cols) * tile, (idx // cols) * tile
            contact.paste(crop, (x + (tile - crop.width) // 2, y + 28))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="red", width=3)
            draw.text((x + 8, y + 6), f"Region {idx + 1}", fill="black")
        contact_path = pathlib.Path(candidate_paths[0]).with_name("candidate_contact_sheet.png")
        contact.save(contact_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(contact_path)},
            {"type": "text", "text": (
                f'Task:\n"{task}"\n\nTarget:\n"{target}"\n\n'
                f"Here are {len(candidate_paths)} candidate regions.\n"
                "Which region is the object the robot should manipulate? "
                "Answer with only the region number."
            )},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[contact], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        match = re.search(r"\b([1-5])\b", answer)
        selected = int(match.group(1)) - 1 if match else 0
        logging.info("Qwen candidate contact-sheet answer: %r -> region %d", answer, selected + 1)
        return max(0, min(selected, len(candidate_paths) - 1))

    return classify


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "safelibero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    safety_level: str = "II" # Task level. Options: I, II
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0]) # Options: [0, 1, 2, 3]
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0]) # Options: [0, 1, 2, 3, 4, ..., 49]
    num_steps_wait: int = 20  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    resume_existing_episodes: bool = False  # Skip complete episode artifacts instead of overwriting them
    save_videos: bool = True
    save_rollout_data: bool = True
    save_perception_diagnostics: bool = True
    fail_on_episode_error: bool = False  # Abort instead of saving a transport/runtime error as a failed rollout
    disable_safety_layer: bool = False  # Run the nominal pi0.5 policy without AEGIS intervention
    use_score_guidance: bool = False  # Evaluate multiple pi0.5 samples and keep the safest one
    score_run_dir: str = str(ROOT / "Safety-value-function/chunk_safety_value_run")
    score_guidance_candidates: int = 4
    score_guidance_device: str = "auto"
    use_flow_guidance: bool = False
    flow_guidance_scale: float = 0.25
    flow_guidance_start_time: float = 0.5
    flow_guidance_run_name: str = "pi05_flow_guided"
    baseline_run_name: str = "pi05_no_safety"
    flow_guidance_translation_only: bool = False
    flow_guidance_orthogonal: bool = False
    use_time_conditioned_guidance: bool = False
    time_conditioned_value_run_dir: str = str(
        ROOT / "Safety-value-function/time_conditioned_clearance_v1"
    )
    time_conditioned_guidance_scale: float = 0.05
    time_conditioned_guidance_time: float = 0.3
    time_conditioned_guidance_times: str = ""
    time_conditioned_clearance_score_weight: float = 0.5
    time_conditioned_guidance_geometry: str = "direct"
    time_conditioned_guidance_normalization: str = "global-rms"
    time_conditioned_guidance_integration: str = "state"
    time_conditioned_guidance_translation_only: bool = True
    time_conditioned_value_backtracking: bool = False
    time_conditioned_margin_scaled: bool = False  # Scale guidance authority by how far the score sits below the gate
    time_conditioned_safety_threshold: float = 0.5
    time_conditioned_value_device: str = "unspecified"
    time_conditioned_run_name: str = "pi05_time_value_guided"
    use_action_expert_guidance: bool = False
    action_expert_critic_run_dir: str = str(ROOT / "Safety-value-function/success_critic_v1")
    # Apply success-critic QPs only in the later denoising stages. The t=0.5
    # intervention was too early and produced large task-degrading corrections.
    action_expert_times: str = "0.3,0.1"
    action_expert_lambda_deviation: float = 1.0
    action_expert_beta_success: float = 10.0
    action_expert_trust_radius: float = 0.05
    action_expert_trust_region_norm: str = "linf"
    action_expert_final_safety_trust_radius: float = 1.0
    action_expert_final_success_trust_radius: float = 0.05
    action_expert_adaptive_safety_trust_radii: str = ""
    action_expert_adaptive_escalate_on_first_barrier_only: bool = False
    action_expert_minimal_intervention: bool = False
    action_expert_translation_only_execution: bool = False
    action_expert_candidates: int = 1
    action_expert_first_step_recovery: bool = False
    action_expert_gamma: float = 0.9
    action_expert_safe_distance: float = 0.01
    action_expert_action_dt: float = 0.05
    # Half-extent of the gripper safety ellipsoid along its local Z axis.
    # Keep this task-independent: the carried-object compound geometry is
    # responsible for extending coverage below the gripper after a grasp.
    action_expert_gripper_z_radius_m: float = 0.11
    action_expert_translation_response_gain: float = 0.22
    action_expert_online_response_gain: bool = False
    action_expert_rotation_response_gain: float = 0.22
    action_expert_safety_resample_attempts: int = 16
    action_expert_continue_on_unsafe: bool = False
    action_expert_pre_execution_qp: bool = True
    action_expert_closed_loop_reprojection: bool = True
    # Number of consecutive actions certified by each receding safety QP.
    # This is independent of replan_steps: pi0.5 still returns H=10, while
    # execution can re-solve a shorter fixed QP window before every action.
    action_expert_qp_horizon: int = 10
    action_expert_debug_rollout_geometry: bool = False
    # Diagnostic only: inspect MuJoCo contacts between physical gripper geoms
    # and the active obstacle. This never changes paper-protocol collision
    # labels, which remain displacement-only.
    diagnose_gripper_obstacle_contacts: bool = False
    diagnostic_top_contact_band_m: float = 0.02
    action_expert_force_axis_aligned_obb: bool = False
    # Shape-only ablation: bypass primitive selection and fit the same MVEE
    # ellipsoid used by the original VLSA implementation.
    action_expert_vlsa_mvee_obstacle: bool = False
    action_expert_obstacle_primitive_kinds: str = "obb,cylinder,capsule"
    # Fit all requested shapes, probe their QPs on one shared pi0.5 sample,
    # then freeze the selected shape for the rest of the episode.
    action_expert_qp_shape_selection: bool = False
    action_expert_shape_selection_lambda_intervention: float = 1.0
    # Optional diagnostic override for the text prompt used by the obstacle
    # detector. When unset, the prompt is resolved from the MuJoCo instance.
    obstacle_prompt_override: str | None = None
    obstacle_selector: str = "simulator"
    obstacle_api_model: str = "glm-4.5v"
    obstacle_api_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    obstacle_api_timeout_s: float = 120.0
    action_expert_obstacle_padding_m: float = 0.005
    # Optional asymmetric padding of only the obstacle's upper world-z face.
    # When unset, the symmetric padding above is used on that face as well.
    action_expert_obstacle_top_padding_m: float | None = None
    # Optional asymmetric padding of only an axis-aligned obstacle box's lower
    # world-z face.  This is baked into the box center and half-extents so all
    # downstream QP and visualization code sees the same geometry.
    action_expert_obstacle_bottom_padding_m: float | None = None
    action_expert_compound_carried_object: bool = False
    # Restrict carried-object perception to the agent camera. Obstacle
    # perception always fuses agentview and backview point clouds.
    action_expert_agentview_only: bool = False
    action_expert_carried_object_prompt: str = "bbq sauce"
    action_expert_carried_object_candidate_top_k: int = 5
    action_expert_qwen_vlm_model: str = ""
    action_expert_qwen_vlm_local_files_only: bool = True
    action_expert_qwen_vlm_device: str = "cpu"
    action_expert_grasp_close_threshold: float = 0.0
    action_expert_grasp_activation_distance: float = 0.18
    # The compound body is estimated lazily from the current agent-view RGB-D
    # frame once the observed jaw width begins decreasing.
    action_expert_grasp_width_delta_threshold: float = 0.00025
    action_expert_grasp_estimation_retry_steps: int = 3
    action_expert_grasp_estimation_window_steps: int = 15
    action_expert_carried_object_crop_radius_m: float = 0.18
    action_expert_carried_object_min_points: int = 30
    action_expert_carried_object_fit_padding_m: float = 0.005
    action_expert_run_name: str = "pi05_action_expert"
    policy_checkpoint_dir: str = ""
    groundingdino_config_path: str = str(
        ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    groundingdino_checkpoint_path: str = str(
        ROOT / "GroundingDINO/groundingdino_swint_ogc.pth"
    )
    groundingdino_hf_model: str = "IDEA-Research/grounding-dino-tiny"
    groundingdino_local_files_only: bool = True
    use_fixed_flow_noise: bool = False
    flow_noise_action_horizon: int = 10
    flow_noise_action_dim: int = 32

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if not args.save_videos and not args.save_rollout_data and not args.use_action_expert_guidance:
        raise ValueError(
            "At least one of --save-videos or --save-rollout-data must be enabled "
            "unless action-expert safety summaries are being saved."
        )
    # Set random seed
    np.random.seed(args.seed)
    if not 0.0 <= args.time_conditioned_safety_threshold <= 1.0:
        raise ValueError("time_conditioned_safety_threshold must be in [0, 1]")
    if args.action_expert_candidates < 1:
        raise ValueError("action_expert_candidates must be positive")
    if args.obstacle_selector not in {"simulator", "vlsa-api"}:
        raise ValueError("obstacle_selector must be 'simulator' or 'vlsa-api'")
    if args.obstacle_selector == "vlsa-api" and args.obstacle_prompt_override:
        raise ValueError(
            "obstacle_prompt_override cannot be used with the VLSA API selector"
        )
    if args.obstacle_api_timeout_s <= 0.0:
        raise ValueError("obstacle_api_timeout_s must be positive")
    if args.action_expert_shape_selection_lambda_intervention < 0.0:
        raise ValueError(
            "action_expert_shape_selection_lambda_intervention must be non-negative"
        )
    if not 1 <= args.replan_steps <= args.flow_noise_action_horizon:
        raise ValueError(
            "replan_steps must be between 1 and the pi0.5 action horizon "
            f"({args.flow_noise_action_horizon})"
        )
    if not 1 <= args.action_expert_qp_horizon <= args.flow_noise_action_horizon:
        raise ValueError(
            "action_expert_qp_horizon must be between 1 and the pi0.5 action "
            f"horizon ({args.flow_noise_action_horizon})"
        )
    if (
        args.action_expert_obstacle_bottom_padding_m is not None
        and args.action_expert_obstacle_bottom_padding_m < 0.0
    ):
        raise ValueError("action_expert_obstacle_bottom_padding_m must be non-negative")
    minimum_remaining_horizon = (
        args.flow_noise_action_horizon - args.replan_steps + 1
    )
    if (
        args.action_expert_closed_loop_reprojection
        and args.action_expert_qp_horizon > minimum_remaining_horizon
    ):
        raise ValueError(
            "A fixed closed-loop QP window would overrun the remaining pi0.5 "
            "chunk: require qp_horizon <= action_horizon - replan_steps + 1, "
            f"got {args.action_expert_qp_horizon} > {minimum_remaining_horizon}"
        )
    obstacle_primitive_kinds = tuple(
        kind.strip()
        for kind in args.action_expert_obstacle_primitive_kinds.split(",")
        if kind.strip()
    )
    supported_obstacle_kinds = {
        "obb",
        "aabb",
        "ellipsoid",
        "cylinder",
        "capsule",
        "sphere",
    }
    if (
        not obstacle_primitive_kinds
        or len(set(obstacle_primitive_kinds)) != len(obstacle_primitive_kinds)
        or not set(obstacle_primitive_kinds) <= supported_obstacle_kinds
    ):
        raise ValueError(
            "action_expert_obstacle_primitive_kinds must be a unique comma-separated "
            f"subset of {sorted(supported_obstacle_kinds)}"
        )
    safety_level = args.safety_level
    task_index = args.task_index
    episode_index = args.episode_index
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=safety_level)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(
        "Task suite: %s, safety level: %s, safety layer enabled: %s",
        args.task_suite_name,
        safety_level,
        not args.disable_safety_layer,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "safelibero_spatial":
        max_steps = 300  
    elif args.task_suite_name == "safelibero_object":
        max_steps = 300  
    elif args.task_suite_name == "safelibero_goal":
        max_steps = 300  
    elif args.task_suite_name == "safelibero_long":
        max_steps = 550  
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    print("OK")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    scorer = None
    guidance_modes = sum(
        (
            bool(args.use_score_guidance),
            bool(args.use_flow_guidance),
            bool(args.use_time_conditioned_guidance),
            bool(args.use_action_expert_guidance),
        )
    )
    if guidance_modes > 1:
        raise ValueError("Candidate, residual-flow, time-conditioned, and action-expert guidance are mutually exclusive.")
    if args.use_action_expert_guidance and not args.disable_safety_layer:
        raise ValueError("Action-expert primitive guidance requires --disable-safety-layer.")
    if args.use_score_guidance:
        if not args.disable_safety_layer:
            raise ValueError("Score guidance is intended for pi0.5-only runs. Set --disable-safety-layer.")
        scorer = ContinuousSafetyScorer.load(pathlib.Path(args.score_run_dir), device=args.score_guidance_device)
        logging.info(
            "Loaded continuous score guidance from %s with %d candidates",
            args.score_run_dir,
            args.score_guidance_candidates,
        )
    model_groundingdino = None
    qwen_vlm_classifier = None
    # Compound geometry is now estimated from a grasp-conditioned RGB-D crop,
    # so it no longer loads Qwen or performs semantic target selection at the
    # beginning of every episode.
    if not args.disable_safety_layer or args.use_action_expert_guidance:
        config_path = pathlib.Path(args.groundingdino_config_path)
        checkpoint_path = pathlib.Path(args.groundingdino_checkpoint_path)
        if config_path.is_file() and checkpoint_path.is_file() and checkpoint_path.stat().st_size > 10_000_000:
            from groundingdino.util.inference import load_model

            model_groundingdino = load_model(str(config_path), str(checkpoint_path))
        else:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            device = os.environ.get("GROUNDINGDINO_DEVICE", "cuda")
            model_groundingdino = {
                "backend": "transformers",
                "device": device,
                "processor": AutoProcessor.from_pretrained(
                    args.groundingdino_hf_model,
                    local_files_only=args.groundingdino_local_files_only,
                ),
                "model": AutoModelForZeroShotObjectDetection.from_pretrained(
                    args.groundingdino_hf_model,
                    local_files_only=args.groundingdino_local_files_only,
                ).to(torch.device(device)).eval(),
            }
            logging.info("Using cached Transformers GroundingDINO model %s", args.groundingdino_hf_model)
    # Start evaluation
    total_episodes, total_successes, total_safesuccesses, total_collisions = 0, 0, 0, 0
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    for task_id in task_index:
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, safety_level, LIBERO_ENV_RESOLUTION, args.seed)
        # Expose the task text to the crop-level VLM prompt.  This keeps the
        # classifier grounded in the actual instruction (e.g. the BBQ sample:
        # put the bbq sauce in the basket).
        args.task_description = task_description
        model = env.sim.model
        data = env.sim.data

        collides = 0
        time_steps = []

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_segment = task_description.replace(" ", "_")



        _out_dir = pathlib.Path(args.video_out_path) / f"{task_segment}"
        _out_dir.mkdir(parents=True, exist_ok=True)
        if args.use_action_expert_guidance:
            run_name = args.action_expert_run_name
        elif args.use_time_conditioned_guidance:
            run_name = args.time_conditioned_run_name
        elif args.use_flow_guidance:
            run_name = args.flow_guidance_run_name
        elif args.use_score_guidance:
            run_name = "pi05_score_guided"
        else:
            run_name = args.baseline_run_name
        out_dir = _out_dir / f"{run_name}_{safety_level}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(
            out_dir,
            run_name=run_name,
            task_description=task_description,
            safety_level=safety_level,
            configuration=dataclasses.asdict(args),
            value_run_dir=(
                pathlib.Path(args.action_expert_critic_run_dir)
                if args.use_action_expert_guidance
                else pathlib.Path(args.time_conditioned_value_run_dir)
                if args.use_time_conditioned_guidance
                else pathlib.Path(args.score_run_dir)
                if args.use_flow_guidance
                else None
            ),
            checkpoint_dir=(pathlib.Path(args.policy_checkpoint_dir) if args.policy_checkpoint_dir else None),
        )



        # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        for episode_idx in episode_index:
            existing_episode_artifacts = sorted(out_dir.glob(f"{episode_idx}_*"))
            annotation_dir = out_dir / str(episode_idx)
            if annotation_dir.exists():
                existing_episode_artifacts.append(annotation_dir)
            if existing_episode_artifacts:
                if args.resume_existing_episodes:
                    base_videos = [
                        path
                        for path in existing_episode_artifacts
                        if path.suffix == ".mp4"
                        and not path.stem.endswith(("_agentview", "_backview"))
                    ]
                    has_video = bool(base_videos) and all(
                        (base_videos[0].parent / f"{base_videos[0].stem}_{view}.mp4").is_file()
                        for view in ("agentview", "backview")
                    )
                    has_rollout = any(
                        path.name.endswith("_last_layer_hidden_states.npz")
                        for path in existing_episode_artifacts
                    )
                    has_annotations = all(
                        (annotation_dir / f"annotated_{view}.png").is_file()
                        for view in ("agentview", "backview")
                    )
                    has_gripper_overlays = all(
                        (annotation_dir / f"gripper_ellipsoid_{view}.png").is_file()
                        for view in ("agentview", "backview")
                    )
                    has_primitive_fit = (annotation_dir / "obstacle_primitive.json").is_file()
                    has_primitive_plot = (annotation_dir / "obstacle_primitive_3d.png").is_file()
                    has_safety_summary = (
                        annotation_dir / "action_expert_safety_summary.json"
                    ).is_file()
                    complete = (
                        (has_video or not args.save_videos)
                        and (has_rollout or not args.save_rollout_data)
                        and (
                            has_annotations
                            or not args.use_action_expert_guidance
                            or not args.save_perception_diagnostics
                        )
                        and (
                            has_gripper_overlays
                            or not args.use_action_expert_guidance
                            or not args.save_perception_diagnostics
                        )
                        and (has_primitive_fit or not args.use_action_expert_guidance)
                        and (
                            has_primitive_plot
                            or not args.use_action_expert_guidance
                            or not args.save_perception_diagnostics
                        )
                        and (has_safety_summary or not args.use_action_expert_guidance)
                    )
                    # A prior run may have completed the rollout and safety
                    # summary but exhausted disk while writing optional
                    # diagnostic overlays. Treat that episode as resumable-
                    # complete so benchmark continuation is not blocked.
                    if has_safety_summary and has_video:
                        complete = True
                    if not complete:
                        raise FileExistsError(
                            f"refusing to skip incomplete episode {episode_idx} in {out_dir}: "
                            f"found {[path.name for path in existing_episode_artifacts]}"
                        )
                    logging.info(
                        "Skipping existing episode %d in %s (%d artifacts)",
                        episode_idx,
                        out_dir,
                        len(existing_episode_artifacts),
                    )
                    continue
                raise FileExistsError(
                    f"refusing to overwrite episode {episode_idx} in {out_dir}: "
                    f"{existing_episode_artifacts[0].name}"
                )
            # A fail-closed policy response closes its WebSocket with status
            # 1011. Start every episode with a fresh connection so one safely
            # rejected chunk cannot turn later episodes into one-frame failures.
            client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            action_expert_chunk_executed = 0
            action_expert_feedback_projection_records = []


            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_backview_images = []
            chunk_start_steps = []
            last_layer_hidden_states = []
            predicted_action_chunks = []
            chunk_infer_ms = []
            guided_candidate_scores = []
            selected_candidate_indices = []
            flow_value_scores = []
            flow_residual_ratios = []
            time_conditioned_scores = []
            time_conditioned_clearance_predictions = []
            time_conditioned_scores_after = []
            time_conditioned_clearance_predictions_after = []
            time_conditioned_gradient_rms = []
            time_conditioned_perturbation_rms = []
            time_conditioned_correction_to_step = []
            time_conditioned_accepted_factors = []
            time_conditioned_injection_counts = []
            time_conditioned_risk_gate_active = []
            time_conditioned_safety_thresholds = []
            action_expert_scores_before = []
            action_expert_scores_after = []
            action_expert_qp_success = []
            action_expert_qp_margins = []
            action_expert_correction_rms = []
            action_expert_barriers_before = []
            action_expert_barriers_nominal = []
            action_expert_intervened = []
            action_expert_min_trajectory_barriers = []
            action_expert_final_trajectory_barriers = []
            action_expert_final_projection_success = []
            action_expert_certified_horizons = []
            action_expert_final_projection_iterations = []
            action_expert_final_projection_correction_rms = []
            action_expert_final_safety_correction_rms = []
            action_expert_final_critic_correction_rms = []
            action_expert_final_critic_refinement_accepted = []
            action_expert_selected_safety_trust_radius = []
            action_expert_attempted_safety_trust_radii = []
            action_expert_safety_resamples = []
            action_expert_critic_records = []
            action_expert_candidate_records = []
            action_expert_rollout_geometry_debug = []
            current_translation_response_gain = float(
                args.action_expert_translation_response_gain
            )
            response_gain_command_window = []
            response_gain_achieved_window = []
            response_gain_records = []
            active_action_expert_debug_chunk = None
            obstacle_primitive_candidates = {}
            obstacle_candidate_frames = {}
            obstacle_shape_selection_record = None
            model = env.sim.model
            data = env.sim.data
            eef_body_id = model.body_name2id("eef_marker")

            # Initial position and orientation of the end-effector ellipsoid
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            r = R.from_quat(eef_quat)
            euler1 = r.as_euler('xyz', degrees=False)
            R1 = R.from_quat(eef_quat).as_matrix()
            offset_local = np.array([0, 0, -0.08])
            offset_world = R1 @ offset_local
            ball_pos = eef_pos + offset_world
            p1 = ball_pos
            env.sim.model.body_pos[eef_body_id] = ball_pos
            env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]
            Q1_diag = np.array(
                [0.06, 0.12, args.action_expert_gripper_z_radius_m],
                dtype=np.float64,
            )

            while t < args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        # img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    
                        # # Save preprocessed image for replay video
                        # replay_images.append(img)
                        t += 1
                        continue
                except Exception:
                    logging.exception("Episode %d failed at simulator step %d", episode_idx, t)
                    if args.fail_on_episode_error:
                        raise
                    break


            
            # Resolve active obstacle identities directly from the trusted
            # simulator state. This removes the legacy external VLM naming
            # dependency while leaving GroundingDINO geometry estimation and
            # the CBF/QP shield unchanged.
            obstacle_names = [
                name.replace("_joint0", "")
                for name in env.sim.model.joint_names
                if "obstacle" in name
            ]
            active_obstacle_names = []
            for obstacle_name in obstacle_names:
                obstacle_position = obs[f"{obstacle_name}_pos"]
                if (
                    obstacle_position[2] > -0.05
                    and -0.5 < obstacle_position[0] < 0.5
                    and -0.5 < obstacle_position[1] < 0.5
                ):
                    active_obstacle_names.append(obstacle_name)
                    print("Obstacle name:", obstacle_name)
            if not active_obstacle_names:
                obstacle_positions = {
                    name: np.asarray(obs[f"{name}_pos"]).tolist()
                    for name in obstacle_names
                }
                skip_path = out_dir / f"{episode_idx}_skipped_no_active_obstacle.txt"
                skip_path.write_text(
                    "No active obstacle found in the workspace.\n"
                    + "\n".join(
                        f"{name}: {position}"
                        for name, position in obstacle_positions.items()
                    )
                    + "\n"
                )
                logging.warning(
                    "Skipping episode %s: no active obstacle; marker=%s",
                    episode_idx,
                    skip_path,
                )
                continue
            initial_obstacle_positions = {
                name: np.asarray(obs[f"{name}_pos"]).copy()
                for name in active_obstacle_names
            }
            diagnostic_gripper_geom_ids, diagnostic_obstacle_geom_ids = (
                _contact_diagnostic_geom_ids(model, active_obstacle_names)
                if args.diagnose_gripper_obstacle_contacts
                else (set(), set())
            )
            if args.diagnose_gripper_obstacle_contacts and (
                not diagnostic_gripper_geom_ids or not diagnostic_obstacle_geom_ids
            ):
                raise RuntimeError(
                    "Could not resolve gripper/active-obstacle collision geoms for contact diagnostics"
                )

            # Detect obstacle geometry for the shield.
            img_out_dir = out_dir / f"{episode_idx}"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            flag_safety_control = False
            # Keep diagnostics and fallback paths well-defined when the
            # detector returns no usable point cloud for a prompt.
            obstacle_top_padding = 0.0
            carried_object_fit = None
            carried_object_active = False
            carried_object_offset = None
            carried_object_activation_step = None
            carried_object_close_steps = []
            carried_object_estimation_attempts = []
            carried_object_last_estimation_step = -10**9
            carried_object_close_cycle_active = False
            carried_object_estimation_window_end = None
            carried_prompt = _carried_object_prompt_from_task(task_description)
            carried_dir = img_out_dir / "carried_object"
            previous_gripper_width = gripper_width(obs["robot0_gripper_qpos"])
            if not args.disable_safety_layer or args.use_action_expert_guidance:
                agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])

                obstacle_selection = select_obstacle_name(
                    mode=args.obstacle_selector,
                    image=agentview_img,
                    instruction=task_description,
                    task_suite_name=args.task_suite_name,
                    simulator_instance=active_obstacle_names[0],
                    prompt_override=args.obstacle_prompt_override,
                    api_model=args.obstacle_api_model,
                    api_base_url=args.obstacle_api_base_url,
                    api_timeout_s=args.obstacle_api_timeout_s,
                )
                obstacle_infromation = obstacle_selection.name
                obstacle_selection_record = obstacle_selection.record()
                # Ground truth is recorded only for offline accuracy analysis;
                # it is never passed into the API selector or GroundingDINO.
                obstacle_selection_record["evaluation_ground_truth_instance"] = (
                    active_obstacle_names[0]
                )
                obstacle_selection_record["ground_truth_used_for_selection"] = False
                (img_out_dir / "obstacle_name_selection.json").write_text(
                    json.dumps(obstacle_selection_record, indent=2, sort_keys=True)
                    + "\n"
                )
                logging.info(
                    "Obstacle selector source=%s answer=%r latency=%s",
                    obstacle_selection.source,
                    obstacle_infromation,
                    (
                        "n/a"
                        if obstacle_selection.latency_s is None
                        else f"{obstacle_selection.latency_s:.3f}s"
                    ),
                )
                # obstacle_infromation = "white storage box"
                agent_view_points = get_point_cloud(
                    agentview_img,
                    agentview_depth,
                    env,
                    "agentview",
                    obstacle_infromation,
                    model_groundingdino,
                    img_out_dir,
                    save_diagnostics=args.save_perception_diagnostics,
                )

                backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
                backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
                back_view_points = get_point_cloud(
                    backview_img,
                    backview_depth,
                    env,
                    "backview",
                    obstacle_infromation,
                    model_groundingdino,
                    img_out_dir,
                    save_diagnostics=args.save_perception_diagnostics,
                )
                if args.save_perception_diagnostics:
                    overlay_gripper_ellipsoid_on_rgb(
                        agentview_img,
                        env,
                        "agentview",
                        p1,
                        R1,
                        Q1_diag,
                        img_out_dir / "gripper_ellipsoid_agentview.png",
                    )
                    overlay_gripper_ellipsoid_on_rgb(
                        backview_img,
                        env,
                        "backview",
                        p1,
                        R1,
                        Q1_diag,
                        img_out_dir / "gripper_ellipsoid_backview.png",
                    )

                # df = pd.DataFrame(back_view_points, columns=["X", "Y", "Z"])
                # df.to_csv("back_view_points.csv", index=False)

                if agent_view_points.shape[1] > 0 and back_view_points.shape[1] > 0:
                    full_points = np.vstack([agent_view_points, back_view_points])    # (Na + Nb, 3)
                elif agent_view_points.shape[1] == 0 and back_view_points.shape[1] > 0:
                    full_points = back_view_points
                elif agent_view_points.shape[1] > 0 and back_view_points.shape[1] == 0:
                    full_points = agent_view_points
                else:
                    full_points = np.array([[]])

                logging.info(
                    "Obstacle point-cloud prompt=%r agent_shape=%s back_shape=%s fused_shape=%s",
                    obstacle_infromation,
                    tuple(agent_view_points.shape),
                    tuple(back_view_points.shape),
                    tuple(full_points.shape),
                )

                # df = pd.DataFrame(full_points, columns=["X", "Y", "Z"])
                # df.to_csv("full_points.csv", index=False)

                # Point cloud filtering
                filter_points = filtering_points(full_points, args.task_suite_name)
                logging.info(
                    "Obstacle point-cloud filtered_shape=%s",
                    tuple(filter_points.shape),
                )
                # print("Number of points after filtering:", filter_points.shape[0])
                flag_safety_control = filter_points.shape[0] > 0
                # import pandas as pd
                # df = pd.DataFrame(filter_points, columns=["X", "Y", "Z"])
                # df.to_csv("filter_points.csv", index=False)

                if flag_safety_control:
                    if args.use_action_expert_guidance:
                        obstacle_top_padding = max(
                            0.0,
                            (
                                args.action_expert_obstacle_top_padding_m
                                - args.action_expert_obstacle_padding_m
                            )
                            if args.action_expert_obstacle_top_padding_m is not None
                            else 0.0,
                        )
                        if args.action_expert_vlsa_mvee_obstacle:
                            primitive_fit = _fit_vlsa_mvee_ellipsoid(filter_points)
                        elif args.action_expert_qp_shape_selection:
                            obstacle_primitive_candidates = fit_primitive_candidates(
                                filter_points,
                                padding=args.action_expert_obstacle_padding_m,
                                allowed_kinds=obstacle_primitive_kinds,
                            )
                            # This provisional fit is used only to construct the
                            # first request.  The shared-noise QP probe below
                            # replaces it and freezes the winner before action 0.
                            primitive_fit = min(
                                obstacle_primitive_candidates.values(),
                                key=lambda fit: fit.score,
                            )
                        else:
                            primitive_fit = fit_best_primitive(
                                filter_points,
                                padding=args.action_expert_obstacle_padding_m,
                                allowed_kinds=obstacle_primitive_kinds,
                            )
                        if (
                            args.action_expert_force_axis_aligned_obb
                            and not args.action_expert_vlsa_mvee_obstacle
                        ):
                            primitive_fit = _fit_axis_aligned_obb(
                                filter_points,
                                padding=args.action_expert_obstacle_padding_m,
                                top_padding=args.action_expert_obstacle_top_padding_m,
                                bottom_padding=args.action_expert_obstacle_bottom_padding_m,
                                selector_scores=primitive_fit.candidate_scores,
                            )
                            # _fit_axis_aligned_obb bakes the asymmetric margin
                            # directly into the box center and half-extents.
                            obstacle_top_padding = 0.0
                        elif (
                            args.action_expert_obstacle_bottom_padding_m is not None
                            and not args.action_expert_vlsa_mvee_obstacle
                        ):
                            primitive_fit = _extend_axis_aligned_box_bottom(
                                primitive_fit,
                                extra_bottom_padding=max(
                                    0.0,
                                    args.action_expert_obstacle_bottom_padding_m
                                    - args.action_expert_obstacle_padding_m,
                                ),
                            )
                        obstacle_kind = primitive_fit.kind
                        p2 = primitive_fit.center
                        R2 = primitive_fit.rotation
                        obstacle_size = primitive_fit.size
                        tracked_obstacle_name = active_obstacle_names[0]
                        initial_obstacle_position = np.asarray(
                            obs[f"{tracked_obstacle_name}_pos"], dtype=np.float64
                        )
                        initial_obstacle_rotation = R.from_quat(
                            np.asarray(
                                obs[f"{tracked_obstacle_name}_quat"],
                                dtype=np.float64,
                            )
                        ).as_matrix()
                        primitive_center_in_obstacle = (
                            initial_obstacle_rotation.T
                            @ (p2 - initial_obstacle_position)
                        )
                        primitive_rotation_in_obstacle = (
                            initial_obstacle_rotation.T @ R2
                        )
                        if obstacle_primitive_candidates:
                            obstacle_candidate_frames = {
                                kind: {
                                    "fit": fit,
                                    "center_in_obstacle": (
                                        initial_obstacle_rotation.T
                                        @ (fit.center - initial_obstacle_position)
                                    ),
                                    "rotation_in_obstacle": (
                                        initial_obstacle_rotation.T @ fit.rotation
                                    ),
                                }
                                for kind, fit in obstacle_primitive_candidates.items()
                            }
                        if args.save_perception_diagnostics:
                            plot_primitive_fit(
                                filter_points,
                                primitive_fit,
                                img_out_dir / "obstacle_primitive_3d.png",
                            )
                        (img_out_dir / "obstacle_primitive.json").write_text(
                            json.dumps(
                                {
                                    "kind": obstacle_kind,
                                    "center": p2.tolist(),
                                    "rotation": R2.tolist(),
                                    "size": obstacle_size.tolist(),
                                    "score": primitive_fit.score,
                                    "surface_rmse": primitive_fit.surface_rmse,
                                    "candidate_scores": primitive_fit.candidate_scores,
                                    "qp_shape_selection": bool(
                                        args.action_expert_qp_shape_selection
                                    ),
                                    "qp_selected": False,
                                    "candidate_geometries": {
                                        kind: {
                                            "center": state["fit"].center.tolist(),
                                            "rotation": state["fit"].rotation.tolist(),
                                            "size": state["fit"].size.tolist(),
                                            "surface_score": state["fit"].score,
                                            "surface_rmse": state["fit"].surface_rmse,
                                        }
                                        for kind, state in obstacle_candidate_frames.items()
                                    },
                                    "allowed_kinds": list(obstacle_primitive_kinds),
                                    "forced_axis_aligned": bool(
                                        args.action_expert_force_axis_aligned_obb
                                    ),
                                    "vlsa_mvee_ellipsoid": bool(
                                        args.action_expert_vlsa_mvee_obstacle
                                    ),
                                    "padding_m": args.action_expert_obstacle_padding_m,
                                    "top_padding_m": (
                                        args.action_expert_obstacle_top_padding_m
                                        if args.action_expert_obstacle_top_padding_m
                                        is not None
                                        else args.action_expert_obstacle_padding_m
                                    ),
                                    "top_padding_support_m": obstacle_top_padding,
                                    "bottom_padding_m": (
                                        args.action_expert_obstacle_bottom_padding_m
                                        if args.action_expert_obstacle_bottom_padding_m
                                        is not None
                                        else args.action_expert_obstacle_padding_m
                                    ),
                                    "bottom_padding_baked_m": max(
                                        0.0,
                                        (
                                            args.action_expert_obstacle_bottom_padding_m
                                            - args.action_expert_obstacle_padding_m
                                        )
                                        if args.action_expert_obstacle_bottom_padding_m
                                        is not None
                                        else 0.0,
                                    ),
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        logging.info(
                            "Selected obstacle primitive: %s; surface RMSE: %.5f m; scores: %s",
                            obstacle_kind,
                            primitive_fit.surface_rmse,
                            primitive_fit.candidate_scores,
                        )
                    else:
                        obstacle_kind = "ellipsoid"
                        p2, R2, obstacle_size = fit_ellipse(
                            filter_points, plot=True, save_path=img_out_dir
                        )
                        Q2_diag = obstacle_size
                    # Control parameter settings
                    z_fixed = (p2 - p1)
                    z_fixed /= np.linalg.norm(z_fixed)
                    p_target = np.array([-0.05, 0.15, 1.05])
                    Kp_pos = 1
                    dt = 0.05

                if (
                    args.use_action_expert_guidance
                    and args.action_expert_compound_carried_object
                ):
                    # Defer carried-object perception until the observed jaws
                    # begin closing around an object.  Obstacle perception
                    # above remains the fused agentview + backview path.
                    carried_dir.mkdir(parents=True, exist_ok=True)
            t = 0
            # print("Joint names (qpos):", env.sim.model.joint_names)

            # Collision follows the SafeLIBERO paper protocol: an episode is
            # unsafe only when an active obstacle is displaced by more than
            # 1 mm (L1 position delta) from its stabilized initial pose.  Do
            # not query MuJoCo contact pairs: transient robot/obstacle contact
            # without obstacle displacement is not a collision for this eval.
            collide_flag = False
            collision_action_steps = []
            per_action_collision_flags = []
            per_action_obstacle_motion_flags = []
            per_action_obstacle_displacement_flags = []
            per_action_unsafe_flags = []
            per_action_obstacle_surface_gaps = []
            per_action_obstacle_barriers = []
            per_action_ellipsoid_surface_gaps = []
            per_action_carried_object_surface_gaps = []
            per_action_carried_object_active = []
            per_action_carried_object_centers = []
            per_action_gripper_obstacle_contact = []
            per_action_gripper_unmodeled_upper_contact = []
            per_action_gripper_actual_top_band_contact = []
            per_action_gripper_contact_with_displacement = []
            gripper_obstacle_contact_records = []
            executed_action_steps = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                      # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)
                        replay_backview_images.append(
                            np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
                        )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    

                    
                    t1 = time.time()
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "__debug_return_last_hidden_state__": True,
                        }
                        if args.use_flow_guidance:
                            element["__safety_value_flow_guidance__"] = {
                                "scale": args.flow_guidance_scale,
                                "start_time": args.flow_guidance_start_time,
                                "translation_only": args.flow_guidance_translation_only,
                                "orthogonal": args.flow_guidance_orthogonal,
                            }
                        if args.use_time_conditioned_guidance:
                            guidance_times = (
                                [
                                    float(value.strip())
                                    for value in args.time_conditioned_guidance_times.split(",")
                                    if value.strip()
                                ]
                                if args.time_conditioned_guidance_times
                                else [args.time_conditioned_guidance_time]
                            )
                            element["__time_conditioned_guidance__"] = {
                                "scale": args.time_conditioned_guidance_scale,
                                "times": guidance_times,
                                "direction_sign": 1,
                                "clearance_score_weight": (
                                    args.time_conditioned_clearance_score_weight
                                ),
                                "translation_only": (
                                    args.time_conditioned_guidance_translation_only
                                ),
                                "geometry": args.time_conditioned_guidance_geometry,
                                "normalization": (
                                    args.time_conditioned_guidance_normalization
                                ),
                                "integration": args.time_conditioned_guidance_integration,
                                "value_backtracking": (
                                    args.time_conditioned_value_backtracking
                                ),
                                "margin_scaled": bool(
                                    args.time_conditioned_margin_scaled
                                ),
                                "safety_threshold": (
                                    args.time_conditioned_safety_threshold
                                ),
                                "denoising_steps": 10,
                            }
                        if args.use_action_expert_guidance and flag_safety_control:
                            controller = env.env.robots[0].controller
                            element["__action_expert_guidance__"] = {
                                "times": [
                                    float(value.strip())
                                    for value in args.action_expert_times.split(",")
                                    if value.strip()
                                ],
                                "lambda_deviation": args.action_expert_lambda_deviation,
                                "beta_success": args.action_expert_beta_success,
                                "trust_radius": args.action_expert_trust_radius,
                                "trust_region_norm": args.action_expert_trust_region_norm,
                                "final_safety_trust_radius": (
                                    args.action_expert_final_safety_trust_radius
                                ),
                                "final_success_trust_radius": (
                                    args.action_expert_final_success_trust_radius
                                ),
                                "adaptive_safety_trust_radii": [
                                    float(value)
                                    for value in args.action_expert_adaptive_safety_trust_radii.split(",")
                                    if value.strip()
                                ],
                                "adaptive_escalate_on_first_barrier_only": (
                                    args.action_expert_adaptive_escalate_on_first_barrier_only
                                ),
                                "minimal_intervention": args.action_expert_minimal_intervention,
                                "translation_only_execution": (
                                    args.action_expert_translation_only_execution
                                ),
                                "first_step_recovery": args.action_expert_first_step_recovery,
                                "gamma": args.action_expert_gamma,
                                "safe_distance": args.action_expert_safe_distance,
                                "action_dt": args.action_expert_action_dt,
                                "controller_input_min": np.asarray(
                                    controller.input_min, dtype=np.float32
                                ),
                                "controller_input_max": np.asarray(
                                    controller.input_max, dtype=np.float32
                                ),
                                "controller_output_min": np.asarray(
                                    controller.output_min, dtype=np.float32
                                ),
                                "controller_output_max": np.asarray(
                                    controller.output_max, dtype=np.float32
                                ),
                                "translation_response_gain": (
                                    current_translation_response_gain
                                ),
                                "rotation_response_gain": (
                                    args.action_expert_rotation_response_gain
                                ),
                                "safety_resample_attempts": (
                                    args.action_expert_safety_resample_attempts
                                ),
                                "continue_on_unsafe": (
                                    args.action_expert_continue_on_unsafe
                                ),
                                "pre_execution_qp": (
                                    args.action_expert_pre_execution_qp
                                ),
                                "qp_horizon": args.action_expert_qp_horizon,
                                "denoising_steps": 10,
                                "geometry": {
                                    "eef_position": np.asarray(eef_pos, dtype=np.float32),
                                    "eef_rotation": np.asarray(R1, dtype=np.float32),
                                    "eef_radii": np.asarray(Q1_diag, dtype=np.float32),
                                    "ellipsoid_offset": np.asarray(
                                        offset_local, dtype=np.float32
                                    ),
                                    "obstacle_center": np.asarray(p2, dtype=np.float32),
                                    "obstacle_rotation": np.asarray(R2, dtype=np.float32),
                                    "obstacle_kind": obstacle_kind,
                                    "obstacle_size": np.asarray(obstacle_size, dtype=np.float32),
                                    "obstacle_top_padding": float(obstacle_top_padding),
                                    "carried_object_active": bool(
                                        carried_object_active
                                    ),
                                    "carried_object_offset": (
                                        np.asarray(
                                            carried_object_offset,
                                            dtype=np.float32,
                                        )
                                        if carried_object_active
                                        else np.zeros(3, dtype=np.float32)
                                    ),
                                    "carried_object_rotation": np.eye(
                                        3, dtype=np.float32
                                    ),
                                    "carried_object_size": (
                                        np.asarray(
                                            carried_object_fit.size,
                                            dtype=np.float32,
                                        )
                                        if carried_object_fit is not None
                                        else np.zeros(3, dtype=np.float32)
                                    ),
                                },
                            }

                        fixed_noise = None
                        if args.use_fixed_flow_noise:
                            fixed_noise = _make_flow_noise(
                                args,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                step=t,
                                candidate_idx=0,
                            )

                        candidate_results = []
                        if (
                            args.action_expert_qp_shape_selection
                            and obstacle_shape_selection_record is None
                            and obstacle_candidate_frames
                        ):
                            current_obstacle_position = np.asarray(
                                obs[f"{tracked_obstacle_name}_pos"], dtype=np.float64
                            )
                            current_obstacle_rotation = R.from_quat(
                                np.asarray(
                                    obs[f"{tracked_obstacle_name}_quat"],
                                    dtype=np.float64,
                                )
                            ).as_matrix()
                            ranked_shapes = []
                            shape_probe_records = []
                            for candidate_kind in obstacle_primitive_kinds:
                                candidate_state = obstacle_candidate_frames[candidate_kind]
                                candidate_fit = candidate_state["fit"]
                                candidate_center = (
                                    current_obstacle_position
                                    + current_obstacle_rotation
                                    @ candidate_state["center_in_obstacle"]
                                )
                                candidate_rotation = (
                                    current_obstacle_rotation
                                    @ candidate_state["rotation_in_obstacle"]
                                )
                                candidate_element = dict(element)
                                candidate_config = dict(
                                    element["__action_expert_guidance__"]
                                )
                                candidate_geometry = dict(candidate_config["geometry"])
                                candidate_geometry.update(
                                    {
                                        "obstacle_center": np.asarray(
                                            candidate_center, dtype=np.float32
                                        ),
                                        "obstacle_rotation": np.asarray(
                                            candidate_rotation, dtype=np.float32
                                        ),
                                        "obstacle_kind": candidate_kind,
                                        "obstacle_size": np.asarray(
                                            candidate_fit.size, dtype=np.float32
                                        ),
                                    }
                                )
                                candidate_config["geometry"] = candidate_geometry
                                candidate_element["__action_expert_guidance__"] = (
                                    candidate_config
                                )
                                candidate_response = client.infer(
                                    candidate_element, noise=fixed_noise
                                )
                                candidate_barriers = np.asarray(
                                    candidate_response[
                                        "action_expert_final_trajectory_barriers"
                                    ],
                                    dtype=np.float64,
                                )
                                certified_horizon = int(
                                    candidate_response.get(
                                        "action_expert_certified_horizon",
                                        len(candidate_barriers),
                                    )
                                )
                                certified = bool(
                                    candidate_response[
                                        "action_expert_final_projection_success"
                                    ]
                                    and certified_horizon == args.action_expert_qp_horizon
                                    and np.all(
                                        candidate_barriers[:certified_horizon] >= -1e-7
                                    )
                                )
                                success_probability = float(
                                    candidate_response.get(
                                        "action_expert_final_success_score_after", 0.0
                                    )
                                )
                                safety_correction = float(
                                    candidate_response.get(
                                        "action_expert_final_safety_correction_rms",
                                        0.0,
                                    )
                                )
                                critic_correction = float(
                                    candidate_response.get(
                                        "action_expert_final_critic_correction_rms",
                                        0.0,
                                    )
                                )
                                correction_rms = math.sqrt(
                                    safety_correction**2 + critic_correction**2
                                )
                                utility = math.log(
                                    max(success_probability, 1e-8)
                                ) - (
                                    args.action_expert_shape_selection_lambda_intervention
                                    * correction_rms**2
                                )
                                minimum_barrier = float(np.min(candidate_barriers))
                                # Feasibility is lexicographically dominant.
                                # Surface score is only a deterministic final
                                # tie-breaker, never the control objective.
                                rank = (
                                    (
                                        1.0,
                                        utility,
                                        minimum_barrier,
                                        -float(candidate_fit.score),
                                    )
                                    if certified
                                    else (
                                        0.0,
                                        minimum_barrier,
                                        utility,
                                        -float(candidate_fit.score),
                                    )
                                )
                                record = {
                                    "kind": candidate_kind,
                                    "certified": certified,
                                    "success_probability": success_probability,
                                    "safety_correction_rms": safety_correction,
                                    "critic_correction_rms": critic_correction,
                                    "combined_correction_rms": correction_rms,
                                    "utility": utility,
                                    "minimum_final_barrier_m": minimum_barrier,
                                    "surface_score": float(candidate_fit.score),
                                    "surface_rmse_m": float(candidate_fit.surface_rmse),
                                }
                                shape_probe_records.append(record)
                                ranked_shapes.append(
                                    (rank, candidate_kind, candidate_response)
                                )
                            _, selected_kind, response = max(
                                ranked_shapes, key=lambda item: item[0]
                            )
                            selected_state = obstacle_candidate_frames[selected_kind]
                            primitive_fit = selected_state["fit"]
                            obstacle_kind = selected_kind
                            obstacle_size = primitive_fit.size
                            primitive_center_in_obstacle = selected_state[
                                "center_in_obstacle"
                            ]
                            primitive_rotation_in_obstacle = selected_state[
                                "rotation_in_obstacle"
                            ]
                            p2 = (
                                current_obstacle_position
                                + current_obstacle_rotation
                                @ primitive_center_in_obstacle
                            )
                            R2 = (
                                current_obstacle_rotation
                                @ primitive_rotation_in_obstacle
                            )
                            obstacle_shape_selection_record = {
                                "selection_step": int(t),
                                "selected_kind": selected_kind,
                                "lambda_intervention": float(
                                    args.action_expert_shape_selection_lambda_intervention
                                ),
                                "shared_fixed_flow_noise": bool(
                                    args.use_fixed_flow_noise
                                ),
                                "candidates": shape_probe_records,
                            }
                            primitive_metadata_path = (
                                img_out_dir / "obstacle_primitive.json"
                            )
                            primitive_metadata = json.loads(
                                primitive_metadata_path.read_text()
                            )
                            primitive_metadata.update(
                                {
                                    "kind": obstacle_kind,
                                    "center": p2.tolist(),
                                    "rotation": R2.tolist(),
                                    "size": obstacle_size.tolist(),
                                    "score": primitive_fit.score,
                                    "surface_rmse": primitive_fit.surface_rmse,
                                    "qp_selected": True,
                                    "qp_selection": obstacle_shape_selection_record,
                                }
                            )
                            primitive_metadata_path.write_text(
                                json.dumps(
                                    primitive_metadata, indent=2, sort_keys=True
                                )
                                + "\n"
                            )
                            logging.info(
                                "One-shot QP shape selector chose %s: %s",
                                selected_kind,
                                shape_probe_records,
                            )
                        elif scorer is not None:
                            base_response = client.infer(element, noise=fixed_noise)
                            candidate_results.append(
                                (
                                    scorer.score_response(base_response),
                                    base_response,
                                    "baseline",
                                )
                            )
                            for candidate_idx in range(max(args.score_guidance_candidates - 1, 0)):
                                noise = _make_flow_noise(
                                    args,
                                    task_id=task_id,
                                    episode_idx=episode_idx,
                                    step=t,
                                    candidate_idx=candidate_idx + 1,
                                )
                                response = client.infer(element, noise=noise)
                                candidate_results.append(
                                    (
                                        scorer.score_response(response),
                                        response,
                                        f"noise_{candidate_idx + 1}",
                                    )
                                )
                            candidate_results.sort(key=lambda item: item[0], reverse=True)
                            selected_score, response, selected_label = candidate_results[0]
                            selected_candidate_indices.append(
                                0 if selected_label == "baseline" else int(selected_label.rsplit("_", 1)[1])
                            )
                            logging.info(
                                "Score guidance selected %s with safety score %.4f among %d candidates",
                                selected_label,
                                selected_score,
                                len(candidate_results),
                            )
                        elif args.use_action_expert_guidance and args.action_expert_candidates > 1:
                            ranked_candidates = []
                            candidate_telemetry = []
                            for candidate_idx in range(args.action_expert_candidates):
                                candidate_noise = _make_flow_noise(
                                    args,
                                    task_id=task_id,
                                    episode_idx=episode_idx,
                                    step=t,
                                    candidate_idx=candidate_idx,
                                )
                                candidate_response = client.infer(element, noise=candidate_noise)
                                candidate_barriers = np.asarray(
                                    candidate_response["action_expert_final_trajectory_barriers"],
                                    dtype=np.float64,
                                )
                                candidate_horizon = int(
                                    candidate_response.get(
                                        "action_expert_certified_horizon",
                                        len(candidate_barriers),
                                    )
                                )
                                candidate_certified = bool(
                                    candidate_response["action_expert_final_projection_success"]
                                    and candidate_horizon >= 1
                                    and np.all(candidate_barriers[:candidate_horizon] >= -1e-7)
                                )
                                candidate_passthrough = not bool(
                                    candidate_response.get("action_expert_intervened", True)
                                )
                                candidate_success = float(
                                    candidate_response.get(
                                        "action_expert_final_success_score_after",
                                        float("-inf"),
                                    )
                                )
                                candidate_correction = float(
                                    candidate_response.get(
                                        "action_expert_final_projection_correction_rms",
                                        float("inf"),
                                    )
                                )
                                candidate_minimum = float(
                                    np.min(candidate_barriers[: max(candidate_horizon, 1)])
                                )
                                rank = (
                                    (
                                        1.0,
                                        float(candidate_passthrough),
                                        candidate_success,
                                        candidate_minimum,
                                        -candidate_correction,
                                    )
                                    if candidate_certified
                                    else (
                                        0.0,
                                        0.0,
                                        candidate_minimum,
                                        candidate_success,
                                        -candidate_correction,
                                    )
                                )
                                ranked_candidates.append((rank, candidate_idx, candidate_response))
                                candidate_telemetry.append(
                                    {
                                        "candidate_index": candidate_idx,
                                        "certified": candidate_certified,
                                        "nominal_passthrough": candidate_passthrough,
                                        "success_probability": candidate_success,
                                        "minimum_certified_barrier_m": candidate_minimum,
                                        "projection_correction_rms": candidate_correction,
                                    }
                                )
                            _, selected_candidate, response = max(
                                ranked_candidates,
                                key=lambda item: item[0],
                            )
                            action_expert_candidate_records.append(
                                {
                                    "chunk_start_step": int(t),
                                    "selected_candidate_index": selected_candidate,
                                    "candidates": candidate_telemetry,
                                }
                            )
                            logging.info(
                                "Action expert selected candidate %d/%d",
                                selected_candidate,
                                args.action_expert_candidates,
                            )
                        else:
                            response = client.infer(element, noise=fixed_noise)
                        action_chunk = np.asarray(response["actions"], dtype=np.float64)
                        if args.action_expert_translation_only_execution:
                            action_chunk = action_chunk.copy()
                            action_chunk[:, 3:6] = 0.0
                        if args.use_action_expert_guidance and flag_safety_control:
                            final_barriers = np.asarray(
                                response["action_expert_final_trajectory_barriers"],
                                dtype=np.float32,
                            )
                            projection_success = bool(
                                response["action_expert_final_projection_success"]
                            )
                            certified_horizon = int(
                                response.get("action_expert_certified_horizon", len(final_barriers))
                            )
                            invalid_final_chunk = (
                                len(final_barriers) != args.action_expert_qp_horizon
                                or certified_horizon != args.action_expert_qp_horizon
                                or not projection_success
                                or np.any(final_barriers[:certified_horizon] < -1e-7)
                            )
                            if invalid_final_chunk and not args.action_expert_continue_on_unsafe:
                                raise RuntimeError(
                                    "Refusing server response without a verified-safe final "
                                    f"H={args.action_expert_qp_horizon} QP prefix: "
                                    f"success={projection_success}, "
                                    f"barriers={final_barriers.tolist()}"
                                )
                            if invalid_final_chunk:
                                logging.warning(
                                    "Continuing with corrected but uncertified H=%d QP prefix: "
                                    "success=%s, minimum_barrier=%.6f",
                                    args.action_expert_qp_horizon,
                                    projection_success,
                                    float(np.min(final_barriers)),
                                )
                            action_expert_final_trajectory_barriers.append(final_barriers)
                            action_expert_final_projection_success.append(projection_success)
                            action_expert_certified_horizons.append(certified_horizon)
                            action_expert_final_projection_iterations.append(
                                int(response["action_expert_final_projection_iterations"])
                            )
                            action_expert_final_projection_correction_rms.append(
                                float(response["action_expert_final_projection_correction_rms"])
                            )
                            action_expert_final_safety_correction_rms.append(
                                float(response.get("action_expert_final_safety_correction_rms", 0.0))
                            )
                            action_expert_final_critic_correction_rms.append(
                                float(response.get("action_expert_final_critic_correction_rms", 0.0))
                            )
                            action_expert_final_critic_refinement_accepted.append(
                                bool(response.get("action_expert_final_critic_refinement_accepted", False))
                            )
                            action_expert_selected_safety_trust_radius.append(
                                float(response.get("action_expert_selected_safety_trust_radius", np.nan))
                            )
                            action_expert_attempted_safety_trust_radii.append(
                                np.asarray(
                                    response.get("action_expert_attempted_safety_trust_radii", []),
                                    dtype=np.float32,
                                ).tolist()
                            )
                            action_expert_safety_resamples.append(
                                int(response.get("action_expert_safety_resamples", 0))
                            )
                            critic_times = np.asarray(
                                response.get("action_expert_times", []), dtype=np.float32
                            )
                            critic_scores_before = np.asarray(
                                response.get("action_expert_success_scores_before", []),
                                dtype=np.float32,
                            )
                            critic_scores_after = np.asarray(
                                response.get("action_expert_success_scores_after", []),
                                dtype=np.float32,
                            )
                            action_expert_critic_records.append(
                                {
                                    "chunk_start_step": int(t),
                                    "planned_execution_steps": list(
                                        range(int(t), int(t + args.replan_steps))
                                    ),
                                    "critic_evaluated": bool(len(critic_times)),
                                    "reason_not_evaluated": (
                                        None if len(critic_times) else "nominal chunk already predicted safe"
                                    ),
                                    "denoising_times": critic_times.tolist(),
                                    "success_probability_before": critic_scores_before.tolist(),
                                    "success_probability_after": critic_scores_after.tolist(),
                                    "pre_execution_success_probability": (
                                        float(critic_scores_after[-1]) if len(critic_scores_after) else None
                                    ),
                                    "final_success_probability_before": float(
                                        response.get("action_expert_final_success_score_before", np.nan)
                                    ),
                                    "final_success_probability_after": float(
                                        response.get("action_expert_final_success_score_after", np.nan)
                                    ),
                                    "final_critic_refinement_accepted": bool(
                                        response.get("action_expert_final_critic_refinement_accepted", False)
                                    ),
                                }
                            )
                            if args.action_expert_debug_rollout_geometry:
                                physical_chunk = np.asarray(action_chunk[:10], dtype=np.float64)
                                normalized_chunk = np.asarray(
                                    response.get("normalized_actions", []),
                                    dtype=np.float64,
                                )
                                # This is the controller-aware rollout of the
                                # raw PI0.5 sample before success-critic and
                                # safety-QP corrections.  Persist it separately
                                # from `physical_actions`, which contains the
                                # final corrected chunk returned for execution.
                                nominal_trajectory_positions = np.asarray(
                                    response.get(
                                        "action_expert_nominal_trajectory_positions",
                                        [],
                                    ),
                                    dtype=np.float64,
                                )
                                controller_rollout = ControllerRollout(
                                    input_min=np.asarray(controller.input_min),
                                    input_max=np.asarray(controller.input_max),
                                    output_min=np.asarray(controller.output_min),
                                    output_max=np.asarray(controller.output_max),
                                    translation_response_gain=np.asarray(
                                        current_translation_response_gain
                                    ),
                                    rotation_response_gain=np.asarray(
                                        args.action_expert_rotation_response_gain
                                    ),
                                )
                                predicted_trajectory = rollout_eef_trajectory(
                                    physical_chunk,
                                    eef_position=np.asarray(eef_pos, dtype=np.float64),
                                    eef_rotation=np.asarray(R1, dtype=np.float64),
                                    ellipsoid_offset=np.asarray(offset_local),
                                    controller=controller_rollout,
                                )
                                controller_clipped = np.clip(
                                    physical_chunk[:, :6], controller.input_min, controller.input_max
                                )
                                controller_scaled_delta = predicted_trajectory.scaled_commands
                                predicted_geometry = []
                                for horizon_index, (eef_prediction, rotation, center) in enumerate(
                                    zip(
                                        predicted_trajectory.positions,
                                        predicted_trajectory.rotations,
                                        predicted_trajectory.ellipsoid_centers,
                                        strict=True,
                                    )
                                ):
                                    terms = _action_expert_geometry_terms(
                                        center,
                                        rotation,
                                        Q1_diag,
                                        p2,
                                        R2,
                                        obstacle_kind,
                                        obstacle_size,
                                        args.action_expert_safe_distance,
                                        obstacle_top_padding=obstacle_top_padding,
                                        carried_object_center=(
                                            np.asarray(center, dtype=np.float64)
                                            + np.asarray(
                                                carried_object_offset,
                                                dtype=np.float64,
                                            )
                                            if carried_object_active
                                            else None
                                        ),
                                        carried_object_rotation=(
                                            np.eye(3, dtype=np.float64)
                                            if carried_object_active
                                            else None
                                        ),
                                        carried_object_size=(
                                            np.asarray(
                                                carried_object_fit.size,
                                                dtype=np.float64,
                                            )
                                            if carried_object_active
                                            and carried_object_fit is not None
                                            else None
                                        ),
                                    )
                                    terms.update(
                                        {
                                            "horizon_index": horizon_index,
                                            "eef_position": eef_prediction.tolist(),
                                            "eef_rotation": rotation.tolist(),
                                            "ellipsoid_center": center.tolist(),
                                            "controller_scaled_command": (
                                                predicted_trajectory.scaled_commands[horizon_index].tolist()
                                            ),
                                            "predicted_achieved_translation": (
                                                predicted_trajectory.achieved_translation_deltas[horizon_index].tolist()
                                            ),
                                            "predicted_achieved_rotation_vector": (
                                                predicted_trajectory.achieved_rotation_vectors[horizon_index].tolist()
                                            ),
                                        }
                                    )
                                    predicted_geometry.append(terms)
                                active_action_expert_debug_chunk = {
                                    "chunk_start_step": t,
                                    "translation_response_gain": float(
                                        current_translation_response_gain
                                    ),
                                    "initial_eef_position": np.asarray(eef_pos).tolist(),
                                    "initial_eef_rotation": np.asarray(R1).tolist(),
                                    "initial_ellipsoid_center": np.asarray(p1).tolist(),
                                    "initial_obstacle_center": np.asarray(p2).tolist(),
                                    "initial_obstacle_rotation": np.asarray(R2).tolist(),
                                    "carried_object_active": bool(
                                        carried_object_active
                                    ),
                                    "carried_object_offset": (
                                        None
                                        if carried_object_offset is None
                                        else np.asarray(
                                            carried_object_offset
                                        ).tolist()
                                    ),
                                    "carried_object_rotation": np.eye(3).tolist(),
                                    "carried_object_size": (
                                        None
                                        if carried_object_fit is None
                                        else np.asarray(
                                            carried_object_fit.size
                                        ).tolist()
                                    ),
                                    "physical_actions": physical_chunk.tolist(),
                                    "normalized_actions": normalized_chunk[:10].tolist(),
                                    "pi05_nominal_trajectory_positions": (
                                        nominal_trajectory_positions[:10].tolist()
                                    ),
                                    "controller_clipped_actions": controller_clipped.tolist(),
                                    "controller_scaled_deltas": controller_scaled_delta.tolist(),
                                    "prediction": predicted_geometry,
                                    "executed": [],
                                }
                                action_expert_rollout_geometry_debug.append(
                                    active_action_expert_debug_chunk
                                )
                        if "last_layer_hidden_state" in response:
                            chunk_start_steps.append(t)
                            last_layer_hidden_states.append(np.asarray(response["last_layer_hidden_state"], dtype=np.float32))
                            predicted_action_chunks.append(
                                np.asarray(action_chunk[: args.replan_steps], dtype=np.float32)
                            )
                            chunk_infer_ms.append(float(response.get("policy_timing", {}).get("infer_ms", np.nan)))
                            if candidate_results:
                                guided_candidate_scores.append(
                                    np.asarray([score for score, _, _ in candidate_results], dtype=np.float32)
                                )
                            if "safety_value_score" in response:
                                flow_value_scores.append(
                                    float(np.asarray(response["safety_value_score"]).reshape(-1)[0])
                                )
                            if "safety_residual_ratio" in response:
                                flow_residual_ratios.append(
                                    float(np.asarray(response["safety_residual_ratio"]))
                                )
                            if "time_conditioned_score_before" in response:
                                time_conditioned_scores.append(
                                    float(response["time_conditioned_score_before"])
                                )
                                time_conditioned_clearance_predictions.append(
                                    float(response["time_conditioned_clearance_before"])
                                )
                                time_conditioned_scores_after.append(
                                    float(response["time_conditioned_score_after"])
                                )
                                time_conditioned_clearance_predictions_after.append(
                                    float(response["time_conditioned_clearance_after"])
                                )
                                time_conditioned_gradient_rms.append(
                                    float(response["time_conditioned_gradient_rms"])
                                )
                                time_conditioned_perturbation_rms.append(
                                    float(response["time_conditioned_perturbation_rms"])
                                )
                                time_conditioned_correction_to_step.append(
                                    float(response["time_conditioned_correction_to_step"])
                                )
                                time_conditioned_accepted_factors.append(
                                    float(response["time_conditioned_accepted_factor"])
                                )
                                time_conditioned_injection_counts.append(
                                    int(response["time_conditioned_injection_count"])
                                )
                                time_conditioned_risk_gate_active.append(
                                    bool(response["time_conditioned_risk_gate_active"])
                                )
                                time_conditioned_safety_thresholds.append(
                                    float(response["time_conditioned_safety_threshold"])
                                )
                            if "action_expert_intervened" in response:
                                action_expert_intervened.append(bool(response["action_expert_intervened"]))
                                trajectory_barriers = np.asarray(
                                    response["action_expert_nominal_trajectory_barriers"], dtype=np.float32
                                )
                                action_expert_min_trajectory_barriers.append(
                                    float(np.min(trajectory_barriers))
                                )
                            if "action_expert_success_scores_before" in response:
                                action_expert_scores_before.append(
                                    np.asarray(response["action_expert_success_scores_before"], dtype=np.float32)
                                )
                                action_expert_scores_after.append(
                                    np.asarray(response["action_expert_success_scores_after"], dtype=np.float32)
                                )
                                action_expert_qp_success.append(
                                    np.asarray(response["action_expert_qp_success"], dtype=np.bool_)
                                )
                                action_expert_qp_margins.append(
                                    np.asarray(response["action_expert_qp_margins"], dtype=np.float32)
                                )
                                action_expert_correction_rms.append(
                                    np.asarray(response["action_expert_correction_rms"], dtype=np.float32)
                                )
                                action_expert_barriers_before.append(
                                    np.asarray(response["action_expert_barriers_before"], dtype=np.float32)
                                )
                                action_expert_barriers_nominal.append(
                                    np.asarray(response["action_expert_barriers_nominal"], dtype=np.float32)
                                )
                        planned_horizon = (
                            10
                            if args.use_action_expert_guidance
                            and args.action_expert_closed_loop_reprojection
                            else args.replan_steps
                        )
                        assert len(action_chunk) >= planned_horizon, (
                            f"Execution requires {planned_horizon} actions, but the policy "
                            f"returned {len(action_chunk)}."
                        )
                        action_plan.extend(action_chunk[:planned_horizon])
                        action_expert_chunk_executed = 0

                    t2 = time.time()
                    # print("t={}, inference time={}".format(t, t2-t1))

                    if (
                        args.use_action_expert_guidance
                        and flag_safety_control
                        and args.action_expert_closed_loop_reprojection
                        and action_expert_chunk_executed > 0
                    ):
                        suffix_before = np.asarray(action_plan, dtype=np.float64)
                        if len(suffix_before) < args.action_expert_qp_horizon:
                            raise RuntimeError(
                                "Not enough actions remain for the fixed receding QP "
                                f"window: need {args.action_expert_qp_horizon}, "
                                f"got {len(suffix_before)}"
                            )
                        feedback_window_before = suffix_before[
                            : args.action_expert_qp_horizon
                        ]
                        controller = env.env.robots[0].controller
                        feedback_controller = ControllerRollout(
                            input_min=np.asarray(controller.input_min),
                            input_max=np.asarray(controller.input_max),
                            output_min=np.asarray(controller.output_min),
                            output_max=np.asarray(controller.output_max),
                            translation_response_gain=np.asarray(
                                current_translation_response_gain
                            ),
                            rotation_response_gain=np.asarray(
                                args.action_expert_rotation_response_gain
                            ),
                        )
                        feedback_obstacle = ActionExpertObstacle(
                            obstacle_kind,
                            np.asarray(p2, dtype=np.float64),
                            np.asarray(R2, dtype=np.float64),
                            np.asarray(obstacle_size, dtype=np.float64),
                            float(obstacle_top_padding),
                        )

                        def feedback_barriers(candidate_actions):
                            return action_expert_trajectory_barriers(
                                candidate_actions,
                                eef_position=np.asarray(eef_pos, dtype=np.float64),
                                eef_rotation=np.asarray(R1, dtype=np.float64),
                                eef_radii=np.asarray(Q1_diag, dtype=np.float64),
                                ellipsoid_offset=np.asarray(offset_local, dtype=np.float64),
                                controller=feedback_controller,
                                obstacle=feedback_obstacle,
                                safe_distance=args.action_expert_safe_distance,
                                carried_object_offset=(
                                    np.asarray(
                                        carried_object_offset,
                                        dtype=np.float64,
                                    )
                                    if carried_object_active
                                    else None
                                ),
                                carried_object_rotation=(
                                    np.eye(3, dtype=np.float64)
                                    if carried_object_active
                                    else None
                                ),
                                carried_object_size=(
                                    np.asarray(
                                        carried_object_fit.size,
                                        dtype=np.float64,
                                    )
                                    if carried_object_active
                                    and carried_object_fit is not None
                                    else None
                                ),
                            )

                        adaptive_feedback_radii = tuple(
                            float(value)
                            for value in args.action_expert_adaptive_safety_trust_radii.split(",")
                            if value.strip()
                        )
                        feedback_nominal_barriers = np.asarray(
                            feedback_barriers(feedback_window_before), dtype=np.float64
                        )
                        feedback_hold_barrier = float(
                            feedback_barriers(np.zeros_like(feedback_window_before))[0]
                        )
                        feedback_recovery_floor = (
                            min(0.0, feedback_hold_barrier + 1e-4)
                            if args.action_expert_minimal_intervention
                            and args.action_expert_first_step_recovery
                            and feedback_hold_barrier < 0.0
                            else None
                        )
                        if (
                            args.action_expert_minimal_intervention
                            and feedback_nominal_barriers[0] >= 0.0
                        ):
                            feedback_projection = None
                            feedback_selected_radius = 0.0
                            feedback_attempted_radii = ()
                        elif adaptive_feedback_radii:
                            adaptive_feedback = project_action_chunk_with_adaptive_radius(
                                feedback_window_before,
                                lambda value: np.asarray(value, dtype=np.float64),
                                feedback_barriers,
                                trust_radii=adaptive_feedback_radii,
                                escalate_only_if_first_barrier_negative=(
                                    args.action_expert_adaptive_escalate_on_first_barrier_only
                                ),
                                enable_nonlinear_fallback=False,
                                first_step_recovery_floor=feedback_recovery_floor,
                            )
                            feedback_projection = adaptive_feedback.projection
                            feedback_selected_radius = adaptive_feedback.selected_trust_radius
                            feedback_attempted_radii = adaptive_feedback.attempted_trust_radii
                        else:
                            feedback_projection = project_action_chunk_with_qp(
                                feedback_window_before,
                                lambda value: np.asarray(value, dtype=np.float64),
                                feedback_barriers,
                                trust_radius=args.action_expert_final_safety_trust_radius,
                                # Feedback execution is deliberately QP-only: the
                                # nominal controller-aware rollout is nonlinear,
                                # but every suffix correction is an affine-
                                # constrained quadratic projection.
                                enable_nonlinear_fallback=False,
                                first_step_recovery_floor=feedback_recovery_floor,
                            )
                            feedback_selected_radius = args.action_expert_final_safety_trust_radius
                            feedback_attempted_radii = (args.action_expert_final_safety_trust_radius,)
                        if feedback_projection is None:
                            action_plan.clear()
                            action_plan.extend(suffix_before)
                            action_expert_feedback_projection_records.append(
                                {
                                    "chunk_start_step": t - action_expert_chunk_executed,
                                    "before_action_step": t,
                                    "executed_actions": action_expert_chunk_executed,
                                    "remaining_horizon": len(suffix_before),
                                    "qp_horizon": len(feedback_window_before),
                                    "success": bool(np.all(feedback_nominal_barriers >= -1e-7)),
                                    "iterations": 0,
                                    "minimum_barrier_before_m": float(np.min(feedback_nominal_barriers)),
                                    "minimum_barrier_after_m": float(np.min(feedback_nominal_barriers)),
                                    "correction_rms": 0.0,
                                    "selected_safety_trust_radius": 0.0,
                                    "attempted_safety_trust_radii": [],
                                    "minimal_intervention_passthrough": True,
                                }
                            )
                        elif (
                            not feedback_projection.success
                            and not args.action_expert_continue_on_unsafe
                        ):
                            raise RuntimeError(
                                "Refusing feedback-reprojected action suffix: "
                                f"executed={action_expert_chunk_executed}, "
                                f"minimum barrier={np.min(feedback_projection.barriers_after):.6f}, "
                                f"status={feedback_projection.status}"
                            )
                        if feedback_projection is not None:
                            suffix_after = suffix_before.copy()
                            suffix_after[: args.action_expert_qp_horizon] = (
                                feedback_projection.actions
                            )
                            action_plan.clear()
                            action_plan.extend(suffix_after)
                            action_expert_feedback_projection_records.append(
                                {
                                    "chunk_start_step": t - action_expert_chunk_executed,
                                    "before_action_step": t,
                                    "executed_actions": action_expert_chunk_executed,
                                    "remaining_horizon": len(suffix_before),
                                    "qp_horizon": len(feedback_window_before),
                                    "success": bool(feedback_projection.success),
                                    "iterations": int(feedback_projection.iterations),
                                    "minimum_barrier_before_m": float(
                                        np.min(feedback_projection.barriers_before)
                                    ),
                                    "minimum_barrier_after_m": float(
                                        np.min(feedback_projection.barriers_after)
                                    ),
                                    "correction_rms": float(
                                        np.sqrt(np.mean(np.square(feedback_projection.correction)))
                                    ),
                                    "selected_safety_trust_radius": float(feedback_selected_radius),
                                    "attempted_safety_trust_radii": list(feedback_attempted_radii),
                                    "minimal_intervention_passthrough": False,
                                }
                            )

                    action = action_plan.popleft()
                    action_expert_chunk_executed += 1
                    t3 = time.time()
                    response_gain_pre_eef_position = np.asarray(
                        obs["robot0_eef_pos"], dtype=np.float64
                    ).copy()
                    response_gain_scaled_command = np.clip(
                        np.asarray(action[:3], dtype=np.float64), -1.0, 1.0
                    ) * 0.05
                    if (
                        args.action_expert_debug_rollout_geometry
                        and active_action_expert_debug_chunk is not None
                    ):
                        debug_pre_eef_position = np.asarray(
                            obs["robot0_eef_pos"], dtype=np.float64
                        ).copy()
                        debug_pre_eef_rotation = R.from_quat(
                            np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
                        ).as_matrix()
                        debug_pre_ellipsoid_center = (
                            debug_pre_eef_position
                            + debug_pre_eef_rotation @ offset_local
                        )
                        debug_controller = env.env.robots[0].controller
                        debug_controller_site_id = model.site_name2id(
                            debug_controller.eef_name
                        )
                        debug_pre_controller_position = np.asarray(
                            data.site_xpos[debug_controller_site_id],
                            dtype=np.float64,
                        ).copy()
                        debug_pre_controller_rotation = np.asarray(
                            data.site_xmat[debug_controller_site_id],
                            dtype=np.float64,
                        ).reshape(3, 3).copy()
                    obstacle_positions_before_action = {
                        name: np.asarray(obs[f"{name}_pos"]).copy()
                        for name in active_obstacle_names
                    }
                    if flag_safety_control and not args.disable_safety_layer:
                        

                        v_ref =  R1.T @ action[:3]
                        u_v_ref = 5 * v_ref
                        omega_ref = action[3:6]
                        u_omega_ref = 5 * omega_ref
                        # print("u_om_ref:", u_omega_ref)

                        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, Q1_diag, R1, p2, Q2_diag, R2, z_fixed)
                        a_u_v = 0.2 * a_v
                        a_u_omega = 0.2 * a_omega
                        

                        u_z_nom = 10 * mu_row
                        u = cp.Variable(9)  # [v_x, v_y, v_z, u_zx, u_zy, u_zz]

                        # --- Weighted cost function ---
                        W = np.diag([1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0, 1.0, 1.0])  
                        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
                        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                        # --- Linear constraints ---
                        constraints = [
                            a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + 10 * h >= 0
                        ]
                        # --- Solve QP ---
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver=cp.OSQP)
                        # --- Read optimization results ---
                        if u.value is not None:
                            u_v = u.value[:3]
                            u_omega = u.value[3:6]
                            u_z = u.value[6:]
                        else:
                            u_v = action[:3]
                            u_omega = action[3:6]
                            u_z = u_z_nom
                            print("No feasible solution")
                            a
                        # print("t={}".format(t))
                        # print("u:", u.value)

                        Id = np.eye(len(z_fixed))
                        dz = (Id - np.outer(z_fixed, z_fixed)) @ u_z
                        z_fixed = z_fixed + dz * dt
                        z_fixed = z_fixed / np.linalg.norm(z_fixed)
                        # print("z_fixed:", z_fixed)

                    

                        action_input = np.zeros(7)
                        action_input[:3] = 0.2 * R1 @ u_v
                        action_input[3:6] = 0.2 * u_omega
                        action_input[6] = action[6]  # Keep gripper closed
                        # print("action_input:", action_input)
                        t4 = time.time()

                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step
                    else:
                        t4 = time.time()

                        obs, reward, done, info = env.step(action.tolist()) # Crucial step


                    
                    # Check both unsafe conditions after every executed action.
                    obstacle_motion = any(
                        np.sum(
                            np.abs(
                                np.asarray(obs[f"{name}_pos"])
                                - obstacle_positions_before_action[name]
                            )
                        )
                        > 0.001
                        for name in active_obstacle_names
                    )
                    obstacle_displacement = any(
                        np.sum(
                            np.abs(
                                np.asarray(obs[f"{name}_pos"])
                                - initial_obstacle_positions[name]
                            )
                        )
                        > 0.001
                        for name in active_obstacle_names
                    )
                    action_collision = obstacle_displacement
                    unsafe_action = obstacle_displacement
                    gripper_obstacle_contact = False
                    gripper_unmodeled_upper_contact = False
                    gripper_actual_top_band_contact = False
                    if args.diagnose_gripper_obstacle_contacts:
                        actual_obstacle_top_z = max(
                            float(data.geom_xpos[geom_id, 2])
                            + float(
                                _geom_world_aabb_half_extents(
                                    model, data, geom_id
                                )[2]
                            )
                            for geom_id in diagnostic_obstacle_geom_ids
                        )
                        estimated_obstacle_top_z = float("nan")
                        if flag_safety_control:
                            _, estimated_obstacle_top_z = _primitive_world_z_bounds(
                                obstacle_kind,
                                p2,
                                R2,
                                obstacle_size,
                                obstacle_top_padding,
                            )
                        for contact_index in range(data.ncon):
                            contact = data.contact[contact_index]
                            geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
                            relevant = (
                                geom_1 in diagnostic_gripper_geom_ids
                                and geom_2 in diagnostic_obstacle_geom_ids
                            ) or (
                                geom_2 in diagnostic_gripper_geom_ids
                                and geom_1 in diagnostic_obstacle_geom_ids
                            )
                            # Positive-distance entries can be speculative margin
                            # contacts; count only physical touching/penetration.
                            if not relevant or float(contact.dist) > 0.0:
                                continue
                            gripper_obstacle_contact = True
                            contact_z = float(contact.pos[2])
                            above_estimated_top = bool(
                                np.isfinite(estimated_obstacle_top_z)
                                and contact_z > estimated_obstacle_top_z + 1e-6
                            )
                            in_actual_top_band = bool(
                                contact_z
                                >= actual_obstacle_top_z
                                - args.diagnostic_top_contact_band_m
                            )
                            gripper_unmodeled_upper_contact |= above_estimated_top
                            gripper_actual_top_band_contact |= in_actual_top_band
                            gripper_geom = (
                                geom_1
                                if geom_1 in diagnostic_gripper_geom_ids
                                else geom_2
                            )
                            obstacle_geom = geom_2 if gripper_geom == geom_1 else geom_1
                            gripper_obstacle_contact_records.append(
                                {
                                    "step": int(t),
                                    "contact_z_m": contact_z,
                                    "penetration_m": max(0.0, -float(contact.dist)),
                                    "gripper_geom": model.geom_id2name(gripper_geom),
                                    "obstacle_geom": model.geom_id2name(obstacle_geom),
                                    "estimated_obstacle_top_z_m": (
                                        None
                                        if not np.isfinite(estimated_obstacle_top_z)
                                        else estimated_obstacle_top_z
                                    ),
                                    "actual_obstacle_top_z_m": actual_obstacle_top_z,
                                    "above_estimated_top": above_estimated_top,
                                    "in_actual_top_band": in_actual_top_band,
                                    "obstacle_displaced": bool(obstacle_displacement),
                                }
                            )
                    executed_action_steps.append(t)
                    per_action_collision_flags.append(action_collision)
                    per_action_obstacle_motion_flags.append(obstacle_motion)
                    per_action_obstacle_displacement_flags.append(obstacle_displacement)
                    per_action_unsafe_flags.append(unsafe_action)
                    per_action_gripper_obstacle_contact.append(
                        gripper_obstacle_contact
                    )
                    per_action_gripper_unmodeled_upper_contact.append(
                        gripper_unmodeled_upper_contact
                    )
                    per_action_gripper_actual_top_band_contact.append(
                        gripper_actual_top_band_contact
                    )
                    per_action_gripper_contact_with_displacement.append(
                        gripper_obstacle_contact and obstacle_displacement
                    )
                    if obstacle_displacement and not collide_flag:
                        print(f"obstacle moved from its initial position at action step {t}")
                        collision_action_steps.append(t)
                    if unsafe_action:
                        collide_flag = True

            

                    
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    r = R.from_quat(eef_quat)
                    euler = r.as_euler('xyz', degrees=False)
                    R1 = R.from_quat(eef_quat).as_matrix()
                    offset_local = np.array([0, 0, -0.08])
                    offset_world = R1 @ offset_local
                    ball_pos = eef_pos + offset_world
                    env.sim.model.body_pos[eef_body_id] = ball_pos
                    env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]
                    p1 = ball_pos

                    if (
                        args.use_action_expert_guidance
                        and args.action_expert_online_response_gain
                    ):
                        response_gain_command_window.append(
                            response_gain_scaled_command.copy()
                        )
                        response_gain_achieved_window.append(
                            np.asarray(eef_pos, dtype=np.float64)
                            - response_gain_pre_eef_position
                        )

                    if (
                        args.action_expert_compound_carried_object
                        and flag_safety_control
                    ):
                        close_signal = bool(
                            float(action[6])
                            > args.action_expert_grasp_close_threshold
                        )
                        current_gripper_width = gripper_width(
                            obs["robot0_gripper_qpos"]
                        )
                        width_delta = previous_gripper_width - current_gripper_width
                        gripper_narrowing = bool(
                            width_delta
                            >= args.action_expert_grasp_width_delta_threshold
                        )
                        if close_signal:
                            carried_object_close_steps.append(int(t))
                        else:
                            # Arm the estimator for the next physical close
                            # cycle only after an open command is observed.
                            carried_object_close_cycle_active = False
                            carried_object_estimation_window_end = None
                        if (
                            close_signal
                            and gripper_narrowing
                            and not carried_object_close_cycle_active
                        ):
                            carried_object_close_cycle_active = True
                            carried_object_estimation_window_end = int(
                                t
                                + args.action_expert_grasp_estimation_window_steps
                            )

                        should_estimate_carried_object = bool(
                            not carried_object_active
                            and close_signal
                            and carried_object_close_cycle_active
                            and carried_object_estimation_window_end is not None
                            and t <= carried_object_estimation_window_end
                            and t - carried_object_last_estimation_step
                            >= args.action_expert_grasp_estimation_retry_steps
                        )
                        if should_estimate_carried_object:
                            carried_object_last_estimation_step = int(t)
                            attempt_dir = carried_dir / f"attempt_step_{t:06d}"
                            attempt_dir.mkdir(parents=True, exist_ok=True)
                            attempt_record = {
                                "step": int(t),
                                "action_close_signal": float(action[6]),
                                "previous_gripper_width": float(
                                    previous_gripper_width
                                ),
                                "current_gripper_width": float(
                                    current_gripper_width
                                ),
                                "width_delta": float(width_delta),
                                "ellipsoid_center_m": np.asarray(
                                    p1, dtype=np.float64
                                ).tolist(),
                                "ellipsoid_rotation": np.asarray(
                                    R1, dtype=np.float64
                                ).tolist(),
                                "ellipsoid_radii_m": np.asarray(
                                    Q1_diag, dtype=np.float64
                                ).tolist(),
                                "success": False,
                            }
                            try:
                                carried_agent_img = np.ascontiguousarray(
                                    obs["agentview_image"][::-1, ::-1]
                                )
                                carried_agent_depth = np.ascontiguousarray(
                                    obs["agentview_depth"][::-1, ::-1]
                                )
                                rgbd_points, rgbd_colors = (
                                    rgbd_to_world_point_cloud(
                                        carried_agent_img,
                                        carried_agent_depth,
                                        env,
                                        "agentview",
                                        stride=2,
                                    )
                                )
                                if args.save_perception_diagnostics:
                                    from PIL import Image

                                    Image.fromarray(carried_agent_img).save(
                                        attempt_dir / "agentview_rgb.png"
                                    )
                                    # Raw RGB-D artifacts are useful for a
                                    # perception audit but dominate disk use
                                    # in full-suite evaluation runs.
                                    np.savez_compressed(
                                        attempt_dir / "point_cloud_rgbd.npz",
                                        points=rgbd_points.astype(np.float32),
                                        colors=rgbd_colors.astype(np.uint8),
                                    )
                                current_obstacle_position_for_crop = np.asarray(
                                    obs[f"{tracked_obstacle_name}_pos"],
                                    dtype=np.float64,
                                )
                                current_obstacle_rotation_for_crop = R.from_quat(
                                    np.asarray(
                                        obs[f"{tracked_obstacle_name}_quat"],
                                        dtype=np.float64,
                                    )
                                ).as_matrix()
                                crop_obstacle_center = (
                                    current_obstacle_position_for_crop
                                    + current_obstacle_rotation_for_crop
                                    @ primitive_center_in_obstacle
                                )
                                crop_obstacle_rotation = (
                                    current_obstacle_rotation_for_crop
                                    @ primitive_rotation_in_obstacle
                                )
                                grasp_cloud = select_grasped_object_points(
                                    rgbd_points,
                                    rgbd_colors,
                                    ellipsoid_center=p1,
                                    ellipsoid_rotation=R1,
                                    ellipsoid_radii=Q1_diag,
                                    task_suite_name=args.task_suite_name,
                                    crop_radius_m=(
                                        args.action_expert_carried_object_crop_radius_m
                                    ),
                                    min_points=(
                                        args.action_expert_carried_object_min_points
                                    ),
                                    max_anchor_distance_m=(
                                        args.action_expert_grasp_activation_distance
                                    ),
                                    obstacle_center=crop_obstacle_center,
                                    obstacle_rotation=crop_obstacle_rotation,
                                    obstacle_half_extents=(
                                        primitive_bounding_box_half_extents(
                                            obstacle_kind,
                                            obstacle_size,
                                        )
                                    ),
                                )
                                # Trim isolated RGB-D edge pixels before the
                                # final AABB, while retaining the visible body.
                                lower = np.quantile(
                                    grasp_cloud.points, 0.01, axis=0
                                )
                                upper = np.quantile(
                                    grasp_cloud.points, 0.99, axis=0
                                )
                                fit_points = grasp_cloud.points[
                                    np.all(
                                        (grasp_cloud.points >= lower)
                                        & (grasp_cloud.points <= upper),
                                        axis=1,
                                    )
                                ]
                                if len(fit_points) < args.action_expert_carried_object_min_points:
                                    raise RuntimeError(
                                        "too few carried-object points after robust trimming"
                                    )
                                candidate_carried_object_fit = _fit_axis_aligned_obb(
                                    fit_points,
                                    padding=(
                                        args.action_expert_carried_object_fit_padding_m
                                    ),
                                    selector_scores={},
                                )
                                attempt_record.update(
                                    {
                                        "rgbd_point_count": int(
                                            len(rgbd_points)
                                        ),
                                        "candidate_point_count": int(
                                            len(grasp_cloud.candidate_points)
                                        ),
                                        "selected_point_count": int(
                                            len(grasp_cloud.points)
                                        ),
                                        "cluster_count": int(
                                            grasp_cloud.cluster_count
                                        ),
                                        "support_plane_z_m": (
                                            None
                                            if grasp_cloud.support_plane_z_m is None
                                            else float(grasp_cloud.support_plane_z_m)
                                        ),
                                        "anchor_distance_m": float(
                                            grasp_cloud.anchor_distance_m
                                        ),
                                        "unvalidated_box_center_m": (
                                            candidate_carried_object_fit.center.tolist()
                                        ),
                                        "unvalidated_box_full_dimensions_m": (
                                            2.0 * candidate_carried_object_fit.size
                                        ).tolist(),
                                    }
                                )
                                grasp_distance = float(
                                    np.linalg.norm(
                                        candidate_carried_object_fit.center - p1
                                    )
                                )
                                if (
                                    grasp_distance
                                    > args.action_expert_grasp_activation_distance
                                ):
                                    raise RuntimeError(
                                        "estimated box center is "
                                        f"{grasp_distance:.4f} m from gripper"
                                    )
                                # Publish geometry to the QP only after every
                                # physical plausibility check has passed.  A
                                # rejected candidate must never appear as an
                                # inactive carried box in rollout metadata.
                                carried_object_fit = candidate_carried_object_fit

                                np.savez_compressed(
                                    attempt_dir / "point_cloud_rgbd.npz",
                                    points=rgbd_points.astype(np.float32),
                                    colors=rgbd_colors.astype(np.uint8),
                                    candidate_points=(
                                        grasp_cloud.candidate_points.astype(np.float32)
                                    ),
                                    candidate_colors=(
                                        grasp_cloud.candidate_colors.astype(np.uint8)
                                    ),
                                    selected_points=(
                                        grasp_cloud.points.astype(np.float32)
                                    ),
                                    selected_colors=(
                                        grasp_cloud.colors.astype(np.uint8)
                                    ),
                                )
                                plot_primitive_fit(
                                    fit_points,
                                    carried_object_fit,
                                    attempt_dir
                                    / "point_cloud_axis_aligned_box_3d.png",
                                )

                                carried_object_active = True
                                carried_object_activation_step = int(t)
                                world_down = np.array(
                                    [0.0, 0.0, -1.0], dtype=np.float64
                                )
                                box_half_extents = np.asarray(
                                    carried_object_fit.size, dtype=np.float64
                                )
                                ellipsoid_local_direction = (
                                    np.asarray(R1, dtype=np.float64).T
                                    @ world_down
                                )
                                ellipsoid_vertical_support = float(
                                    np.linalg.norm(
                                        np.asarray(Q1_diag, dtype=np.float64)
                                        * ellipsoid_local_direction
                                    )
                                )
                                carried_object_offset = world_down * (
                                    ellipsoid_vertical_support
                                    + float(box_half_extents[2])
                                )
                                attempt_record.update(
                                    {
                                        "success": True,
                                        "box_center_m": (
                                            carried_object_fit.center.tolist()
                                        ),
                                        "box_full_dimensions_m": (
                                            2.0 * carried_object_fit.size
                                        ).tolist(),
                                        "ellipsoid_center_offset_m": (
                                            carried_object_offset.tolist()
                                        ),
                                    }
                                )
                                (carried_dir / "carried_object_primitive.json").write_text(
                                    json.dumps(
                                        {
                                            "perception_mode": (
                                                "grasp_conditioned_agentview_rgbd"
                                            ),
                                            "task_target_label": carried_prompt,
                                            "activation_step": int(t),
                                            "kind": "obb",
                                            "center_at_estimation": (
                                                carried_object_fit.center.tolist()
                                            ),
                                            "rotation": np.eye(3).tolist(),
                                            "size": carried_object_fit.size.tolist(),
                                            "full_dimensions_m": (
                                                2.0 * carried_object_fit.size
                                            ).tolist(),
                                            "anchor_distance_m": float(
                                                grasp_cloud.anchor_distance_m
                                            ),
                                            "selected_point_count": int(
                                                len(grasp_cloud.points)
                                            ),
                                            "padding_m": (
                                                args.action_expert_carried_object_fit_padding_m
                                            ),
                                        },
                                        indent=2,
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                # The remaining suffix was certified for the
                                # bare gripper. Replan immediately with the
                                # newly observed compound body.
                                action_plan.clear()
                                action_expert_chunk_executed = 0
                                logging.info(
                                    "Activated grasp-conditioned RGB-D AABB "
                                    "at step %d: width %.5f -> %.5f m, "
                                    "anchor distance=%.4f m, dimensions=%s m, "
                                    "offset=%s",
                                    t,
                                    previous_gripper_width,
                                    current_gripper_width,
                                    grasp_cloud.anchor_distance_m,
                                    2.0 * carried_object_fit.size,
                                    carried_object_offset,
                                )
                            except (RuntimeError, ValueError) as exc:
                                attempt_record["error"] = str(exc)
                                logging.warning(
                                    "Grasp-conditioned RGB-D estimate failed "
                                    "at step %d; will retry: %s",
                                    t,
                                    exc,
                                )
                            carried_object_estimation_attempts.append(
                                attempt_record
                            )
                            (attempt_dir / "estimation.json").write_text(
                                json.dumps(
                                    attempt_record,
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        previous_gripper_width = current_gripper_width

                    if args.use_action_expert_guidance and flag_safety_control:
                        current_obstacle_position = np.asarray(
                            obs[f"{tracked_obstacle_name}_pos"], dtype=np.float64
                        )
                        current_obstacle_rotation = R.from_quat(
                            np.asarray(
                                obs[f"{tracked_obstacle_name}_quat"],
                                dtype=np.float64,
                            )
                        ).as_matrix()
                        # Keep the fitted primitive rigidly attached to a
                        # moving obstacle for every supported shape.
                        p2 = (
                            current_obstacle_position
                            + current_obstacle_rotation
                            @ primitive_center_in_obstacle
                        )
                        R2 = (
                            current_obstacle_rotation
                            @ primitive_rotation_in_obstacle
                        )
                        ellipsoid_surface_gap = ellipsoid_obstacle_gap(
                            ActionExpertEllipsoid(p1, R1, Q1_diag),
                            ActionExpertObstacle(
                                obstacle_kind,
                                p2,
                                R2,
                                obstacle_size,
                                float(obstacle_top_padding),
                            ),
                        )
                        carried_object_center = None
                        carried_object_surface_gap = float("nan")
                        if carried_object_active:
                            carried_object_center = (
                                np.asarray(p1, dtype=np.float64)
                                + np.asarray(
                                    carried_object_offset,
                                    dtype=np.float64,
                                )
                            )
                            carried_object_surface_gap = primitive_obstacle_gap(
                                ActionExpertObstacle(
                                    "obb",
                                    carried_object_center,
                                    np.eye(3, dtype=np.float64),
                                    np.asarray(
                                        carried_object_fit.size,
                                        dtype=np.float64,
                                    ),
                                ),
                                ActionExpertObstacle(
                                    obstacle_kind,
                                    p2,
                                    R2,
                                    obstacle_size,
                                    float(obstacle_top_padding),
                                ),
                            )
                            actual_surface_gap = min(
                                ellipsoid_surface_gap,
                                carried_object_surface_gap,
                            )
                        else:
                            actual_surface_gap = ellipsoid_surface_gap
                        actual_barrier = (
                            actual_surface_gap - args.action_expert_safe_distance
                        )
                    else:
                        ellipsoid_surface_gap = float("nan")
                        carried_object_surface_gap = float("nan")
                        carried_object_center = None
                        actual_surface_gap = float("nan")
                        actual_barrier = float("nan")
                    per_action_obstacle_surface_gaps.append(actual_surface_gap)
                    per_action_obstacle_barriers.append(actual_barrier)
                    per_action_ellipsoid_surface_gaps.append(
                        ellipsoid_surface_gap
                    )
                    per_action_carried_object_surface_gaps.append(
                        carried_object_surface_gap
                    )
                    per_action_carried_object_active.append(
                        bool(carried_object_active)
                    )
                    per_action_carried_object_centers.append(
                        None
                        if carried_object_center is None
                        else carried_object_center.tolist()
                    )

                    if (
                        args.use_action_expert_guidance
                        and args.action_expert_closed_loop_reprojection
                        and action_expert_chunk_executed >= args.replan_steps
                    ):
                        if (
                            args.action_expert_online_response_gain
                            and response_gain_command_window
                        ):
                            commands = np.asarray(
                                response_gain_command_window, dtype=np.float64
                            )
                            achieved = np.asarray(
                                response_gain_achieved_window, dtype=np.float64
                            )
                            denominator = float(np.sum(commands * commands))
                            measured_gain = (
                                float(np.sum(commands * achieved) / denominator)
                                if denominator > 1e-10
                                else current_translation_response_gain
                            )
                            measured_gain = float(
                                np.clip(measured_gain, 0.02, 1.0)
                            )
                            response_gain_records.append(
                                {
                                    "after_action_step": int(t),
                                    "previous_gain": float(
                                        current_translation_response_gain
                                    ),
                                    "measured_gain": measured_gain,
                                    "command_rms_m": float(
                                        np.sqrt(np.mean(np.square(commands)))
                                    ),
                                    "achieved_rms_m": float(
                                        np.sqrt(np.mean(np.square(achieved)))
                                    ),
                                }
                            )
                            current_translation_response_gain = measured_gain
                            response_gain_command_window.clear()
                            response_gain_achieved_window.clear()
                            logging.info(
                                "Updated online translation response gain after step %d: %.4f",
                                t,
                                current_translation_response_gain,
                            )
                        # The EEF pose above was read directly from the
                        # post-env.step observation. After five feedback steps,
                        # discard the unused suffix and ask VLA for a new H=10.
                        action_plan.clear()
                        action_expert_chunk_executed = 0
                    if (
                        args.action_expert_debug_rollout_geometry
                        and active_action_expert_debug_chunk is not None
                    ):
                        horizon_index = len(
                            active_action_expert_debug_chunk["executed"]
                        )
                        if horizon_index < len(
                            active_action_expert_debug_chunk["prediction"]
                        ):
                            predicted_terms = active_action_expert_debug_chunk[
                                "prediction"
                            ][horizon_index]
                            actual_terms = _action_expert_geometry_terms(
                                p1,
                                R1,
                                Q1_diag,
                                p2,
                                R2,
                                obstacle_kind,
                                obstacle_size,
                                args.action_expert_safe_distance,
                                obstacle_top_padding=obstacle_top_padding,
                                carried_object_center=carried_object_center,
                                carried_object_rotation=(
                                    np.eye(3, dtype=np.float64)
                                    if carried_object_center is not None
                                    else None
                                ),
                                carried_object_size=(
                                    np.asarray(
                                        carried_object_fit.size,
                                        dtype=np.float64,
                                    )
                                    if carried_object_center is not None
                                    and carried_object_fit is not None
                                    else None
                                ),
                            )
                            controller = env.env.robots[0].controller
                            post_controller_position = np.asarray(
                                data.site_xpos[debug_controller_site_id],
                                dtype=np.float64,
                            ).copy()
                            post_controller_rotation = np.asarray(
                                data.site_xmat[debug_controller_site_id],
                                dtype=np.float64,
                            ).reshape(3, 3).copy()
                            raw_arm_action = np.asarray(action[:6], dtype=np.float64)
                            clipped_arm_action = np.clip(raw_arm_action, -1.0, 1.0)
                            scaled_delta = clipped_arm_action * np.array(
                                [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                                dtype=np.float64,
                            )
                            predicted_eef_position = np.asarray(
                                predicted_terms["eef_position"], dtype=np.float64
                            )
                            predicted_rotation = np.asarray(
                                predicted_terms["eef_rotation"], dtype=np.float64
                            )
                            predicted_center = np.asarray(
                                predicted_terms["ellipsoid_center"], dtype=np.float64
                            )
                            nominal_positions = active_action_expert_debug_chunk.get(
                                "pi05_nominal_trajectory_positions", []
                            )
                            nominal_eef_position = (
                                np.asarray(
                                    nominal_positions[horizon_index],
                                    dtype=np.float64,
                                )
                                if horizon_index < len(nominal_positions)
                                else None
                            )
                            executed_terms = {
                                "action_step": t,
                                "horizon_index": horizon_index,
                                "horizon": horizon_index + 1,
                                "raw_controller_action": raw_arm_action.tolist(),
                                "clipped_controller_action": clipped_arm_action.tolist(),
                                "controller_scaled_delta": scaled_delta.tolist(),
                                "saturated_channels": np.flatnonzero(
                                    np.abs(raw_arm_action) > 1.0
                                ).tolist(),
                                "pre_eef_position": debug_pre_eef_position.tolist(),
                                "post_eef_position": np.asarray(eef_pos).tolist(),
                                "achieved_eef_translation": (
                                    np.asarray(eef_pos) - debug_pre_eef_position
                                ).tolist(),
                                "commanded_goal_position": np.asarray(
                                    controller.goal_pos
                                ).tolist(),
                                "goal_position_tracking_error_m": float(
                                    np.linalg.norm(
                                        np.asarray(controller.goal_pos)
                                        - post_controller_position
                                    )
                                ),
                                "pre_controller_position": (
                                    debug_pre_controller_position.tolist()
                                ),
                                "post_controller_position": (
                                    post_controller_position.tolist()
                                ),
                                "achieved_controller_translation": (
                                    post_controller_position
                                    - debug_pre_controller_position
                                ).tolist(),
                                "pre_eef_rotation": debug_pre_eef_rotation.tolist(),
                                "post_eef_rotation": np.asarray(R1).tolist(),
                                "commanded_goal_rotation": np.asarray(
                                    controller.goal_ori
                                ).tolist(),
                                "goal_orientation_tracking_error_deg": (
                                    _rotation_error_degrees(
                                        post_controller_rotation,
                                        controller.goal_ori,
                                    )
                                ),
                                "pre_controller_rotation": (
                                    debug_pre_controller_rotation.tolist()
                                ),
                                "post_controller_rotation": (
                                    post_controller_rotation.tolist()
                                ),
                                "actual_ellipsoid_center": np.asarray(p1).tolist(),
                                "predicted_eef_position": predicted_eef_position.tolist(),
                                "pi05_nominal_eef_position": (
                                    None
                                    if nominal_eef_position is None
                                    else nominal_eef_position.tolist()
                                ),
                                "actual_eef_position": np.asarray(eef_pos).tolist(),
                                "predicted_eef_rotation": predicted_rotation.tolist(),
                                "actual_eef_rotation": np.asarray(R1).tolist(),
                                "predicted_ellipsoid_center": predicted_center.tolist(),
                                "predicted_eef_support_radius_m": float(
                                    predicted_terms["gripper_support"]
                                ),
                                "actual_eef_support_radius_m": float(
                                    actual_terms["gripper_support"]
                                ),
                                "predicted_surface_gap_m": float(
                                    predicted_terms["surface_gap"]
                                ),
                                "actual_surface_gap_m": float(
                                    actual_terms["surface_gap"]
                                ),
                                "predicted_barrier_m": float(
                                    predicted_terms["barrier"]
                                ),
                                "actual_barrier_m": float(actual_terms["barrier"]),
                                "actual_obstacle_center": np.asarray(p2).tolist(),
                                "actual_obstacle_rotation": np.asarray(R2).tolist(),
                                "eef_position_error_m": float(
                                    np.linalg.norm(
                                        predicted_eef_position - np.asarray(eef_pos)
                                    )
                                ),
                                "orientation_error_deg": _rotation_error_degrees(
                                    predicted_rotation, R1
                                ),
                                "ellipsoid_center_error_m": float(
                                    np.linalg.norm(predicted_center - np.asarray(p1))
                                ),
                                "gripper_support_error_m": float(
                                    actual_terms["gripper_support"]
                                    - predicted_terms["gripper_support"]
                                ),
                                "obstacle_support_error_m": float(
                                    actual_terms["obstacle_support"]
                                    - predicted_terms["obstacle_support"]
                                ),
                                "barrier_error_m": float(
                                    actual_terms["barrier"]
                                    - predicted_terms["barrier"]
                                ),
                                "position_error_mm": 1000.0 * float(
                                    np.linalg.norm(
                                        predicted_eef_position - np.asarray(eef_pos)
                                    )
                                ),
                                "ellipsoid_center_error_mm": 1000.0 * float(
                                    np.linalg.norm(predicted_center - np.asarray(p1))
                                ),
                                "support_radius_error_mm": 1000.0 * float(
                                    actual_terms["gripper_support"]
                                    - predicted_terms["gripper_support"]
                                ),
                                "barrier_error_mm": 1000.0 * float(
                                    actual_terms["barrier"]
                                    - predicted_terms["barrier"]
                                ),
                                "collision": action_collision,
                                "collision_contact": None,
                                "collision_definition": "obstacle_displacement",
                                "geometry": actual_terms,
                            }
                            active_action_expert_debug_chunk["executed"].append(
                                executed_terms
                            )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception(
                        "Episode %d failed during policy/environment step %d",
                        episode_idx,
                        t,
                    )
                    if args.fail_on_episode_error:
                        raise
                    break

            task_episodes += 1
            total_episodes += 1

            time_steps.append(t)
            if collide_flag == True:
                collides += 1
                total_collisions += 1

            suffix = "success" if done else "failure"
            safe = "safe" if not collide_flag else "unsafe"
            video_path = out_dir / f"{episode_idx}_{suffix}_{safe}.mp4"
            if args.save_videos:
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=30,
                )
                imageio.mimwrite(
                    video_path.with_name(f"{video_path.stem}_agentview.mp4"),
                    [np.asarray(x) for x in replay_images],
                    fps=30,
                )
                imageio.mimwrite(
                    video_path.with_name(f"{video_path.stem}_backview.mp4"),
                    [np.asarray(x) for x in replay_backview_images],
                    fps=30,
                )
            hidden_state_path = video_path.parent / f"{video_path.stem}_last_layer_hidden_states.npz"
            hidden_state_array = (
                np.stack(last_layer_hidden_states, axis=0)
                if last_layer_hidden_states
                else np.empty((0, 0, 0), dtype=np.float32)
            )
            _save_npz_if_enabled(
                args.save_rollout_data,
                hidden_state_path,
                last_layer_hidden_states=hidden_state_array,
                action_chunks=(
                    np.stack(predicted_action_chunks, axis=0)
                    if predicted_action_chunks
                    else np.empty((0, args.replan_steps, 7), dtype=np.float32)
                ),
                chunk_start_steps=np.asarray(chunk_start_steps, dtype=np.int32),
                chunk_safety_scores=np.asarray(
                    [
                        0.0
                        if any(
                            unsafe
                            for action_step, unsafe in zip(
                                executed_action_steps, per_action_unsafe_flags
                            )
                            if start <= action_step < start + args.replan_steps
                        )
                        else 1.0
                        for start in chunk_start_steps
                    ],
                    dtype=np.float32,
                ),
                executed_action_steps=np.asarray(executed_action_steps, dtype=np.int32),
                per_action_collision_flags=np.asarray(
                    per_action_collision_flags, dtype=np.bool_
                ),
                per_action_obstacle_motion_flags=np.asarray(
                    per_action_obstacle_motion_flags, dtype=np.bool_
                ),
                per_action_obstacle_displacement_flags=np.asarray(
                    per_action_obstacle_displacement_flags, dtype=np.bool_
                ),
                per_action_unsafe_flags=np.asarray(
                    per_action_unsafe_flags, dtype=np.bool_
                ),
                per_action_gripper_obstacle_contact=np.asarray(
                    per_action_gripper_obstacle_contact, dtype=np.bool_
                ),
                per_action_gripper_unmodeled_upper_contact=np.asarray(
                    per_action_gripper_unmodeled_upper_contact, dtype=np.bool_
                ),
                per_action_gripper_actual_top_band_contact=np.asarray(
                    per_action_gripper_actual_top_band_contact, dtype=np.bool_
                ),
                per_action_gripper_contact_with_displacement=np.asarray(
                    per_action_gripper_contact_with_displacement, dtype=np.bool_
                ),
                per_action_obstacle_surface_gaps=np.asarray(
                    per_action_obstacle_surface_gaps, dtype=np.float32
                ),
                per_action_obstacle_barriers=np.asarray(
                    per_action_obstacle_barriers, dtype=np.float32
                ),
                collision_action_steps=np.asarray(
                    collision_action_steps, dtype=np.int32
                ),
                chunk_infer_ms=np.asarray(chunk_infer_ms, dtype=np.float32),
                guided_candidate_scores=(
                    np.asarray(guided_candidate_scores, dtype=object)
                    if guided_candidate_scores
                    else np.empty((0,), dtype=np.float32)
                ),
                selected_candidate_indices=np.asarray(selected_candidate_indices, dtype=np.int32),
                flow_value_scores=np.asarray(flow_value_scores, dtype=np.float32),
                flow_residual_ratios=np.asarray(flow_residual_ratios, dtype=np.float32),
                time_conditioned_scores=np.asarray(
                    time_conditioned_scores, dtype=np.float32
                ),
                time_conditioned_clearance_predictions=np.asarray(
                    time_conditioned_clearance_predictions, dtype=np.float32
                ),
                time_conditioned_scores_after=np.asarray(
                    time_conditioned_scores_after, dtype=np.float32
                ),
                time_conditioned_clearance_predictions_after=np.asarray(
                    time_conditioned_clearance_predictions_after, dtype=np.float32
                ),
                time_conditioned_gradient_rms=np.asarray(
                    time_conditioned_gradient_rms, dtype=np.float32
                ),
                time_conditioned_perturbation_rms=np.asarray(
                    time_conditioned_perturbation_rms, dtype=np.float32
                ),
                time_conditioned_correction_to_step=np.asarray(
                    time_conditioned_correction_to_step, dtype=np.float32
                ),
                time_conditioned_accepted_factors=np.asarray(
                    time_conditioned_accepted_factors, dtype=np.float32
                ),
                time_conditioned_injection_counts=np.asarray(
                    time_conditioned_injection_counts, dtype=np.int32
                ),
                time_conditioned_risk_gate_active=np.asarray(
                    time_conditioned_risk_gate_active, dtype=np.bool_
                ),
                time_conditioned_safety_thresholds=np.asarray(
                    time_conditioned_safety_thresholds, dtype=np.float32
                ),
                action_expert_scores_before=np.asarray(action_expert_scores_before, dtype=np.float32),
                action_expert_scores_after=np.asarray(action_expert_scores_after, dtype=np.float32),
                action_expert_qp_success=np.asarray(action_expert_qp_success, dtype=np.bool_),
                action_expert_qp_margins=np.asarray(action_expert_qp_margins, dtype=np.float32),
                action_expert_correction_rms=np.asarray(action_expert_correction_rms, dtype=np.float32),
                action_expert_barriers_before=np.asarray(action_expert_barriers_before, dtype=np.float32),
                action_expert_barriers_nominal=np.asarray(action_expert_barriers_nominal, dtype=np.float32),
                action_expert_intervened=np.asarray(action_expert_intervened, dtype=np.bool_),
                action_expert_min_trajectory_barriers=np.asarray(
                    action_expert_min_trajectory_barriers, dtype=np.float32
                ),
                action_expert_certified_horizons=np.asarray(
                    action_expert_certified_horizons, dtype=np.int32
                ),
                action_expert_perception_available=np.asarray(flag_safety_control, dtype=np.bool_),
                episode_steps=np.asarray(t, dtype=np.int32),
                flow_guidance_scale=np.asarray(args.flow_guidance_scale, dtype=np.float32),
                flow_guidance_start_time=np.asarray(args.flow_guidance_start_time, dtype=np.float32),
                flow_guidance_translation_only=np.asarray(args.flow_guidance_translation_only),
                flow_guidance_orthogonal=np.asarray(args.flow_guidance_orthogonal),
                success=np.asarray(done),
                collision=np.asarray(collide_flag),
                safe_success=np.asarray(done and not collide_flag),
            )
            if args.use_action_expert_guidance:
                final_minimums = [
                    float(np.min(values[:certified_horizon]))
                    for values, certified_horizon in zip(
                        action_expert_final_trajectory_barriers,
                        action_expert_certified_horizons,
                        strict=True,
                    )
                ]
                (img_out_dir / "action_expert_safety_summary.json").write_text(
                    json.dumps(
                        {
                            "all_returned_chunks_verified_safe": bool(
                                action_expert_final_projection_success
                                and all(action_expert_final_projection_success)
                                and all(value >= -1e-7 for value in final_minimums)
                            ),
                            "chunks_returned": len(action_expert_final_trajectory_barriers),
                            "task_suite": args.task_suite_name,
                            "task_index": task_id,
                            "obstacle_name_selection": obstacle_selection_record,
                            "episode_index": episode_idx,
                            "safety_level": safety_level,
                            "success": bool(done),
                            "collision": bool(collide_flag),
                            "safe_success": bool(done and not collide_flag),
                            "collision_definition": (
                                "active_obstacle_position_l1_displacement_gt_0.001_m"
                            ),
                            "robot_obstacle_contact": None,
                            "robot_obstacle_contact_tracked": bool(
                                args.diagnose_gripper_obstacle_contacts
                            ),
                            "gripper_obstacle_contact": bool(
                                any(per_action_gripper_obstacle_contact)
                            ),
                            "gripper_obstacle_contact_steps": int(
                                sum(per_action_gripper_obstacle_contact)
                            ),
                            "gripper_unmodeled_upper_contact": bool(
                                any(per_action_gripper_unmodeled_upper_contact)
                            ),
                            "gripper_unmodeled_upper_contact_steps": int(
                                sum(per_action_gripper_unmodeled_upper_contact)
                            ),
                            "gripper_actual_top_band_contact": bool(
                                any(per_action_gripper_actual_top_band_contact)
                            ),
                            "gripper_actual_top_band_contact_steps": int(
                                sum(per_action_gripper_actual_top_band_contact)
                            ),
                            "gripper_contact_with_displacement": bool(
                                any(per_action_gripper_contact_with_displacement)
                            ),
                            "gripper_contact_with_displacement_steps": int(
                                sum(per_action_gripper_contact_with_displacement)
                            ),
                            "gripper_contact_diagnostic": {
                                "enabled": bool(
                                    args.diagnose_gripper_obstacle_contacts
                                ),
                                "actual_top_band_m": float(
                                    args.diagnostic_top_contact_band_m
                                ),
                                "unmodeled_upper_definition": (
                                    "penetrating MuJoCo gripper-obstacle contact "
                                    "with contact_z above current estimated primitive top"
                                ),
                                "records": gripper_obstacle_contact_records,
                            },
                            "obstacle_displaced": bool(
                                any(per_action_obstacle_displacement_flags)
                            ),
                            "paper_protocol_collision": bool(
                                any(per_action_obstacle_displacement_flags)
                            ),
                            "paper_protocol_safe_success": bool(
                                done and not any(per_action_obstacle_displacement_flags)
                            ),
                            "episode_steps": int(t),
                            "certified_horizons": action_expert_certified_horizons,
                            "d_safe_m": args.action_expert_safe_distance,
                            "compound_carried_object": {
                                "enabled": bool(
                                    args.action_expert_compound_carried_object
                                ),
                                "prompt": _carried_object_prompt_from_task(task_description),
                                "perception_mode": (
                                    "grasp_conditioned_agentview_rgbd"
                                ),
                                "activation_step": carried_object_activation_step,
                                "close_signal_steps": carried_object_close_steps,
                                "estimation_attempts": (
                                    carried_object_estimation_attempts
                                ),
                                "activation_distance_m": (
                                    args.action_expert_grasp_activation_distance
                                ),
                                "close_threshold": (
                                    args.action_expert_grasp_close_threshold
                                ),
                                "width_delta_threshold_m": (
                                    args.action_expert_grasp_width_delta_threshold
                                ),
                                "estimation_retry_steps": (
                                    args.action_expert_grasp_estimation_retry_steps
                                ),
                                "estimation_window_steps": (
                                    args.action_expert_grasp_estimation_window_steps
                                ),
                                "crop_radius_m": (
                                    args.action_expert_carried_object_crop_radius_m
                                ),
                                "axis_aligned_rotation": np.eye(3).tolist(),
                                "box_half_extents_m": (
                                    None
                                    if carried_object_fit is None
                                    else carried_object_fit.size.tolist()
                                ),
                                "ellipsoid_center_offset_m": (
                                    None
                                    if carried_object_offset is None
                                    else carried_object_offset.tolist()
                                ),
                            },
                            "minimum_final_barrier_m": (
                                min(final_minimums) if final_minimums else None
                            ),
                            "final_projection_correction_rms": (
                                action_expert_final_projection_correction_rms
                            ),
                            "final_safety_correction_rms": action_expert_final_safety_correction_rms,
                            "final_critic_correction_rms": action_expert_final_critic_correction_rms,
                            "final_critic_refinement_accepted": (
                                action_expert_final_critic_refinement_accepted
                            ),
                            "selected_safety_trust_radius": (
                                action_expert_selected_safety_trust_radius
                            ),
                            "attempted_safety_trust_radii": (
                                action_expert_attempted_safety_trust_radii
                            ),
                            "safety_resamples": action_expert_safety_resamples,
                            "final_projection_iterations": (
                                action_expert_final_projection_iterations
                            ),
                            "final_trajectory_barriers_m": [
                                values.tolist()
                                for values in action_expert_final_trajectory_barriers
                            ],
                            "actual_per_action_surface_gaps_m": (
                                per_action_obstacle_surface_gaps
                            ),
                            "actual_per_action_barriers_m": (
                                per_action_obstacle_barriers
                            ),
                            "actual_per_action_ellipsoid_surface_gaps_m": (
                                per_action_ellipsoid_surface_gaps
                            ),
                            "actual_per_action_carried_object_surface_gaps_m": [
                                None if not np.isfinite(value) else float(value)
                                for value in per_action_carried_object_surface_gaps
                            ],
                            "actual_per_action_carried_object_active": (
                                per_action_carried_object_active
                            ),
                            "actual_per_action_carried_object_centers_m": (
                                per_action_carried_object_centers
                            ),
                            "collision_surface_gaps_m": [
                                gap
                                for gap, collision in zip(
                                    per_action_obstacle_surface_gaps,
                                    per_action_collision_flags,
                                    strict=True,
                                )
                                if collision
                            ],
                            "feedback_reprojections": (
                                action_expert_feedback_projection_records
                            ),
                            "online_translation_response_gain": {
                                "enabled": bool(
                                    args.action_expert_online_response_gain
                                ),
                                "initial_gain": float(
                                    args.action_expert_translation_response_gain
                                ),
                                "final_gain": float(
                                    current_translation_response_gain
                                ),
                                "records": response_gain_records,
                            },
                            "success_critic": {
                                "checkpoint": str(args.action_expert_critic_run_dir),
                                "beta_success": args.action_expert_beta_success,
                                "records": action_expert_critic_records,
                                "note": (
                                    "The critic is evaluated at VLA chunk generation. The final pre-execution "
                                    "probability applies to the generated five-action execution window; "
                                    "client-side feedback QPs do not recompute the critic."
                                ),
                            },
                            "candidate_selection": {
                                "candidate_count": args.action_expert_candidates,
                                "records": action_expert_candidate_records,
                            },
                            "obstacle_shape_selection": {
                                "enabled": bool(
                                    args.action_expert_qp_shape_selection
                                ),
                                "one_shot": True,
                                "lambda_intervention": float(
                                    args.action_expert_shape_selection_lambda_intervention
                                ),
                                "record": obstacle_shape_selection_record,
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                if args.action_expert_debug_rollout_geometry:
                    controller = env.env.robots[0].controller
                    (
                        img_out_dir
                        / "action_expert_rollout_geometry_debug.json"
                    ).write_text(
                        json.dumps(
                            {
                                "control_timestep_s": float(
                                    env.env.control_timestep
                                ),
                                "simulation_timestep_s": float(
                                    env.env.model_timestep
                                ),
                                "controller_type": type(controller).__name__,
                                "controller_input_min": np.asarray(
                                    controller.input_min
                                ).tolist(),
                                "controller_input_max": np.asarray(
                                    controller.input_max
                                ).tolist(),
                                "controller_output_min": np.asarray(
                                    controller.output_min
                                ).tolist(),
                                "controller_output_max": np.asarray(
                                    controller.output_max
                                ).tolist(),
                                "controller_use_delta": bool(controller.use_delta),
                                "online_translation_response_gain": {
                                    "enabled": bool(
                                        args.action_expert_online_response_gain
                                    ),
                                    "initial_gain": float(
                                        args.action_expert_translation_response_gain
                                    ),
                                    "final_gain": float(
                                        current_translation_response_gain
                                    ),
                                    "records": response_gain_records,
                                },
                                "ellipsoid_offset_local_m": offset_local.tolist(),
                                "ellipsoid_radii_m": Q1_diag.tolist(),
                                "compound_carried_object_enabled": bool(
                                    args.action_expert_compound_carried_object
                                ),
                                "carried_object_perception_mode": (
                                    "grasp_conditioned_agentview_rgbd"
                                ),
                                "carried_object_prompt": (
                                    _carried_object_prompt_from_task(task_description)
                                ),
                                "carried_object_estimation_attempts": (
                                    carried_object_estimation_attempts
                                ),
                                "carried_object_activation_step": (
                                    carried_object_activation_step
                                ),
                                "carried_object_offset_from_ellipsoid_m": (
                                    None
                                    if carried_object_offset is None
                                    else carried_object_offset.tolist()
                                ),
                                "carried_object_rotation": np.eye(3).tolist(),
                                "carried_object_box_half_extents_m": (
                                    None
                                    if carried_object_fit is None
                                    else carried_object_fit.size.tolist()
                                ),
                                "chunks": action_expert_rollout_geometry_debug,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"Collision: {collide_flag}")
            ss = done and not collide_flag
            if ss:
                total_safesuccesses += 1
            logging.info(f"SS (Safe Success): {ss}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"# task collides: {collides} ({collides / task_episodes * 100:.1f}%)")
            logging.info(f"# total collides: {total_collisions} ({total_collisions / total_episodes * 100:.1f}%)")
            logging.info(f"# safesuccesses: {total_safesuccesses} ({total_safesuccesses / total_episodes * 100:.1f}%)")

            print("collide_flag:", collide_flag)
            print("collision_action_steps:", collision_action_steps)


        # Log final results
        if task_episodes:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        else:
            logging.info("Current task success rate: unavailable (all episodes skipped)")
        if total_episodes:
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    else:
        logging.info("Total success rate: unavailable (no completed episodes)")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Time steps: {time_steps}")


def _rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(second, dtype=np.float64) @ np.asarray(
        first, dtype=np.float64
    ).T
    return float(np.degrees(R.from_matrix(relative).magnitude()))


def _fit_axis_aligned_obb(
    points: np.ndarray,
    *,
    padding: float,
    selector_scores: dict[str, float],
    top_padding: float | None = None,
    bottom_padding: float | None = None,
) -> PrimitiveFit:
    """Fit a world-axis-aligned OBB while retaining selector diagnostics."""
    points = np.asarray(points, dtype=np.float64)
    raw_lower = np.min(points, axis=0)
    raw_upper = np.max(points, axis=0)
    padded_lower = raw_lower - padding
    padded_upper = raw_upper + padding
    if top_padding is not None:
        if top_padding < 0.0:
            raise ValueError("top_padding must be non-negative")
        padded_upper[2] = raw_upper[2] + top_padding
    if bottom_padding is not None:
        if bottom_padding < 0.0:
            raise ValueError("bottom_padding must be non-negative")
        padded_lower[2] = raw_lower[2] - bottom_padding
    center = 0.5 * (padded_lower + padded_upper)
    half_extents = np.maximum(0.5 * (padded_upper - padded_lower), 0.0)

    offset = np.abs(points - center) - half_extents
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    inside = np.minimum(np.max(offset, axis=1), 0.0)
    distances = np.abs(outside + inside)
    cutoff = np.quantile(distances, 0.95)
    trimmed = distances[distances <= cutoff]
    rmse = float(np.sqrt(np.mean(np.square(trimmed))))
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-6)
    score = float(
        np.log((rmse / scale) ** 2 + 1e-12)
        + 9 * np.log(max(len(trimmed), 2)) / max(len(trimmed), 1)
    )
    scores = dict(selector_scores)
    scores["obb_axis_aligned"] = score
    return PrimitiveFit(
        kind="obb",
        center=center,
        rotation=np.eye(3, dtype=np.float64),
        size=half_extents,
        score=score,
        surface_rmse=rmse,
        candidate_scores=scores,
    )


def _fit_vlsa_mvee_ellipsoid(points: np.ndarray) -> PrimitiveFit:
    """Fit the original VLSA minimum-volume enclosing ellipsoid."""
    points = np.asarray(points, dtype=np.float64)
    center, rotation, radii = fit_ellipse(points, plot=False)
    local = (points - center) @ rotation
    levels = np.linalg.norm(local / np.maximum(radii, 1e-12), axis=1)
    projected = local / np.maximum(levels[:, None], 1e-12)
    distances = np.linalg.norm(local - projected, axis=1)
    cutoff = np.quantile(distances, 0.95)
    trimmed = distances[distances <= cutoff]
    rmse = float(np.sqrt(np.mean(np.square(trimmed))))
    return PrimitiveFit(
        kind="ellipsoid",
        center=np.asarray(center, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        size=np.asarray(radii, dtype=np.float64),
        score=rmse,
        surface_rmse=rmse,
        candidate_scores={"vlsa_mvee_ellipsoid": rmse},
    )


def _extend_axis_aligned_box_bottom(
    primitive_fit: PrimitiveFit,
    *,
    extra_bottom_padding: float,
) -> PrimitiveFit:
    """Extend only the lower world-z face of an axis-aligned box."""
    if extra_bottom_padding < 0.0:
        raise ValueError("extra_bottom_padding must be non-negative")
    if extra_bottom_padding == 0.0:
        return primitive_fit
    if primitive_fit.kind not in {"aabb", "obb"} or not np.allclose(
        primitive_fit.rotation, np.eye(3), atol=1e-8
    ):
        raise ValueError(
            "asymmetric bottom padding requires an axis-aligned box; "
            f"selected primitive was {primitive_fit.kind!r}"
        )
    center = np.asarray(primitive_fit.center, dtype=np.float64).copy()
    half_extents = np.asarray(primitive_fit.size, dtype=np.float64).copy()
    center[2] -= 0.5 * extra_bottom_padding
    half_extents[2] += 0.5 * extra_bottom_padding
    return dataclasses.replace(
        primitive_fit,
        center=center,
        size=half_extents,
    )


def _action_expert_geometry_terms(
    gripper_center: np.ndarray,
    gripper_rotation: np.ndarray,
    gripper_radii: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_rotation: np.ndarray,
    obstacle_kind: str,
    obstacle_size: np.ndarray,
    safe_distance: float,
    obstacle_top_padding: float = 0.0,
    carried_object_center: np.ndarray | None = None,
    carried_object_rotation: np.ndarray | None = None,
    carried_object_size: np.ndarray | None = None,
) -> dict:
    details = ellipsoid_obstacle_gap_details(
        ActionExpertEllipsoid(
            np.asarray(gripper_center, dtype=np.float64),
            np.asarray(gripper_rotation, dtype=np.float64),
            np.asarray(gripper_radii, dtype=np.float64),
        ),
        ActionExpertObstacle(
            obstacle_kind,
            np.asarray(obstacle_center, dtype=np.float64),
            np.asarray(obstacle_rotation, dtype=np.float64),
            np.asarray(obstacle_size, dtype=np.float64),
            float(obstacle_top_padding),
        ),
    )
    ellipsoid_surface_gap = float(details["surface_gap"])
    carried_surface_gap = None
    if carried_object_center is not None:
        if carried_object_rotation is None or carried_object_size is None:
            raise ValueError("complete carried-object geometry is required")
        carried_details = primitive_obstacle_gap_details(
            ActionExpertObstacle(
                "obb",
                np.asarray(carried_object_center, dtype=np.float64),
                np.asarray(carried_object_rotation, dtype=np.float64),
                np.asarray(carried_object_size, dtype=np.float64),
            ),
            ActionExpertObstacle(
                obstacle_kind,
                np.asarray(obstacle_center, dtype=np.float64),
                np.asarray(obstacle_rotation, dtype=np.float64),
                np.asarray(obstacle_size, dtype=np.float64),
                float(obstacle_top_padding),
            ),
        )
        carried_surface_gap = float(carried_details["surface_gap"])
    compound_surface_gap = (
        ellipsoid_surface_gap
        if carried_surface_gap is None
        else min(ellipsoid_surface_gap, carried_surface_gap)
    )
    result = {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in {
            **details,
            "surface_gap": compound_surface_gap,
            "barrier": compound_surface_gap - safe_distance,
        }.items()
    }
    result.update(
        {
            "ellipsoid_surface_gap": ellipsoid_surface_gap,
            "carried_object_active": carried_surface_gap is not None,
            "carried_object_surface_gap": carried_surface_gap,
            "carried_object_active_direction": (
                None
                if carried_surface_gap is None
                else np.asarray(
                    carried_details["active_direction"], dtype=np.float64
                ).tolist()
            ),
            "carried_object_support": (
                None
                if carried_surface_gap is None
                else float(carried_details["carried_object_support"])
            ),
            "carried_object_obstacle_support": (
                None
                if carried_surface_gap is None
                else float(carried_details["obstacle_support"])
            ),
            "active_component": (
                "carried_object_box"
                if carried_surface_gap is not None
                and carried_surface_gap < ellipsoid_surface_gap
                else "gripper_ellipsoid"
            ),
            "carried_object_center": (
                None
                if carried_object_center is None
                else np.asarray(carried_object_center, dtype=np.float64).tolist()
            ),
        }
    )
    return result


def _save_npz_if_enabled(enabled: bool, path: pathlib.Path, **arrays) -> None:
    """Write the optional rollout sidecar without complicating episode control flow."""
    if enabled:
        np.savez_compressed(path, **arrays)


def _make_flow_noise(
    args: Args,
    *,
    task_id: int,
    episode_idx: int,
    step: int,
    candidate_idx: int,
) -> np.ndarray:
    """Generate reproducible pi0.5 latent noise with the model's padded action dimension."""
    level_id = 1 if args.safety_level == "I" else 2
    seed = np.random.SeedSequence(
        [args.seed, level_id, task_id, episode_idx, step, candidate_idx]
    )
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (args.flow_noise_action_horizon, args.flow_noise_action_dim)
    ).astype(np.float32)


def _obstacle_prompt_from_instance(obstacle_name: str) -> str:
    """Map a SafeLIBERO simulator instance to a deterministic detector prompt."""
    return simulator_obstacle_prompt(obstacle_name)


def _carried_object_prompt_from_task(task_description: str) -> str:
    """Infer the object manipulated by a SafeLIBERO instruction."""
    text = task_description.lower()
    # Longest/more specific phrases first to avoid matching "mug" in names.
    phrases = (
        ("orange juice", "orange juice carton"),
        ("bbq sauce", "bbq sauce"),
        ("chocolate pudding", "chocolate pudding"),
        ("milk", "red milk carton"),
        ("yellow and white mug", "yellow and white mug"),
        ("white mug", "white mug"),
        ("black bowl", "black bowl"),
        ("bowl", "bowl"),
        ("mug", "mug"),
    )
    for needle, prompt in phrases:
        if needle in text:
            return prompt
    return "object to manipulate"


def _get_libero_env(task, level, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(task_description)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    # 手动调用函数
    eval_libero(args)
