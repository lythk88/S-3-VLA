"""Evaluate obstacle-perception primitive coverage on SafeLIBERO initial states.

This stops immediately after the same two-view GroundingDINO -> RGB-D cloud ->
primitive fitting path used by ``main_aegis.py``.  It compares both the raw
selected primitive and the currently executed axis-aligned OBB against the
active obstacle's MuJoCo collision geoms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from main_aegis import _fit_axis_aligned_obb, _obstacle_prompt_from_instance
from primitive_fitting import PrimitiveFit, fit_best_primitive
from utils import filtering_points, get_point_cloud


SUITES = (
    "safelibero_object",
    "safelibero_spatial",
    "safelibero_goal",
    "safelibero_long",
)
LEVELS = ("I", "II")
AXES = ("x", "y", "z")
BOX_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)


def _load_detector(
    config: Path,
    checkpoint: Path,
    device: str,
    backend: str,
    hf_model: str,
):
    if backend in {"native", "native-pytorch"}:
        from groundingdino.util.inference import load_model

        model = load_model(str(config), str(checkpoint), device=device)
        if backend == "native-pytorch":
            # The installed GroundingDINO package has no custom _C binary for
            # Blackwell. Its own reference PyTorch implementation is inference-
            # equivalent and runs on CUDA without that extension.
            from groundingdino.models.GroundingDINO import ms_deform_attn

            def pure_torch_apply(
                value,
                value_spatial_shapes,
                _value_level_start_index,
                sampling_locations,
                attention_weights,
                _im2col_step,
            ):
                return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                    value,
                    value_spatial_shapes,
                    sampling_locations,
                    attention_weights,
                )

            ms_deform_attn.MultiScaleDeformableAttnFunction.apply = staticmethod(
                pure_torch_apply
            )
        return model
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    return {
        "backend": "transformers",
        "device": device,
        "processor": AutoProcessor.from_pretrained(hf_model, local_files_only=False),
        "model": AutoModelForZeroShotObjectDetection.from_pretrained(
            hf_model, local_files_only=False
        ).to(torch.device(device)).eval(),
    }


def _active_obstacles(env, obs) -> list[str]:
    names = [
        name.replace("_joint0", "")
        for name in env.sim.model.joint_names
        if "obstacle" in name
    ]
    return [
        name
        for name in names
        if (
            np.asarray(obs[f"{name}_pos"])[2] > -0.05
            and -0.5 < np.asarray(obs[f"{name}_pos"])[0] < 0.5
            and -0.5 < np.asarray(obs[f"{name}_pos"])[1] < 0.5
        )
    ]


def _geom_half_aabb(model, data, geom_id: int) -> np.ndarray:
    """World-axis AABB half extents for common MuJoCo collision primitives."""
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    # mjtGeom values: sphere=2, capsule=3, ellipsoid=4, cylinder=5, box=6.
    if geom_type == 2:
        return np.full(3, size[0], dtype=np.float64)
    if geom_type == 3:
        axis = rotation[:, 2]
        return size[0] + size[1] * np.abs(axis)
    if geom_type == 4:
        return np.sqrt(np.square(rotation) @ np.square(size[:3]))
    if geom_type == 5:
        axis = rotation[:, 2]
        return size[1] * np.abs(axis) + size[0] * np.sqrt(
            np.maximum(1.0 - np.square(axis), 0.0)
        )
    if geom_type == 6:
        return np.abs(rotation) @ size[:3]
    raise ValueError(f"unsupported collision geom type {geom_type}")


def _actual_collision_geometry(env, obstacle_name: str) -> dict:
    model, data = env.sim.model, env.sim.data
    lowers, uppers, support_points, geom_names = [], [], [], []
    unsupported = []
    prefix = f"{obstacle_name}_g"
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id) or ""
        if not name.startswith(prefix) or int(model.geom_group[geom_id]) != 0:
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        try:
            half_aabb = _geom_half_aabb(model, data, geom_id)
        except ValueError:
            unsupported.append(name)
            continue
        lower, upper = center - half_aabb, center + half_aabb
        lowers.append(lower)
        uppers.append(upper)
        geom_names.append(name)
        # All current SafeLIBERO obstacle collision geoms are boxes. For boxes,
        # checking all eight vertices against a convex fitted primitive is an
        # exact containment test for the complete collision geom.
        if int(model.geom_type[geom_id]) == 6:
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            local_corners = BOX_SIGNS * np.asarray(model.geom_size[geom_id], dtype=np.float64)[:3]
            support_points.append(local_corners @ rotation.T + center)
        else:
            support_points.append(BOX_SIGNS * half_aabb + center)
    if not lowers:
        raise RuntimeError(
            f"no supported group-0 collision geoms found for {obstacle_name}; "
            f"unsupported={unsupported}"
        )
    return {
        "lower": np.min(np.stack(lowers), axis=0),
        "upper": np.max(np.stack(uppers), axis=0),
        "support_points": np.vstack(support_points),
        "geom_names": geom_names,
        "unsupported_geom_names": unsupported,
    }


def _primitive_half_aabb(fit: PrimitiveFit) -> np.ndarray:
    rotation = np.asarray(fit.rotation, dtype=np.float64)
    if fit.kind == "obb":
        return np.abs(rotation) @ np.asarray(fit.size, dtype=np.float64)
    radius, half_length = np.asarray(fit.size, dtype=np.float64)
    axis = rotation[:, 2]
    if fit.kind == "cylinder":
        return half_length * np.abs(axis) + radius * np.sqrt(
            np.maximum(1.0 - np.square(axis), 0.0)
        )
    if fit.kind == "capsule":
        return radius + half_length * np.abs(axis)
    raise ValueError(f"unsupported fitted primitive {fit.kind!r}")


def _primitive_contains(fit: PrimitiveFit, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    local = (np.asarray(points, dtype=np.float64) - fit.center) @ fit.rotation
    if fit.kind == "obb":
        return np.all(np.abs(local) <= np.asarray(fit.size) + tolerance, axis=1)
    radius, half_length = np.asarray(fit.size, dtype=np.float64)
    if fit.kind == "cylinder":
        return (np.linalg.norm(local[:, :2], axis=1) <= radius + tolerance) & (
            np.abs(local[:, 2]) <= half_length + tolerance
        )
    if fit.kind == "capsule":
        axial_excess = np.maximum(np.abs(local[:, 2]) - half_length, 0.0)
        distance = np.sqrt(np.sum(np.square(local[:, :2]), axis=1) + axial_excess**2)
        return distance <= radius + tolerance
    raise ValueError(f"unsupported fitted primitive {fit.kind!r}")


def _coverage(prefix: str, fit: PrimitiveFit, actual: dict) -> dict:
    half_aabb = _primitive_half_aabb(fit)
    estimated_lower = np.asarray(fit.center) - half_aabb
    estimated_upper = np.asarray(fit.center) + half_aabb
    actual_lower = actual["lower"]
    actual_upper = actual["upper"]
    intersection = np.maximum(
        0.0,
        np.minimum(estimated_upper, actual_upper) - np.maximum(estimated_lower, actual_lower),
    )
    actual_span = np.maximum(actual_upper - actual_lower, 1e-12)
    result = {
        f"{prefix}_kind": fit.kind,
        f"{prefix}_center_m": np.asarray(fit.center).tolist(),
        f"{prefix}_size_m": np.asarray(fit.size).tolist(),
        f"{prefix}_aabb_lower_m": estimated_lower.tolist(),
        f"{prefix}_aabb_upper_m": estimated_upper.tolist(),
        f"{prefix}_full_geometry_cover": bool(
            np.all(_primitive_contains(fit, actual["support_points"]))
        ),
    }
    axis_flags = []
    for index, axis in enumerate(AXES):
        covered = bool(
            estimated_lower[index] <= actual_lower[index] + 1e-6
            and estimated_upper[index] >= actual_upper[index] - 1e-6
        )
        axis_flags.append(covered)
        result[f"{prefix}_{axis}_cover"] = covered
        result[f"{prefix}_{axis}_coverage_fraction"] = float(
            intersection[index] / actual_span[index]
        )
        result[f"{prefix}_{axis}_lower_margin_m"] = float(
            actual_lower[index] - estimated_lower[index]
        )
        result[f"{prefix}_{axis}_upper_margin_m"] = float(
            estimated_upper[index] - actual_upper[index]
        )
    result[f"{prefix}_xyz_cover"] = bool(all(axis_flags))
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items()):
        successful = [row for row in group_rows if row.get("perception_success")]
        item = dict(zip(group_keys, group))
        item.update(
            {
                "cases": len(group_rows),
                "perception_successes": len(successful),
                "perception_success_rate": len(successful) / len(group_rows),
                "raw_kind_counts": dict(Counter(row["raw_kind"] for row in successful)),
            }
        )
        for prefix in ("raw", "executed"):
            for metric in (
                "x_cover",
                "y_cover",
                "z_cover",
                "xyz_cover",
                "full_geometry_cover",
            ):
                values = [bool(row[f"{prefix}_{metric}"]) for row in successful]
                item[f"{prefix}_{metric}_count"] = int(sum(values))
                item[f"{prefix}_{metric}_rate_successful"] = (
                    float(np.mean(values)) if values else None
                )
                item[f"{prefix}_{metric}_rate_all"] = sum(values) / len(group_rows)
            for axis in AXES:
                fractions = [
                    float(row[f"{prefix}_{axis}_coverage_fraction"])
                    for row in successful
                ]
                item[f"{prefix}_{axis}_mean_coverage_fraction"] = (
                    float(np.mean(fractions)) if fractions else None
                )
                for side in ("lower", "upper"):
                    key = f"{prefix}_{axis}_{side}_margin_m"
                    margins = [float(row[key]) for row in successful]
                    covered = [margin >= -1e-6 for margin in margins]
                    item[f"{prefix}_{axis}_{side}_cover_count"] = int(sum(covered))
                    item[f"{prefix}_{axis}_{side}_cover_rate_successful"] = (
                        float(np.mean(covered)) if covered else None
                    )
                    item[f"{prefix}_{axis}_{side}_cover_rate_all"] = (
                        sum(covered) / len(group_rows)
                    )
                    item[f"{prefix}_{axis}_{side}_margin_mean_m"] = (
                        float(np.mean(margins)) if margins else None
                    )
                    item[f"{prefix}_{axis}_{side}_margin_median_m"] = (
                        float(np.median(margins)) if margins else None
                    )
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "obstacle_geometry_coverage_320",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--padding-m", type=float, default=0.01)
    parser.add_argument("--top-padding-m", type=float, default=0.02)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--detector-device",
        choices=("cpu", "cuda"),
        default=os.environ.get("GROUNDINGDINO_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--detector-backend",
        choices=("native", "native-pytorch", "transformers"),
        default="native",
    )
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--groundingdino-hf-model", default="IDEA-Research/grounding-dino-tiny"
    )
    parser.add_argument(
        "--groundingdino-config",
        type=Path,
        default=root / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--groundingdino-checkpoint",
        type=Path,
        default=root / "GroundingDINO/groundingdino_swint_ogc.pth",
    )
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "coverage_rows.jsonl"
    rows_by_key = {}
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["suite"], row["level"], row["task_index"], row["episode_index"])
            rows_by_key[key] = row

    os.environ["GROUNDINGDINO_DEVICE"] = args.detector_device
    if args.detector_device == "cpu":
        import torch

        torch.set_num_threads(
            max(1, args.torch_threads or os.cpu_count() or 1)
        )
    detector = _load_detector(
        args.groundingdino_config,
        args.groundingdino_checkpoint,
        args.detector_device,
        args.detector_backend,
        args.groundingdino_hf_model,
    )
    full_expected = sum(
        benchmark.get_benchmark_dict()[suite](safety_level=level).n_tasks
        for suite in SUITES
        for level in LEVELS
    ) * args.episodes
    total_expected = len(range(args.shard_index, full_expected, args.shard_count))
    completed_this_run = 0
    case_ordinal = 0
    started = time.time()
    stop = False
    for suite_name in SUITES:
        if stop:
            break
        for level in LEVELS:
            if stop:
                break
            suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level)
            for task_index in range(suite.n_tasks):
                if stop:
                    break
                eligible_episodes = [
                    episode_index
                    for episode_index in range(args.episodes)
                    if (case_ordinal + episode_index) % args.shard_count
                    == args.shard_index
                ]
                case_ordinal += args.episodes
                if not eligible_episodes:
                    continue
                task = suite.get_task(task_index)
                initial_states = suite.get_task_init_states(task_index)
                bddl_path = (
                    Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                    camera_depths=True,
                )
                env.seed(args.seed)
                try:
                    for episode_index in eligible_episodes:
                        key = (suite_name, level, task_index, episode_index)
                        if key in rows_by_key:
                            continue
                        if args.max_cases is not None and completed_this_run >= args.max_cases:
                            stop = True
                            break
                        case_started = time.time()
                        row = {
                            "suite": suite_name,
                            "level": level,
                            "task_index": task_index,
                            "task": task.language,
                            "episode_index": episode_index,
                            "perception_success": False,
                        }
                        try:
                            env.reset()
                            obs = env.set_init_state(initial_states[episode_index])
                            active = _active_obstacles(env, obs)
                            if len(active) != 1:
                                raise RuntimeError(f"expected one active obstacle, found {active}")
                            obstacle_name = active[0]
                            prompt = _obstacle_prompt_from_instance(obstacle_name)
                            actual = _actual_collision_geometry(env, obstacle_name)
                            points_by_view = []
                            view_counts = {}
                            for view in ("agentview", "backview"):
                                image = np.ascontiguousarray(obs[f"{view}_image"][::-1, ::-1])
                                depth = np.ascontiguousarray(obs[f"{view}_depth"][::-1, ::-1])
                                points = get_point_cloud(
                                    image,
                                    depth,
                                    env,
                                    view,
                                    prompt,
                                    detector,
                                    args.output_dir,
                                    save_diagnostics=False,
                                )
                                valid = (
                                    isinstance(points, np.ndarray)
                                    and points.ndim == 2
                                    and points.shape[1:] == (3,)
                                    and len(points) > 0
                                )
                                view_counts[view] = int(len(points)) if valid else 0
                                if valid:
                                    points_by_view.append(points)
                            if not points_by_view:
                                raise RuntimeError("GroundingDINO returned no valid points in either view")
                            filtered = filtering_points(np.vstack(points_by_view), suite_name)
                            if len(filtered) < 16:
                                raise RuntimeError(f"only {len(filtered)} filtered points")
                            raw_fit = fit_best_primitive(
                                filtered,
                                allowed_kinds=("obb", "cylinder", "capsule"),
                            )
                            executed_fit = _fit_axis_aligned_obb(
                                filtered,
                                padding=args.padding_m,
                                top_padding=args.top_padding_m,
                                selector_scores=raw_fit.candidate_scores,
                            )
                            row.update(
                                {
                                    "perception_success": True,
                                    "obstacle_name": obstacle_name,
                                    "prompt": prompt,
                                    "agentview_points": view_counts["agentview"],
                                    "backview_points": view_counts["backview"],
                                    "filtered_points": int(len(filtered)),
                                    "actual_geom_count": len(actual["geom_names"]),
                                    "actual_aabb_lower_m": actual["lower"].tolist(),
                                    "actual_aabb_upper_m": actual["upper"].tolist(),
                                    "actual_full_dimensions_m": (
                                        actual["upper"] - actual["lower"]
                                    ).tolist(),
                                    "raw_candidate_scores": raw_fit.candidate_scores,
                                }
                            )
                            row.update(_coverage("raw", raw_fit, actual))
                            row.update(_coverage("executed", executed_fit, actual))
                        except Exception as exc:  # Keep the 320-case audit resumable.
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        row["elapsed_s"] = time.time() - case_started
                        rows_by_key[key] = row
                        completed_this_run += 1
                        ordered_rows = [rows_by_key[k] for k in sorted(rows_by_key)]
                        rows_path.write_text(
                            "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered_rows)
                        )
                        elapsed = time.time() - started
                        rate = completed_this_run / max(elapsed, 1e-9)
                        remaining = max(total_expected - len(rows_by_key), 0)
                        eta = remaining / max(rate, 1e-9)
                        print(
                            json.dumps(
                                {
                                    "done": len(rows_by_key),
                                    "total": total_expected,
                                    "suite": suite_name,
                                    "level": level,
                                    "task": task_index,
                                    "episode": episode_index,
                                    "ok": row["perception_success"],
                                    "executed_xyz_cover": row.get("executed_xyz_cover"),
                                    "elapsed_s": round(row["elapsed_s"], 2),
                                    "eta_min": round(eta / 60.0, 1),
                                }
                            ),
                            flush=True,
                        )
                finally:
                    env.close()

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    _write_csv(args.output_dir / "coverage_rows.csv", rows)
    summary = {
        "configuration": {
            "suites": SUITES,
            "levels": LEVELS,
            "episodes_per_task": args.episodes,
            "resolution": args.resolution,
            "seed": args.seed,
            "detector_device": args.detector_device,
            "detector_backend": args.detector_backend,
            "torch_threads": args.torch_threads,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "views": ["agentview", "backview"],
            "raw_padding_m": 0.005,
            "executed_shape": "world-axis-aligned OBB",
            "executed_padding_m": args.padding_m,
            "executed_top_padding_m": args.top_padding_m,
            "axis_cover_definition": "estimated AABB interval contains MuJoCo collision-geom AABB interval",
            "full_geometry_cover_definition": "all vertices of all group-0 MuJoCo collision boxes lie inside fitted primitive",
        },
        "completed_cases": len(rows),
        "expected_cases": total_expected,
        "full_benchmark_expected_cases": full_expected,
        "by_task_level": _aggregate(rows, ("suite", "level", "task_index", "task")),
        "by_suite_level": _aggregate(rows, ("suite", "level")),
        "by_level": _aggregate(rows, ("level",)),
        "overall": _aggregate(rows, tuple()),
    }
    (args.output_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "coverage_by_task_level.csv", summary["by_task_level"])
    _write_csv(args.output_dir / "coverage_by_suite_level.csv", summary["by_suite_level"])
    print(json.dumps({"finished": True, "completed_cases": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    # GroundingDINO reads this variable internally.
    os.environ.setdefault("GROUNDINGDINO_DEVICE", "cuda")
    main()
