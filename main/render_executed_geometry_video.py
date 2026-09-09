"""Render an executed gripper ellipsoid and obstacle primitive beside a rollout video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BOX_SIGNS = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
)
BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)
BOX_FACES = (
    (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
    (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5),
)


def _box_vertices(center: np.ndarray, rotation: np.ndarray, size: np.ndarray) -> np.ndarray:
    return center + (BOX_SIGNS * size) @ rotation.T


def _draw_box(ax, center, rotation, size, color, *, alpha=0.18, linewidth=1.6):
    vertices = _box_vertices(center, rotation, size)
    ax.add_collection3d(
        Poly3DCollection(
            [[vertices[index] for index in face] for face in BOX_FACES],
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )
    for first, second in BOX_EDGES:
        segment = vertices[[first, second]]
        ax.plot(*segment.T, color=color, linewidth=linewidth, alpha=0.95)


def _draw_ellipsoid(ax, center, rotation, radii, color):
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 11)
    local = np.stack(
        [
            radii[0] * np.outer(np.cos(u), np.sin(v)),
            radii[1] * np.outer(np.sin(u), np.sin(v)),
            radii[2] * np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    world = local @ rotation.T + center
    ax.plot_wireframe(
        world[..., 0], world[..., 1], world[..., 2],
        rstride=2, cstride=2, color=color, linewidth=0.8, alpha=0.88,
    )


# Collision boxes from safelibero's moka_pot_obstacle.xml (half-extents, m).
MOKA_POT_COLLISION_BOXES = (
    ((0.0, 0.0, 0.00070), (0.04000, 0.04000, 0.10648)),
    ((0.0, -0.04150, 0.00070), (0.00808, 0.04000, 0.10648)),
    ((0.0, 0.04728, 0.00070), (0.00808, 0.04000, 0.10648)),
    ((0.04827, 0.02006, 0.00070), (0.00672, 0.02750, 0.10648)),
    ((-0.04694, -0.02245, 0.00070), (0.00672, 0.02750, 0.10648)),
    ((0.04739, -0.02245, 0.00070), (0.00672, 0.02750, 0.10648)),
    ((-0.04694, 0.02414, 0.00070), (0.00672, 0.02750, 0.10648)),
    ((0.0, -0.05746, 0.06816), (0.00923, 0.02819, 0.03922)),
    ((0.0, -0.07856, 0.07390), (0.01154, 0.01283, 0.02198)),
    ((0.0, 0.06654, 0.08398), (0.01154, 0.01634, 0.02198)),
)


def _load_records(debug_path: Path):
    payload = json.loads(debug_path.read_text())
    records = [row for chunk in payload["chunks"] for row in chunk["executed"]]
    records.sort(key=lambda row: int(row["action_step"]))
    if not records:
        raise ValueError(f"No executed geometry records in {debug_path}")
    return payload, records


def _ffprobe(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def render(
    debug_path: Path,
    primitive_path: Path,
    original_video: Path,
    output: Path,
    frame_stride: int,
    case_label: str,
    source_summary_path: Path | None,
    actual_mujoco_moka: bool = False,
) -> dict:
    payload, records = _load_records(debug_path)
    primitive = json.loads(primitive_path.read_text())
    if primitive["kind"] not in {"obb", "aabb", "ellipsoid"}:
        raise ValueError(
            "This renderer currently requires a box or ellipsoid primitive, "
            f"got {primitive['kind']!r}"
        )

    steps = np.asarray([row["action_step"] for row in records], dtype=int)
    centers = np.asarray([row["actual_ellipsoid_center"] for row in records], dtype=float)
    rotations = np.asarray([row["actual_eef_rotation"] for row in records], dtype=float)
    obstacle_centers = np.asarray([row["actual_obstacle_center"] for row in records], dtype=float)
    obstacle_rotations = np.asarray([row["actual_obstacle_rotation"] for row in records], dtype=float)
    barriers = np.asarray([row["actual_barrier_m"] for row in records], dtype=float)
    surface_gaps = np.asarray([row["actual_surface_gap_m"] for row in records], dtype=float)
    active_directions = np.asarray([row["geometry"]["active_direction"] for row in records], dtype=float)
    gripper_supports = np.asarray([row["geometry"]["gripper_support"] for row in records], dtype=float)
    obstacle_supports = np.asarray([row["geometry"]["obstacle_support"] for row in records], dtype=float)
    carried_active = np.asarray(
        [bool(row["geometry"].get("carried_object_active", False)) for row in records],
        dtype=bool,
    )
    carried_centers = np.asarray(
        [
            row["geometry"].get("carried_object_center")
            if row["geometry"].get("carried_object_center") is not None
            else [np.nan, np.nan, np.nan]
            for row in records
        ],
        dtype=float,
    )
    active_components = [
        row["geometry"].get("active_component", "gripper_ellipsoid")
        for row in records
    ]
    carried_directions = np.asarray(
        [
            row["geometry"].get("carried_object_active_direction")
            if row["geometry"].get("carried_object_active_direction") is not None
            else [np.nan, np.nan, np.nan]
            for row in records
        ],
        dtype=float,
    )
    carried_supports = np.asarray(
        [
            row["geometry"].get("carried_object_support")
            if row["geometry"].get("carried_object_support") is not None
            else np.nan
            for row in records
        ],
        dtype=float,
    )
    carried_obstacle_supports = np.asarray(
        [
            row["geometry"].get("carried_object_obstacle_support")
            if row["geometry"].get("carried_object_obstacle_support") is not None
            else np.nan
            for row in records
        ],
        dtype=float,
    )
    carried_box_size_value = payload.get("carried_object_box_half_extents_m")
    carried_box_size = (
        None
        if carried_box_size_value is None
        else np.asarray(carried_box_size_value, dtype=float)
    )
    carried_object_prompt = payload.get("carried_object_prompt", "carried object")
    radii = np.asarray(payload["ellipsoid_radii_m"], dtype=float)
    obstacle_size = np.asarray(primitive["size"], dtype=float)
    safe_distance = float(surface_gaps[0] - barriers[0])
    obstacle_displacements = np.sum(np.abs(obstacle_centers - obstacle_centers[0]), axis=1)
    displaced = obstacle_displacements > 0.001

    source_summary = None
    source_barriers = None
    barrier_replay_error = None
    if source_summary_path is not None and source_summary_path.is_file():
        source_summary = json.loads(source_summary_path.read_text())
        source_barriers = np.asarray(source_summary["actual_per_action_barriers_m"], dtype=float)
        compared = min(len(source_barriers), len(barriers))
        barrier_replay_error = barriers[:compared] - source_barriers[:compared]

    video_info = _ffprobe(original_video)
    video_frames = int(video_info.get("nb_frames", len(records)))
    frame_count = min(len(records), video_frames)
    frame_indices = np.arange(0, frame_count, frame_stride, dtype=int)
    output_fps = 30.0 / frame_stride

    if primitive["kind"] == "ellipsoid":
        all_obstacle_vertices = np.concatenate(
            [
                np.vstack(
                    (
                        c - np.sqrt(np.diag(r @ np.diag(np.square(obstacle_size)) @ r.T)),
                        c + np.sqrt(np.diag(r @ np.diag(np.square(obstacle_size)) @ r.T)),
                    )
                )
                for c, r in zip(obstacle_centers, obstacle_rotations)
            ]
        )
    else:
        all_obstacle_vertices = np.concatenate(
            [_box_vertices(c, r, obstacle_size) for c, r in zip(obstacle_centers, obstacle_rotations)]
        )
    scene_points = np.concatenate([centers[:frame_count], all_obstacle_vertices], axis=0)
    if actual_mujoco_moka:
        actual_vertices = np.concatenate([
            _box_vertices(obstacle_centers[0] + obstacle_rotations[0] @ np.asarray(local_center), obstacle_rotations[0], np.asarray(size))
            for local_center, size in MOKA_POT_COLLISION_BOXES
        ])
        scene_points = np.concatenate([scene_points, actual_vertices], axis=0)
    if carried_box_size is not None and np.any(carried_active[:frame_count]):
        active_carried_centers = carried_centers[:frame_count][carried_active[:frame_count]]
        carried_vertices = np.concatenate(
            [
                _box_vertices(center, np.eye(3), carried_box_size)
                for center in active_carried_centers
            ]
        )
        scene_points = np.concatenate([scene_points, carried_vertices], axis=0)
    scene_min = scene_points.min(axis=0) - np.array([0.09, 0.09, 0.09])
    scene_max = scene_points.max(axis=0) + np.array([0.09, 0.09, 0.09])
    scene_span = scene_max - scene_min

    fig = plt.figure(figsize=(6.4, 6.4), dpi=80, facecolor="#faf9f6")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.89, bottom=0.115)

    def draw(render_index):
        index = int(frame_indices[render_index])
        ax.clear()
        fig.legends.clear()
        center = centers[index]
        obstacle_center = obstacle_centers[index]
        barrier = float(barriers[index])
        surface_gap = float(surface_gaps[index])
        active_component = active_components[index]
        if active_component == "carried_object_box" and carried_active[index]:
            direction = carried_directions[index]
            first_support = carried_centers[index] + direction * carried_supports[index]
            second_support = obstacle_center - direction * carried_obstacle_supports[index]
        else:
            direction = active_directions[index]
            first_support = center + direction * gripper_supports[index]
            second_support = obstacle_center - direction * obstacle_supports[index]
        geometry_safe = barrier >= 0.0
        moved = bool(displaced[index])
        gripper_color = "#1565c0" if geometry_safe else "#d32f2f"
        obstacle_color = "#7b1fa2" if moved else "#ef6c00"

        if primitive["kind"] == "ellipsoid":
            _draw_ellipsoid(
                ax, obstacle_centers[0], obstacle_rotations[0], obstacle_size,
                "#90a4ae",
            )
        else:
            _draw_box(
                ax, obstacle_centers[0], obstacle_rotations[0], obstacle_size,
                "#90a4ae", alpha=0.035, linewidth=0.8,
            )
        if actual_mujoco_moka:
            for local_center, box_size in MOKA_POT_COLLISION_BOXES:
                _draw_box(
                    ax,
                    obstacle_center + obstacle_rotations[index] @ np.asarray(local_center),
                    obstacle_rotations[index],
                    np.asarray(box_size),
                    "#1565c0",
                    alpha=0.16,
                    linewidth=0.8,
                )
        if safe_distance > 0.0 and primitive["kind"] != "ellipsoid":
            _draw_box(
                ax, obstacle_center, obstacle_rotations[index], obstacle_size + safe_distance,
                "#ffb74d", alpha=0.025, linewidth=0.65,
            )
        if primitive["kind"] == "ellipsoid":
            _draw_ellipsoid(
                ax, obstacle_center, obstacle_rotations[index], obstacle_size,
                obstacle_color,
            )
        else:
            _draw_box(
                ax, obstacle_center, obstacle_rotations[index], obstacle_size,
                obstacle_color, alpha=0.23, linewidth=1.8,
            )
        _draw_ellipsoid(ax, center, rotations[index], radii, gripper_color)
        if carried_active[index] and carried_box_size is not None:
            _draw_box(
                ax,
                carried_centers[index],
                np.eye(3),
                carried_box_size,
                "#00897b",
                alpha=0.25,
                linewidth=1.8,
            )
        ax.scatter(*center, color=gripper_color, s=24)
        ax.scatter(*obstacle_center, color=obstacle_color, marker="x", s=38)

        tail_start = max(0, index - 50)
        if index > tail_start:
            ax.plot(*centers[tail_start:index + 1].T, color="#3949ab", linewidth=2.0, alpha=0.72)
            ax.plot(
                *obstacle_centers[tail_start:index + 1].T,
                color="#7b1fa2", linewidth=1.6, alpha=0.72,
            )
        ax.plot(
            [first_support[0], second_support[0]],
            [first_support[1], second_support[1]],
            [first_support[2], second_support[2]],
            color="#2e7d32" if surface_gap >= 0.0 else "#c62828",
            linewidth=3.0,
        )
        ax.scatter(*first_support, color="#212121", s=13)
        ax.scatter(*second_support, color="#212121", s=13)

        ax.set_xlim(scene_min[0], scene_max[0])
        ax.set_ylim(scene_min[1], scene_max[1])
        ax.set_zlim(scene_min[2], scene_max[2])
        ax.set_box_aspect(scene_span)
        ax.view_init(elev=28, azim=48)
        ax.set_xlabel("world x (m)", labelpad=2)
        ax.set_ylabel("world y (m)", labelpad=2)
        ax.set_zlabel("world z (m)", labelpad=2)
        ax.grid(True, alpha=0.22)
        source_barrier = (
            float(source_barriers[index])
            if source_barriers is not None and index < len(source_barriers)
            else None
        )
        source_safe = source_barrier is None or source_barrier >= 0.0
        geometry_status = "SOURCE DCBF SAFE" if source_safe else "SOURCE DCBF VIOLATION"
        motion_status = "OBSTACLE DISPLACED" if moved else "obstacle stable"
        barrier_text = (
            f"source h={source_barrier * 1000:+.1f} mm · {geometry_status}"
            if source_barrier is not None
            else f"replay h={barrier * 1000:+.1f} mm · {geometry_status}"
        )
        title = (
            f"{case_label} · executed geometry\n"
            f"step {steps[index]:03d} · replay gap={surface_gap * 1000:+.1f} mm · "
            f"{barrier_text}\n"
            f"obstacle L1 displacement={obstacle_displacements[index] * 1000:.2f} mm · "
            f"{motion_status} · active={active_component}"
        )
        ax.set_title(
            title,
            color="#b71c1c" if (not source_safe or moved) else "#1b5e20",
            fontsize=10.5,
            fontweight="bold",
            pad=4,
        )
        fig.legend(
            handles=[
                Patch(facecolor="#ef6c00", alpha=0.35, label=(
                    f"fitted obstacle {primitive['kind'].upper()}: "
                    + " × ".join(f"{2*x*100:.1f}" for x in obstacle_size)
                    + " cm"
                )),
                Line2D([0], [0], color=gripper_color, label=(
                    "gripper ellipsoid radii: "
                    + " × ".join(f"{x*100:.1f}" for x in radii)
                    + " cm"
                )),
                Line2D([0], [0], color="#2e7d32", linewidth=3, label="modeled closest surface gap"),
                Line2D([0], [0], color="#7b1fa2", label="obstacle trajectory (>1 mm = collision)"),
                Line2D([0], [0], color="#90a4ae", label="initial obstacle pose"),
                Patch(
                    facecolor="#00897b",
                    alpha=0.35,
                    label=(
                        f"attached {carried_object_prompt} AABB"
                        if carried_box_size is None
                        else f"attached {carried_object_prompt} AABB: "
                        + " × ".join(f"{2*x*100:.1f}" for x in carried_box_size)
                        + " cm"
                    ),
                ),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2,
            fontsize=7.5, frameon=False,
        )
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    geometry_video = output.with_name(output.stem + "_3d_only.mp4")
    movie = animation.FuncAnimation(
        fig, draw, frames=len(frame_indices), interval=1000.0 / output_fps, blit=False,
    )
    movie.save(
        geometry_video,
        writer=animation.FFMpegWriter(
            fps=output_fps, codec="libx264", bitrate=2600,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
    )
    plt.close(fig)

    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(original_video), "-i", str(geometry_video),
            "-filter_complex",
            f"[0:v]fps={output_fps},scale=512:512:force_original_aspect_ratio=decrease,"
            "pad=512:512:(ow-iw)/2:(oh-ih)/2[left];"
            "[1:v]scale=512:512[right];[left][right]hstack=inputs=2[out]",
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-movflags", "+faststart", "-shortest", str(output),
        ],
        check=True,
    )

    negative = np.flatnonzero(barriers[:frame_count] < 0.0)
    motion = np.flatnonzero(displaced[:frame_count])
    report = {
        "case": case_label,
        "source_video": str(original_video.resolve()),
        "geometry_replay": str(debug_path.resolve()),
        "source_frames": video_frames,
        "geometry_frames": int(len(records)),
        "synchronized_frames": int(frame_count),
        "gripper_ellipsoid_radii_m": radii.tolist(),
        "obstacle_primitive": primitive,
        "actual_mujoco_moka_boxes_drawn": bool(actual_mujoco_moka),
        "safe_distance_m": safe_distance,
        "minimum_surface_gap_m": float(np.min(surface_gaps[:frame_count])),
        "minimum_barrier_m": float(np.min(barriers[:frame_count])),
        "negative_barrier_frames": int(len(negative)),
        "first_negative_barrier_step": int(steps[negative[0]]) if len(negative) else None,
        "maximum_obstacle_l1_displacement_m": float(np.max(obstacle_displacements[:frame_count])),
        "first_obstacle_displacement_step": int(steps[motion[0]]) if len(motion) else None,
        "source_summary_success": source_summary.get("success") if source_summary else None,
        "source_summary_collision": source_summary.get("collision") if source_summary else None,
        "source_summary_episode_steps": source_summary.get("episode_steps") if source_summary else None,
        "replay_barrier_comparison_frames": int(len(barrier_replay_error)) if barrier_replay_error is not None else 0,
        "replay_barrier_mae_m": float(np.mean(np.abs(barrier_replay_error))) if barrier_replay_error is not None else None,
        "replay_barrier_max_abs_error_m": float(np.max(np.abs(barrier_replay_error))) if barrier_replay_error is not None else None,
        "compound_carried_object_enabled": bool(
            payload.get("compound_carried_object_enabled", False)
        ),
        "carried_object_activation_step": payload.get(
            "carried_object_activation_step"
        ),
        "carried_object_box_full_dimensions_m": (
            None
            if carried_box_size is None
            else (2.0 * carried_box_size).tolist()
        ),
        "carried_object_active_frames": int(np.sum(carried_active[:frame_count])),
        "carried_object_limiting_frames": int(
            sum(
                component == "carried_object_box"
                for component in active_components[:frame_count]
            )
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-json", type=Path, required=True)
    parser.add_argument("--primitive-json", type=Path, required=True)
    parser.add_argument("--original-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--actual-mujoco-moka", action="store_true")
    args = parser.parse_args()
    if args.frame_stride < 1 or 30 % args.frame_stride:
        parser.error("--frame-stride must be a positive divisor of 30")
    report = render(
        args.debug_json,
        args.primitive_json,
        args.original_video,
        args.output,
        args.frame_stride,
        args.case_label,
        args.source_summary,
        args.actual_mujoco_moka,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
