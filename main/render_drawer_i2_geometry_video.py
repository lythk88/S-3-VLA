"""Render the drawer goal I/2 gripper trajectory beside its real milk geometry."""

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


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CASE = (
    ROOT
    / "results/smcbf_joint_dcbf_l2_first10/safelibero_goal"
    / "open_the_top_drawer_and_put_the_bowl_inside/smcbf_joint_dcbf_l2_first10_I"
)

# Ground-truth geometry extracted from the compiled MuJoCo model after the
# standard 20 settling actions for safelibero_goal, level I, episode 2, seed 7.
MILK_VISUAL_CENTER = np.array([-0.1401012578, 0.1694603357, 1.0394413649])
MILK_VISUAL_HALF_SIZE = 0.5 * np.array([0.1082318964, 0.2855367033, 0.1098932098])
MILK_COLLISION_CENTER = np.array([-0.1398309037, 0.1699999451, 1.0311609546])
MILK_COLLISION_HALF_SIZE = 0.5 * np.array([0.10614, 0.26238, 0.105])
MILK_ROTATION = np.array(
    [
        [1.0, -3.1651310148e-7, -9.8412673415e-10],
        [-9.8413971554e-10, -4.1013776708e-8, -1.0],
        [3.1651310144e-7, 1.0, -4.1013776930e-8],
    ]
)
BOWL_CENTER = np.array([-5.0855687003, -0.0012304728, -0.0015929249])
CABINET_CENTER = np.array([0.021, -0.255, 1.015])
CABINET_HALF_SIZE = np.array([0.125, 0.105, 0.111])
GRIPPER_RADII = np.array([0.06, 0.12, 0.11])


def _vertices(center, rotation, half_size):
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=float,
    )
    return center + (signs * half_size) @ rotation.T


EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)
FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _draw_box(ax, center, rotation, half_size, color, *, alpha=0.08, linewidth=1.5):
    verts = _vertices(center, rotation, half_size)
    ax.add_collection3d(
        Poly3DCollection(
            [[verts[i] for i in face] for face in FACES],
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )
    for first, second in EDGES:
        segment = verts[[first, second]]
        ax.plot(*segment.T, color=color, linewidth=linewidth)


def _draw_ellipsoid(ax, center, rotation, color):
    u = np.linspace(0.0, 2.0 * np.pi, 16)
    v = np.linspace(0.0, np.pi, 9)
    local = np.stack(
        [
            GRIPPER_RADII[0] * np.outer(np.cos(u), np.sin(v)),
            GRIPPER_RADII[1] * np.outer(np.sin(u), np.sin(v)),
            GRIPPER_RADII[2] * np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    world = local @ rotation.T + center
    ax.plot_wireframe(
        world[..., 0], world[..., 1], world[..., 2],
        rstride=2, cstride=2, color=color, linewidth=0.8, alpha=0.85,
    )


def _load_trajectory(debug_path: Path):
    payload = json.loads(debug_path.read_text())
    records = []
    fitted_by_step = {}
    for chunk in payload["chunks"]:
        chunk_step = int(chunk["chunk_start_step"])
        fitted_by_step[chunk_step] = (
            np.asarray(chunk["initial_obstacle_center"], dtype=float),
            np.asarray(chunk["initial_obstacle_rotation"], dtype=float),
        )
        for row in chunk["executed"]:
            records.append(row)
    records.sort(key=lambda row: int(row["action_step"]))
    if not records:
        raise ValueError(f"No executed geometry records in {debug_path}")

    centers = np.asarray([row["actual_ellipsoid_center"] for row in records], dtype=float)
    rotations = np.asarray([row["actual_eef_rotation"] for row in records], dtype=float)
    barriers = np.asarray([row["actual_barrier_m"] for row in records], dtype=float)
    steps = np.asarray([row["action_step"] for row in records], dtype=int)
    return payload, steps, centers, rotations, barriers, fitted_by_step


def render(
    debug_path: Path,
    fitted_path: Path,
    original_video: Path,
    output: Path,
    frame_stride: int,
    case_label: str,
    milk_visual_center: np.ndarray,
    milk_collision_center: np.ndarray,
    milk_rotation: np.ndarray,
    bowl_center: np.ndarray,
    obstacle_label: str,
    visual_half_size: np.ndarray,
    collision_half_size: np.ndarray,
    target_structure_label: str,
    target_structure_center: np.ndarray,
    target_structure_half_size: np.ndarray,
):
    payload, steps, centers, rotations, barriers, fitted_by_step = _load_trajectory(debug_path)
    fitted = json.loads(fitted_path.read_text())
    fitted_size = np.asarray(fitted["size"], dtype=float)
    bowl_in_workspace = bool(
        -0.5 < bowl_center[0] < 0.5
        and -0.5 < bowl_center[1] < 0.5
        and bowl_center[2] > 0.5
    )

    fig = plt.figure(figsize=(6.4, 6.4), dpi=80, facecolor="#faf9f6")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.91, bottom=0.10)

    frame_indices = np.arange(0, len(steps), frame_stride, dtype=int)
    output_fps = 30.0 / frame_stride
    scene_points = [centers, _vertices(milk_visual_center, milk_rotation, visual_half_size)]
    scene_points.append(_vertices(milk_collision_center, milk_rotation, collision_half_size))
    scene_points.append(_vertices(target_structure_center, np.eye(3), target_structure_half_size))
    scene_points.extend(
        _vertices(center, rotation, fitted_size)[None, ...]
        for center, rotation in fitted_by_step.values()
    )
    if bowl_in_workspace:
        scene_points.append(bowl_center[None, :])
    scene_points = np.concatenate([np.reshape(points, (-1, 3)) for points in scene_points])
    scene_min = scene_points.min(axis=0) - np.array([0.08, 0.08, 0.08])
    scene_max = scene_points.max(axis=0) + np.array([0.08, 0.08, 0.08])
    scene_span = scene_max - scene_min

    def update(render_index):
        frame_index = int(frame_indices[render_index])
        ax.clear()
        fig.legends.clear()
        step = int(steps[frame_index])
        center = centers[frame_index]
        rotation = rotations[frame_index]
        barrier = float(barriers[frame_index])
        chunk_step = max(key for key in fitted_by_step if key <= step)
        fitted_center, fitted_rotation = fitted_by_step[chunk_step]

        _draw_box(ax, target_structure_center, np.eye(3), target_structure_half_size, "#8d5a2b", alpha=0.18)
        _draw_box(ax, milk_visual_center, milk_rotation, visual_half_size, "#00897b", alpha=0.10, linewidth=2.1)
        _draw_box(ax, milk_collision_center, milk_rotation, collision_half_size, "#2e7d32", alpha=0.04, linewidth=1.0)
        _draw_box(ax, fitted_center, fitted_rotation, fitted_size, "#f4511e", alpha=0.11, linewidth=1.8)
        if bowl_in_workspace:
            ax.scatter(*bowl_center, color="#111111", s=38, marker="o")

        tail_start = max(0, frame_index - 45)
        tail = centers[tail_start : frame_index + 1]
        if len(tail) > 1:
            ax.plot(*tail.T, color="#3949ab", linewidth=2.2, alpha=0.75)
        gripper_color = "#1565c0" if barrier >= 0.0 else "#d32f2f"
        _draw_ellipsoid(ax, center, rotation, gripper_color)
        ax.scatter(*center, color=gripper_color, s=24)

        ax.set_xlim(*scene_min[[0]], *scene_max[[0]])
        ax.set_ylim(*scene_min[[1]], *scene_max[[1]])
        ax.set_zlim(*scene_min[[2]], *scene_max[[2]])
        ax.set_box_aspect(scene_span)
        ax.view_init(elev=28, azim=48)
        ax.set_xlabel("world x (m)", labelpad=2)
        ax.set_ylabel("world y (m)", labelpad=2)
        ax.set_zlabel("world z (m)", labelpad=2)
        ax.grid(True, alpha=0.22)
        status = "SAFE" if barrier >= 0.0 else "DCBF VIOLATION"
        bowl_status = (
            f"target bowl = [{bowl_center[0]:+.3f}, {bowl_center[1]:+.3f}, {bowl_center[2]:+.3f}] m"
            if bowl_in_workspace
            else f"target bowl is outside workspace: x = {bowl_center[0]:+.3f} m"
        )
        ax.set_title(
            f"{case_label} · actual executed gripper geometry\n"
            f"step {step:03d} · h = {barrier * 1000:+.1f} mm · {status}\n"
            f"{bowl_status}",
            color="#1b5e20" if barrier >= 0.0 else "#b71c1c",
            fontsize=11,
            fontweight="bold",
            pad=5,
        )
        fig.legend(
            handles=[
                Patch(facecolor="#f4511e", alpha=0.25, label=f"pipeline fitted OBB: {' × '.join(f'{2*x*100:.1f}' for x in fitted_size)} cm"),
                Patch(facecolor="#00897b", alpha=0.20, label=f"real {obstacle_label} visual bounds: {' × '.join(f'{2*x*100:.1f}' for x in visual_half_size)} cm"),
                Line2D([0], [0], color="#2e7d32", label=f"real {obstacle_label} collision bounds: {' × '.join(f'{2*x*100:.1f}' for x in collision_half_size)} cm"),
                Line2D([0], [0], color=gripper_color, label="gripper ellipsoid: radii 6 × 12 × 11 cm"),
                Patch(facecolor="#8d5a2b", alpha=0.22, label=target_structure_label),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            fontsize=7.5,
            frameon=False,
        )
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    geometry_video = output.with_name(output.stem + "_3d_only.mp4")
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / output_fps,
        blit=False,
    )
    movie.save(
        geometry_video,
        writer=animation.FFMpegWriter(
            fps=output_fps, codec="libx264", bitrate=2500,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
    )
    plt.close(fig)

    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(original_video), "-i", str(geometry_video),
            "-filter_complex",
            f"[0:v]fps={output_fps},scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2[left];"
            "[1:v]scale=512:512[right];[left][right]hstack=inputs=2[out]",
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-movflags", "+faststart", "-shortest", str(output),
        ],
        check=True,
    )

    report = {
        "case": case_label,
        "original_video": str(original_video.resolve()),
        "debug_geometry": str(debug_path.resolve()),
        "source_geometry_frames": int(len(steps)),
        "rendered_frames": int(len(frame_indices)),
        "output_fps": output_fps,
        "pipeline_fitted_obb_full_dimensions_m": (2.0 * fitted_size).tolist(),
        "pipeline_fitted_obb_volume_m3": float(8.0 * np.prod(fitted_size)),
        "obstacle_label": obstacle_label,
        "real_visual_bounds_full_dimensions_m": (2.0 * visual_half_size).tolist(),
        "real_visual_bounds_volume_m3": float(8.0 * np.prod(visual_half_size)),
        "real_collision_bounds_full_dimensions_m": (2.0 * collision_half_size).tolist(),
        "real_collision_bounds_volume_m3": float(8.0 * np.prod(collision_half_size)),
        "fit_to_visual_bounding_volume_ratio": float(np.prod(fitted_size) / np.prod(visual_half_size)),
        "minimum_executed_pipeline_barrier_m": float(np.min(barriers)),
        "fraction_executed_pipeline_barrier_negative": float(np.mean(barriers < 0.0)),
        "target_bowl_center_m": bowl_center.tolist(),
        "target_bowl_in_workspace": bowl_in_workspace,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-json", type=Path, required=True)
    parser.add_argument("--fitted-json", type=Path, required=True)
    parser.add_argument("--original-video", type=Path, default=ORIGINAL_CASE / "2_failure_safe_backview.mp4")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/analysis/drawer_goal_I2_backview_plus_real_3d_geometry.mp4",
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--case-label", default="Drawer goal I/2")
    parser.add_argument(
        "--milk-visual-center", type=float, nargs=3,
        default=MILK_VISUAL_CENTER.tolist(), metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--milk-collision-center", type=float, nargs=3,
        default=MILK_COLLISION_CENTER.tolist(), metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--milk-rotation", type=float, nargs=9,
        default=MILK_ROTATION.reshape(-1).tolist(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument(
        "--bowl-center", type=float, nargs=3,
        default=BOWL_CENTER.tolist(), metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--obstacle-label", default="milk carton")
    parser.add_argument(
        "--real-visual-half-size", type=float, nargs=3,
        default=MILK_VISUAL_HALF_SIZE.tolist(), metavar=("HX", "HY", "HZ"),
    )
    parser.add_argument(
        "--real-collision-half-size", type=float, nargs=3,
        default=MILK_COLLISION_HALF_SIZE.tolist(), metavar=("HX", "HY", "HZ"),
    )
    parser.add_argument("--target-structure-label", default="cabinet")
    parser.add_argument(
        "--target-structure-center", type=float, nargs=3,
        default=CABINET_CENTER.tolist(), metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--target-structure-half-size", type=float, nargs=3,
        default=CABINET_HALF_SIZE.tolist(), metavar=("HX", "HY", "HZ"),
    )
    args = parser.parse_args()
    if args.frame_stride < 1 or 30 % args.frame_stride:
        parser.error("--frame-stride must be a positive divisor of 30")
    render(
        args.debug_json,
        args.fitted_json,
        args.original_video,
        args.output,
        args.frame_stride,
        args.case_label,
        np.asarray(args.milk_visual_center, dtype=float),
        np.asarray(args.milk_collision_center, dtype=float),
        np.asarray(args.milk_rotation, dtype=float).reshape(3, 3),
        np.asarray(args.bowl_center, dtype=float),
        args.obstacle_label,
        np.asarray(args.real_visual_half_size, dtype=float),
        np.asarray(args.real_collision_half_size, dtype=float),
        args.target_structure_label,
        np.asarray(args.target_structure_center, dtype=float),
        np.asarray(args.target_structure_half_size, dtype=float),
    )


if __name__ == "__main__":
    main()
