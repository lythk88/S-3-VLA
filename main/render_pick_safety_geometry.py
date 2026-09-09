"""Render a two-view 3D diagnostic of the pick-path safety geometry.

The default scene state is the stabilized initial state from SafeLIBERO goal,
level I, episode 2 (seed 7), paired with that episode's saved obstacle fit.
The animation evaluates every displayed pose with the same support-function
gap used by the action-expert DCBF implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation, Slerp

from openpi.policies.action_expert_qp import Ellipsoid, ObstaclePrimitive
from openpi.policies.action_expert_qp import ellipsoid_obstacle_gap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_DIR = (
    ROOT
    / "results/smcbf_joint_dcbf_l2_first10/safelibero_goal"
    / "put_the_bowl_on_top_of_the_cabinet/smcbf_joint_dcbf_l2_first10_I/2"
)

# Stabilized simulator state after the same 20 dummy actions used by main_aegis.py.
EEF_POSITION = np.array([-0.2046407977, 0.0097062211, 1.1852918068])
EEF_QUATERNION_XYZW = np.array([0.9995822197, 0.0005104206, -0.0288984059, 0.0000872624])
BOWL_CENTER = np.array([-0.0798554073, -0.0135798345, 0.8984041502])
CABINET_TOP_CENTER = np.array([0.0210660576, -0.2547002016, 1.12652])
CABINET_TOP_HALF_SIZE = np.array([0.12534, 0.09438, 0.00147])
CABINET_BASE_CENTER = np.array([0.021, -0.255, 1.015])
CABINET_BASE_HALF_SIZE = np.array([0.125, 0.105, 0.111])
ELLIPSOID_OFFSET_LOCAL = np.array([0.0, 0.0, -0.08])
ELLIPSOID_RADII = np.array([0.06, 0.12, 0.11])
SAFE_DISTANCE = 0.01


def _box_vertices(center: np.ndarray, rotation: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    signs = np.array(
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
        dtype=float,
    )
    return center + (signs * half_size) @ rotation.T


def _box_faces(vertices: np.ndarray) -> list[np.ndarray]:
    return [
        vertices[[0, 1, 2, 3]],
        vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]],
        vertices[[2, 3, 7, 6]],
        vertices[[1, 2, 6, 5]],
        vertices[[0, 3, 7, 4]],
    ]


def _draw_box(ax, center, rotation, half_size, *, color, alpha, edgecolor, linewidth=0.8):
    collection = Poly3DCollection(
        _box_faces(_box_vertices(center, rotation, half_size)),
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def _draw_ellipsoid(ax, center, rotation, radii, *, color, alpha=0.42):
    u = np.linspace(0.0, 2.0 * np.pi, 25)
    v = np.linspace(0.0, np.pi, 15)
    local = np.stack(
        [
            radii[0] * np.outer(np.cos(u), np.sin(v)),
            radii[1] * np.outer(np.sin(u), np.sin(v)),
            radii[2] * np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    world = local @ rotation.T + center
    ax.plot_surface(
        world[..., 0],
        world[..., 1],
        world[..., 2],
        color=color,
        alpha=alpha,
        linewidth=0.15,
        edgecolor=color,
        shade=True,
    )


def _draw_bowl(ax):
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    z = np.linspace(BOWL_CENTER[2] - 0.006, BOWL_CENTER[2] + 0.045, 10)
    tt, zz = np.meshgrid(theta, z)
    radius = 0.047 * (0.72 + 0.28 * (zz - z.min()) / (z.max() - z.min()))
    xx = BOWL_CENTER[0] + radius * np.cos(tt)
    yy = BOWL_CENTER[1] + radius * np.sin(tt)
    ax.plot_surface(xx, yy, zz, color="#161616", alpha=0.92, linewidth=0)


def _barrier(center, rotation, obstacle) -> float:
    gripper = Ellipsoid(center, rotation, ELLIPSOID_RADII)
    return float(ellipsoid_obstacle_gap(gripper, obstacle) - SAFE_DISTANCE)


def _best_yaw_at_target(target_center, initial_rotation, obstacle):
    yaws = np.linspace(-180.0, 180.0, 721)
    rotations = [Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ initial_rotation for yaw in yaws]
    barriers = np.array([_barrier(target_center, rot, obstacle) for rot in rotations])
    index = int(np.argmax(barriers))
    return float(yaws[index]), rotations[index], float(barriers[index])


def render(episode_dir: Path, output: Path, frames: int, fps: int) -> dict:
    primitive = json.loads((episode_dir / "obstacle_primitive.json").read_text())
    obstacle = ObstaclePrimitive(
        primitive["kind"],
        np.asarray(primitive["center"], dtype=float),
        np.asarray(primitive["rotation"], dtype=float),
        np.asarray(primitive["size"], dtype=float),
    )
    initial_rotation = Rotation.from_quat(EEF_QUATERNION_XYZW).as_matrix()
    initial_center = EEF_POSITION + initial_rotation @ ELLIPSOID_OFFSET_LOCAL
    # A representative pre-grasp pose: the ellipsoid center is 10 cm above the bowl center.
    grasp_center = BOWL_CENTER + np.array([0.0, 0.0, 0.10])
    yaw, feasible_rotation, feasible_barrier = _best_yaw_at_target(
        grasp_center, initial_rotation, obstacle
    )

    path_t = np.linspace(0.0, 1.0, 201)
    direct_path = initial_center[None] * (1.0 - path_t[:, None]) + grasp_center[None] * path_t[:, None]
    path_barriers = np.array([_barrier(center, initial_rotation, obstacle) for center in direct_path])
    blocked = np.flatnonzero(path_barriers < 0.0)
    crossing_fraction = float(path_t[blocked[0]]) if len(blocked) else None

    # A conservative center-space envelope for the fixed-orientation ellipsoid.
    obstacle_axes = obstacle.rotation.T
    gripper_support = np.array(
        [np.linalg.norm(ELLIPSOID_RADII * (initial_rotation.T @ axis)) for axis in obstacle_axes]
    )
    forbidden_half_size = obstacle.size + gripper_support + SAFE_DISTANCE

    key_rots = Rotation.from_matrix(np.stack([initial_rotation, feasible_rotation]))
    rotation_slerp = Slerp([0.0, 1.0], key_rots)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f7f7f5")
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
    fig.subplots_adjust(left=0.015, right=0.985, top=0.85, bottom=0.08, wspace=0.015)

    def state_for_frame(frame):
        phase = frame / max(frames - 1, 1)
        if phase < 0.12:
            return initial_center, initial_rotation, "Initial geometry"
        if phase < 0.60:
            progress = (phase - 0.12) / 0.48
            return (1.0 - progress) * initial_center + progress * grasp_center, initial_rotation, "Direct pick approach · wrist fixed"
        if phase < 0.73:
            return grasp_center, initial_rotation, "Requested grasp pose · DCBF infeasible"
        if phase < 0.93:
            progress = (phase - 0.73) / 0.20
            return grasp_center, rotation_slerp([progress]).as_matrix()[0], "Diagnostic only · rotate wrist at target"
        return grasp_center, feasible_rotation, "Reoriented grasp geometry · feasible"

    def draw(frame):
        center, gripper_rotation, phase_label = state_for_frame(frame)
        barrier = _barrier(center, gripper_rotation, obstacle)
        is_safe = barrier >= 0.0
        gripper_color = "#1976d2" if is_safe else "#e53935"

        for index, ax in enumerate(axes):
            ax.clear()
            ax.set_facecolor("#f7f7f5")
            _draw_box(
                ax,
                CABINET_BASE_CENTER,
                np.eye(3),
                CABINET_BASE_HALF_SIZE,
                color="#8d5a2b",
                alpha=0.22,
                edgecolor="#6d3f18",
            )
            _draw_box(
                ax,
                CABINET_TOP_CENTER,
                np.eye(3),
                CABINET_TOP_HALF_SIZE,
                color="#b97837",
                alpha=0.55,
                edgecolor="#6d3f18",
            )
            _draw_bowl(ax)
            _draw_box(
                ax,
                obstacle.center,
                obstacle.rotation,
                obstacle.size,
                color="#ff7043",
                alpha=0.62,
                edgecolor="#a32600",
                linewidth=1.3,
            )
            _draw_box(
                ax,
                obstacle.center,
                obstacle.rotation,
                forbidden_half_size,
                color="#ef5350",
                alpha=0.075,
                edgecolor="#d32f2f",
                linewidth=0.8,
            )
            _draw_ellipsoid(ax, center, gripper_rotation, ELLIPSOID_RADII, color=gripper_color)

            safe_segments = path_barriers[:-1] >= 0.0
            for j in range(len(direct_path) - 1):
                ax.plot(
                    direct_path[j : j + 2, 0],
                    direct_path[j : j + 2, 1],
                    direct_path[j : j + 2, 2],
                    color="#2e7d32" if safe_segments[j] else "#d32f2f",
                    linewidth=2.4,
                    alpha=0.8,
                )
            ax.scatter(*BOWL_CENTER, color="black", s=32)
            ax.scatter(*grasp_center, color="#6a1b9a", marker="*", s=85)
            ax.scatter(*obstacle.center, color="#a32600", marker="x", s=48)
            ax.set_xlim(-0.32, 0.24)
            ax.set_ylim(-0.41, 0.35)
            ax.set_zlim(0.84, 1.38)
            ax.set_box_aspect((0.56, 0.76, 0.54))
            ax.set_xlabel("world x (m)", labelpad=4)
            ax.set_ylabel("world y (m)", labelpad=4)
            ax.set_zlabel("world z (m)", labelpad=4)
            ax.grid(True, alpha=0.25)
            if index == 0:
                ax.view_init(elev=26, azim=44)
                ax.set_title("Perspective", fontsize=12, pad=4)
            else:
                ax.view_init(elev=88, azim=-90)
                ax.set_title("Top-down", fontsize=12, pad=4)

        status = "SAFE / QP permits" if is_safe else "BLOCKED / QP must correct"
        fig.suptitle(
            "Goal I · episode 2: can the gripper safety ellipsoid reach the bowl?\n"
            f"{phase_label}   |   barrier h = {barrier * 1000:+.1f} mm   |   {status}",
            fontsize=15,
            fontweight="bold",
            color="#1b5e20" if is_safe else "#b71c1c",
            y=0.965,
        )
        fig.legend(
            handles=[
                Patch(facecolor="#1976d2", alpha=0.45, label="gripper ellipsoid (safe)"),
                Patch(facecolor="#e53935", alpha=0.45, label="gripper ellipsoid (violating)"),
                Patch(facecolor="#ff7043", alpha=0.62, label="fitted active obstacle: milk carton OBB"),
                Patch(facecolor="#ef5350", alpha=0.10, label="fixed-wrist forbidden center envelope"),
                Patch(facecolor="#161616", alpha=0.92, label="target bowl"),
                Patch(facecolor="#8d5a2b", alpha=0.35, label="cabinet (not in shield)"),
            ],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.002),
            fontsize=9,
            frameon=False,
        )
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(output, writer=writer)
    plt.close(fig)

    report = {
        "case": "safelibero_goal level I episode 2",
        "source_episode_dir": str(episode_dir.resolve()),
        "active_obstacle": "milk_obstacle_1",
        "obstacle_primitive": primitive,
        "gripper_initial_center_m": initial_center.tolist(),
        "gripper_ellipsoid_radii_m": ELLIPSOID_RADII.tolist(),
        "target_bowl_center_m": BOWL_CENTER.tolist(),
        "representative_grasp_ellipsoid_center_m": grasp_center.tolist(),
        "safe_distance_m": SAFE_DISTANCE,
        "direct_path_first_blocked_fraction": crossing_fraction,
        "initial_barrier_m": _barrier(initial_center, initial_rotation, obstacle),
        "fixed_wrist_target_barrier_m": _barrier(grasp_center, initial_rotation, obstacle),
        "best_world_z_yaw_deg": yaw,
        "reoriented_target_barrier_m": feasible_barrier,
        "conclusion": (
            "The fixed-wrist gripper ellipsoid makes the direct pick pose infeasible. "
            "A wrist reorientation makes the same center feasible, but the evaluated pipeline "
            "zeros rotation during execution."
        ),
        "caveat": (
            "The grasp center is a representative target 0.10 m above the saved bowl center; "
            "the original run did not save the full XYZ action trajectory."
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/analysis/cabinet_goal_I2_pick_safety_geometry_2view.mp4",
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    report = render(args.episode_dir, args.output, args.frames, args.fps)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
