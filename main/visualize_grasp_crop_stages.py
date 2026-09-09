"""Visualize every mask stage used by grasp-conditioned RGB-D geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from grasp_rgbd import horizontal_support_plane_keep_mask, select_grasped_object_points


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


def _draw_box(ax, center, rotation, size, color, *, alpha=0.12, linewidth=1.2):
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
        ax.plot(*segment.T, color=color, linewidth=linewidth, alpha=0.9)


def _draw_ellipsoid(ax, center, rotation, radii, color="#6a1b9a"):
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 13)
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
        rstride=2, cstride=2, color=color, linewidth=0.55, alpha=0.5,
    )


def _executed_record(debug_payload: dict, action_step: int) -> dict:
    records = [row for chunk in debug_payload["chunks"] for row in chunk["executed"]]
    return next(row for row in records if int(row["action_step"]) == action_step)


def _subset_mask(points: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Map an exactly persisted float32 point subset back to the RGB-D cloud."""
    point_lookup: dict[tuple[float, float, float], int] = {
        tuple(point): index for index, point in enumerate(points)
    }
    mask = np.zeros(len(points), dtype=bool)
    for point in subset:
        index = point_lookup.get(tuple(point))
        if index is not None:
            mask[index] = True
    return mask


def _pixel_coordinates(point_count: int, image_shape: tuple[int, int], stride: int):
    height, width = image_shape
    sampled_height = len(range(0, height, stride))
    sampled_width = len(range(0, width, stride))
    if point_count != sampled_height * sampled_width:
        raise ValueError(
            "The RGB-D cloud contains invalid-depth filtering, so its flattened "
            "points cannot be mapped directly back to image pixels."
        )
    sampled_index = np.arange(point_count)
    internal_row = (sampled_index // sampled_width) * stride
    internal_column = (sampled_index % sampled_width) * stride
    # rgbd_to_world_point_cloud reverses the saved evaluator image before sampling.
    display_row = (height - 1) - internal_row
    display_column = (width - 1) - internal_column
    return display_column, display_row


def render(
    attempt_dir: Path,
    obstacle_primitive_path: Path,
    debug_path: Path,
    output: Path,
    task_suite_name: str,
    stride: int = 2,
    output_3d: Path | None = None,
    output_final_3d: Path | None = None,
) -> dict:
    image = np.asarray(Image.open(attempt_dir / "agentview_rgb.png").convert("RGB"))
    cloud = np.load(attempt_dir / "point_cloud_rgbd.npz")
    points = np.asarray(cloud["points"], dtype=np.float32)
    colors = np.asarray(cloud["colors"], dtype=np.uint8)
    estimation = json.loads((attempt_dir / "estimation.json").read_text())
    primitive = json.loads(obstacle_primitive_path.read_text())
    debug = json.loads(debug_path.read_text())
    action_step = int(estimation["step"])
    executed = _executed_record(debug, action_step)

    center = np.asarray(estimation["ellipsoid_center_m"], dtype=np.float64)
    rotation = np.asarray(estimation["ellipsoid_rotation"], dtype=np.float64)
    radii = np.asarray(estimation["ellipsoid_radii_m"], dtype=np.float64)
    points64 = points.astype(np.float64)
    world_down = np.array([0.0, 0.0, -1.0])
    vertical_support = float(np.linalg.norm(radii * (rotation.T @ world_down)))
    bottom_point = center + world_down * vertical_support
    relative = points64 - bottom_point
    horizontal_distance = np.linalg.norm(relative[:, :2], axis=1)

    masks: list[tuple[str, str, np.ndarray]] = []
    mask = np.all(np.isfinite(points64), axis=1)
    mask &= horizontal_distance <= 0.18
    masks.append(("1. Horizontal ROI", "distance to bottom XY <= 18 cm", mask.copy()))
    previous = mask.copy()
    mask &= (relative[:, 2] <= 0.035) & (relative[:, 2] >= -0.25)
    masks.append(("2. Vertical ROI", "bottom -25 cm <= Z <= bottom +3.5 cm", mask.copy()))
    support_keep, support_plane_z = horizontal_support_plane_keep_mask(points64, mask)
    mask &= support_keep
    plane_text = (
        "no dominant plane detected"
        if support_plane_z is None
        else f"remove only |Z - {support_plane_z*100:.2f} cm| <= 0.3 cm"
    )
    masks.append(("3. Support plane only", plane_text, mask.copy()))

    ellipsoid_local = (points64 - center) @ rotation
    ellipsoid_level = np.sum(np.square(ellipsoid_local / radii), axis=1)
    mask &= ellipsoid_level > 1.02**2
    masks.append(("4. Remove ellipsoid", "keep level > 1.02^2", mask.copy()))

    obstacle_center = np.asarray(executed["actual_obstacle_center"], dtype=np.float64)
    obstacle_rotation = np.asarray(executed["actual_obstacle_rotation"], dtype=np.float64)
    obstacle_local = (points64 - obstacle_center) @ obstacle_rotation
    inside_obstacle = np.all(
        np.abs(obstacle_local) <= np.asarray(primitive["size"], dtype=np.float64) + 0.004,
        axis=1,
    )
    mask &= ~inside_obstacle
    masks.append(("5. Remove obstacle", "outside fitted OBB +4 mm", mask.copy()))

    grasp_cloud = select_grasped_object_points(
        points,
        colors,
        ellipsoid_center=center,
        ellipsoid_rotation=rotation,
        ellipsoid_radii=radii,
        task_suite_name=task_suite_name,
        crop_radius_m=0.18,
        min_points=30,
        max_anchor_distance_m=0.18,
        obstacle_center=obstacle_center,
        obstacle_rotation=obstacle_rotation,
        obstacle_half_extents=np.asarray(primitive["size"], dtype=np.float64),
    )
    candidate_points = np.asarray(grasp_cloud.candidate_points, dtype=np.float32)
    selected_points = np.asarray(grasp_cloud.points, dtype=np.float32)
    candidate_mask = _subset_mask(points, candidate_points)
    selected_mask = _subset_mask(points, selected_points)
    lower = np.quantile(selected_points, 0.01, axis=0)
    upper = np.quantile(selected_points, 0.99, axis=0)
    fit_points = selected_points[
        np.all((selected_points >= lower) & (selected_points <= upper), axis=1)
    ]
    fit_minimum = np.min(fit_points, axis=0)
    fit_maximum = np.max(fit_points, axis=0)
    final_center = (fit_minimum + fit_maximum) / 2.0
    final_half_extents = (fit_maximum - fit_minimum) / 2.0 + 0.005
    final_dimensions = 2.0 * final_half_extents
    fit_mask = _subset_mask(points, fit_points)
    masks.extend(
        [
            ("6. Voxelize", "one point per 3 mm voxel", candidate_mask),
            ("7. DBSCAN component", "selected closest dense component", selected_mask),
            ("8. 1-99% trim", "points used for final AABB", fit_mask),
        ]
    )

    columns, rows = _pixel_coordinates(len(points), image.shape[:2], stride)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=130)
    previous_mask = np.ones(len(points), dtype=bool)
    counts = {}
    for axis, (title, condition, current_mask) in zip(axes.flat, masks):
        removed = previous_mask & ~current_mask
        axis.imshow(image)
        if np.any(removed):
            axis.scatter(
                columns[removed], rows[removed], s=1.0, c="#ef5350", alpha=0.16,
                linewidths=0, rasterized=True,
            )
        if np.any(current_mask):
            axis.scatter(
                columns[current_mask], rows[current_mask], s=4.0, c="#00e676", alpha=0.82,
                linewidths=0, rasterized=True,
            )
        axis.set_title(f"{title}: {int(current_mask.sum()):,} points", fontsize=13, weight="bold")
        axis.text(
            0.5, -0.035, condition, transform=axis.transAxes, ha="center", va="top",
            fontsize=10,
        )
        axis.set_axis_off()
        counts[title] = int(current_mask.sum())
        previous_mask = current_mask

    dimensions_cm = final_dimensions * 100.0
    fig.suptitle(
        f"Milk ep2 carried-object crop at rollout step {action_step}\n"
        f"green = retained, red = removed at this stage | final AABB = "
        f"{dimensions_cm[0]:.2f} x {dimensions_cm[1]:.2f} x {dimensions_cm[2]:.2f} cm",
        fontsize=17,
        weight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.04, wspace=0.025, hspace=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    if output_3d is not None:
        fig_3d = plt.figure(figsize=(22, 12), dpi=130, facecolor="#fafafa")
        previous_mask = np.ones(len(points), dtype=bool)
        x_limits = (float(bottom_point[0] - 0.19), float(bottom_point[0] + 0.19))
        y_limits = (float(bottom_point[1] - 0.19), float(bottom_point[1] + 0.19))
        z_limits = (
            float(max(0.0, np.min(points64[masks[0][2], 2]) - 0.01)),
            float(max(center[2] + radii.max() + 0.03, 0.32)),
        )
        for plot_index, (title, condition, current_mask) in enumerate(masks, start=1):
            axis = fig_3d.add_subplot(2, 4, plot_index, projection="3d")
            removed = previous_mask & ~current_mask
            retained_points = points64[current_mask]
            removed_points = points64[removed]
            if len(removed_points):
                # Downsample only the rejected context so retained geometry stays crisp.
                rejected_stride = max(1, len(removed_points) // 12000)
                rejected = removed_points[::rejected_stride]
                axis.scatter(
                    rejected[:, 0], rejected[:, 1], rejected[:, 2],
                    s=0.6, c="#ef5350", alpha=0.09, depthshade=False,
                )
            if len(retained_points):
                point_size = 0.7 if len(retained_points) > 5000 else 4.0
                axis.scatter(
                    retained_points[:, 0], retained_points[:, 1], retained_points[:, 2],
                    s=point_size, c="#00897b", alpha=0.9, depthshade=False,
                )
            _draw_ellipsoid(axis, center, rotation, radii)
            axis.scatter(*bottom_point, c="#212121", marker="x", s=28)
            if plot_index >= 5:
                _draw_box(
                    axis, obstacle_center, obstacle_rotation,
                    np.asarray(primitive["size"], dtype=np.float64), "#ef6c00",
                    alpha=0.035, linewidth=0.65,
                )
            if plot_index == len(masks):
                _draw_box(
                    axis, final_center, np.eye(3), final_half_extents, "#1565c0",
                    alpha=0.22, linewidth=2.0,
                )
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_zlim(*z_limits)
            axis.set_box_aspect(
                (x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0])
            )
            axis.view_init(elev=24, azim=-54)
            axis.set_title(f"{title}\n{int(current_mask.sum()):,} retained", fontsize=12, weight="bold")
            axis.text2D(0.5, -0.04, condition, transform=axis.transAxes, ha="center", fontsize=9)
            axis.set_xlabel("world X (m)", fontsize=8, labelpad=0)
            axis.set_ylabel("world Y (m)", fontsize=8, labelpad=0)
            axis.set_zlabel("world Z (m)", fontsize=8, labelpad=0)
            axis.tick_params(labelsize=7, pad=0)
            axis.grid(True, alpha=0.18)
            previous_mask = current_mask
        fig_3d.suptitle(
            f"Milk ep2 RGB-D point-cloud filtering at rollout step {action_step}\n"
            "teal = retained | red = removed at current stage | purple = gripper ellipsoid | "
            "blue = final carried-object AABB",
            fontsize=17,
            weight="bold",
        )
        fig_3d.subplots_adjust(
            left=0.015, right=0.985, top=0.90, bottom=0.06, wspace=0.08, hspace=0.14
        )
        output_3d.parent.mkdir(parents=True, exist_ok=True)
        fig_3d.savefig(output_3d, bbox_inches="tight")
        plt.close(fig_3d)

    if output_final_3d is not None:
        final_fig = plt.figure(figsize=(9, 8), dpi=150, facecolor="#fafafa")
        final_axis = final_fig.add_subplot(111, projection="3d")
        final_axis.scatter(
            fit_points[:, 0], fit_points[:, 1], fit_points[:, 2],
            s=10, c="#00897b", alpha=0.9, depthshade=False, label="selected Milk cloud",
        )
        _draw_box(
            final_axis, final_center, np.eye(3), final_half_extents, "#1565c0",
            alpha=0.22, linewidth=2.2,
        )
        _draw_ellipsoid(final_axis, center, rotation, radii)
        final_axis.scatter(*bottom_point, c="#212121", marker="x", s=50, label="ellipsoid bottom")
        scene_points = np.vstack(
            (
                fit_points,
                _box_vertices(final_center, np.eye(3), final_half_extents),
                center[None],
                bottom_point[None],
            )
        )
        scene_minimum = scene_points.min(axis=0) - np.array([0.025, 0.025, 0.025])
        scene_maximum = scene_points.max(axis=0) + np.array([0.025, 0.025, 0.025])
        final_axis.set_xlim(scene_minimum[0], scene_maximum[0])
        final_axis.set_ylim(scene_minimum[1], scene_maximum[1])
        final_axis.set_zlim(scene_minimum[2], scene_maximum[2])
        final_axis.set_box_aspect(scene_maximum - scene_minimum)
        final_axis.view_init(elev=22, azim=-58)
        final_axis.set_xlabel("world X (m)")
        final_axis.set_ylabel("world Y (m)")
        final_axis.set_zlabel("world Z (m)")
        final_axis.set_title(
            "Milk ep2 · plane-only support filtering\n"
            f"new AABB = {dimensions_cm[0]:.2f} x {dimensions_cm[1]:.2f} x "
            f"{dimensions_cm[2]:.2f} cm",
            fontsize=15,
            weight="bold",
        )
        final_axis.legend(loc="upper left", fontsize=9)
        final_axis.grid(True, alpha=0.2)
        output_final_3d.parent.mkdir(parents=True, exist_ok=True)
        final_fig.savefig(output_final_3d, bbox_inches="tight")
        plt.close(final_fig)

    report = {
        "action_step": action_step,
        "bottom_point_m": bottom_point.tolist(),
        "vertical_roi_world_z_m": [float(bottom_point[2] - 0.25), float(bottom_point[2] + 0.035)],
        "support_plane_z_m": support_plane_z,
        "counts": counts,
        "selected_z_range_m": [float(selected_points[:, 2].min()), float(selected_points[:, 2].max())],
        "selected_z_span_m": float(np.ptp(selected_points[:, 2])),
        "final_box_center_m": final_center.tolist(),
        "final_box_full_dimensions_m": final_dimensions.tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--obstacle-primitive", type=Path, required=True)
    parser.add_argument("--debug-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-3d", type=Path)
    parser.add_argument("--output-final-3d", type=Path)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    report = render(
        args.attempt_dir,
        args.obstacle_primitive,
        args.debug_json,
        args.output,
        args.task_suite_name,
        args.stride,
        args.output_3d,
        args.output_final_3d,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
