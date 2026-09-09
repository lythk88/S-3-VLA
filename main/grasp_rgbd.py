"""Grasp-conditioned RGB-D geometry for a carried object.

The carried object is deliberately estimated only after the physical gripper
starts closing.  This avoids relying on an open-loop semantic detection made
at the beginning of an episode, when visually similar cartons are common.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.cluster import DBSCAN


@dataclasses.dataclass(frozen=True)
class GraspPointCloud:
    """Selected carried-object cloud plus diagnostics from the RGB-D crop."""

    points: np.ndarray
    colors: np.ndarray
    candidate_points: np.ndarray
    candidate_colors: np.ndarray
    bottom_point: np.ndarray
    anchor_distance_m: float
    cluster_count: int
    support_plane_z_m: float | None = None


def horizontal_support_plane_keep_mask(
    points: np.ndarray,
    roi_mask: np.ndarray,
    *,
    bin_width_m: float = 0.002,
    tolerance_m: float = 0.003,
    minimum_fraction: float = 0.10,
) -> tuple[np.ndarray, float | None]:
    """Keep protruding geometry while removing a dominant world-horizontal plane.

    A fixed workspace-Z cutoff also removes the lower side of an object resting
    on the support.  Instead, find the densest narrow Z band in the current
    grasp ROI and remove only points within ``tolerance_m`` of that plane.  The
    density requirement prevents a horizontal slice through an ordinary object
    from being mistaken for a support surface.
    """
    points = np.asarray(points, dtype=np.float64)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    roi_z = points[roi_mask, 2]
    keep = np.ones(len(points), dtype=bool)
    if len(roi_z) == 0 or not np.all(np.isfinite(roi_z)):
        return keep, None

    z_min = float(np.min(roi_z))
    z_max = float(np.max(roi_z))
    if z_max - z_min < bin_width_m:
        return keep, None
    edges = np.arange(z_min, z_max + 2.0 * bin_width_m, bin_width_m)
    counts, edges = np.histogram(roi_z, bins=edges)
    dominant_index = int(np.argmax(counts))
    required = max(30, int(np.ceil(minimum_fraction * len(roi_z))))
    if int(counts[dominant_index]) < required:
        return keep, None

    band_lower = edges[dominant_index]
    band_upper = edges[dominant_index + 1]
    band = roi_mask & (points[:, 2] >= band_lower) & (points[:, 2] < band_upper)
    plane_z = float(np.median(points[band, 2]))
    keep &= np.abs(points[:, 2] - plane_z) > tolerance_m
    return keep, plane_z


def gripper_width(qpos: np.ndarray) -> float:
    """Return a sign-independent opening width proxy for a two-jaw gripper."""
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    return float(np.sum(np.abs(values)))


def rgbd_to_world_point_cloud(
    image: np.ndarray,
    depth: np.ndarray,
    env,
    view: str,
    *,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project the evaluator's rotated RGB-D view into colored XYZ points.

    ``main_aegis`` rotates simulator images by 180 degrees before passing them
    here.  Undoing that rotation and using the same vertical-pixel convention
    as ``utils.get_point_cloud`` keeps this cloud in the MuJoCo world frame.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    if stride < 1:
        raise ValueError("stride must be at least 1")
    rgb = np.asarray(image, dtype=np.uint8)[::-1, ::-1]
    metric_depth = get_real_depth_map(env.sim, np.asarray(depth))[::-1, ::-1]
    metric_depth = np.asarray(metric_depth, dtype=np.float64).squeeze()
    height, width = metric_depth.shape
    rows, columns = np.indices((height, width))
    sampled_rows = rows[::stride, ::stride]
    sampled_columns = columns[::stride, ::stride]
    # Match the MuJoCo / robosuite pixel convention already used by the
    # obstacle point-cloud path.
    sampled_v = (height - 1) - sampled_rows
    sampled_depth = metric_depth[::stride, ::stride]
    sampled_rgb = rgb[::stride, ::stride]

    inverse_intrinsics = np.linalg.inv(
        get_camera_intrinsic_matrix(env.sim, view, height, width)
    )
    pixels = np.stack(
        (
            sampled_columns.reshape(-1),
            sampled_v.reshape(-1),
            np.ones(sampled_columns.size, dtype=np.float64),
        ),
        axis=0,
    )
    depth_values = sampled_depth.reshape(-1)
    camera_points = inverse_intrinsics @ pixels * depth_values
    camera_homogeneous = np.vstack(
        (camera_points, np.ones(depth_values.size, dtype=np.float64))
    )
    camera_to_world = get_camera_extrinsic_matrix(env.sim, view)
    world_points = (camera_to_world @ camera_homogeneous)[:3].T
    colors = sampled_rgb.reshape(-1, 3)
    valid = (
        np.all(np.isfinite(world_points), axis=1)
        & np.isfinite(depth_values)
        & (depth_values > 0.0)
    )
    return world_points[valid], colors[valid]


def workspace_z_bounds(task_suite_name: str) -> tuple[float, float]:
    """Return the same broad vertical workspace used by obstacle perception."""
    if "spatial" in task_suite_name or "goal" in task_suite_name:
        return 0.92, 1.5
    if "object" in task_suite_name:
        return 0.05, 0.5
    if "long" in task_suite_name:
        return 0.43, 0.8
    return -np.inf, np.inf


def select_grasped_object_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    ellipsoid_center: np.ndarray,
    ellipsoid_rotation: np.ndarray,
    ellipsoid_radii: np.ndarray,
    task_suite_name: str,
    crop_radius_m: float = 0.18,
    min_points: int = 30,
    max_anchor_distance_m: float = 0.18,
    obstacle_center: np.ndarray | None = None,
    obstacle_rotation: np.ndarray | None = None,
    obstacle_half_extents: np.ndarray | None = None,
) -> GraspPointCloud:
    """Select the RGB-D component immediately below a closing gripper.

    Selection is geometric rather than semantic: crop around the gripper's
    lowest world-z support, remove the known gripper and obstacle volumes,
    voxelize, then cluster colored XYZ samples.  The component whose visible
    surface is closest to the grasp point is treated as the carried object.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(colors):
        raise ValueError("points/colors must have matching shapes (N,3)")
    center = np.asarray(ellipsoid_center, dtype=np.float64)
    rotation = np.asarray(ellipsoid_rotation, dtype=np.float64)
    radii = np.asarray(ellipsoid_radii, dtype=np.float64)
    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    vertical_support = float(np.linalg.norm(radii * (rotation.T @ world_down)))
    bottom_point = center + world_down * vertical_support

    relative = points - bottom_point
    horizontal_distance = np.linalg.norm(relative[:, :2], axis=1)
    mask = (
        np.all(np.isfinite(points), axis=1)
        & (horizontal_distance <= crop_radius_m)
        & (relative[:, 2] <= 0.035)
        & (relative[:, 2] >= -0.25)
    )

    # Remove only the dominant horizontal support plane.  Do not apply a
    # global workspace-Z cutoff: any geometry protruding from the plane may be
    # part of the carried object and must remain eligible for clustering.
    support_keep, support_plane_z = horizontal_support_plane_keep_mask(points, mask)
    mask &= support_keep

    # Remove the known gripper safety volume.  The lower object surface stays
    # available even when its top is partly occluded by the fingers.
    ellipsoid_local = (points - center) @ rotation
    ellipsoid_level = np.sum(np.square(ellipsoid_local / radii), axis=1)
    mask &= ellipsoid_level > 1.02**2

    # The obstacle cloud can be very close to the grasp ROI.  It has already
    # been estimated separately, so exclude its padded OBB before clustering.
    if obstacle_center is not None:
        if obstacle_rotation is None or obstacle_half_extents is None:
            raise ValueError("obstacle rotation and half extents are required")
        obstacle_local = (
            points - np.asarray(obstacle_center, dtype=np.float64)
        ) @ np.asarray(obstacle_rotation, dtype=np.float64)
        inside_obstacle = np.all(
            np.abs(obstacle_local)
            <= np.asarray(obstacle_half_extents, dtype=np.float64) + 0.004,
            axis=1,
        )
        mask &= ~inside_obstacle

    candidate_points = points[mask]
    candidate_colors = colors[mask]
    if len(candidate_points) < min_points:
        raise RuntimeError(
            f"only {len(candidate_points)} RGB-D points in the grasp volume"
        )

    # One sample per 3 mm voxel keeps DBSCAN fast at 1024x1024 while retaining
    # both XYZ and RGB.  Color is a soft feature: it separates an object from
    # an adjacent support surface without fragmenting printed packaging.
    voxel = np.floor(candidate_points / 0.003).astype(np.int64)
    _, voxel_indices = np.unique(voxel, axis=0, return_index=True)
    voxel_points = candidate_points[voxel_indices]
    voxel_colors = candidate_colors[voxel_indices]
    xyz_features = (voxel_points - bottom_point) / 0.010
    rgb_features = voxel_colors.astype(np.float64) / 255.0 * 0.45
    features = np.column_stack((xyz_features, rgb_features))
    labels = DBSCAN(eps=1.35, min_samples=4, n_jobs=1).fit_predict(features)
    valid_labels = np.unique(labels[labels >= 0])
    if len(valid_labels) == 0:
        raise RuntimeError("RGB-D grasp crop has no dense object component")

    choices: list[tuple[float, int, np.ndarray]] = []
    for label in valid_labels:
        indices = np.flatnonzero(labels == label)
        if len(indices) < min_points:
            continue
        cluster = voxel_points[indices]
        dimensions = np.ptp(cluster, axis=0)
        # Reject large background structures and nearly flat support patches.
        if np.any(dimensions > np.array([0.22, 0.22, 0.28])):
            continue
        if dimensions[2] < 0.006 and np.prod(dimensions[:2]) > 0.004:
            continue
        distances = np.linalg.norm(cluster - bottom_point, axis=1)
        anchor_distance = float(np.quantile(distances, 0.05))
        centroid_distance = float(np.linalg.norm(np.mean(cluster, axis=0) - bottom_point))
        score = anchor_distance + 0.15 * centroid_distance - 0.001 * np.log1p(len(indices))
        choices.append((score, int(label), indices))
    if not choices:
        raise RuntimeError("RGB-D grasp crop has no physically plausible object component")

    _, selected_label, selected_indices = min(choices, key=lambda item: item[0])
    selected_points = voxel_points[selected_indices]
    selected_colors = voxel_colors[selected_indices]
    anchor_distance = float(
        np.quantile(np.linalg.norm(selected_points - bottom_point, axis=1), 0.05)
    )
    if anchor_distance > max_anchor_distance_m:
        raise RuntimeError(
            f"nearest RGB-D component is {anchor_distance:.4f} m from grasp point"
        )
    return GraspPointCloud(
        points=selected_points,
        colors=selected_colors,
        candidate_points=voxel_points,
        candidate_colors=voxel_colors,
        bottom_point=bottom_point,
        anchor_distance_m=anchor_distance,
        cluster_count=int(len(valid_labels)),
        support_plane_z_m=support_plane_z,
    )
