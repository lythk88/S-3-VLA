import numpy as np

from grasp_rgbd import (
    gripper_width,
    horizontal_support_plane_keep_mask,
    select_grasped_object_points,
)


def _grid_box(center, half_extents, color):
    axes = [
        np.linspace(center[index] - half_extents[index], center[index] + half_extents[index], 15)
        for index in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None], len(points), axis=0)
    return points, colors


def test_gripper_width_is_sign_independent():
    assert np.isclose(gripper_width([0.04, -0.04]), 0.08)
    assert gripper_width([0.01, -0.01]) < gripper_width([0.04, -0.04])


def test_horizontal_support_filter_keeps_points_protruding_from_plane():
    plane_x, plane_y = np.meshgrid(
        np.linspace(-0.15, 0.15, 80), np.linspace(-0.15, 0.15, 80), indexing="ij"
    )
    plane = np.column_stack(
        (plane_x.ravel(), plane_y.ravel(), np.full(plane_x.size, 0.05))
    )
    protrusion, _ = _grid_box(
        np.array([0.0, 0.0, 0.105]), np.array([0.025, 0.025, 0.05]), [200, 0, 0]
    )
    points = np.vstack((plane, protrusion))

    keep, plane_z = horizontal_support_plane_keep_mask(
        points, np.ones(len(points), dtype=bool)
    )

    assert np.isclose(plane_z, 0.05)
    assert not np.any(keep[: len(plane)])
    assert np.all(keep[len(plane):][protrusion[:, 2] > 0.055])


def test_grasp_rgbd_selects_near_object_and_rejects_obstacle():
    # Identity ellipsoid has its lowest point at z=1.05.  The target touches
    # that point; a larger obstacle sits nearby in x and must be removed.
    target_points, target_colors = _grid_box(
        np.array([0.0, 0.0, 1.00]), np.array([0.025, 0.03, 0.05]), [130, 55, 20]
    )
    obstacle_points, obstacle_colors = _grid_box(
        np.array([0.10, 0.0, 1.04]), np.array([0.035, 0.035, 0.10]), [20, 20, 20]
    )
    points = np.vstack((target_points, obstacle_points))
    colors = np.vstack((target_colors, obstacle_colors))

    selected = select_grasped_object_points(
        points,
        colors,
        ellipsoid_center=np.array([0.0, 0.0, 1.16]),
        ellipsoid_rotation=np.eye(3),
        ellipsoid_radii=np.array([0.06, 0.12, 0.11]),
        task_suite_name="safelibero_goal",
        min_points=20,
        obstacle_center=np.array([0.10, 0.0, 1.04]),
        obstacle_rotation=np.eye(3),
        obstacle_half_extents=np.array([0.04, 0.04, 0.11]),
    )

    np.testing.assert_allclose(np.median(selected.points, axis=0)[:2], [0.0, 0.0], atol=0.01)
    assert selected.anchor_distance_m < 0.03
    assert np.max(selected.points[:, 0]) < 0.05
