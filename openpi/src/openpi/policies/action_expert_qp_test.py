import numpy as np

import openpi.policies.action_expert_qp as action_expert_qp
from openpi.policies.action_expert_qp import ControllerRollout
from openpi.policies.action_expert_qp import Ellipsoid
from openpi.policies.action_expert_qp import ObstaclePrimitive
from openpi.policies.action_expert_qp import ellipsoid_gap
from openpi.policies.action_expert_qp import ellipsoid_obstacle_gap
from openpi.policies.action_expert_qp import ellipsoid_obstacle_gap_details
from openpi.policies.action_expert_qp import predict_eef_positions
from openpi.policies.action_expert_qp import primitive_obstacle_gap
from openpi.policies.action_expert_qp import project_action_chunk_with_qp
from openpi.policies.action_expert_qp import project_action_chunk_with_adaptive_radius
from openpi.policies.action_expert_qp import rollout_eef_trajectory
from openpi.policies.action_expert_qp import solve_action_expert_qp
from openpi.policies.action_expert_qp import trajectory_barriers
from openpi.policies.action_expert_qp import trajectory_cbf_constraints


def _controller(step=0.05, rotation_step=0.5, response=1.0):
    return ControllerRollout(
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.array([step] * 3 + [rotation_step] * 3),
        output_max=np.array([step] * 3 + [rotation_step] * 3),
        translation_response_gain=np.asarray(response),
        rotation_response_gain=np.asarray(response),
    )


def test_ellipsoid_gap_for_spheres():
    first = Ellipsoid(np.zeros(3), np.eye(3), np.full(3, 0.1))
    second = Ellipsoid(np.array([0.5, 0.0, 0.0]), np.eye(3), np.full(3, 0.2))
    assert np.isclose(ellipsoid_gap(first, second), 0.2)


def test_gap_diagnostics_match_existing_obstacle_gap():
    gripper = Ellipsoid(
        np.array([0.1, -0.2, 0.3]),
        np.eye(3),
        np.array([0.05, 0.08, 0.1]),
    )
    obstacle = ObstaclePrimitive(
        "obb",
        np.array([0.4, 0.1, 0.25]),
        np.eye(3),
        np.array([0.08, 0.06, 0.12]),
    )

    details = ellipsoid_obstacle_gap_details(gripper, obstacle)

    assert np.isclose(details["surface_gap"], ellipsoid_obstacle_gap(gripper, obstacle))
    assert np.isclose(np.linalg.norm(details["center_direction"]), 1.0)
    assert np.isclose(np.linalg.norm(details["active_direction"]), 1.0)


def test_trajectory_barrier_decreases_toward_obstacle():
    obstacle = ObstaclePrimitive("cylinder", np.array([0.5, 0.0, 0.0]), np.eye(3), np.array([0.05, 0.05]))
    actions = np.zeros((10, 7))
    actions[:, 0] = 1.0
    values = trajectory_barriers(
        actions,
        eef_position=np.zeros(3),
        eef_rotation=np.eye(3),
        eef_radii=np.full(3, 0.05),
        ellipsoid_offset=np.zeros(3),
        controller=_controller(step=0.01),
        obstacle=obstacle,
        safe_distance=0.01,
    )
    assert np.all(np.diff(values) < 0.0)


def test_axis_aligned_carried_box_gap_to_obstacle_box():
    carried = ObstaclePrimitive(
        "obb", np.zeros(3), np.eye(3), np.array([0.1, 0.2, 0.3])
    )
    obstacle = ObstaclePrimitive(
        "obb", np.array([0.5, 0.0, 0.0]), np.eye(3), np.array([0.1, 0.1, 0.1])
    )

    assert np.isclose(primitive_obstacle_gap(carried, obstacle), 0.3)


def test_compound_trajectory_uses_carried_box_when_it_is_closer():
    obstacle = ObstaclePrimitive(
        "obb", np.array([0.5, 0.0, 0.0]), np.eye(3), np.full(3, 0.05)
    )
    actions = np.zeros((2, 7))
    values = trajectory_barriers(
        actions,
        eef_position=np.zeros(3),
        eef_rotation=np.eye(3),
        eef_radii=np.full(3, 0.05),
        ellipsoid_offset=np.zeros(3),
        controller=_controller(step=0.01),
        obstacle=obstacle,
        safe_distance=0.01,
        carried_object_offset=np.array([0.25, 0.0, 0.0]),
        carried_object_rotation=np.eye(3),
        carried_object_size=np.full(3, 0.05),
    )

    # The bare ellipsoid barrier is 0.39 m; the attached box barrier is 0.14 m.
    np.testing.assert_allclose(values, [0.14, 0.14], atol=1e-8)


def test_predict_eef_positions_integrates_all_ten_translational_velocities():
    actions = np.zeros((10, 7))
    actions[:, 0] = np.arange(1.0, 11.0)
    actions[:, 1] = 2.0
    actions[:, 2] = -1.0
    actions[:, 3:] = 1_000.0  # Non-translational channels must have no effect.
    initial = np.array([0.5, -0.25, 1.0])

    positions = predict_eef_positions(actions, eef_position=initial, action_dt=0.05)

    assert positions.shape == (10, 3)
    expected = initial + 0.05 * np.cumsum(actions[:, :3], axis=0)
    np.testing.assert_allclose(positions, expected)
    np.testing.assert_allclose(positions[-1], [3.25, 0.75, 0.5])


def test_controller_rollout_clips_scales_and_applies_achieved_gain_recursively():
    actions = np.zeros((2, 7))
    actions[:, 0] = [2.0, -2.0]
    trajectory = rollout_eef_trajectory(
        actions,
        eef_position=np.array([0.4, 0.0, 0.0]),
        eef_rotation=np.eye(3),
        ellipsoid_offset=np.zeros(3),
        controller=_controller(step=0.05, response=0.22),
    )
    np.testing.assert_allclose(trajectory.scaled_commands[:, 0], [0.05, -0.05])
    np.testing.assert_allclose(trajectory.achieved_translation_deltas[:, 0], [0.011, -0.011])
    np.testing.assert_allclose(trajectory.positions[:, 0], [0.411, 0.4])


def test_controller_rollout_left_multiplies_rotation_and_rotates_center_offset():
    actions = np.zeros((1, 7))
    actions[0, 5] = 1.0
    trajectory = rollout_eef_trajectory(
        actions,
        eef_position=np.zeros(3),
        eef_rotation=np.eye(3),
        ellipsoid_offset=np.array([0.0, -0.08, 0.0]),
        controller=_controller(rotation_step=np.pi / 2),
    )
    np.testing.assert_allclose(trajectory.ellipsoid_centers[0], [0.08, 0.0, 0.0], atol=1e-7)


def test_trajectory_barriers_check_a_late_horizon_violation():
    obstacle = ObstaclePrimitive("cylinder", np.array([0.5, 0.0, 0.0]), np.eye(3), np.array([0.05, 0.05]))
    actions = np.zeros((10, 7))
    actions[:, 0] = 1.0

    values = trajectory_barriers(
        actions,
        eef_position=np.zeros(3),
        eef_rotation=np.eye(3),
        eef_radii=np.full(3, 0.05),
        ellipsoid_offset=np.zeros(3),
        controller=_controller(step=0.04),
        obstacle=obstacle,
        safe_distance=0.01,
    )

    assert np.all(values[:9] >= 0.0)
    assert values[9] < 0.0


def test_gap_uses_each_obstacle_primitive_support_function():
    gripper = Ellipsoid(np.zeros(3), np.eye(3), np.full(3, 0.1))
    center = np.array([1.0, 0.0, 0.0])
    obb = ObstaclePrimitive("obb", center, np.eye(3), np.array([0.2, 0.3, 0.4]))
    cylinder = ObstaclePrimitive("cylinder", center, np.eye(3), np.array([0.2, 0.4]))
    capsule = ObstaclePrimitive("capsule", center, np.eye(3), np.array([0.2, 0.4]))
    ellipsoid = ObstaclePrimitive("ellipsoid", center, np.eye(3), np.array([0.2, 0.3, 0.4]))

    assert np.isclose(ellipsoid_obstacle_gap(gripper, obb), 0.7)
    assert np.isclose(ellipsoid_obstacle_gap(gripper, cylinder), 0.7)
    assert np.isclose(ellipsoid_obstacle_gap(gripper, capsule), 0.7)
    assert np.isclose(ellipsoid_obstacle_gap(gripper, ellipsoid), 0.7)


def test_axial_support_distinguishes_cylinder_and_capsule():
    gripper = Ellipsoid(np.zeros(3), np.eye(3), np.full(3, 0.1))
    center = np.array([0.0, 0.0, 1.0])
    cylinder = ObstaclePrimitive("cylinder", center, np.eye(3), np.array([0.2, 0.4]))
    capsule = ObstaclePrimitive("capsule", center, np.eye(3), np.array([0.2, 0.4]))

    assert np.isclose(ellipsoid_obstacle_gap(gripper, cylinder), 0.5)
    assert np.isclose(ellipsoid_obstacle_gap(gripper, capsule), 0.3)


def test_sphere_support_and_top_only_padding_leave_bottom_and_xy_unchanged():
    obstacle = ObstaclePrimitive(
        "sphere",
        np.zeros(3),
        np.eye(3),
        np.array([0.2]),
        top_padding=0.02,
    )

    assert np.isclose(action_expert_qp.obstacle_support(np.array([1.0, 0.0, 0.0]), obstacle), 0.2)
    assert np.isclose(action_expert_qp.obstacle_support(np.array([0.0, 0.0, -1.0]), obstacle), 0.2)
    assert np.isclose(action_expert_qp.obstacle_support(np.array([0.0, 0.0, 1.0]), obstacle), 0.22)


def test_gap_uses_a_valid_shape_axis_when_centerline_is_false_overlap():
    gripper = Ellipsoid(
        np.zeros(3),
        np.eye(3),
        np.array([0.50, 0.05, 0.05]),
    )
    obstacle = ObstaclePrimitive(
        "obb",
        np.array([0.40, 0.0, 0.20]),
        np.eye(3),
        np.array([0.10, 0.10, 0.10]),
    )

    centerline = obstacle.center / np.linalg.norm(obstacle.center)
    centerline_gap = (
        np.linalg.norm(obstacle.center)
        - action_expert_qp.ellipsoid_support(centerline, gripper.rotation, gripper.radii)
        - action_expert_qp.obstacle_support(-centerline, obstacle)
    )
    assert centerline_gap < 0.0
    # The z projections are separated by 5 cm, which is a sound certificate
    # even though the centerline projection overlaps.
    assert np.isclose(ellipsoid_obstacle_gap(gripper, obstacle), 0.05)


def test_qp_prefers_success_direction_inside_safe_halfspace():
    result = solve_action_expert_qp(
        np.array([1.0, 0.0]),
        np.array([[-1.0, 0.0]]),
        np.array([-0.1]),
        lambda_deviation=1.0,
        beta_success=1.0,
        trust_radius=1.0,
    )
    assert result.success
    assert np.isclose(result.correction[0], 0.1, atol=1e-5)


def test_qp_returns_zero_when_trust_region_is_infeasible():
    result = solve_action_expert_qp(
        np.array([1.0]),
        np.array([[1.0]]),
        np.array([2.0]),
        lambda_deviation=1.0,
        beta_success=1.0,
        trust_radius=1.0,
    )
    assert not result.success
    np.testing.assert_array_equal(result.correction, np.zeros(1))


def test_qp_l2_trust_region_limits_joint_correction_norm():
    result = solve_action_expert_qp(
        np.ones(2),
        np.zeros((0, 2)),
        np.zeros(0),
        lambda_deviation=1.0,
        beta_success=10.0,
        trust_radius=0.2,
        trust_region_norm="l2",
    )

    assert result.success
    assert np.isclose(np.linalg.norm(result.correction), 0.2, atol=1e-6)
    np.testing.assert_allclose(result.correction[0], result.correction[1], atol=1e-6)


def test_qp_keeps_a_feasible_candidate_when_slsqp_does_not_certify_optimality(monkeypatch):
    class Result:
        x = np.array([0.25])
        success = False
        message = "Positive directional derivative for linesearch"

    monkeypatch.setattr(action_expert_qp, "minimize", lambda *args, **kwargs: Result())
    result = solve_action_expert_qp(
        np.zeros(1),
        np.ones((1, 1)),
        np.array([0.2]),
        lambda_deviation=1.0,
        beta_success=0.0,
        trust_radius=1.0,
    )

    assert result.success
    np.testing.assert_allclose(result.correction, [0.25])
    assert "Positive directional derivative" in result.status


def test_trajectory_cbf_constraints_use_previous_action_step():
    barriers = np.array([0.2, 0.05])
    jacobian = np.eye(2)
    matrix, rhs = trajectory_cbf_constraints(barriers, jacobian, 0.4, gamma=0.5)
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [-0.5, 1.0]])
    np.testing.assert_allclose(rhs, [0.0, 0.05])


def test_final_qp_projection_makes_every_action_step_safe():
    obstacle = ObstaclePrimitive("cylinder", np.array([0.5, 0.0, 0.0]), np.eye(3), np.array([0.05, 0.05]))
    actions = np.zeros((10, 7))
    actions[:, 0] = 1.0

    def barrier_function(physical_actions):
        return trajectory_barriers(
            physical_actions,
            eef_position=np.zeros(3),
            eef_rotation=np.eye(3),
            eef_radii=np.full(3, 0.05),
            ellipsoid_offset=np.zeros(3),
            controller=_controller(step=0.04),
            obstacle=obstacle,
            safe_distance=0.01,
        )

    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        barrier_function,
        trust_radius=1.0,
    )

    assert result.success
    assert result.iterations >= 1
    assert result.barriers_before[-1] < 0.0
    assert np.all(result.barriers_after >= -1e-7)
    assert np.any(np.abs(result.correction) > 1e-6)
    assert not np.array_equal(result.actions, actions)
    np.testing.assert_allclose(result.barriers_after, barrier_function(result.actions), atol=1e-7)


def test_final_qp_projection_leaves_an_already_safe_chunk_exactly_unchanged():
    actions = np.zeros((10, 7), dtype=np.float32)

    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        lambda physical_actions: np.ones(len(physical_actions)),
        trust_radius=0.05,
    )

    assert result.success
    assert result.iterations == 0
    np.testing.assert_array_equal(result.actions, actions)
    np.testing.assert_array_equal(result.correction, np.zeros((10, 3), dtype=np.float32))


def test_safe_chunk_uses_success_gradient_without_losing_safety():
    actions = np.zeros((3, 7), dtype=np.float64)

    def barriers(physical_actions):
        return 0.5 - physical_actions[:, 0]

    gradient = np.zeros((3, 3), dtype=np.float64)
    gradient[:, 0] = 1.0
    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        barriers,
        trust_radius=0.2,
        success_gradient=gradient,
        lambda_deviation=1.0,
        beta_success=1.0,
        enable_nonlinear_fallback=False,
    )

    assert result.success
    assert np.all(result.actions[:, 0] > 0.0)
    assert np.all(result.barriers_after >= 0.0)
    assert "safe success-guided projection" in result.status


def test_fail_soft_uses_success_gradient_within_best_barrier_floor():
    actions = np.zeros((2, 7), dtype=np.float64)

    def barriers(physical_actions):
        # Only token 0 determines the common safety floor. Token 1 remains
        # available for the secondary success objective.
        return np.array([physical_actions[0, 0] - 2.0, physical_actions[0, 0] - 2.0])

    gradient = np.zeros((2, 3), dtype=np.float64)
    gradient[1, 1] = 1.0
    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        barriers,
        trust_radius=0.5,
        success_gradient=gradient,
        lambda_deviation=1.0,
        beta_success=1.0,
        enable_nonlinear_fallback=False,
    )

    assert not result.success
    np.testing.assert_allclose(result.actions[0, 0], 0.5, atol=2e-6)
    assert result.actions[1, 1] > 0.0
    np.testing.assert_allclose(result.barriers_after, -1.5, atol=2e-6)


def test_adaptive_projection_stops_at_first_safe_radius():
    actions = np.zeros((2, 7), dtype=np.float64)

    def barriers(physical_actions):
        return physical_actions[:, 0] - 0.15

    result = project_action_chunk_with_adaptive_radius(
        actions,
        lambda value: value,
        barriers,
        trust_radii=(0.1, 0.2, 0.5),
        enable_nonlinear_fallback=False,
    )

    assert result.projection.success
    assert result.selected_trust_radius == 0.2
    assert result.attempted_trust_radii == (0.1, 0.2)
    assert np.all(result.projection.barriers_after >= -1e-7)


def test_adaptive_projection_does_not_escalate_for_only_distant_violation():
    actions = np.zeros((2, 7), dtype=np.float64)

    def barriers(physical_actions):
        return np.array([0.1, physical_actions[1, 0] - 0.3])

    result = project_action_chunk_with_adaptive_radius(
        actions,
        lambda value: value,
        barriers,
        trust_radii=(0.1, 0.2, 0.3),
        escalate_only_if_first_barrier_negative=True,
        enable_nonlinear_fallback=False,
    )

    assert not result.projection.success
    assert result.attempted_trust_radii == (0.1,)
    assert result.selected_trust_radius == 0.1


def test_adaptive_projection_stops_after_minimum_norm_first_step_recovery():
    actions = np.zeros((2, 7), dtype=np.float64)

    def barriers(physical_actions):
        return physical_actions[:, 0] - 0.2

    result = project_action_chunk_with_adaptive_radius(
        actions,
        lambda value: value,
        barriers,
        trust_radii=(0.02, 0.1, 0.3),
        first_step_recovery_floor=-0.19,
        enable_nonlinear_fallback=False,
    )

    assert not result.projection.success
    assert result.selected_trust_radius == 0.02
    assert result.attempted_trust_radii == (0.02,)
    np.testing.assert_allclose(result.projection.actions[0, 0], 0.01, atol=2e-6)
    np.testing.assert_allclose(result.projection.actions[1, :3], 0.0, atol=2e-6)
    assert result.projection.barriers_after[0] >= -0.19 - 1e-7
    assert "first-step recovery floor" in result.projection.status


def test_final_qp_projection_fails_closed_when_whole_chunk_cannot_be_made_safe():
    actions = np.zeros((10, 7))
    actions[:, 0] = 1.0

    def impossible_barrier(physical_actions):
        return -np.ones(len(physical_actions))

    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        impossible_barrier,
        trust_radius=0.05,
    )

    assert not result.success
    assert np.all(result.barriers_after < 0.0)


def test_infeasible_projection_returns_barrier_improving_correction():
    actions = np.zeros((5, 7), dtype=np.float64)

    def unsafe_but_improvable(physical_actions):
        return physical_actions[:, 0] - 2.0

    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        unsafe_but_improvable,
        trust_radius=0.5,
        enable_nonlinear_fallback=False,
    )

    assert not result.success
    assert np.min(result.barriers_before) == -2.0
    np.testing.assert_allclose(result.actions[:, 0], 0.5, atol=2e-6)
    np.testing.assert_allclose(result.barriers_after, -1.5, atol=2e-6)
    assert np.min(result.barriers_after) > np.min(result.barriers_before)
    assert "accepted fail-soft correction" in result.status


def test_final_projection_can_switch_safe_components_with_nonlinear_fallback(monkeypatch):
    actions = np.zeros((10, 7), dtype=np.float64)
    actions[:, 0] = 0.5

    def disjunctive_barrier(physical_actions):
        positions = np.cumsum(physical_actions[:, :3], axis=0)
        # Safe on either side of a square obstacle.  The max makes this a
        # union of separating-axis half-spaces, like the primitive barrier.
        return np.maximum(np.abs(positions[:, 0]), np.abs(positions[:, 1])) - 0.6

    original_solver = action_expert_qp.solve_action_expert_qp
    calls = 0

    def reject_linear_subproblems(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_solver(*args, **kwargs)
        return action_expert_qp.QPResult(
            correction=np.zeros_like(result.correction),
            success=False,
            status="forced local rejection",
            minimum_linearized_margin=result.minimum_linearized_margin,
            objective=result.objective,
        )

    monkeypatch.setattr(action_expert_qp, "solve_action_expert_qp", reject_linear_subproblems)
    result = project_action_chunk_with_qp(
        actions,
        lambda value: value,
        disjunctive_barrier,
        trust_radius=1.0,
    )

    assert calls >= 1
    assert result.success
    assert "nonlinear multi-axis projection" in result.status
    assert np.all(result.barriers_after >= -1e-7)
