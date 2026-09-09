"""Geometry and the safety-first action-expert quadratic program."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Ellipsoid:
    center: np.ndarray
    rotation: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class ObstaclePrimitive:
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    # One-sided Minkowski extrusion along world +Z. This increases only the
    # obstacle's upper support; its lower face and X/Y footprint are unchanged.
    top_padding: float = 0.0


@dataclass(frozen=True)
class QPResult:
    correction: np.ndarray
    success: bool
    status: str
    minimum_linearized_margin: float
    objective: float


@dataclass(frozen=True)
class ChunkProjectionResult:
    actions: np.ndarray
    correction: np.ndarray
    barriers_before: np.ndarray
    barriers_after: np.ndarray
    success: bool
    iterations: int
    status: str


@dataclass(frozen=True)
class AdaptiveProjectionResult:
    projection: ChunkProjectionResult
    selected_trust_radius: float
    attempted_trust_radii: tuple[float, ...]


@dataclass(frozen=True)
class ControllerRollout:
    """Robosuite OSC command convention plus calibrated one-step tracking."""

    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    translation_response_gain: np.ndarray
    rotation_response_gain: np.ndarray


@dataclass(frozen=True)
class EefTrajectory:
    positions: np.ndarray
    rotations: np.ndarray
    ellipsoid_centers: np.ndarray
    scaled_commands: np.ndarray
    achieved_translation_deltas: np.ndarray
    achieved_rotation_vectors: np.ndarray


def ellipsoid_support(direction: np.ndarray, rotation: np.ndarray, radii: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=np.float64)
    shape = (
        np.asarray(rotation, dtype=np.float64) @ np.diag(np.square(radii)) @ np.asarray(rotation, dtype=np.float64).T
    )
    return float(np.sqrt(max(float(direction @ shape @ direction), 0.0)))


def obb_support(direction: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray) -> float:
    return float(np.sum(np.asarray(half_extents) * np.abs(np.asarray(rotation).T @ direction)))


def capsule_support(direction: np.ndarray, axis: np.ndarray, radius: float, half_length: float) -> float:
    return float(half_length * abs(float(np.asarray(axis) @ direction)) + radius)


def cylinder_support(direction: np.ndarray, axis: np.ndarray, radius: float, half_length: float) -> float:
    axial = float(np.clip(np.asarray(axis) @ direction, -1.0, 1.0))
    return float(half_length * abs(axial) + radius * np.sqrt(max(1.0 - axial * axial, 0.0)))


def ellipsoid_gap(first: Ellipsoid, second: Ellipsoid) -> float:
    displacement = np.asarray(second.center) - np.asarray(first.center)
    distance = float(np.linalg.norm(displacement))
    if distance < 1e-9:
        return -float(np.max(first.radii) + np.max(second.radii))
    direction = displacement / distance
    return (
        distance
        - ellipsoid_support(direction, first.rotation, first.radii)
        - ellipsoid_support(direction, second.rotation, second.radii)
    )


def obstacle_support(direction: np.ndarray, obstacle: ObstaclePrimitive) -> float:
    """Support distance, including an optional one-sided world-Z margin."""
    direction = np.asarray(direction, dtype=np.float64)
    if obstacle.kind == "ellipsoid":
        radii = np.asarray(obstacle.size, dtype=np.float64)
        if radii.shape != (3,):
            raise ValueError("ellipsoid size must contain three radii")
        base_support = ellipsoid_support(direction, obstacle.rotation, radii)
    elif obstacle.kind in {"obb", "aabb"}:
        if np.asarray(obstacle.size).shape != (3,):
            raise ValueError("box size must contain three half-extents")
        base_support = obb_support(direction, obstacle.rotation, obstacle.size)
    elif obstacle.kind == "cylinder":
        radius, half_length = np.asarray(obstacle.size, dtype=np.float64)
        base_support = cylinder_support(direction, obstacle.rotation[:, 2], radius, half_length)
    elif obstacle.kind == "capsule":
        radius, half_length = np.asarray(obstacle.size, dtype=np.float64)
        base_support = capsule_support(direction, obstacle.rotation[:, 2], radius, half_length)
    elif obstacle.kind == "sphere":
        size = np.asarray(obstacle.size, dtype=np.float64)
        if size.shape != (1,):
            raise ValueError("sphere size must contain one radius")
        base_support = float(size[0])
    else:
        raise ValueError(f"unsupported obstacle primitive: {obstacle.kind!r}")
    top_padding = float(obstacle.top_padding)
    if not np.isfinite(top_padding) or top_padding < 0.0:
        raise ValueError("obstacle top_padding must be finite and non-negative")
    return base_support + top_padding * max(float(direction[2]), 0.0)


def primitive_obstacle_gap(
    carried_object: ObstaclePrimitive,
    obstacle: ObstaclePrimitive,
) -> float:
    """Conservative separating gap between a carried primitive and an obstacle.

    The grasp-conditioned implementation currently supplies an axis-aligned
    OBB for ``carried_object``.  Keeping this generic makes the distance code
    use the same support functions as the gripper ellipsoid path.
    """
    displacement = np.asarray(obstacle.center) - np.asarray(carried_object.center)
    carried_axes = np.asarray(carried_object.rotation, dtype=np.float64).T
    obstacle_axes = np.asarray(obstacle.rotation, dtype=np.float64).T
    candidates = [displacement]
    candidates.extend(carried_axes)
    candidates.extend(obstacle_axes)
    candidates.extend(
        np.cross(carried_axis, obstacle_axis)
        for carried_axis in carried_axes
        for obstacle_axis in obstacle_axes
    )
    directions = []
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            continue
        direction = np.asarray(candidate, dtype=np.float64) / norm
        if float(displacement @ direction) < 0.0:
            direction = -direction
        if not any(
            abs(float(direction @ previous)) > 1.0 - 1e-8
            for previous in directions
        ):
            directions.append(direction)
    if not directions:
        return -float(np.max(carried_object.size) + np.max(obstacle.size))
    return max(
        float(displacement @ direction)
        - obstacle_support(direction, carried_object)
        - obstacle_support(-direction, obstacle)
        for direction in directions
    )


def primitive_obstacle_gap_details(
    carried_object: ObstaclePrimitive,
    obstacle: ObstaclePrimitive,
) -> dict[str, np.ndarray | float]:
    """Return the active separating-axis terms for primitive diagnostics."""
    displacement = np.asarray(obstacle.center) - np.asarray(carried_object.center)
    carried_axes = np.asarray(carried_object.rotation, dtype=np.float64).T
    obstacle_axes = np.asarray(obstacle.rotation, dtype=np.float64).T
    candidates = [displacement]
    candidates.extend(carried_axes)
    candidates.extend(obstacle_axes)
    candidates.extend(
        np.cross(carried_axis, obstacle_axis)
        for carried_axis in carried_axes
        for obstacle_axis in obstacle_axes
    )
    directions = []
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            continue
        direction = np.asarray(candidate, dtype=np.float64) / norm
        if float(displacement @ direction) < 0.0:
            direction = -direction
        if not any(
            abs(float(direction @ previous)) > 1.0 - 1e-8
            for previous in directions
        ):
            directions.append(direction)
    if not directions:
        return {
            "active_direction": np.zeros(3, dtype=np.float64),
            "projected_center_distance": 0.0,
            "carried_object_support": float(np.max(carried_object.size)),
            "obstacle_support": float(np.max(obstacle.size)),
            "surface_gap": -float(
                np.max(carried_object.size) + np.max(obstacle.size)
            ),
        }
    terms = []
    for direction in directions:
        projected = float(displacement @ direction)
        carried_support = obstacle_support(direction, carried_object)
        other_support = obstacle_support(-direction, obstacle)
        terms.append(
            (
                projected - carried_support - other_support,
                direction,
                projected,
                carried_support,
                other_support,
            )
        )
    surface_gap, direction, projected, carried_support, other_support = max(
        terms, key=lambda item: item[0]
    )
    return {
        "active_direction": np.asarray(direction, dtype=np.float64),
        "projected_center_distance": projected,
        "carried_object_support": carried_support,
        "obstacle_support": other_support,
        "surface_gap": surface_gap,
    }


def _separating_directions(
    gripper: Ellipsoid,
    obstacle: ObstaclePrimitive,
) -> list[np.ndarray]:
    """Return conservative candidate axes for convex-shape separation.

    A positive projection gap on *any* axis proves that the two convex shapes
    are separated.  Testing only the center-to-center axis is unnecessarily
    conservative for elongated, rotated shapes and was the source of false
    overlaps in the pilot.  These axes include the standard box axes and cross
    axes, plus the centerline and the gripper principal axes.  The finite set
    remains conservative: it can miss a valid separating axis, but it cannot
    invent separation where the projected intervals overlap.
    """
    displacement = np.asarray(obstacle.center) - np.asarray(gripper.center)
    candidates = [displacement]
    gripper_axes = np.asarray(gripper.rotation, dtype=np.float64).T
    obstacle_axes = np.asarray(obstacle.rotation, dtype=np.float64).T
    candidates.extend(gripper_axes)
    candidates.extend(obstacle_axes)
    candidates.extend(
        np.cross(gripper_axis, obstacle_axis) for gripper_axis in gripper_axes for obstacle_axis in obstacle_axes
    )
    if obstacle.kind in {"cylinder", "capsule"}:
        obstacle_axis = np.asarray(obstacle.rotation, dtype=np.float64)[:, 2]
        candidates.append(displacement - obstacle_axis * float(displacement @ obstacle_axis))

    directions = []
    for candidate in candidates:
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            continue
        direction = np.asarray(candidate, dtype=np.float64) / norm
        if float(displacement @ direction) < 0.0:
            direction = -direction
        if not any(abs(float(direction @ previous)) > 1.0 - 1e-8 for previous in directions):
            directions.append(direction)
    return directions


def ellipsoid_obstacle_gap(gripper: Ellipsoid, obstacle: ObstaclePrimitive) -> float:
    """Conservative separating gap for the gripper and obstacle primitive."""
    displacement = np.asarray(obstacle.center) - np.asarray(gripper.center)
    distance = float(np.linalg.norm(displacement))
    if distance < 1e-9:
        return -float(np.max(gripper.radii) + np.max(obstacle.size))
    return max(
        float(displacement @ direction)
        - ellipsoid_support(direction, gripper.rotation, gripper.radii)
        - obstacle_support(-direction, obstacle)
        for direction in _separating_directions(gripper, obstacle)
    )


def ellipsoid_obstacle_gap_details(
    gripper: Ellipsoid,
    obstacle: ObstaclePrimitive,
) -> dict[str, np.ndarray | float]:
    """Expose the terms of the existing gap calculation for diagnostics.

    This intentionally calls the same support functions and separating-axis
    enumeration as :func:`ellipsoid_obstacle_gap`; it does not define a new
    collision metric or change the QP formulation.
    """
    displacement = np.asarray(obstacle.center, dtype=np.float64) - np.asarray(gripper.center, dtype=np.float64)
    center_distance = float(np.linalg.norm(displacement))
    if center_distance < 1e-9:
        center_direction = np.zeros(3, dtype=np.float64)
        active_direction = center_direction.copy()
        gripper_support = float(np.max(gripper.radii))
        obstacle_radius = float(np.max(obstacle.size))
        projected_center_distance = 0.0
        surface_gap = -(gripper_support + obstacle_radius)
    else:
        center_direction = displacement / center_distance
        candidates = []
        for direction in _separating_directions(gripper, obstacle):
            projected = float(displacement @ direction)
            gripper_radius = ellipsoid_support(direction, gripper.rotation, gripper.radii)
            obstacle_radius = obstacle_support(-direction, obstacle)
            candidates.append(
                (
                    projected - gripper_radius - obstacle_radius,
                    direction,
                    projected,
                    gripper_radius,
                    obstacle_radius,
                )
            )
        (
            surface_gap,
            active_direction,
            projected_center_distance,
            gripper_support,
            obstacle_radius,
        ) = max(candidates, key=lambda item: item[0])
    centerline_gripper_support = (
        ellipsoid_support(center_direction, gripper.rotation, gripper.radii)
        if center_distance >= 1e-9
        else float(np.max(gripper.radii))
    )
    centerline_obstacle_support = (
        obstacle_support(-center_direction, obstacle) if center_distance >= 1e-9 else float(np.max(obstacle.size))
    )
    return {
        "center_distance": center_distance,
        "center_direction": np.asarray(center_direction, dtype=np.float64),
        "active_direction": np.asarray(active_direction, dtype=np.float64),
        "projected_center_distance": float(projected_center_distance),
        "gripper_support": float(gripper_support),
        "obstacle_support": float(obstacle_radius),
        "surface_gap": float(surface_gap),
        "centerline_gripper_support": float(centerline_gripper_support),
        "centerline_obstacle_support": float(centerline_obstacle_support),
        "centerline_surface_gap": float(center_distance - centerline_gripper_support - centerline_obstacle_support),
    }


def _controller_scaled_commands(
    physical_actions: np.ndarray,
    *,
    controller: ControllerRollout,
) -> np.ndarray:
    """Apply Robosuite ``Controller.scale_action`` channel by channel."""
    actions = np.asarray(physical_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError("physical_actions must have shape (H, D) with D >= 6")
    input_min = np.asarray(controller.input_min, dtype=np.float64)
    input_max = np.asarray(controller.input_max, dtype=np.float64)
    output_min = np.asarray(controller.output_min, dtype=np.float64)
    output_max = np.asarray(controller.output_max, dtype=np.float64)
    if any(value.shape != (6,) for value in (input_min, input_max, output_min, output_max)):
        raise ValueError("controller input/output limits must each contain six arm channels")
    clipped = np.clip(actions[:, :6], input_min, input_max)
    scale = np.divide(output_max - output_min, input_max - input_min)
    return (clipped - (input_max + input_min) / 2.0) * scale + (output_max + output_min) / 2.0


def rollout_eef_trajectory(
    physical_actions: np.ndarray,
    *,
    eef_position: np.ndarray,
    eef_rotation: np.ndarray,
    ellipsoid_offset: np.ndarray,
    controller: ControllerRollout,
) -> EefTrajectory:
    """Recursively predict achieved OSC pose and the offset collision center.

    OSC constructs ``goal_pos = current_pos + scaled_delta[:3]`` and
    ``goal_ori = Exp(scaled_delta[3:]) @ current_ori``. The calibrated gains
    predict the fraction reached in one control interval.
    """
    actions = np.asarray(physical_actions, dtype=np.float64)
    position = np.asarray(eef_position, dtype=np.float64)
    rotation = np.asarray(eef_rotation, dtype=np.float64)
    offset = np.asarray(ellipsoid_offset, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError("physical_actions must have shape (H, D) with D >= 6")
    if not len(actions):
        raise ValueError("physical_actions must contain at least one future action")
    if position.shape != (3,):
        raise ValueError("eef_position must have shape (3,)")
    if rotation.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("eef_rotation and ellipsoid_offset must have shapes (3,3) and (3,)")

    scaled = _controller_scaled_commands(actions, controller=controller)
    translation_gain = np.broadcast_to(np.asarray(controller.translation_response_gain, dtype=np.float64), (3,))
    rotation_gain = np.broadcast_to(np.asarray(controller.rotation_response_gain, dtype=np.float64), (3,))
    achieved_translation = scaled[:, :3] * translation_gain
    achieved_rotation_vectors = scaled[:, 3:6] * rotation_gain
    positions, rotations, centers = [], [], []
    for delta_position, delta_rotation in zip(achieved_translation, achieved_rotation_vectors, strict=True):
        position = position + delta_position
        # Robosuite left-multiplies the world-frame axis-angle error.
        rotation = Rotation.from_rotvec(delta_rotation).as_matrix() @ rotation
        positions.append(position.copy())
        rotations.append(rotation.copy())
        centers.append(position + rotation @ offset)
    return EefTrajectory(
        positions=np.asarray(positions),
        rotations=np.asarray(rotations),
        ellipsoid_centers=np.asarray(centers),
        scaled_commands=scaled,
        achieved_translation_deltas=achieved_translation,
        achieved_rotation_vectors=achieved_rotation_vectors,
    )


def predict_eef_positions(
    physical_actions: np.ndarray,
    *,
    eef_position: np.ndarray,
    action_dt: float,
) -> np.ndarray:
    """Legacy ideal translational rollout retained for ablations."""
    actions = np.asarray(physical_actions, dtype=np.float64)
    position = np.asarray(eef_position, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise ValueError("physical_actions must have shape (H, D) with D >= 3")
    if not len(actions):
        raise ValueError("physical_actions must contain at least one future action")
    if position.shape != (3,):
        raise ValueError("eef_position must have shape (3,)")
    if not np.isfinite(action_dt) or action_dt <= 0.0:
        raise ValueError("action_dt must be a finite positive control timestep")
    return position[None, :] + action_dt * np.cumsum(actions[:, :3], axis=0)


def trajectory_barriers(
    physical_actions: np.ndarray,
    *,
    eef_position: np.ndarray,
    eef_rotation: np.ndarray,
    eef_radii: np.ndarray,
    ellipsoid_offset: np.ndarray,
    controller: ControllerRollout,
    obstacle: ObstaclePrimitive,
    safe_distance: float,
    carried_object_offset: np.ndarray | None = None,
    carried_object_rotation: np.ndarray | None = None,
    carried_object_size: np.ndarray | None = None,
) -> np.ndarray:
    """Full-pose barriers for the gripper or grasp-conditioned compound body.

    When carried-object geometry is supplied, its center translates rigidly
    with the gripper ellipsoid center while its rotation remains fixed in the
    world frame.  The compound barrier is the minimum of the ellipsoid and
    carried-box gaps to the active obstacle.
    """
    trajectory = rollout_eef_trajectory(
        physical_actions,
        eef_position=eef_position,
        eef_rotation=eef_rotation,
        ellipsoid_offset=ellipsoid_offset,
        controller=controller,
    )
    ellipsoid_barriers = np.asarray(
        [ellipsoid_obstacle_gap(
                Ellipsoid(
                    center,
                    rotation,
                    np.asarray(eef_radii),
                ),
                obstacle,
            )
            - safe_distance
            for center, rotation in zip(trajectory.ellipsoid_centers, trajectory.rotations, strict=True)
        ],
        dtype=np.float64,
    )
    supplied = (
        carried_object_offset is not None,
        carried_object_rotation is not None,
        carried_object_size is not None,
    )
    if not any(supplied):
        return ellipsoid_barriers
    if not all(supplied):
        raise ValueError("carried-object offset, rotation, and size must be supplied together")
    offset = np.asarray(carried_object_offset, dtype=np.float64)
    rotation = np.asarray(carried_object_rotation, dtype=np.float64)
    size = np.asarray(carried_object_size, dtype=np.float64)
    if offset.shape != (3,) or rotation.shape != (3, 3) or size.shape != (3,):
        raise ValueError("carried-object geometry must have shapes (3,), (3,3), and (3,)")
    carried_barriers = np.asarray(
        [
            primitive_obstacle_gap(
                ObstaclePrimitive("obb", center + offset, rotation, size),
                obstacle,
            )
            - safe_distance
            for center in trajectory.ellipsoid_centers
        ],
        dtype=np.float64,
    )
    return np.minimum(ellipsoid_barriers, carried_barriers)


def linearize_barriers(
    normalized_actions: np.ndarray,
    normalized_to_physical,
    barrier_function,
    *,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return B(a) and dB/d(a_xyz) in the normalized action convention."""
    action = np.asarray(normalized_actions, dtype=np.float64)
    base = np.asarray(barrier_function(normalized_to_physical(action)), dtype=np.float64)
    horizon = action.shape[0]
    jacobian = np.empty((len(base), horizon * 3), dtype=np.float64)
    for flat_index in range(horizon * 3):
        token, channel = divmod(flat_index, 3)
        plus, minus = action.copy(), action.copy()
        plus[token, channel] += epsilon
        minus[token, channel] -= epsilon
        jacobian[:, flat_index] = (
            np.asarray(barrier_function(normalized_to_physical(plus)))
            - np.asarray(barrier_function(normalized_to_physical(minus)))
        ) / (2.0 * epsilon)
    return base, jacobian


def trajectory_cbf_constraints(
    nominal_barriers: np.ndarray,
    barrier_jacobian: np.ndarray,
    initial_barrier: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the paper's CBF evolution along an action chunk."""
    barriers = np.asarray(nominal_barriers, dtype=np.float64)
    jacobian = np.asarray(barrier_jacobian, dtype=np.float64)
    if jacobian.shape[0] != len(barriers):
        raise ValueError("one barrier Jacobian row is required per trajectory step")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    decay = 1.0 - gamma
    matrix = jacobian.copy()
    rhs = np.empty_like(barriers)
    rhs[0] = decay * initial_barrier - barriers[0]
    if len(barriers) > 1:
        matrix[1:] -= decay * jacobian[:-1]
        rhs[1:] = decay * barriers[:-1] - barriers[1:]
    return matrix, rhs


def solve_action_expert_qp(
    success_gradient: np.ndarray,
    barrier_jacobian: np.ndarray,
    barrier_rhs: np.ndarray,
    *,
    lambda_deviation: float,
    beta_success: float,
    trust_radius: float,
    trust_region_norm: str = "linf",
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    tolerance: float = 1e-7,
) -> QPResult:
    """Solve Eq. (16) of the draft over the 10x3 translation correction."""
    gradient = np.asarray(success_gradient, dtype=np.float64).reshape(-1)
    matrix = np.asarray(barrier_jacobian, dtype=np.float64)
    rhs = np.asarray(barrier_rhs, dtype=np.float64).reshape(-1)
    if matrix.shape != (len(rhs), len(gradient)):
        raise ValueError(f"constraint shape {matrix.shape} does not match {(len(rhs), len(gradient))}")
    if lambda_deviation <= 0.0 or trust_radius < 0.0 or beta_success < 0.0:
        raise ValueError("lambda_deviation must be positive and beta/trust non-negative")
    if trust_region_norm not in {"linf", "l2"}:
        raise ValueError("trust_region_norm must be 'linf' or 'l2'")

    lower = (
        np.full_like(gradient, -trust_radius)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    )
    upper = (
        np.full_like(gradient, trust_radius)
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    )
    if lower.shape != gradient.shape or upper.shape != gradient.shape:
        raise ValueError("QP correction bounds must match the flattened gradient")
    if np.any(lower > upper):
        raise ValueError("QP lower bounds must not exceed upper bounds")

    if trust_region_norm == "l2":
        correction_variable = cp.Variable(len(gradient))
        constraints = [
            correction_variable >= lower,
            correction_variable <= upper,
            cp.norm(correction_variable, 2) <= trust_radius,
        ]
        if len(rhs):
            constraints.append(matrix @ correction_variable >= rhs)
        problem = cp.Problem(
            cp.Minimize(
                lambda_deviation * cp.sum_squares(correction_variable)
                - beta_success * gradient @ correction_variable
            ),
            constraints,
        )
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except cp.error.SolverError as exc:
            return QPResult(
                correction=np.zeros_like(gradient, dtype=np.float32),
                success=False,
                status=f"CLARABEL solver error: {exc}",
                minimum_linearized_margin=float("-inf"),
                objective=0.0,
            )
        value = correction_variable.value
        correction = np.zeros_like(gradient) if value is None else np.asarray(value, dtype=np.float64)
        margin = float(np.min(matrix @ correction - rhs)) if len(rhs) else float("inf")
        feasible = bool(
            problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and np.all(np.isfinite(correction))
            and margin >= -tolerance
            and np.linalg.norm(correction) <= trust_radius + tolerance
        )
        if not feasible:
            correction = np.zeros_like(gradient)
        objective = lambda_deviation * float(correction @ correction) - beta_success * float(
            gradient @ correction
        )
        return QPResult(
            correction=correction.astype(np.float32),
            success=feasible,
            status=f"CLARABEL: {problem.status}",
            minimum_linearized_margin=margin,
            objective=objective,
        )

    unconstrained = beta_success * gradient / (2.0 * lambda_deviation)
    initial = np.clip(unconstrained, lower, upper)

    # SLSQP is much more reliable when it starts from a feasible point.  More
    # importantly, its status reports whether it believes it found an optimum,
    # not whether the returned point is safe.  The old implementation treated
    # statuses such as "positive directional derivative" as infeasible and
    # threw away a constraint-feasible candidate.  First solve the bounded
    # linear feasibility problem with HiGHS, then let SLSQP reduce the QP cost.
    feasibility_status = ""
    if len(rhs):
        feasibility = linprog(
            np.zeros_like(gradient),
            A_ub=-matrix,
            b_ub=-rhs,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        feasibility_status = str(feasibility.message)
        if feasibility.success and feasibility.x is not None:
            initial = np.asarray(feasibility.x, dtype=np.float64)

    def objective(value):
        return lambda_deviation * float(value @ value) - beta_success * float(gradient @ value)

    def derivative(value):
        return 2.0 * lambda_deviation * value - beta_success * gradient

    constraints = []
    if len(rhs):
        constraints.append(LinearConstraint(matrix, rhs, np.full_like(rhs, np.inf)))
    result = minimize(
        objective,
        initial,
        jac=derivative,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    correction = np.asarray(result.x if result.x is not None else initial, dtype=np.float64)
    margin = float(np.min(matrix @ correction - rhs)) if len(rhs) else float("inf")
    # Safety depends on feasibility, not on SLSQP's optimality flag.  Exact
    # nonlinear trajectory barriers are re-evaluated by the caller before this
    # action chunk can be returned.
    feasible = bool(np.all(np.isfinite(correction)) and margin >= -tolerance)
    if not feasible:
        correction = np.zeros_like(gradient)
    status = str(result.message)
    if feasibility_status:
        status = f"linear feasibility: {feasibility_status}; QP: {status}"
    return QPResult(
        correction=correction.astype(np.float32),
        success=feasible,
        status=status,
        minimum_linearized_margin=margin,
        objective=float(objective(correction)),
    )


def solve_maximin_barrier_qp(
    nominal_barriers: np.ndarray,
    barrier_jacobian: np.ndarray,
    *,
    trust_radius: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    success_gradient: np.ndarray | None = None,
    lambda_deviation: float = 1.0,
    beta_success: float = 0.0,
    tolerance: float = 1e-7,
) -> QPResult:
    """Find the minimum-norm correction with the best attainable barrier floor.

    This is the fail-soft counterpart to the hard safety projection. First an
    LP finds the largest common affine barrier value ``z``. Then the existing
    convex quadratic projection finds the minimum-norm correction attaining
    that floor. A negative ``z`` means full safety is infeasible, but the
    returned correction is still the locally least-unsafe one.
    """
    barriers = np.asarray(nominal_barriers, dtype=np.float64).reshape(-1)
    jacobian = np.asarray(barrier_jacobian, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    if jacobian.shape != (len(barriers), len(lower)) or upper.shape != lower.shape:
        raise ValueError("max-min barrier dimensions do not match")

    # B_i + J_i d >= z  <=>  -J_i d + z <= B_i.
    objective = np.zeros(len(lower) + 1, dtype=np.float64)
    objective[-1] = -1.0
    inequalities = np.column_stack((-jacobian, np.ones(len(barriers))))
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=barriers,
        bounds=[*zip(lower, upper, strict=True), (None, None)],
        method="highs",
    )
    if not result.success or result.x is None:
        return QPResult(
            correction=np.zeros_like(lower, dtype=np.float32),
            success=False,
            status=f"max-min LP failed: {result.message}",
            minimum_linearized_margin=float("-inf"),
            objective=0.0,
        )

    best_floor = float(result.x[-1])
    # Allow numerical tolerance below the LP optimum so the quadratic solve
    # is not rejected due to HiGHS / SLSQP feasibility tolerances.
    floor = best_floor - max(tolerance, 1e-9)
    gradient = (
        np.zeros_like(lower)
        if success_gradient is None
        else np.asarray(success_gradient, dtype=np.float64).reshape(-1)
    )
    if gradient.shape != lower.shape:
        raise ValueError("max-min success gradient must match correction dimensions")
    qp = solve_action_expert_qp(
        gradient,
        jacobian,
        floor - barriers,
        lambda_deviation=lambda_deviation,
        beta_success=beta_success,
        trust_radius=trust_radius,
        lower_bounds=lower,
        upper_bounds=upper,
        tolerance=tolerance,
    )
    return QPResult(
        correction=qp.correction,
        success=qp.success,
        status=f"max-min barrier floor={best_floor:.9f}; {qp.status}",
        minimum_linearized_margin=qp.minimum_linearized_margin,
        objective=qp.objective,
    )


def project_action_chunk_with_qp(
    normalized_actions: np.ndarray,
    normalized_to_physical,
    barrier_function,
    *,
    trust_radius: float,
    max_iterations: int = 5,
    safety_buffer: float = 1e-4,
    tolerance: float = 1e-7,
    enable_nonlinear_fallback: bool = True,
    success_gradient: np.ndarray | None = None,
    lambda_deviation: float = 1.0,
    beta_success: float = 0.0,
    first_step_recovery_floor: float | None = None,
) -> ChunkProjectionResult:
    """Project a final action chunk until every exact H-step barrier is safe.

    Each sequential QP has one constraint for every future action position.
    The returned success flag is based on a fresh nonlinear barrier evaluation,
    not merely the linearized QP constraints.
    """
    original = np.asarray(normalized_actions, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] < 3 or not len(original):
        raise ValueError("normalized_actions must have shape (H, D), H > 0, D >= 3")
    if trust_radius < 0.0 or max_iterations < 1 or safety_buffer < 0.0:
        raise ValueError("projection settings must be non-negative and iterations positive")

    candidate = original.copy()
    total_correction = np.zeros((len(candidate), 3), dtype=np.float64)
    barriers_before = np.asarray(barrier_function(normalized_to_physical(candidate)), dtype=np.float64)
    barriers_after = barriers_before.copy()
    if np.all(barriers_after >= -tolerance):
        if success_gradient is not None and beta_success > 0.0:
            base_barriers, jacobian = linearize_barriers(
                candidate, normalized_to_physical, barrier_function
            )
            result = solve_action_expert_qp(
                success_gradient,
                jacobian,
                -base_barriers,
                lambda_deviation=lambda_deviation,
                beta_success=beta_success,
                trust_radius=trust_radius,
                tolerance=tolerance,
            )
            if result.success:
                step = result.correction.reshape(len(candidate), 3).astype(np.float64)
                guided = candidate.copy()
                guided[:, :3] += step
                guided_barriers = np.asarray(
                    barrier_function(normalized_to_physical(guided)), dtype=np.float64
                )
                if np.all(guided_barriers >= -tolerance):
                    return ChunkProjectionResult(
                        actions=guided.astype(np.float32),
                        correction=step.astype(np.float32),
                        barriers_before=barriers_before.astype(np.float32),
                        barriers_after=guided_barriers.astype(np.float32),
                        success=True,
                        iterations=1,
                        status=f"safe success-guided projection: {result.status}",
                    )
        return ChunkProjectionResult(
            actions=candidate.astype(np.float32),
            correction=total_correction.astype(np.float32),
            barriers_before=barriers_before.astype(np.float32),
            barriers_after=barriers_after.astype(np.float32),
            success=True,
            iterations=0,
            status="already safe",
        )

    statuses = []
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        base_barriers, jacobian = linearize_barriers(
            candidate,
            normalized_to_physical,
            barrier_function,
        )
        flattened_total = total_correction.reshape(-1)
        result = solve_action_expert_qp(
            (
                np.zeros_like(flattened_total)
                if success_gradient is None
                else np.asarray(success_gradient, dtype=np.float64).reshape(-1)
            ),
            jacobian,
            safety_buffer - base_barriers,
            lambda_deviation=lambda_deviation,
            beta_success=beta_success,
            trust_radius=trust_radius,
            lower_bounds=-trust_radius - flattened_total,
            upper_bounds=trust_radius - flattened_total,
            tolerance=tolerance,
        )
        statuses.append(result.status)
        if not result.success:
            if first_step_recovery_floor is not None:
                statuses.append("hard horizon projection infeasible; trying first-step recovery")
                break
            # Full safety can be unreachable when the measured state already
            # violates the margin. Do not silently execute the nominal suffix:
            # solve a second convex problem for the greatest attainable
            # minimum barrier and retain it only after exact reevaluation.
            fail_soft = solve_maximin_barrier_qp(
                base_barriers,
                jacobian,
                trust_radius=trust_radius,
                lower_bounds=-trust_radius - flattened_total,
                upper_bounds=trust_radius - flattened_total,
                success_gradient=success_gradient,
                lambda_deviation=lambda_deviation,
                beta_success=beta_success,
                tolerance=tolerance,
            )
            statuses.append(fail_soft.status)
            if not fail_soft.success:
                break
            step = fail_soft.correction.reshape(len(candidate), 3).astype(np.float64)
            fail_soft_candidate = candidate.copy()
            fail_soft_candidate[:, :3] += step
            fail_soft_barriers = np.asarray(
                barrier_function(normalized_to_physical(fail_soft_candidate)), dtype=np.float64
            )
            if np.min(fail_soft_barriers) <= np.min(barriers_after) + tolerance:
                statuses.append("max-min correction rejected: exact minimum barrier did not improve")
                break
            candidate = fail_soft_candidate
            total_correction += step
            barriers_after = fail_soft_barriers
            statuses.append(
                "accepted fail-soft correction: exact minimum barrier "
                f"{np.min(barriers_before):.9f} -> {np.min(barriers_after):.9f}"
            )
            if np.max(np.abs(step)) <= tolerance:
                break
            continue
        step = result.correction.reshape(len(candidate), 3).astype(np.float64)
        candidate[:, :3] += step
        total_correction += step
        barriers_after = np.asarray(barrier_function(normalized_to_physical(candidate)), dtype=np.float64)
        if np.all(barriers_after >= -tolerance):
            return ChunkProjectionResult(
                actions=candidate.astype(np.float32),
                correction=total_correction.astype(np.float32),
                barriers_before=barriers_before.astype(np.float32),
                barriers_after=barriers_after.astype(np.float32),
                success=True,
                iterations=iterations,
                status="; ".join(statuses),
            )
        if np.max(np.abs(step)) <= tolerance:
            break

    if first_step_recovery_floor is not None:
        base_barriers, jacobian = linearize_barriers(
            original,
            normalized_to_physical,
            barrier_function,
        )
        recovery = solve_action_expert_qp(
            np.zeros(len(original) * 3, dtype=np.float64),
            jacobian[:1],
            np.asarray([first_step_recovery_floor - base_barriers[0]], dtype=np.float64),
            lambda_deviation=1.0,
            beta_success=0.0,
            trust_radius=trust_radius,
            tolerance=tolerance,
        )
        if recovery.success:
            recovery_correction = recovery.correction.reshape(len(original), 3).astype(np.float64)
            recovery_candidate = original.copy()
            recovery_candidate[:, :3] += recovery_correction
            recovery_barriers = np.asarray(
                barrier_function(normalized_to_physical(recovery_candidate)),
                dtype=np.float64,
            )
            if recovery_barriers[0] >= first_step_recovery_floor - tolerance:
                statuses.append(
                    "first-step recovery floor "
                    f"{first_step_recovery_floor:.9f} achieved at {recovery_barriers[0]:.9f}"
                )
                return ChunkProjectionResult(
                    actions=recovery_candidate.astype(np.float32),
                    correction=recovery_correction.astype(np.float32),
                    barriers_before=barriers_before.astype(np.float32),
                    barriers_after=recovery_barriers.astype(np.float32),
                    success=bool(np.all(recovery_barriers >= -tolerance)),
                    iterations=iterations,
                    status="; ".join(statuses),
                )
        statuses.append(f"first-step recovery failed: {recovery.status}")

    # The exact multi-axis barrier is a maximum over valid separating axes.
    # Consequently, the safe set is a union of convex regions.  A sequential
    # QP around one active axis may report its local half-space infeasible even
    # though a safe hold or a detour through another separating axis exists.
    # Use bounded multi-start nonlinear projection only as that fallback.  The
    # objective remains minimum deviation from pi0.5 and every accepted result
    # is checked with the exact full-H barrier below.
    if not enable_nonlinear_fallback:
        barriers_after = np.asarray(barrier_function(normalized_to_physical(candidate)), dtype=np.float64)
        return ChunkProjectionResult(
            actions=candidate.astype(np.float32),
            correction=total_correction.astype(np.float32),
            barriers_before=barriers_before.astype(np.float32),
            barriers_after=barriers_after.astype(np.float32),
            success=bool(np.all(barriers_after >= -tolerance)),
            iterations=iterations,
            status="; ".join(statuses) if statuses else "projection was not attempted",
        )

    flattened_size = len(candidate) * 3
    lower = np.full(flattened_size, -trust_radius, dtype=np.float64)
    upper = np.full(flattened_size, trust_radius, dtype=np.float64)

    def corrected_actions(flattened):
        value = original.copy()
        value[:, :3] += np.asarray(flattened, dtype=np.float64).reshape(-1, 3)
        return value

    def exact_constraints(flattened):
        return (
            np.asarray(
                barrier_function(normalized_to_physical(corrected_actions(flattened))),
                dtype=np.float64,
            )
            - safety_buffer
        )

    physical_original = np.asarray(normalized_to_physical(original), dtype=np.float64)
    normalized_slopes = np.empty((len(original), 3), dtype=np.float64)
    slope_epsilon = 1e-3
    for token in range(len(original)):
        for channel in range(3):
            perturbed = original.copy()
            perturbed[token, channel] += slope_epsilon
            physical_perturbed = np.asarray(normalized_to_physical(perturbed), dtype=np.float64)
            normalized_slopes[token, channel] = (
                physical_perturbed[token, channel] - physical_original[token, channel]
            ) / slope_epsilon

    def correction_for_physical(desired_translation):
        desired = np.asarray(desired_translation, dtype=np.float64)
        correction = np.divide(
            desired - physical_original[:, :3],
            normalized_slopes,
            out=np.zeros_like(normalized_slopes),
            where=np.abs(normalized_slopes) > 1e-9,
        )
        return np.clip(correction.reshape(-1), lower, upper)

    starts = [
        np.clip(total_correction.reshape(-1), lower, upper),
        np.zeros(flattened_size, dtype=np.float64),
        correction_for_physical(np.zeros((len(original), 3), dtype=np.float64)),
    ]
    # Axis maneuvers provide starts in each connected component of the
    # separating-axis safe set.  Tapering them preserves a useful terminal
    # position and avoids the large persistent drift of a fixed half-space.
    taper = np.linspace(1.0, 0.0, len(original), dtype=np.float64)[:, None]
    for axis in np.eye(3):
        for sign in (-1.0, 1.0):
            for magnitude in (0.5, 1.0):
                desired = physical_original[:, :3].copy()
                desired += sign * magnitude * taper * axis
                starts.append(correction_for_physical(desired))

    feasible_candidates = []
    for initial in starts:
        result = minimize(
            lambda value: float(value @ value),
            initial,
            jac=lambda value: 2.0 * value,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=[{"type": "ineq", "fun": exact_constraints}],
            options={"ftol": 1e-10, "maxiter": 300, "disp": False},
        )
        value = np.asarray(result.x if result.x is not None else initial, dtype=np.float64)
        margins = exact_constraints(value)
        if np.all(np.isfinite(value)) and np.min(margins) >= -tolerance:
            feasible_candidates.append((float(value @ value), value, str(result.message)))

    if feasible_candidates:
        _, flattened, nonlinear_status = min(feasible_candidates, key=lambda item: item[0])
        candidate = corrected_actions(flattened)
        total_correction = flattened.reshape(-1, 3)
        barriers_after = np.asarray(barrier_function(normalized_to_physical(candidate)), dtype=np.float64)
        if np.all(barriers_after >= -tolerance):
            return ChunkProjectionResult(
                actions=candidate.astype(np.float32),
                correction=total_correction.astype(np.float32),
                barriers_before=barriers_before.astype(np.float32),
                barriers_after=barriers_after.astype(np.float32),
                success=True,
                iterations=iterations,
                status=("; ".join(statuses) + f"; nonlinear multi-axis projection: {nonlinear_status}"),
            )

    barriers_after = np.asarray(barrier_function(normalized_to_physical(candidate)), dtype=np.float64)
    return ChunkProjectionResult(
        actions=candidate.astype(np.float32),
        correction=total_correction.astype(np.float32),
        barriers_before=barriers_before.astype(np.float32),
        barriers_after=barriers_after.astype(np.float32),
        success=bool(np.all(barriers_after >= -tolerance)),
        iterations=iterations,
        status="; ".join(statuses) if statuses else "projection was not attempted",
    )


def refine_action_chunk_for_success(
    normalized_actions: np.ndarray,
    normalized_to_physical,
    barrier_function,
    success_gradient: np.ndarray,
    *,
    trust_radius: float,
    lambda_deviation: float,
    beta_success: float,
    trust_region_norm: str = "linf",
    barrier_tolerance: float = 1e-7,
) -> ChunkProjectionResult:
    """Improve critic score locally without reducing the achieved barrier floor.

    This stage is deliberately separate from safety recovery: ``trust_radius``
    limits critic-only motion, while the input action chunk is the output of
    the potentially larger safety projection.
    """
    original = np.asarray(normalized_actions, dtype=np.float64)
    barriers_before, jacobian = linearize_barriers(
        original, normalized_to_physical, barrier_function
    )
    floor = float(np.min(barriers_before))
    result = solve_action_expert_qp(
        success_gradient,
        jacobian,
        floor - barriers_before,
        lambda_deviation=lambda_deviation,
        beta_success=beta_success,
        trust_radius=trust_radius,
        trust_region_norm=trust_region_norm,
        tolerance=barrier_tolerance,
    )
    correction = result.correction.reshape(len(original), 3).astype(np.float64)
    candidate = original.copy()
    candidate[:, :3] += correction
    barriers_after = np.asarray(
        barrier_function(normalized_to_physical(candidate)), dtype=np.float64
    )
    preserved = bool(
        result.success
        and np.min(barriers_after) >= floor - barrier_tolerance
    )
    if not preserved:
        candidate = original.copy()
        correction = np.zeros_like(correction)
        barriers_after = barriers_before.copy()
    return ChunkProjectionResult(
        actions=candidate.astype(np.float32),
        correction=correction.astype(np.float32),
        barriers_before=barriers_before.astype(np.float32),
        barriers_after=barriers_after.astype(np.float32),
        success=preserved,
        iterations=1,
        status=(
            f"critic refinement preserved barrier floor {floor:.9f}; {result.status}"
            if preserved
            else f"critic refinement rejected at barrier floor {floor:.9f}; {result.status}"
        ),
    )


def project_action_chunk_with_adaptive_radius(
    normalized_actions: np.ndarray,
    normalized_to_physical,
    barrier_function,
    *,
    trust_radii: tuple[float, ...] | list[float],
    escalate_only_if_first_barrier_negative: bool = False,
    **projection_kwargs,
) -> AdaptiveProjectionResult:
    """Increase correction authority only until exact safety is recovered."""
    radii = tuple(float(value) for value in trust_radii)
    if not radii or any(value < 0.0 for value in radii):
        raise ValueError("adaptive trust radii must be a non-empty non-negative sequence")
    if any(right <= left for left, right in zip(radii, radii[1:])):
        raise ValueError("adaptive trust radii must be strictly increasing")
    if escalate_only_if_first_barrier_negative:
        initial_barriers = np.asarray(
            barrier_function(normalized_to_physical(normalized_actions)),
            dtype=np.float64,
        )
        if len(initial_barriers) == 0:
            raise ValueError("adaptive projection requires at least one barrier")
        if initial_barriers[0] >= 0.0:
            radii = radii[:1]
    attempts = []
    for radius in radii:
        result = project_action_chunk_with_qp(
            normalized_actions,
            normalized_to_physical,
            barrier_function,
            trust_radius=radius,
            **projection_kwargs,
        )
        attempts.append((radius, result))
        if result.success:
            return AdaptiveProjectionResult(result, radius, tuple(x[0] for x in attempts))
        if "first-step recovery floor" in result.status:
            return AdaptiveProjectionResult(result, radius, tuple(x[0] for x in attempts))
    radius, result = max(
        attempts,
        key=lambda item: float(np.min(item[1].barriers_after)),
    )
    return AdaptiveProjectionResult(result, radius, tuple(x[0] for x in attempts))
