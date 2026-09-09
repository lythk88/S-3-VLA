"""Success-guided, geometry-constrained corrections inside pi0.5 denoising."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies.action_expert_qp import ControllerRollout
from openpi.policies.action_expert_qp import ObstaclePrimitive
from openpi.policies.action_expert_qp import linearize_barriers
from openpi.policies.action_expert_qp import project_action_chunk_with_adaptive_radius
from openpi.policies.action_expert_qp import project_action_chunk_with_qp
from openpi.policies.action_expert_qp import refine_action_chunk_for_success
from openpi.policies.action_expert_qp import rollout_eef_trajectory
from openpi.policies.action_expert_qp import solve_action_expert_qp
from openpi.policies.action_expert_qp import trajectory_barriers
from openpi.policies.action_expert_qp import trajectory_cbf_constraints
from openpi.shared import nnx_utils


class _UnsafeActionChunkError(RuntimeError):
    """One stochastic pi0.5 sample could not be projected safely."""


class ActionExpertGuidancePolicy:
    """Apply the draft QP at selected denoising times, then resume the flow."""

    def __init__(self, base_policy, success_critic: torch.nn.Module, *, torch_device: str = "cpu"):
        self._base = base_policy
        self._critic = success_critic.to(torch_device).eval()
        self._torch_device = torch.device(torch_device)
        sampler = getattr(base_policy._model, "sample_actions_with_denoising_trace", None)
        if sampler is None:
            raise RuntimeError("The pi0 denoising-trace method was not installed")
        self._sample_with_trace = nnx_utils.module_jit(sampler, static_argnames=("num_steps",))

    @property
    def metadata(self):
        return self._base.metadata

    def _sample(self, rng, observation, action, start_time, steps):
        return self._sample_with_trace(
            rng,
            observation,
            start_noisy_action=jnp.asarray(action)[None],
            start_time=jnp.asarray(start_time, dtype=jnp.float32),
            num_steps=steps,
        )

    @staticmethod
    def _state(trace, requested_time):
        _, hidden, noisy, times, flow, active = trace
        times = np.asarray(times[0], dtype=np.float32)
        candidates = np.flatnonzero(np.asarray(active[0], dtype=np.bool_))
        index = int(candidates[np.argmin(np.abs(times[candidates] - requested_time))])
        if abs(float(times[index]) - requested_time) > 0.051:
            raise RuntimeError(f"guidance time {requested_time} absent from trace")
        return (
            np.asarray(hidden[0, index], dtype=np.float32),
            np.asarray(noisy[0, index], dtype=np.float32),
            float(times[index]),
            np.asarray(flow[0, index], dtype=np.float32),
        )

    def _success(self, hidden, action, denoising_time, *, gradient):
        hidden_tensor = torch.from_numpy(np.array(hidden, copy=True))[None].to(self._torch_device)
        action_tensor = torch.from_numpy(np.array(action, copy=True))[None].to(self._torch_device)
        time_tensor = torch.tensor([denoising_time], dtype=torch.float32, device=self._torch_device)
        action_tensor.requires_grad_(gradient)
        with torch.set_grad_enabled(gradient):
            logit = self._critic(hidden_tensor, action_tensor, time_tensor)
            objective = torch.nn.functional.logsigmoid(logit)
            derivative = torch.autograd.grad(objective.sum(), action_tensor)[0][0] if gradient else None
        return (
            None if derivative is None else derivative.detach().cpu().numpy().astype(np.float32),
            float(torch.sigmoid(logit).detach().cpu()),
        )

    def infer(self, obs: dict) -> dict:
        config = obs.get("__action_expert_guidance__")
        if config is None:
            return self._base.infer(obs)
        attempts = int(config.get("safety_resample_attempts", 16))
        if attempts < 1:
            raise ValueError("safety_resample_attempts must be positive")
        rejection_statuses = []
        for attempt in range(attempts):
            try:
                outputs = self._infer_once(obs)
                outputs["action_expert_safety_resamples"] = np.asarray(attempt, dtype=np.int32)
                return outputs
            except _UnsafeActionChunkError as exc:
                rejection_statuses.append(str(exc))
        raise RuntimeError(
            f"Refusing to execute after {attempts} independently sampled chunks all "
            f"failed the exact H=10 safety projection. Last rejection: "
            f"{rejection_statuses[-1]}"
        )

    def _infer_once(self, obs: dict) -> dict:
        inputs = dict(obs)
        config = inputs.pop("__action_expert_guidance__", None)
        if config is None:
            return self._base.infer(inputs)
        debug_noise = inputs.pop("__debug_noise__", None)
        inputs.pop("__debug_return_last_hidden_state__", None)
        steps = int(config.get("denoising_steps", 10))
        if steps != 10:
            raise ValueError("success critic was trained with ten denoising steps")
        action_horizon = int(self._base._model.action_horizon)
        qp_horizon = int(config.get("qp_horizon", action_horizon))
        if not 1 <= qp_horizon <= action_horizon:
            raise ValueError(
                f"qp_horizon must be in [1, {action_horizon}], got {qp_horizon}"
            )
        requested_times = sorted({float(x) for x in config.get("times", [0.5, 0.3, 0.1])}, reverse=True)
        if not requested_times:
            raise ValueError("At least one action-expert guidance time is required")
        geometry = config["geometry"]
        gamma = float(config.get("gamma", 0.9))
        trust_radius = float(config.get("trust_radius", 0.05))
        trust_region_norm = str(config.get("trust_region_norm", "linf"))
        final_safety_trust_radius = float(config.get("final_safety_trust_radius", 1.0))
        final_success_trust_radius = float(config.get("final_success_trust_radius", 0.05))
        adaptive_safety_trust_radii = tuple(
            float(value) for value in config.get("adaptive_safety_trust_radii", [])
        )
        adaptive_escalate_on_first_barrier_only = bool(
            config.get("adaptive_escalate_on_first_barrier_only", False)
        )
        minimal_intervention = bool(config.get("minimal_intervention", False))
        translation_only_execution = bool(config.get("translation_only_execution", False))
        first_step_recovery = bool(config.get("first_step_recovery", False))
        lambda_deviation = float(config.get("lambda_deviation", 1.0))
        beta_success = float(config.get("beta_success", 10.0))
        safe_distance = float(config.get("safe_distance", 0.01))
        continue_on_unsafe = bool(config.get("continue_on_unsafe", False))
        pre_execution_qp = bool(config.get("pre_execution_qp", True))
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if final_safety_trust_radius < trust_radius:
            raise ValueError("final_safety_trust_radius must be at least the guidance trust_radius")

        transformed = self._base._input_transform(jax.tree.map(lambda x: x, inputs))
        transformed = jax.tree.map(lambda x: jnp.asarray(x)[None], transformed)
        state = np.asarray(transformed["state"][0])
        observation = _model.Observation.from_dict(transformed)
        self._base._rng, sample_rng = jax.random.split(self._base._rng)
        if debug_noise is None:
            noise = np.asarray(
                jax.random.normal(
                    sample_rng,
                    (self._base._model.action_horizon, self._base._model.action_dim),
                    dtype=jnp.float32,
                )
            )
        else:
            noise = np.asarray(debug_noise, dtype=np.float32)

        eef_position = np.asarray(geometry["eef_position"], dtype=np.float64)
        eef_rotation = np.asarray(geometry["eef_rotation"], dtype=np.float64)
        eef_radii = np.asarray(geometry["eef_radii"], dtype=np.float64)
        ellipsoid_offset = np.asarray(geometry["ellipsoid_offset"], dtype=np.float64)
        controller = ControllerRollout(
            input_min=np.asarray(config["controller_input_min"], dtype=np.float64),
            input_max=np.asarray(config["controller_input_max"], dtype=np.float64),
            output_min=np.asarray(config["controller_output_min"], dtype=np.float64),
            output_max=np.asarray(config["controller_output_max"], dtype=np.float64),
            translation_response_gain=np.asarray(config.get("translation_response_gain", 0.22), dtype=np.float64),
            rotation_response_gain=np.asarray(config.get("rotation_response_gain", 0.22), dtype=np.float64),
        )
        obstacle = ObstaclePrimitive(
            str(geometry["obstacle_kind"]),
            np.asarray(geometry["obstacle_center"], dtype=np.float64),
            np.asarray(geometry["obstacle_rotation"], dtype=np.float64),
            np.asarray(geometry["obstacle_size"], dtype=np.float64),
            float(geometry.get("obstacle_top_padding", 0.0)),
        )
        carried_object_active = bool(geometry.get("carried_object_active", False))
        carried_object_offset = (
            np.asarray(geometry["carried_object_offset"], dtype=np.float64)
            if carried_object_active
            else None
        )
        carried_object_rotation = (
            np.asarray(geometry["carried_object_rotation"], dtype=np.float64)
            if carried_object_active
            else None
        )
        carried_object_size = (
            np.asarray(geometry["carried_object_size"], dtype=np.float64)
            if carried_object_active
            else None
        )

        def to_physical(normalized):
            physical = np.asarray(
                self._base._output_transform({"state": state, "actions": np.asarray(normalized, dtype=np.float32)})[
                    "actions"
                ],
                dtype=np.float64,
            )
            if translation_only_execution:
                physical = physical.copy()
                physical[:, 3:6] = 0.0
            return physical

        def barriers(physical):
            return trajectory_barriers(
                physical,
                eef_position=eef_position,
                eef_rotation=eef_rotation,
                eef_radii=eef_radii,
                ellipsoid_offset=ellipsoid_offset,
                controller=controller,
                obstacle=obstacle,
                safe_distance=safe_distance,
                carried_object_offset=carried_object_offset,
                carried_object_rotation=carried_object_rotation,
                carried_object_size=carried_object_size,
            )

        def qp_prefix(actions):
            actions = np.asarray(actions)
            if len(actions) < qp_horizon:
                raise ValueError(
                    f"QP requires {qp_horizon} actions, but received {len(actions)}"
                )
            return actions[:qp_horizon]

        def replace_qp_prefix(actions, corrected_prefix):
            result = np.asarray(actions).copy()
            corrected_prefix = np.asarray(corrected_prefix)
            if len(corrected_prefix) != qp_horizon:
                raise ValueError(
                    f"Corrected QP prefix must contain {qp_horizon} actions, "
                    f"got {len(corrected_prefix)}"
                )
            result[:qp_horizon] = corrected_prefix
            return result

        start = time.monotonic()
        nominal_trace = self._sample(sample_rng, observation, noise, 1.0, steps)
        nominal_actions = np.asarray(nominal_trace[0][0], dtype=np.float32)
        nominal_physical_actions = to_physical(nominal_actions)
        nominal_trajectory = rollout_eef_trajectory(
            nominal_physical_actions,
            eef_position=eef_position,
            eef_rotation=eef_rotation,
            ellipsoid_offset=ellipsoid_offset,
            controller=controller,
        )
        nominal_trajectory_positions = nominal_trajectory.positions
        # pi0.5 and the success critic always retain their trained H=10 input.
        # Safety QPs certify only the receding prefix requested by qp_horizon.
        nominal_trajectory_barriers = barriers(qp_prefix(nominal_physical_actions))

        if minimal_intervention and nominal_trajectory_barriers[0] >= 0.0:
            nominal_hidden = np.asarray(nominal_trace[1][0], dtype=np.float32)
            nominal_active = np.asarray(nominal_trace[5][0], dtype=np.bool_)
            active_indices = np.flatnonzero(nominal_active)
            last_hidden = nominal_hidden[active_indices[-1]] if len(active_indices) else nominal_hidden[-1]
            _, nominal_success = self._success(last_hidden, nominal_actions, 0.0, gradient=False)
            # Keep the diagnostic arrays rectangular across passthrough and
            # corrected chunks. A passthrough still evaluates the critic at
            # every configured denoising time, but never applies its gradient.
            passthrough_times = []
            passthrough_scores = []
            passthrough_barriers = []
            dt = -1.0 / steps
            for requested_time in requested_times:
                hidden, current, actual_time, flow = self._state(nominal_trace, requested_time)
                stage_action = current + dt * flow
                _, stage_success = self._success(hidden, stage_action, actual_time, gradient=False)
                stage_physical = to_physical(qp_prefix(stage_action))
                passthrough_times.append(actual_time)
                passthrough_scores.append(stage_success)
                passthrough_barriers.append(float(np.min(barriers(stage_physical))))
            if pre_execution_qp:
                passthrough_times.append(0.0)
                passthrough_scores.append(nominal_success)
                passthrough_barriers.append(float(np.min(nominal_trajectory_barriers)))
            telemetry_length = len(passthrough_times)
            outputs = self._base._output_transform({"state": state, "actions": nominal_actions})
            outputs.update(
                {
                    "normalized_actions": nominal_actions,
                    "last_layer_hidden_state": last_hidden,
                    "action_expert_intervened": np.asarray(0, dtype=np.bool_),
                    "action_expert_nominal_trajectory_positions": nominal_trajectory_positions.astype(np.float32),
                    "action_expert_nominal_trajectory_barriers": nominal_trajectory_barriers.astype(np.float32),
                    "action_expert_final_trajectory_barriers": nominal_trajectory_barriers.astype(np.float32),
                    "action_expert_final_projection_success": np.asarray(
                        np.all(nominal_trajectory_barriers >= -1e-7), dtype=np.bool_
                    ),
                    "action_expert_certified_horizon": np.asarray(qp_horizon, dtype=np.int32),
                    "action_expert_final_projection_iterations": np.asarray(0, dtype=np.int32),
                    "action_expert_final_projection_correction_rms": np.asarray(0.0, dtype=np.float32),
                    "action_expert_final_safety_correction_rms": np.asarray(0.0, dtype=np.float32),
                    "action_expert_final_critic_correction_rms": np.asarray(0.0, dtype=np.float32),
                    "action_expert_final_critic_refinement_accepted": np.asarray(False, dtype=np.bool_),
                    "action_expert_selected_safety_trust_radius": np.asarray(0.0, dtype=np.float32),
                    "action_expert_attempted_safety_trust_radii": np.asarray([], dtype=np.float32),
                    "action_expert_final_success_score_before": np.asarray(nominal_success, dtype=np.float32),
                    "action_expert_final_success_score_after": np.asarray(nominal_success, dtype=np.float32),
                    "action_expert_times": np.asarray(passthrough_times, dtype=np.float32),
                    "action_expert_success_scores_before": np.asarray(passthrough_scores, dtype=np.float32),
                    "action_expert_success_scores_after": np.asarray(passthrough_scores, dtype=np.float32),
                    "action_expert_qp_success": np.zeros(telemetry_length, dtype=np.bool_),
                    "action_expert_qp_margins": np.full(telemetry_length, np.inf, dtype=np.float32),
                    "action_expert_correction_rms": np.zeros(telemetry_length, dtype=np.float32),
                    "action_expert_barriers_before": np.full(
                        telemetry_length, nominal_trajectory_barriers[0], dtype=np.float32
                    ),
                    "action_expert_barriers_nominal": np.asarray(passthrough_barriers, dtype=np.float32),
                    "policy_timing": {"infer_ms": (time.monotonic() - start) * 1000.0},
                }
            )
            return outputs

        working_trace = nominal_trace
        telemetry = []
        dt = -1.0 / steps
        initial_barrier = float(
            np.min(barriers(np.zeros((qp_horizon, 7), dtype=np.float64)))
        )
        for requested_time in requested_times:
            hidden, current, actual_time, flow = self._state(working_trace, requested_time)
            nominal = current + dt * flow
            if minimal_intervention:
                _, score = self._success(hidden, nominal, actual_time, gradient=False)
                nominal_barriers = barriers(to_physical(qp_prefix(nominal)))
                telemetry.append(
                    {
                        "time": actual_time,
                        "score_before": score,
                        "score_after": score,
                        "qp_success": False,
                        "qp_margin": np.inf,
                        "correction_rms": 0.0,
                        "barrier_before": initial_barrier,
                        "barrier_nominal": float(np.min(nominal_barriers)),
                        "status": "minimal-intervention critic evaluation; denoising QP skipped",
                    }
                )
                continue
            gradient, score_before = self._success(hidden, nominal, actual_time, gradient=True)
            assert gradient is not None
            nominal_prefix = qp_prefix(nominal)
            nominal_barriers, jacobian = linearize_barriers(
                nominal_prefix, to_physical, barriers
            )
            constraint_matrix, rhs = trajectory_cbf_constraints(nominal_barriers, jacobian, initial_barrier, gamma)
            result = solve_action_expert_qp(
                gradient[:qp_horizon, :3],
                constraint_matrix,
                rhs,
                lambda_deviation=lambda_deviation,
                beta_success=beta_success,
                trust_radius=trust_radius,
                trust_region_norm=trust_region_norm,
            )
            correction = np.zeros_like(nominal)
            correction[:qp_horizon, :3] = result.correction.reshape(-1, 3)
            corrected = nominal + correction
            _, score_after = self._success(hidden, corrected, actual_time, gradient=False)
            telemetry.append(
                {
                    "time": actual_time,
                    "score_before": score_before,
                    "score_after": score_after,
                    "qp_success": result.success,
                    "qp_margin": result.minimum_linearized_margin,
                    "correction_rms": float(
                        np.sqrt(np.mean(np.square(correction[:qp_horizon, :3])))
                    ),
                    "barrier_before": initial_barrier,
                    "barrier_nominal": float(np.min(nominal_barriers)),
                    "status": result.status,
                }
            )
            working_trace = self._sample(sample_rng, observation, corrected, max(actual_time + dt, 0.0), steps)

        guided_actions, guided_hidden, _, _, _, active = working_trace
        denoised_actions = np.asarray(guided_actions[0], dtype=np.float32)
        active_indices = np.flatnonzero(np.asarray(active[0], dtype=np.bool_))
        pre_execution_hidden = (
            np.asarray(guided_hidden[0, active_indices[-1]], dtype=np.float32) if len(active_indices) else hidden
        )
        if pre_execution_qp and minimal_intervention:
            _, score = self._success(
                pre_execution_hidden,
                denoised_actions,
                0.0,
                gradient=False,
            )
            barriers_at_zero = barriers(to_physical(qp_prefix(denoised_actions)))
            telemetry.append(
                {
                    "time": 0.0,
                    "score_before": score,
                    "score_after": score,
                    "qp_success": False,
                    "qp_margin": np.inf,
                    "correction_rms": 0.0,
                    "barrier_before": initial_barrier,
                    "barrier_nominal": float(np.min(barriers_at_zero)),
                    "status": "minimal-intervention critic evaluation; pre-execution denoising QP skipped",
                }
            )
        elif pre_execution_qp:
            gradient, score_before = self._success(
                pre_execution_hidden,
                denoised_actions,
                0.0,
                gradient=True,
            )
            assert gradient is not None
            denoised_prefix = qp_prefix(denoised_actions)
            barriers_at_zero, jacobian = linearize_barriers(
                denoised_prefix,
                to_physical,
                barriers,
            )
            constraint_matrix, rhs = trajectory_cbf_constraints(
                barriers_at_zero,
                jacobian,
                initial_barrier,
                gamma,
            )
            result = solve_action_expert_qp(
                gradient[:qp_horizon, :3],
                constraint_matrix,
                rhs,
                lambda_deviation=lambda_deviation,
                beta_success=beta_success,
                trust_radius=trust_radius,
                trust_region_norm=trust_region_norm,
            )
            correction = np.zeros_like(denoised_actions)
            correction[:qp_horizon, :3] = result.correction.reshape(-1, 3)
            corrected_at_zero = denoised_actions + correction
            _, score_after = self._success(
                pre_execution_hidden,
                corrected_at_zero,
                0.0,
                gradient=False,
            )
            telemetry.append(
                {
                    "time": 0.0,
                    "score_before": score_before,
                    "score_after": score_after,
                    "qp_success": result.success,
                    "qp_margin": result.minimum_linearized_margin,
                    "correction_rms": float(
                        np.sqrt(np.mean(np.square(correction[:qp_horizon, :3])))
                    ),
                    "barrier_before": initial_barrier,
                    "barrier_nominal": float(np.min(barriers_at_zero)),
                    "status": result.status,
                }
            )
            denoised_actions = corrected_at_zero
        if adaptive_safety_trust_radii:
            adaptive_projection = project_action_chunk_with_adaptive_radius(
                qp_prefix(denoised_actions),
                to_physical,
                barriers,
                trust_radii=adaptive_safety_trust_radii,
                escalate_only_if_first_barrier_negative=(
                    adaptive_escalate_on_first_barrier_only
                ),
                enable_nonlinear_fallback=not continue_on_unsafe,
                first_step_recovery_floor=(
                    min(0.0, initial_barrier + 1e-4)
                    if minimal_intervention and first_step_recovery and initial_barrier < 0.0
                    else None
                ),
            )
            final_projection = adaptive_projection.projection
            selected_safety_trust_radius = adaptive_projection.selected_trust_radius
            attempted_safety_trust_radii = adaptive_projection.attempted_trust_radii
        else:
            final_projection = project_action_chunk_with_qp(
                qp_prefix(denoised_actions),
                to_physical,
                barriers,
                trust_radius=final_safety_trust_radius,
                enable_nonlinear_fallback=not continue_on_unsafe,
                first_step_recovery_floor=(
                    min(0.0, initial_barrier + 1e-4)
                    if minimal_intervention and first_step_recovery and initial_barrier < 0.0
                    else None
                ),
            )
            selected_safety_trust_radius = final_safety_trust_radius
            attempted_safety_trust_radii = (final_safety_trust_radius,)
        projected_actions = replace_qp_prefix(
            denoised_actions, final_projection.actions
        )
        final_gradient, final_score_before = self._success(
            pre_execution_hidden,
            projected_actions,
            0.0,
            gradient=True,
        )
        assert final_gradient is not None
        success_refinement = refine_action_chunk_for_success(
            final_projection.actions,
            to_physical,
            barriers,
            final_gradient[:qp_horizon, :3],
            trust_radius=final_success_trust_radius,
            lambda_deviation=lambda_deviation,
            beta_success=beta_success,
            trust_region_norm=trust_region_norm,
        )
        refined_actions = replace_qp_prefix(
            projected_actions, success_refinement.actions
        )
        _, final_score_after = self._success(
            pre_execution_hidden,
            refined_actions,
            0.0,
            gradient=False,
        )
        refinement_accepted = bool(
            success_refinement.success
            and final_score_after > final_score_before
        )
        if refinement_accepted:
            final_actions = refined_actions
            final_trajectory_barriers = success_refinement.barriers_after
        else:
            final_actions = projected_actions
            final_trajectory_barriers = final_projection.barriers_after
            final_score_after = final_score_before
        if not final_projection.success and not continue_on_unsafe:
            raise _UnsafeActionChunkError(
                f"Refusing to execute an action chunk whose final {qp_horizon}-step QP prefix "
                f"is unsafe: minimum barrier before={np.min(final_projection.barriers_before):.6f}, "
                f"after={np.min(final_projection.barriers_after):.6f}; "
                f"projection status={final_projection.status}"
            )
        normalized_actions = final_actions
        final_physical_actions = to_physical(normalized_actions)
        final_trajectory_barriers = barriers(qp_prefix(final_physical_actions))
        if not np.all(final_trajectory_barriers >= -1e-7) and not continue_on_unsafe:
            raise RuntimeError("Internal safety error: accepted final chunk failed exact barrier validation")
        outputs = self._base._output_transform({"state": state, "actions": normalized_actions})
        last_hidden = pre_execution_hidden
        outputs.update(
            {
                "normalized_actions": normalized_actions,
                "last_layer_hidden_state": last_hidden,
                "action_expert_intervened": np.asarray(1, dtype=np.bool_),
                "action_expert_nominal_trajectory_positions": nominal_trajectory_positions.astype(np.float32),
                "action_expert_nominal_trajectory_barriers": nominal_trajectory_barriers.astype(np.float32),
                "action_expert_final_trajectory_barriers": final_trajectory_barriers.astype(np.float32),
                "action_expert_final_projection_success": np.asarray(final_projection.success, dtype=np.bool_),
                "action_expert_certified_horizon": np.asarray(len(final_trajectory_barriers), dtype=np.int32),
                "action_expert_final_projection_iterations": np.asarray(final_projection.iterations, dtype=np.int32),
                "action_expert_final_projection_correction_rms": np.asarray(
                    np.sqrt(np.mean(np.square(final_projection.correction))), dtype=np.float32
                ),
                "action_expert_final_safety_correction_rms": np.asarray(
                    np.sqrt(np.mean(np.square(final_projection.correction))), dtype=np.float32
                ),
                "action_expert_final_critic_correction_rms": np.asarray(
                    np.sqrt(np.mean(np.square(success_refinement.correction)))
                    if refinement_accepted
                    else 0.0,
                    dtype=np.float32,
                ),
                "action_expert_final_critic_refinement_accepted": np.asarray(
                    refinement_accepted, dtype=np.bool_
                ),
                "action_expert_selected_safety_trust_radius": np.asarray(
                    selected_safety_trust_radius, dtype=np.float32
                ),
                "action_expert_attempted_safety_trust_radii": np.asarray(
                    attempted_safety_trust_radii, dtype=np.float32
                ),
                "action_expert_final_success_score_before": np.asarray(final_score_before, dtype=np.float32),
                "action_expert_final_success_score_after": np.asarray(final_score_after, dtype=np.float32),
                "action_expert_times": np.asarray([x["time"] for x in telemetry], dtype=np.float32),
                "action_expert_success_scores_before": np.asarray(
                    [x["score_before"] for x in telemetry], dtype=np.float32
                ),
                "action_expert_success_scores_after": np.asarray(
                    [x["score_after"] for x in telemetry], dtype=np.float32
                ),
                "action_expert_qp_success": np.asarray([x["qp_success"] for x in telemetry], dtype=np.bool_),
                "action_expert_qp_margins": np.asarray([x["qp_margin"] for x in telemetry], dtype=np.float32),
                "action_expert_correction_rms": np.asarray([x["correction_rms"] for x in telemetry], dtype=np.float32),
                "action_expert_barriers_before": np.asarray([x["barrier_before"] for x in telemetry], dtype=np.float32),
                "action_expert_barriers_nominal": np.asarray(
                    [x["barrier_nominal"] for x in telemetry], dtype=np.float32
                ),
                "policy_timing": {"infer_ms": (time.monotonic() - start) * 1000.0},
            }
        )
        return outputs
