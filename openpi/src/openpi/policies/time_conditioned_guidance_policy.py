"""Online pi0.5 guidance from a time-conditioned PyTorch safety value.

The value gradient is taken only through the normalized noisy-action input.
The pi0.5 hidden state is fixed, matching the training and offline-gradient-
gate convention.  This diagnostic wrapper supports several explicitly named
ways of inserting that gradient into the pi0.5 denoising trajectory so the
integration rule can be screened independently of the frozen value model.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.shared import nnx_utils


_TRAINED_TIMES = (0.1, 0.3, 0.5)
_GEOMETRIES = {"direct", "orthogonal", "task-compatible"}
_NORMALIZATIONS = {"global-rms", "per-token-rms", "task-step-rms"}
_INTEGRATIONS = {"state", "flow-step"}


class TimeConditionedGuidancePolicy:
    """Wrap a pi0.5 policy with bounded noisy-action value guidance."""

    def __init__(self, base_policy, value_model: torch.nn.Module, *, torch_device: str = "cpu"):
        self._base = base_policy
        self._value_model = value_model.to(torch_device).eval()
        self._torch_device = torch.device(torch_device)
        sampler = getattr(base_policy._model, "sample_actions_with_denoising_trace", None)
        if sampler is None:
            raise RuntimeError("The pi0 denoising-trace method was not installed")
        self._sample_with_trace = nnx_utils.module_jit(
            sampler, static_argnames=("num_steps",)
        )

    @property
    def metadata(self):
        return self._base.metadata

    def _value_outputs(
        self,
        hidden_state: np.ndarray,
        noisy_action: np.ndarray,
        denoising_time: float,
        clearance_score_weight: float,
        *,
        require_gradient: bool,
    ) -> tuple[np.ndarray | None, float, float, float]:
        # Arrays decoded from JAX / msgpack may be read-only. Copying also
        # makes the PyTorch inputs independent of later JAX buffers.
        hidden = torch.from_numpy(
            np.array(hidden_state, dtype=np.float32, copy=True)
        )[None]
        action = torch.from_numpy(
            np.array(noisy_action, dtype=np.float32, copy=True)
        )[None]
        time_tensor = torch.tensor([denoising_time], dtype=torch.float32)
        hidden = hidden.to(self._torch_device)
        action = action.to(self._torch_device)
        time_tensor = time_tensor.to(self._torch_device)
        if require_gradient:
            action.requires_grad_(True)
        with torch.set_grad_enabled(require_gradient):
            logit, clearance = self._value_model(hidden, action, time_tensor)
            log_safety = torch.nn.functional.logsigmoid(logit)
            objective = log_safety + clearance_score_weight * clearance
            gradient = (
                torch.autograd.grad(objective.sum(), action)[0][0]
                if require_gradient
                else None
            )
        gradient_np = None
        if gradient is not None:
            gradient_np = (
                torch.nan_to_num(gradient).detach().cpu().numpy().astype(np.float32)
            )
        return (
            gradient_np,
            float(objective.detach().cpu().item()),
            float(torch.sigmoid(logit).detach().cpu().item()),
            float(clearance.detach().cpu().item()),
        )

    @staticmethod
    def _active_view(value: np.ndarray, translation_only: bool) -> np.ndarray:
        return value[:, :3] if translation_only else value

    @classmethod
    def _shape_gradient(
        cls,
        gradient: np.ndarray,
        nominal_step: np.ndarray,
        *,
        translation_only: bool,
        geometry: str,
    ) -> np.ndarray:
        shaped = np.asarray(gradient, dtype=np.float32).copy()
        if translation_only:
            shaped[:, 3:] = 0.0
        gradient_view = cls._active_view(shaped, translation_only)
        step_view = cls._active_view(nominal_step, translation_only)
        flat_gradient = gradient_view.reshape(-1)
        flat_step = step_view.reshape(-1)
        step_norm_sq = float(np.dot(flat_step, flat_step))
        if geometry == "orthogonal" and step_norm_sq > 1e-12:
            flat_gradient -= (
                float(np.dot(flat_gradient, flat_step)) / step_norm_sq
            ) * flat_step
        elif geometry == "task-compatible" and step_norm_sq > 1e-12:
            alignment = float(np.dot(flat_gradient, flat_step))
            if alignment < 0.0:
                flat_gradient -= (alignment / step_norm_sq) * flat_step
        gradient_view[...] = flat_gradient.reshape(gradient_view.shape)
        return shaped

    @classmethod
    def _make_correction(
        cls,
        shaped_gradient: np.ndarray,
        nominal_step: np.ndarray,
        *,
        translation_only: bool,
        normalization: str,
        scale: float,
        direction_sign: float,
    ) -> tuple[np.ndarray, float, float, float]:
        gradient_view = cls._active_view(shaped_gradient, translation_only)
        gradient_rms = float(np.sqrt(np.mean(np.square(gradient_view))))
        correction = np.zeros_like(shaped_gradient)
        if math.isfinite(gradient_rms) and gradient_rms >= 1e-8:
            if normalization == "per-token-rms":
                token_rms = np.sqrt(np.mean(np.square(gradient_view), axis=1, keepdims=True))
                normalized_view = np.divide(
                    gradient_view,
                    np.maximum(token_rms, 1e-8),
                    out=np.zeros_like(gradient_view),
                    where=np.isfinite(token_rms),
                )
                cls._active_view(correction, translation_only)[...] = normalized_view
                correction *= scale
            else:
                correction = shaped_gradient / gradient_rms
                if normalization == "task-step-rms":
                    step_view = cls._active_view(nominal_step, translation_only)
                    target_rms = scale * float(np.sqrt(np.mean(np.square(step_view))))
                    correction *= target_rms
                else:
                    correction *= scale
            correction *= direction_sign
        correction_view = cls._active_view(correction, translation_only)
        correction_rms = float(np.sqrt(np.mean(np.square(correction_view))))
        step_rms = float(
            np.sqrt(
                np.mean(
                    np.square(cls._active_view(nominal_step, translation_only))
                )
            )
        )
        correction_to_step = correction_rms / max(step_rms, 1e-8)
        return correction, gradient_rms, correction_rms, correction_to_step

    def _sample_trace(
        self,
        sample_rng,
        observation,
        noisy_action: np.ndarray,
        start_time: float,
        denoising_steps: int,
    ):
        return self._sample_with_trace(
            sample_rng,
            observation,
            start_noisy_action=jnp.asarray(noisy_action)[None, ...],
            start_time=jnp.asarray(start_time, dtype=jnp.float32),
            num_steps=denoising_steps,
        )

    @staticmethod
    def _trace_state(trace, requested_time: float):
        _, hidden, noisy, times, task_flows, active_mask = trace
        times_np = np.asarray(times[0], dtype=np.float32)
        active_np = np.asarray(active_mask[0], dtype=np.bool_)
        candidates = np.flatnonzero(active_np)
        if not len(candidates):
            raise RuntimeError("Denoising trace has no active state")
        index = int(candidates[np.argmin(np.abs(times_np[candidates] - requested_time))])
        actual_time = float(times_np[index])
        if abs(actual_time - requested_time) > 0.051:
            raise RuntimeError(
                f"Requested guidance time {requested_time} is absent from trace; "
                f"closest active time is {actual_time}"
            )
        return (
            np.asarray(hidden[0, index], dtype=np.float32),
            np.asarray(noisy[0, index], dtype=np.float32),
            actual_time,
            np.asarray(task_flows[0, index], dtype=np.float32),
        )

    def infer(self, obs: dict) -> dict:
        inputs = dict(obs)
        guidance = inputs.pop("__time_conditioned_guidance__", None)
        if guidance is None:
            return self._base.infer(inputs)
        inputs.pop("__debug_return_last_hidden_state__", None)
        debug_noise = inputs.pop("__debug_noise__", None)

        denoising_steps = int(guidance.get("denoising_steps", 10))
        requested_times = guidance.get("times", [guidance.get("time", 0.3)])
        requested_times = sorted({float(value) for value in requested_times}, reverse=True)
        scale = float(guidance.get("scale", 0.05))
        direction_sign = float(guidance.get("direction_sign", 1.0))
        clearance_score_weight = float(guidance.get("clearance_score_weight", 0.5))
        translation_only = bool(guidance.get("translation_only", True))
        geometry = str(guidance.get("geometry", "direct"))
        normalization = str(guidance.get("normalization", "global-rms"))
        integration = str(guidance.get("integration", "state"))
        value_backtracking = bool(guidance.get("value_backtracking", False))
        margin_scaled = bool(guidance.get("margin_scaled", False))
        safety_threshold_value = guidance.get("safety_threshold")
        safety_threshold = (
            None
            if safety_threshold_value is None
            else float(safety_threshold_value)
        )
        if denoising_steps != 10:
            raise ValueError("The trained guidance convention requires 10 denoising steps")
        if not requested_times:
            raise ValueError("At least one guidance time is required")
        for requested_time in requested_times:
            if min(abs(requested_time - trained) for trained in _TRAINED_TIMES) > 1e-6:
                raise ValueError(
                    f"Guidance time {requested_time} was not used in training; "
                    f"choose from {_TRAINED_TIMES}"
                )
        if scale < 0.0:
            raise ValueError("Guidance scale must be non-negative")
        if direction_sign not in (-1.0, 0.0, 1.0):
            raise ValueError("direction_sign must be -1, 0, or 1")
        if geometry not in _GEOMETRIES:
            raise ValueError(f"Unknown geometry {geometry!r}; choose from {sorted(_GEOMETRIES)}")
        if normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization {normalization!r}; choose from {sorted(_NORMALIZATIONS)}"
            )
        if integration not in _INTEGRATIONS:
            raise ValueError(
                f"Unknown integration {integration!r}; choose from {sorted(_INTEGRATIONS)}"
            )
        if value_backtracking and integration != "state":
            raise ValueError("Fixed-context value backtracking is defined only for state integration")
        if safety_threshold is not None and not 0.0 <= safety_threshold <= 1.0:
            raise ValueError("safety_threshold must be in [0, 1]")

        transformed = jax.tree.map(lambda x: x, inputs)
        transformed = self._base._input_transform(transformed)
        transformed = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...], transformed
        )
        self._base._rng, sample_rng = jax.random.split(self._base._rng)
        observation = _model.Observation.from_dict(transformed)

        start = time.monotonic()
        if debug_noise is None:
            noise_np = np.asarray(
                jax.random.normal(
                    sample_rng,
                    (
                        1,
                        self._base._model.action_horizon,
                        self._base._model.action_dim,
                    ),
                    dtype=jnp.float32,
                )[0],
                dtype=np.float32,
            )
            external_noise = False
        else:
            noise_np = np.asarray(debug_noise, dtype=np.float32)
            external_noise = True
        nominal = self._sample_trace(
            sample_rng, observation, noise_np, 1.0, denoising_steps
        )
        working_trace = nominal
        telemetry = []
        last_injection_hidden = None
        dt = -1.0 / denoising_steps

        for requested_time in requested_times:
            hidden_state, noisy_action, actual_time, task_flow = self._trace_state(
                working_trace, requested_time
            )
            _, objective_before, score_before, clearance_before = (
                self._value_outputs(
                    hidden_state,
                    noisy_action,
                    actual_time,
                    clearance_score_weight,
                    require_gradient=False,
                )
            )
            risk_gate_active = (
                safety_threshold is None or score_before < safety_threshold
            )
            nominal_step = dt * task_flow
            correction = np.zeros_like(noisy_action)
            gradient_rms = 0.0
            correction_rms = 0.0
            correction_to_step = 0.0
            accepted_factor = 0.0
            objective_after = objective_before
            score_after = score_before
            clearance_after = clearance_before

            # A binary gate spends the same authority on a borderline score as
            # on a confident one, and measured gate precision is well under
            # half. Margin scaling ties authority to how far below the
            # threshold the score sits: no effect at the boundary, full scale
            # only when the value model is confident the state is unsafe.
            effective_scale = scale
            if risk_gate_active and margin_scaled and safety_threshold:
                margin = (safety_threshold - score_before) / safety_threshold
                effective_scale = scale * float(np.clip(margin, 0.0, 1.0))

            if risk_gate_active:
                raw_gradient, _, _, _ = self._value_outputs(
                    hidden_state,
                    noisy_action,
                    actual_time,
                    clearance_score_weight,
                    require_gradient=True,
                )
                assert raw_gradient is not None
                shaped_gradient = self._shape_gradient(
                    raw_gradient,
                    nominal_step,
                    translation_only=translation_only,
                    geometry=geometry,
                )
                correction, gradient_rms, correction_rms, correction_to_step = (
                    self._make_correction(
                        shaped_gradient,
                        nominal_step,
                        translation_only=translation_only,
                        normalization=normalization,
                        scale=effective_scale,
                        direction_sign=direction_sign,
                    )
                )

                accepted_factor = 1.0
                if value_backtracking:
                    accepted_factor = 0.0
                    for factor in (1.0, 0.5, 0.25, 0.125):
                        candidate = noisy_action + factor * correction
                        _, candidate_objective, candidate_score, candidate_clearance = (
                            self._value_outputs(
                                hidden_state,
                                candidate,
                                actual_time,
                                clearance_score_weight,
                                require_gradient=False,
                            )
                        )
                        if candidate_objective >= objective_before - 1e-7:
                            accepted_factor = factor
                            objective_after = candidate_objective
                            score_after = candidate_score
                            clearance_after = candidate_clearance
                            break
                    correction *= accepted_factor
                    correction_rms *= accepted_factor
                    correction_to_step *= accepted_factor
                elif integration == "state":
                    _, objective_after, score_after, clearance_after = self._value_outputs(
                        hidden_state,
                        noisy_action + correction,
                        actual_time,
                        clearance_score_weight,
                        require_gradient=False,
                    )

            if integration == "state":
                guided_noisy_action = noisy_action + correction
                resume_time = actual_time
            else:
                # dt is negative, so a positive correction ascends the value while
                # the nominal Euler motion remains dt * task_flow.
                guided_noisy_action = noisy_action + nominal_step + correction
                resume_time = max(0.0, actual_time + dt)
            if risk_gate_active:
                working_trace = self._sample_trace(
                    sample_rng,
                    observation,
                    guided_noisy_action,
                    resume_time,
                    denoising_steps,
                )
            last_injection_hidden = hidden_state
            telemetry.append(
                {
                    "time": actual_time,
                    "gradient_rms": gradient_rms,
                    "correction_rms": correction_rms,
                    "correction_to_step": correction_to_step,
                    "accepted_factor": accepted_factor,
                    "risk_gate_active": risk_gate_active,
                    "objective_before": objective_before,
                    "objective_after": objective_after,
                    "score_before": score_before,
                    "score_after": score_after,
                    "clearance_before": clearance_before,
                    "clearance_after": clearance_after,
                }
            )

        guided_actions, guided_hidden, guided_noisy, guided_times, _, guided_active = (
            working_trace
        )
        model_time = time.monotonic() - start
        normalized_actions = np.asarray(guided_actions[0], dtype=np.float32)
        outputs = self._base._output_transform(
            {
                "state": np.asarray(transformed["state"][0]),
                "actions": normalized_actions,
            }
        )
        guided_active_np = np.asarray(guided_active[0], dtype=np.bool_)
        active_indices = np.flatnonzero(guided_active_np)
        if len(active_indices):
            output_hidden = np.asarray(
                guided_hidden[0, int(active_indices[-1])], dtype=np.float32
            )
        else:
            assert last_injection_hidden is not None
            output_hidden = last_injection_hidden
        first = telemetry[0]
        outputs.update(
            {
                "normalized_actions": normalized_actions,
                "last_layer_hidden_state": output_hidden,
                "time_conditioned_score_before": np.asarray(
                    first["score_before"], dtype=np.float32
                ),
                "time_conditioned_clearance_before": np.asarray(
                    first["clearance_before"], dtype=np.float32
                ),
                "time_conditioned_score_after": np.asarray(
                    telemetry[-1]["score_after"], dtype=np.float32
                ),
                "time_conditioned_clearance_after": np.asarray(
                    telemetry[-1]["clearance_after"], dtype=np.float32
                ),
                "time_conditioned_gradient_rms": np.asarray(
                    np.mean([item["gradient_rms"] for item in telemetry]), dtype=np.float32
                ),
                "time_conditioned_perturbation_rms": np.asarray(
                    np.mean([item["correction_rms"] for item in telemetry]), dtype=np.float32
                ),
                "time_conditioned_correction_to_step": np.asarray(
                    np.max([item["correction_to_step"] for item in telemetry]),
                    dtype=np.float32,
                ),
                "time_conditioned_accepted_factor": np.asarray(
                    np.min([item["accepted_factor"] for item in telemetry]),
                    dtype=np.float32,
                ),
                "time_conditioned_injection_count": np.asarray(
                    sum(item["risk_gate_active"] for item in telemetry), dtype=np.int32
                ),
                "time_conditioned_risk_gate_active": np.asarray(
                    any(item["risk_gate_active"] for item in telemetry), dtype=np.bool_
                ),
                "time_conditioned_safety_threshold": np.asarray(
                    np.nan if safety_threshold is None else safety_threshold,
                    dtype=np.float32,
                ),
                "time_conditioned_guidance_time": np.asarray(
                    first["time"], dtype=np.float32
                ),
                "time_conditioned_guidance_scale": np.asarray(scale, dtype=np.float32),
                "time_conditioned_direction_sign": np.asarray(
                    direction_sign, dtype=np.float32
                ),
                "time_conditioned_injection_times": np.asarray(
                    [item["time"] for item in telemetry], dtype=np.float32
                ),
                "time_conditioned_objectives_before": np.asarray(
                    [item["objective_before"] for item in telemetry], dtype=np.float32
                ),
                "time_conditioned_objectives_after": np.asarray(
                    [item["objective_after"] for item in telemetry], dtype=np.float32
                ),
                "nominal_normalized_actions": np.asarray(
                    nominal[0][0], dtype=np.float32
                ),
                "time_conditioned_external_noise": np.asarray(
                    external_noise, dtype=np.bool_
                ),
                "guided_denoising_hidden_states": np.asarray(
                    guided_hidden[0], dtype=np.float32
                ),
                "guided_denoising_noisy_actions": np.asarray(
                    guided_noisy[0], dtype=np.float32
                ),
                "guided_denoising_times": np.asarray(guided_times[0], dtype=np.float32),
                "guided_denoising_active_mask": guided_active_np,
                "policy_timing": {"infer_ms": model_time * 1000},
            }
        )
        return outputs
