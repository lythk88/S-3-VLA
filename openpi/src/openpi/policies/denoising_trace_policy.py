"""Policy wrapper exposing pi0.5 denoising traces over the debug protocol."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import nnx_utils


class DenoisingTracePolicy:
    """Delegate ordinary inference and handle trace requests explicitly."""

    def __init__(self, base_policy):
        self._base = base_policy
        sampler = getattr(base_policy._model, "sample_actions_with_denoising_trace", None)
        if sampler is None:
            raise RuntimeError("The pi0 denoising-trace method was not installed")
        self._sample_with_trace = nnx_utils.module_jit(sampler, static_argnames=("num_steps",))

    @property
    def metadata(self):
        return self._base.metadata

    def infer(self, obs: dict) -> dict:
        inputs = dict(obs)
        trace_config = inputs.pop("__debug_denoising_trace__", None)
        if trace_config is None:
            return self._base.infer(inputs)
        debug_noise = inputs.pop("__debug_noise__", None)
        start_noisy_action = trace_config.get("start_noisy_action", debug_noise)
        if start_noisy_action is None:
            raise ValueError("Trace inference requires fixed noise or start_noisy_action")
        start_time = float(trace_config.get("start_time", 1.0))
        num_steps = int(trace_config.get("num_steps", 10))

        transformed = jax.tree.map(lambda x: x, inputs)
        transformed = self._base._input_transform(transformed)
        transformed = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
        self._base._rng, sample_rng = jax.random.split(self._base._rng)
        observation = _model.Observation.from_dict(transformed)

        start = time.monotonic()
        (
            actions,
            hidden_states,
            noisy_actions,
            times,
            task_flows,
            active_mask,
        ) = self._sample_with_trace(
            sample_rng,
            observation,
            start_noisy_action=jnp.asarray(start_noisy_action)[None, ...],
            start_time=jnp.asarray(start_time, dtype=jnp.float32),
            num_steps=num_steps,
        )
        model_time = time.monotonic() - start

        normalized_actions = np.asarray(actions[0])
        hidden_np = np.asarray(hidden_states[0], dtype=np.float32)
        noisy_np = np.asarray(noisy_actions[0], dtype=np.float32)
        times_np = np.asarray(times[0], dtype=np.float32)
        flows_np = np.asarray(task_flows[0], dtype=np.float32)
        active_np = np.asarray(active_mask[0], dtype=np.bool_)
        return_times = trace_config.get("return_times")
        if return_times:
            candidates = np.flatnonzero(active_np)
            indices = np.asarray(
                [
                    int(candidates[np.argmin(np.abs(times_np[candidates] - float(requested)))])
                    for requested in return_times
                ],
                dtype=np.int32,
            )
            hidden_np = hidden_np[indices]
            noisy_np = noisy_np[indices]
            times_np = times_np[indices]
            flows_np = flows_np[indices]
            active_np = active_np[indices]
        outputs = self._base._output_transform(
            {
                "state": np.asarray(transformed["state"][0]),
                "actions": normalized_actions,
            }
        )
        outputs.update(
            {
                "normalized_actions": normalized_actions.astype(np.float32),
                "denoising_hidden_states": hidden_np,
                "denoising_noisy_actions": noisy_np,
                "denoising_times": times_np,
                "denoising_task_flows": flows_np,
                "denoising_active_mask": active_np,
                "policy_timing": {"infer_ms": model_time * 1000},
            }
        )
        return outputs
