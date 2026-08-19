"""Runtime-installed pi0.5 sampler that exposes the denoising trajectory.

This module intentionally lives beside, rather than edits, ``pi0.py`` so a
long-running baseline process can keep an immutable copy of its loaded model
code while denoising-value data collection is developed independently.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0


def sample_actions_with_denoising_trace(
    self: _pi0.Pi0,
    rng,
    observation: _model.Observation,
    *,
    start_noisy_action,
    start_time,
    num_steps: int = 10,
):
    """Denoise from ``(start_noisy_action, start_time)`` and return every step.

    Arrays retain the model's normalized padded action convention. Inactive
    entries occur when starting below t=1; callers must use ``active_mask``.
    """
    del rng  # The caller supplies the exact noisy state for reproducibility.
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    x_start = jnp.asarray(start_noisy_action)
    if x_start.ndim == 2:
        x_start = x_start[None, ...]
    time_start = jnp.broadcast_to(jnp.asarray(start_time, dtype=jnp.float32), (batch_size,))

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
    )

    def step(carry, _):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate(
            [prefix_to_suffix_mask, suffix_attn_mask], axis=-1
        )
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        hidden_state = suffix_out[:, -self.action_horizon :]
        task_flow = self.action_out_proj(hidden_state)
        active = time >= (-dt / 2)
        active_action = active[:, None, None]
        next_x = jnp.where(active_action, x_t + dt * task_flow, x_t)
        next_time = jnp.where(active, time + dt, time)
        emitted = (
            x_t.astype(jnp.float32),
            time.astype(jnp.float32),
            hidden_state.astype(jnp.float32),
            task_flow.astype(jnp.float32),
            active,
        )
        return (next_x, next_time), emitted

    (actions, _), trace = jax.lax.scan(
        step, (x_start, time_start), xs=None, length=num_steps
    )
    noisy_actions, times, hidden_states, task_flows, active_mask = (
        jnp.swapaxes(value, 0, 1) for value in trace
    )
    return (
        actions,
        hidden_states,
        noisy_actions,
        times,
        task_flows,
        active_mask,
    )


def install() -> None:
    """Install the trace method on Pi0 before policy construction."""
    if not hasattr(_pi0.Pi0, "sample_actions_with_denoising_trace"):
        setattr(
            _pi0.Pi0,
            "sample_actions_with_denoising_trace",
            sample_actions_with_denoising_trace,
        )

