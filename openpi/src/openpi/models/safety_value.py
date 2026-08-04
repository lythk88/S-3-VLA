"""Differentiable JAX forward pass for the continuous safety-value MLP."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    return x @ weight.T + bias


def _layer_norm(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * weight + bias


def hidden_feature(hidden_state: jax.Array) -> jax.Array:
    """Match the feature construction used to train the PyTorch value model."""
    hidden_state = hidden_state.astype(jnp.float32)
    token_mean = jnp.mean(hidden_state, axis=1)
    # ``jnp.std`` has an undefined derivative when all token values in one
    # hidden dimension are equal. Quantized/bfloat16 transformer activations
    # do contain such dimensions, so use the equivalent stabilized expression.
    token_std = jnp.sqrt(
        jnp.mean(jnp.square(hidden_state - token_mean[:, None, :]), axis=1)
        + 1e-12
    )
    return jnp.concatenate(
        [
            token_mean,
            token_std,
            jnp.max(hidden_state, axis=1),
            hidden_state[:, -1],
        ],
        axis=-1,
    )


def safety_value_logit(
    hidden_state: jax.Array,
    parameters: dict[str, jax.Array],
) -> jax.Array:
    """Return one safety logit per batch item for hidden states shaped (B, T, D)."""
    x = hidden_feature(hidden_state)
    x = (x - parameters["feature_mean"]) / parameters["feature_std"]
    x = _linear(x, parameters["net.0.weight"], parameters["net.0.bias"])
    x = _layer_norm(x, parameters["net.1.weight"], parameters["net.1.bias"])
    x = jax.nn.relu(x)
    x = _linear(x, parameters["net.4.weight"], parameters["net.4.bias"])
    x = _layer_norm(x, parameters["net.5.weight"], parameters["net.5.bias"])
    x = jax.nn.relu(x)
    x = _linear(x, parameters["net.8.weight"], parameters["net.8.bias"])
    return jnp.squeeze(x, axis=-1)
