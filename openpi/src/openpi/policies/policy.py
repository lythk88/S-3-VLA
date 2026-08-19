from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        safety_value_run_dir: pathlib.Path | str | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._safety_value_parameters = None
        if safety_value_run_dir is not None:
            archive_path = pathlib.Path(safety_value_run_dir) / "jax_guidance_model.npz"
            with np.load(archive_path) as archive:
                self._safety_value_parameters = {
                    key: jnp.asarray(archive[key]) for key in archive.files
                }

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_last_hidden_state = getattr(model, "sample_actions_with_last_hidden_state", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            sample_actions_with_last_hidden_state = getattr(model, "sample_actions_with_last_hidden_state", None)
            self._sample_actions_with_last_hidden_state = (
                nnx_utils.module_jit(sample_actions_with_last_hidden_state)
                if sample_actions_with_last_hidden_state is not None
                else None
            )
            safety_guided_sampler = getattr(model, "sample_actions_with_safety_value_guidance", None)
            self._sample_actions_with_safety_value_guidance = (
                nnx_utils.module_jit(safety_guided_sampler)
                if safety_guided_sampler is not None
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = dict(obs)
        return_last_hidden_state = bool(inputs.pop("__debug_return_last_hidden_state__", False))
        debug_noise = inputs.pop("__debug_noise__", None)
        flow_guidance = inputs.pop("__safety_value_flow_guidance__", None)
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if debug_noise is not None:
            noise = debug_noise
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        base_outputs = {
            "state": inputs["state"],
        }
        extra_outputs = {}
        if flow_guidance is not None:
            if self._is_pytorch_model or self._sample_actions_with_safety_value_guidance is None:
                raise NotImplementedError("Safety-value residual flow guidance requires the JAX pi0.5 sampler.")
            if self._safety_value_parameters is None:
                raise RuntimeError("The policy server was not given --safety-value-run-dir.")
            actions, last_hidden_state, safety_score, residual_ratio = (
                self._sample_actions_with_safety_value_guidance(
                    sample_rng_or_pytorch_device,
                    observation,
                    safety_value_parameters=self._safety_value_parameters,
                    guidance_scale=jnp.asarray(flow_guidance.get("scale", 0.25), dtype=jnp.float32),
                    guidance_start_time=jnp.asarray(flow_guidance.get("start_time", 0.5), dtype=jnp.float32),
                    guidance_translation_only=jnp.asarray(
                        flow_guidance.get("translation_only", False), dtype=jnp.bool_
                    ),
                    guidance_orthogonal=jnp.asarray(
                        flow_guidance.get("orthogonal", False), dtype=jnp.bool_
                    ),
                    **sample_kwargs,
                )
            )
            base_outputs["actions"] = actions
            extra_outputs["last_layer_hidden_state"] = last_hidden_state
            extra_outputs["safety_value_score"] = safety_score
            extra_outputs["safety_residual_ratio"] = residual_ratio
        elif return_last_hidden_state:
            if self._sample_actions_with_last_hidden_state is None:
                raise NotImplementedError("This policy does not expose last hidden state debugging outputs.")
            actions, last_hidden_state = self._sample_actions_with_last_hidden_state(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
            base_outputs["actions"] = actions
            extra_outputs["last_layer_hidden_state"] = last_hidden_state
        else:
            base_outputs["actions"] = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            base_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), base_outputs)
            extra_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), extra_outputs)
        else:
            base_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), base_outputs)
            extra_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), extra_outputs)

        outputs = self._output_transform(base_outputs)
        outputs.update(extra_outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
