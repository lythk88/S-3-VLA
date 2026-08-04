# Continuous Safety-Value Residual Flow Guidance

## Goal

Improve the plain pi0.5 policy on the original SafeLIBERO tasks by adding a
continuous learned safety residual to the flow-matching vector field:

\[
v_{\mathrm{total}}(x,a_t,t)=v_{\mathrm{task}}(x,a_t,t)+v_{\mathrm{safe}}(x,a_t,t).
\]

The first evaluation is restricted to `safelibero_spatial`.

## Current value function

The frozen value model consumes the pi0.5 action-token hidden state shaped
`(10, 1024)`. It constructs a 4096-dimensional feature by concatenating token
mean, standard deviation, maximum, and the final token, normalizes that feature,
and applies `4096 -> 1024 -> 256 -> 1` with LayerNorm, ReLU, and a sigmoid. Its
output is a continuous predicted safety probability in `[0, 1]`.

## Residual flow

At every guided denoising step, evaluate the value model on the current action
hidden state and differentiate `log V` with respect to the noisy action state
`a_t`. Since pi0.5 samples from `t=1` to `t=0` with a negative Euler step, use

\[
v_{\mathrm{safe}}=-\lambda(t)\nabla_{a_t}\log V(x,a_t,t).
\]

The gradient is RMS-normalized and scaled relative to the RMS magnitude of the
task flow. A late quadratic schedule applies guidance only for `t <= 0.5`, where
hidden states are closest to the final-step hidden states used to train the
value model. The initial pilot uses `guidance_scale=0.25`.

## Paired spatial evaluation

1. Export the PyTorch value model to a framework-neutral NPZ for JAX inference.
2. Start the pi0.5 policy server with the frozen value parameters.
3. Run all four spatial tasks at safety levels I and II for episode seed 0.
4. Run baseline and guided policies with identical deterministic `(10, 32)`
   initial flow noise at every replanning step.
5. Compare task success, collision, safe success, predicted safety value, and
   inference time. If the eight-rollout pilot is stable, expand the episode list
   before drawing benchmark-level conclusions.

## Commands

```bash
main/.venv/bin/python Safety-value-function/export_jax_guidance.py \
  --run-dir Safety-value-function/chunk_safety_value_run

srun --partition=main --gres=gpu:1 --mem=64G --ntasks=1 \
  bash scripts/run_spatial_flow_guidance_pilot.sh
```

Set `EPISODES="0 1 2 3 4"` to expand the paired pilot to 40 rollouts.

The completed episode-0 pilot and its interpretation are documented in
`SPATIAL_FLOW_GUIDANCE_RESULTS.md`.
