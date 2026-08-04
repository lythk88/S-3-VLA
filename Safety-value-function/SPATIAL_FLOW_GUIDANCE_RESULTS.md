# Spatial SafeLIBERO Residual-Flow Pilot Results

## Setup

- Policy: pi0.5 LIBERO checkpoint.
- Suite: all four `safelibero_spatial` tasks at safety levels I and II.
- Episodes: episode 0 for each task/level, giving eight paired rollouts.
- Pairing: baseline and guided runs used the same deterministic `(10, 32)`
  initial flow noise at each replanning point.
- Residual: `v_safe = -lambda(t) * grad(log V)` with RMS normalization,
  `guidance_scale=0.25`, and a quadratic late schedule for `t <= 0.5`.
- This is a functional pilot, not a statistically sufficient benchmark result.

## Primary outcomes

| Metric | pi0.5 baseline | Residual guided | Delta |
|---|---:|---:|---:|
| Task success | 62.5% (5/8) | 50.0% (4/8) | -12.5 pp |
| Rollout collision | 87.5% (7/8) | 87.5% (7/8) | 0 pp |
| Safe success | 12.5% (1/8) | 12.5% (1/8) | 0 pp |
| Mean predicted safety, rollout-balanced | 0.5646 | 0.6861 | +0.1215 |

The only binary outcome change was spatial task 2 at level II: the baseline
succeeded with a collision, while the guided policy failed with a collision.

## Diagnostics

- Every one of the eight matched first action chunks increased its predicted
  safety value. The average increased from 0.6476 to 0.7402 (+0.0926).
- Across every collected chunk, the mean learned score increased from 0.5337
  to 0.6609.
- The residual was finite after stabilization and averaged 0.06375 of the task
  flow RMS magnitude across all denoising steps.
- Physical contact occupancy worsened from 181/1688 executed actions (10.72%)
  to 210/1662 (12.64%), driven mainly by task 0 level I.
- Excluding one-time JAX compilation, mean server inference increased from
  about 31.0 ms to 52.0 ms per action chunk (1.68x).
- On baseline pilot chunks, the frozen value model separated immediate safe and
  unsafe chunk labels with ROC AUC 0.778, so it has useful correlation. That
  correlation did not make its input gradient a reliable causal safety control.

## Interpretation

The implementation is mechanically doing what the equation requests: the
gradient residual consistently raises the learned value. The pilot nevertheless
does **not** improve pi0.5-only physical results. Increasing a discriminative
hidden-state score is not equivalent to moving actions away from obstacles.
The guided hidden states are also off the distribution used to train the value
model, so the policy can move in score-increasing directions that are not safer
in the simulator.

The current value was trained only on final denoising hidden states and binary
five-action collision labels. Applying it inside intermediate denoising steps
introduces a time-distribution mismatch, even with late guidance. The score also
lacks an explicit action-deviation or task-progress constraint, which explains
the observed loss in task success.

## Recommended next iteration

1. Retrain a time-conditioned value `V(x, a_t, t)` using hidden states sampled
   from all denoising times, with continuous minimum-distance/contact targets.
2. Add guided/on-policy samples during value training to prevent classifier
   exploitation under value-gradient inference.
3. Constrain the residual with an action-deviation or task-value trust region.
4. Only then sweep small scales such as 0.05, 0.10, and 0.15 on paired spatial
   seeds before launching the 400-rollout spatial benchmark.

## Outputs

- Videos and rollout arrays: `results/spatial_flow_guidance_pilot/`
- Logs and generated analysis: `logs/spatial_flow_guidance_pilot/`
- Quarantined invalid pre-stabilization run: `results/invalid_nan_guidance/`
