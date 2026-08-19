# Continuous Safety-Value Residual Flow Guidance

## Diagnostic integration selection and held-out run (2026-08-09)

The strongest available time-conditioned checkpoint failed the pre-registered
clearance confidence-bound gate, so it is not safety-validated. With the
user's explicit authorization, it is being used only for a clearly labeled
diagnostic comparison. The frozen model, objective, and checksum do not change
between integration methods.

The completed development screen compares 11 ways to insert
`d[logsigmoid(safety_logit) + 0.5*normalized_clearance]/d noisy_action_t`:

- one-shot state injection at trained times `t=0.1, 0.3, 0.5`;
- global and per-token RMS normalization;
- correction magnitude relative to the nominal pi0.5 Euler step;
- direct, task-flow-orthogonal, and task-compatible gradient geometry;
- literal flow-step residual integration;
- sequential recomputation at `t=0.5,0.3,0.1`;
- fixed-hidden nonlinear value backtracking;
- XYZ-only versus full 32-dimensional action-state guidance.

Development selection uses episodes 0 and 1 across all eight Spatial
task/level strata, exact fixed flow noise, and 16 rollouts per method. Ranking
is safe success, then task success, then fewer collisions. Full results are in
`results/time_conditioned_gradient_method_screen_dev2/`.

The promoted rule is a single XYZ state correction at `t=0.3`:

```text
nominal_step = -0.1 * pi05_task_flow
direction = grad_objective / max(XYZ_RMS(grad_objective), 1e-8)
correction = direction * 0.5 * XYZ_RMS(nominal_step)
noisy_action_t <- noisy_action_t + correction
```

This makes the correction exactly 50% of the local nominal step RMS, after
which pi0.5 resumes denoising to `t=0`. The hidden state is fixed during value
differentiation. The value gradient runs on CPU with
`torch.manual_seed(0)` and deterministic PyTorch algorithms; pi0.5 remains on
one H100. The run uses the frozen 20-episode test split
`8-13,15-19,21-29`, all four tasks, both levels, exact fixed noise, and no
method selection on those test episodes. Outputs are written under
`results/spatial_flow_guidance_pilot/` with exact run name
`pi05_time_value_trust_t03_r050_heldout20_diag`.

This held-out run estimates behavior; it does not retroactively pass the
offline clearance gate or justify a safety claim.

### Completed held-out result

The diagnostic held-out run completed 160/160 guided rollouts. Relative to the
exact 160-rollout plain comparator, success changed from 94 to 105, collisions
from 107 to 104, and safe successes from 46 to 53. The corresponding paired
episode-clustered intervals are documented in `SPATIAL_FLOW_GUIDANCE_RESULTS.md`
and in the JSON report under `results/spatial_flow_guidance_pilot/`.

The aggregate point estimates favor guidance, but the collision and safe-
success confidence intervals cross zero and the success interval touches zero.
The next scientific action is therefore not another held-out hyperparameter
search. It is to improve the value model using more clearance-informative
same-state perturbations and rerun the unchanged offline clearance gate. The
current trust integration remains a diagnostic candidate only.

## Goal

Improve the plain pi0.5 policy on the original SafeLIBERO tasks by adding a
continuous learned safety residual to the flow-matching vector field:

\[
v_{\mathrm{total}}(x,a_t,t)=v_{\mathrm{task}}(x,a_t,t)+v_{\mathrm{safe}}(x,a_t,t).
\]

The first evaluation is restricted to `safelibero_spatial`.

## Active time-conditioned implementation (2026-08-05)

The final-state MLP described below is retained as bootstrap history, not as
the model being promoted. The active model is:

```text
V_theta(hidden_t[10,1024], noisy_action_t[10,32], t)
    -> (safety_logit, normalized_minimum_clearance)
```

Each token combines a LayerNorm/linear projection of the pi0.5 hidden token, a
two-layer projection of its 32-dimensional normalized noisy action, a Fourier
embedding of denoising time, and a learned token-position embedding. Two
pre-norm transformer encoder blocks (width 256, four heads) preserve action
token order. Attention pooling feeds separate safety and clearance heads.

The action convention is pi0.5's normalized padded `(10,32)` denoising state.
Clearance is the minimum MuJoCo narrow-phase signed robot-to-active-obstacle
distance, capped above at 0.03 m: nominal chunks use the next 20 executed
actions, while same-state perturbation branches use the next five replanned
actions. Negative clearance means contact penetration. Near-contact sensing sets obstacle geom
margin and gap to 0.03 m without activating a constraint at positive distance.

### Data and splits

- Bootstrap corpus: `training_dataset/pi05_hidden_chunks`, 245 rollout groups.
- Recollected trace corpus: `training_dataset/pi05_denoising_value_v1`.
- Every nominal chunk stores all ten `(hidden_t, noisy_action_t, t)` states.
- Each episode retains one far-clearance context and up to three contexts whose
  same-state nominal preview approaches or contacts an obstacle.
- At `t in {0.1, 0.3, 0.5}`, four paired `+/-` XYZ perturbations are denoised
  and simulated from the identical MuJoCo state. Targets include signed minimum
  clearance, contact, action deviation, and a discounted 20-action future
  hazard value.
- Train/validation assignment is by complete rollout, stratified by task and
  safety level with seed 7. No chunk or perturbation pair crosses the split.
- The final Spatial test uses 20 common obstacle-active initial-state episodes
  outside the source corpus. The exact IDs and per-stratum obstacle audit are
  frozen in `results/spatial_heldout20_split.json` before evaluation. The audit
  selected episodes `8-13, 15-19, 21-29` and rejected 14 and 20; its SHA-256 is
  `914cea912e5d5c934f5736cd109c9294f5bd6388cf229e282aa695ded0439f1e`.

Training first initializes the hidden representation for ten epochs from the
bootstrap future-hazard labels. It then fine-tunes for up to 50 epochs on the
time-conditioned trace corpus with early stopping. The fine-tuning objective is
binary cross entropy plus `0.25 * Brier`, clearance Huber loss, and same-state
paired clearance ranking/difference losses. The selected checkpoint is
`Safety-value-function/time_conditioned_clearance_v1/best_model.pt` and its
manifest records the Git commit, dirty-diff identity, complete configuration,
data audit checksum, split, runtime/GPU, and checkpoint checksum.

### Mandatory gates and online intervention

The checkpoint is not benchmark-eligible after validation loss alone. First,
the held-out branch gate tests whether its direct noisy-action derivative picks
the higher-clearance member of measured `+/-` pairs, with rollout-clustered
confidence intervals. Next, a fresh simulator gate repeats symmetric local
branches from validation trajectories and requires both the lower 95% bound of
direction accuracy to exceed 0.5 and the lower bound of clearance gain to
exceed zero.

Online guidance uses the same derivative convention: `hidden_t` is held fixed,
the derivative of `log(sigmoid(safety_logit)) + 0.5 * normalized_clearance` is
taken with respect to `noisy_action_t`, non-XYZ channels are masked, and the
remaining gradient is normalized to unit XYZ RMS. At `t=0.3`:

```text
noisy_action_t <- noisy_action_t + 0.05 * gradient / max(XYZ_RMS(gradient), 1e-8)
```

pi0.5 then resumes flow matching from the perturbed noisy state to `t=0`. The
server verifies the value-checkpoint checksum and refuses the final benchmark
unless the live gate status is `live_gradient_gate_passed`.

### H100 execution chain

- Job `36205` (`value-trace-full-v2`): clean full 245-episode trace/branch
  collection. The rejected eight-file partial from job `36195` is preserved in
  `training_dataset/failed_collections/pi05_denoising_value_v1_manifest_drift_job36195`.
- Job `36206` (`value-train-v1`): audit, one-H100 training, and offline gradient
  gate; queued with `afterok:36205`.
- Job `36207` (`value-live-gate`): one-H100 fresh simulator gradient gate,
  queued with `afterok:36206`.
- Job `36214` (`pi05-heldout20`): second-H100 plain held-out comparator, 160
  rollouts, writing to `results/pi05_spatial_heldout_20ep`.
- Job `36215` (`pi05-time-guided20`): final 160-rollout guided benchmark and
  paired report, queued behind both `36214` and `36207`.
- A paired one-H100 guided benchmark follows only if both the live gate and the
  plain held-out comparator succeed. It evaluates all four Spatial tasks, both
  safety levels, and exactly 20 fixed-noise episodes per task/level.

## Current value function

The frozen value model consumes the pi0.5 action-token hidden state shaped
`(10, 1024)`. It constructs a 4096-dimensional feature by concatenating token
mean, standard deviation, maximum, and the final token, normalizes that feature,
and applies `4096 -> 1024 -> 256 -> 1` with LayerNorm, ReLU, and a sigmoid. Its
output is a continuous predicted safety probability in `[0, 1]`.

## One-H100 value-training run (2026-08-05)

One NVIDIA H100 80 GB GPU was used to train a new bootstrap value model
from:

```text
/home/lythk/safe-flow-matching/training_dataset/pi05_hidden_chunks
```

The corpus contains 245 completed rollouts, 11,110 action chunks, 17 tasks,
55,212 executed actions, and 11,825 robot-obstacle contact actions. All 245
NPZ files pass the chunk/hidden/action alignment check.

### Split and external test

Training and validation both come only from `pi05_hidden_chunks`. The split is
grouped by complete rollout and stratified by task and safety level, so chunks
from one rollout can never appear in both sets. With split seed 7 and a nominal
20% validation fraction, small strata are rounded up, producing:

- training: 185 rollouts;
- validation: 60 rollouts;
- represented strata: all 31 task/level combinations in each split;
- internal test: none.

Fresh rollouts from the original `safelibero_spatial` simulator are the only
test set. They are not used for early stopping, hyperparameter selection, or
normalization.

### What this H100 run trains

The source files contain final action-token hidden states and executed contact
traces, but do not contain intermediate noisy actions `a_t`, denoising times
`t`, or continuous geometric clearance. Consequently this run is explicitly a
**final-state bootstrap value**, not the planned time-conditioned
`V(h_t, a_t, t)` model.

The input is the final `(10, 1024)` pi0.5 hidden sequence pooled into
mean/std/max/last features. The target is a 20-action discounted future-safety
value with `tau=10`; a collision farther in the future has a smaller effect
than an immediate collision. Training uses unweighted soft BCE plus `0.25`
Brier loss, AdamW at `1e-4`, batch size 256, rollout-grouped validation early
stopping, and seed 7. The selected checkpoint is exported to the existing JAX
MLP format for a fresh SafeLIBERO test.

Output directory:

```text
Safety-value-function/chunk_safety_value_external_test_v1/
```

It will contain `best_model.pt`, `normalizer.npz`, `chunk_pairs.json`,
`jax_guidance_model.npz`, and `training_manifest.json`. The manifest records
the dataset SHA-256 fingerprint, exact rollout split, Git/diff identity,
runtime and GPU, hyperparameters, and artifact checksums. Its status remains
`external_test_pending` until fresh simulator evaluation finishes.

Executed training command:

```bash
/home/lythk/vlsa-aegis/.run_venv/bin/python \
  Safety-value-function/train_chunk_safety_value.py \
  --data-root training_dataset/pi05_hidden_chunks \
  --output-dir Safety-value-function/chunk_safety_value_external_test_v1 \
  --external-test --validation-fraction 0.2 --split-seed 7 \
  --target future_hazard --hazard-lookahead 20 --hazard-tau 10 \
  --loss soft_bce --learning-rate 1e-4 \
  --batch-size 256 --epochs 100 --early-stop-patience 12 \
  --seed 7 --device cuda
```

This checkpoint must not be promoted merely for improving validation AUC or
Brier error. It remains subject to the original SafeLIBERO gradient/clearance
gate described in `SPATIAL_FLOW_GUIDANCE_RESULTS.md`.

### Completed training result

- Slurm job: `35856` (`value-ext-v1`).
- Hardware: one NVIDIA H100 80 GB.
- Status: completed successfully; JAX export completed.
- Runtime: 15 seconds including feature loading, training, hashing, and export.
- Split: 8,265 training chunks from 185 rollouts; 2,845 validation chunks from
  60 rollouts; zero internal-test chunks.
- Best checkpoint: epoch 1, validation loss `0.499295`.
- Early stopping: epoch 13 after 12 stale epochs. Training loss continued to
  fall while validation loss rose, showing immediate overfitting after epoch 1.

Validation comparison on the fixed split:

| Model | Soft-target Brier | Future-event AUC | Immediate-safe AUC | Immediate Brier |
|---|---:|---:|---:|---:|
| Previous chunk MLP | 0.1188 | **0.8753** | **0.9193** | 0.1294 |
| New external-test bootstrap | **0.1055** | 0.8574 | 0.8892 | **0.1120** |

The new checkpoint improves calibration but reduces ranking AUC. It is saved
as `trained_validation_selected_external_test_pending`; this mixed result is
not sufficient for promotion. Fresh original SafeLIBERO testing remains
required. Its first valid simulator evaluation is job `36142`: 20 plain,
value-guided episodes at each of safety levels I and II for spatial task 0.
That run uses XYZ-only guidance at scale `0.05` for `t <= 0.3`; it is an
external behavior check, not a promotion decision by itself.

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
