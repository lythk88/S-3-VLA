# Spatial SafeLIBERO Residual-Flow Pilot Results

## Time-conditioned gradient-integration screen (2026-08-09)

The user explicitly authorized a diagnostic bypass of the failed offline gate
to test different ways of inserting the frozen value gradient into pi0.5. This
does not change the scientific status of the model: checkpoint
`a8f08b91e7e73fa9042f2a6df727e42b4596a1f45db7b37605430f3c3b51f7e7`
still has 73.24% held-out gradient-direction accuracy, but its simulated
clearance 95% CI is `[-1.673, +3.611] mm` and therefore crosses zero.

Two H100s ran development episodes 0 and 1 for all four Spatial tasks and both
safety levels. Every method used the same deterministic `(10,32)` initial flow
noise, pi0.5 only, and 16 rollouts. The screen produced 192 complete rollout
artifacts and no skips or runtime failures. Exact manifests, artifacts, and the
machine-readable paired report are in
`results/time_conditioned_gradient_method_screen_dev2/`.

| Method | Success | Collision | Safe success | Mean chunk inference |
|---|---:|---:|---:|---:|
| Plain pi0.5 | 11/16 | 14/16 | 2/16 | 45.2 ms |
| Orthogonal XYZ, `t=.3`, global RMS `.05` | 10/16 | 13/16 | 3/16 | 93.0 ms |
| XYZ trust ratio, `t=.3`, 50% of nominal step | 10/16 | 13/16 | 3/16 | 93.7 ms |
| Sequential XYZ, `t=.5,.3,.1`, 25% per step | 10/16 | 13/16 | 3/16 | 211.7 ms |
| Flow-step XYZ, `t=.3`, 50% of nominal step | 9/16 | 13/16 | 3/16 | 91.9 ms |
| Late XYZ, `t=.1`, global RMS `.05` | 8/16 | **12/16** | 3/16 | 93.4 ms |
| Direct XYZ, `t=.3`, global RMS `.05` | 8/16 | 13/16 | 3/16 | 101.0 ms |
| Full 32-D, `t=.3`, 25% of nominal step | **13/16** | 14/16 | 2/16 | 93.3 ms |
| Value-backtracked XYZ, `t=.3`, global RMS `.05` | 11/16 | 14/16 | 2/16 | 92.5 ms |
| Task-compatible XYZ, `t=.3`, global RMS `.05` | 10/16 | 13/16 | 2/16 | 94.3 ms |
| Early XYZ, `t=.5`, global RMS `.05` | 9/16 | 14/16 | 2/16 | 94.5 ms |
| Per-token XYZ, `t=.3`, RMS `.03` | 8/16 | 15/16 | 1/16 | 107.0 ms |

All methods raised the model's own fixed-hidden objective except flow-step,
whose post-step objective is not evaluated under the same hidden state. The
largest score increase did not produce the best simulator result. In
particular, sequential guidance raised the score most but only tied the much
cheaper trust method, and per-token guidance raised the score while worsening
all three episode-level metrics.

The screen also exposed sensitivity that must not be hidden. Direct `t=.3`
and value-backtracked `t=.3` should be identical because backtracking accepted
factor `1.0` for every chunk. CUDA value-gradient roundoff nevertheless changed
their first chunks by roughly `5e-4` to `8e-4` action RMS, after which chaotic
simulation produced different outcomes. The held-out diagnostic therefore
uses deterministic PyTorch CPU value gradients while pi0.5 remains on H100.

The promoted integration is the XYZ trust-ratio method: one injection at
`t=.3`, direct state integration, and correction RMS capped at 50% of the
nominal pi0.5 Euler-step RMS. It tied orthogonal guidance on episode outcomes,
retained 10/11 plain successes, is explicitly magnitude-bounded, and costs less
than half as much as sequential guidance. The 20-episode-per-task/level run is
named `pi05_time_value_trust_t03_r050_heldout20_diag` and writes under
`results/spatial_flow_guidance_pilot/`. It remains diagnostic because the
underlying value model failed the clearance gate.

### Held-out diagnostic outcome

The promoted run completed all 160 guided rollouts on two H100s (Slurm
allocations `37480` and `37486`), with deterministic CPU value gradients and
pi0.5 on GPU. There were no skipped episodes, error artifacts, or evaluator
exceptions. The complete paired result is:

| Method | Success | Collision | Safe success | Mean chunk inference |
|---|---:|---:|---:|---:|
| Plain pi0.5 | 94/160 (58.75%) | 107/160 (66.88%) | 46/160 (28.75%) | 32.4 ms |
| Trust-guided pi0.5 | **105/160 (65.63%)** | **104/160 (65.00%)** | **53/160 (33.13%)** | 787.3 ms |

Paired guided-minus-plain effects with episode-clustered 95% bootstrap CIs:

- task success: `+6.88 pp`, CI `[+0.00, +13.75]`; 24 improved pairs and 13 worsened;
- collision: `-1.88 pp`, CI `[-8.13, +4.38]`; 16 improved pairs and 13 worsened;
- safe success: `+4.38 pp`, CI `[-1.88, +10.63]`; 16 improved pairs and 9 worsened.

Thus the guided point estimate beats plain pi0.5 on all three aggregate
metrics, most clearly on task success. None of the clustered intervals is
strictly separated from zero, and per-stratum behavior is heterogeneous, so
the run does not establish a statistically reliable safety improvement. The
observed mean chunk latency is 25.2 times the stored plain baseline latency;
this deterministic CPU-gradient configuration is not deployment-ready.

The model's own score increased by `+0.0123` per chunk on average, and its
normalized-clearance head increased by `+0.0152`; these are model predictions,
not measured simulator-clearance gains. The original failed clearance gate
continues to control the scientific interpretation.

Final files:

```text
results/spatial_flow_guidance_pilot/TIME_CONDITIONED_TRUST_T03_R050_HELDOUT20_RESULTS.md
results/spatial_flow_guidance_pilot/time_conditioned_trust_t03_r050_heldout20_comparison.json
```

Every active run directory contains an identity-checked `manifest.json`. The
audit verified 160 exact episode artifacts, eight manifests, a common source
diff hash `12d22d821c963dbbda8072ed68221b896b47e1eeeb857bf70535938bad63fc58`,
one injection per chunk, and an observed correction/nominal-step ratio in
`[0.49999988, 0.50000012]`. The pre-restart zero-artifact manifests affected by
the first-path serialization bug are retained under
`results/failed_runs/time_conditioned_trust_heldout20_manifest_path_bug_jobs37480_37486/`.

## Continuation outcome (2026-08-09)

The original `36430 -> 36431 -> 36432` chain could not be resumed in place:
`36430` had already ended `OUT_OF_MEMORY`, and the two downstream `afterok`
dependencies could never become eligible. The completed trace corpus was
retained and the stale dependent jobs were cancelled.

The data stage is complete and auditable. Retry job `36429` collected all 245
source rollout groups into `training_dataset/pi05_denoising_value_v1`; the
audit passed with 11,553 chunks, 4,770 same-context perturbation pairs, and
115,530 denoising states. The first trainer repeatedly decompressed a full NPZ
tensor for every selected sample and exceeded 128 GiB before epoch one. The
loader now caches each NPZ tensor once and was measured at 3.43 GiB peak RSS
for the complete train/validation materialization.

Replacement job `37233` then trained successfully on one H100, but failed the
unchanged offline gradient gate. A fixed-hidden pair-loss correction in job
`37236` produced the strongest candidate:

| Offline gradient metric | Result |
|---|---:|
| Informative held-out pairs | 71 / 1,122 |
| Independent rollout clusters | 31 |
| Direction accuracy | 73.24% |
| Cluster-bootstrap accuracy 95% CI | [61.82%, 82.90%] |
| Mean selected-minus-rejected clearance | +1.135 mm |
| Cluster-bootstrap clearance 95% CI | [-1.673, +3.611] mm |
| Gate | **Failed** |

The accuracy condition passed, but the lower confidence bound for simulated
clearance remained below zero. Independent seeds 11, 19, and 23, all seed
ensembles, a lower fine-tuning rate, a hidden-conditioned bilinear action head,
an action-affine head, and all-pair continuous regression were also evaluated;
none passed both unchanged confidence-bound conditions. Their manifests and
gate metrics are retained under
`Safety-value-function/time_conditioned_candidates/`; negative intermediate
runs are retained under `Safety-value-function/failed_runs/`.

Consequently, replacement live-gate job `37234` and guided held-out job
`37235` were cancelled instead of silently bypassing the scientific gate. At
that decision point, no guided held-out episodes were launched. The plain
held-out comparator remains complete at `results/pi05_spatial_heldout_20ep`:
160/160 episodes, 94 successes (58.75%), 107 collision episodes (66.88%), and
46 safe successes (28.75%), recomputed directly from all NPZ artifacts.

The defensible next step is to collect substantially more clearance-informative
same-context directions per rollout before retraining. Running the live gate or
the 160 guided held-out episodes with the current checkpoint would require an
explicit decision to weaken or bypass the pre-registered offline gate.


`main/main_aegis.py` now accepts `--flow-guidance-run-name`, and
`scripts/run_spatial_flow_guidance_pilot.sh` exposes it as `GUIDED_RUN_NAME`.

## Time-conditioned value-guidance run in progress (2026-08-05)

The hidden-only pilots below are retained as negative/diagnostic history. The
active experiment trains `V(hidden_t, noisy_action_t, t)` with signed minimum
clearance and same-state paired perturbations, as specified in
`CONTINUOUS_SCORE_GUIDANCE_PLAN.md`.

Current durable artifacts and jobs:

- Plain `0–19` pilot: job `36147`, completed in 3:04:04 at
  `results/pi05_spatial_20ep`. It contains 159 valid rollout pairs plus one
  explicit `no active obstacle` skip marker, so it is not the final held-out
  comparator. Across the 159 valid rollouts it achieved 97 successes (61.0%),
  111 collision rollouts (69.8%), and 44 safe successes (27.7%).
- Validated hazard-aware trace pilot:
  `training_dataset/pi05_denoising_value_pilot_v4`.
- Full 245-episode trace corpus: job `36205`, writing a clean run to
  `training_dataset/pi05_denoising_value_v1`. Job `36195` stopped after eight
  files because its manifest mistakenly treated unrelated dirty-worktree edits
  as producer drift; that partial is quarantined under
  `training_dataset/failed_collections/` and is excluded from training.
- Value audit/training/offline gate: job `36206`, dependent on `36205`, writing
  to `Safety-value-function/time_conditioned_clearance_v1`.
- Fresh same-state simulator gradient gate: job `36207`, dependent on `36206`.
- Held-out split manifest: `results/spatial_heldout20_split.json`. The selected
  IDs must be absent from the corresponding `pi05_hidden_chunks` strata and
  have an active obstacle in all eight Spatial task/level strata. The completed
  audit selected `8-13, 15-19, 21-29`; SHA-256
  `914cea912e5d5c934f5736cd109c9294f5bd6388cf229e282aa695ded0439f1e`.
- Plain held-out output: `results/pi05_spatial_heldout_20ep`, run name
  `pi05_plain_heldout20`, job `36214`.
- Guided held-out output: `results/spatial_flow_guidance_pilot`, run name
  `pi05_time_value_guided_heldout20`, dependency-gated job `36215`.

The final comparison is 160 exactly paired rollouts per method: four tasks,
two safety levels, and 20 common held-out initial states. Both policies use the
same deterministic `(10,32)` initial denoising noise at every replanning step.
The guided launch is dependency-gated and cannot start if either gradient gate
fails. Final aggregate, per-stratum, paired confidence-interval, manifest, and
checksum results will be written to:

```text
results/spatial_flow_guidance_pilot/time_conditioned_heldout20_comparison.json
results/spatial_flow_guidance_pilot/TIME_CONDITIONED_HELDOUT20_RESULTS.md
```


- Composition: `v_total = v_task + v_safe`
### Planned value-model improvement

The current training set contains 11,110 chunks from 245 rollouts. The present
MLP pools each `(10, 1024)` hidden sequence into mean/std/max/last features and
uses five-action binary contact labels. A compact token-attention model is the
next model trial: retain token order and train against a discounted
future-hazard target so a collision just beyond the executed five-action chunk
produces an early warning signal. A small model is preferred over a large
transformer because the grouped dataset has only 245 independent rollouts.

### Future-hazard value retraining

The existing MLP was retrained on an H100 with a 20-action discounted
future-hazard target (`tau=10`). This changes 8.35% of the 11,110 chunks from a
binary endpoint into a continuous early-warning target.

The first attempt reused the asymmetric focal/ranking loss and balanced
sampling. It overfit after one epoch and was rejected: held-out grouped
future-hazard AUC changed from 0.8114 (old model) to 0.8069, while Brier error
worsened from 0.1426 to 0.1580. Its artifacts remain in
`Safety-value-function/chunk_safety_value_future_hazard/` as a negative result.


| Value model | Future-hazard AUC | Immediate-collision AUC | Future Brier | Immediate Brier |
|---|---:|---:|---:|---:|
| Original chunk MLP | 0.8114 | 0.8629 | 0.1426 | 0.1563 |
| Future-hazard soft-loss MLP | **0.8403** | **0.8817** | **0.1106** | **0.1185** |

The improved checkpoint and JAX export are in
`Safety-value-function/chunk_safety_value_future_hazard_soft/`. A separate
`new-try-2` rollout will use the same composition as `new-try` and change only
the value checkpoint, isolating the value-model effect.


## Setup

- Policy: pi0.5 LIBERO checkpoint.
- Suite: all four `safelibero_spatial` tasks at safety levels I and II.
- Episodes: episode 0 for each task/level, giving eight paired rollouts.
- Pairing: baseline and guided runs used the same deterministic `(10, 32)`
  initial flow noise at each replanning point.
- Residual: `v_safe = -lambda(t) * grad(log V)` with RMS normalization,
  `guidance_scale=0.25`, and a quadratic late schedule for `t <= 0.5`.
- This is a functional pilot, not a statistically sufficient benchmark result.

### 3. Value model to train

Train a compact time-conditioned value

```text
V_theta(h_t, a_t, t) -> (safety probability, minimum clearance)
```

rather than another final-state pooled MLP. Each of the ten action tokens will
fuse a projected hidden token, its 32-dimensional noisy action, and a Fourier
embedding of `t`. Two small self-attention blocks will preserve token order;
attention pooling followed by two output heads will predict the safety value
and clearance. Width, depth, and parameter count will remain small because
the independent sample count is the number of rollouts, not the much larger
number of correlated denoising states.

The planned loss is:

```text
L = soft BCE(safety) + 0.25 * Brier(safety)
    + lambda_clearance * Huber(predicted_clearance, measured_clearance)
    + lambda_rank * paired_clearance_ranking
    + lambda_grad * gradient_alignment
    + lambda_dev * action_deviation_regularization
```

The paired ranking term orders two perturbations from the same observation by
measured clearance. The gradient-alignment term uses finite-difference local
perturbations and penalizes a score gradient whose ascent direction lowers
simulated clearance. Action-deviation regularization limits exploitation far
from the pi0.5 action manifold. Hyperparameters are selected on rollout-grouped
validation loss and gradient alignment, with early stopping; the held-out
episodes remain untouched until the final gate.
