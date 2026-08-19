# Safety-value training: updated Phase 1 and backup 39396 Phase 2

## Data

- Phase 1: `training_dataset/pi05_hidden_chunks` (248 rollout groups,
  11,477 action chunks).
- Phase 2:
  `training_dataset/regeneration_backups/pi05_denoising_value_v1_pre_regeneration_39396`
  (241 rollout groups, 11,370 nominal chunks, 9,432 counterfactual branches,
  and 4,716 symmetric perturbation pairs).

The Phase-2 backup is missing seven rollouts from one Level-I long-horizon
stratum. They remain valid Phase-1-only examples. Phase-2 groups are split
first by complete rollout and task/level stratum. Every overlapping Phase-1
group inherits that assignment, so no Phase-2 validation rollout is seen in
Phase-1 training. The seven Phase-1-only groups receive a separate deterministic
6/1 train/validation split. The final counts are:

- Phase 1: 8,609 train chunks and 2,868 validation chunks.
- Phase 2: 33,147 train examples and 10,395 validation examples.
- Informative paired branches (absolute clearance difference at least 0.5 mm):
  293 train and 75 validation pairs.

## Model

The model is

```text
V_theta(hidden_t[10,1024], noisy_action_t[10,32], t)
    -> (safety_logit, normalized_minimum_clearance)
```

It projects hidden and noisy-action tokens to width 256, adds a Fourier time
embedding and learned token positions, applies two pre-norm four-head
Transformer blocks, attention-pools the sequence, and uses separate safety and
clearance heads.

## Phase 1: hidden-state bootstrap

For a chunk starting at action `s`, find the first collision offset `d` in the
next 20 executed actions. The soft future-safety target is

```text
y = 1                              if no collision occurs
y = 1 - exp(-d / 10)               otherwise
```

Thus immediate contact has target zero, later contact is discounted, and a
collision-free horizon has target one. Phase 1 supplies the final hidden state,
zero noisy action, and `t=0`. Its loss is

```text
L_phase1 = BCEWithLogits(safety_logit, y)
           + 0.25 * mean((sigmoid(safety_logit) - y)^2)
```

This phase initializes the hidden representation; it does not claim to teach
an action gradient.

## Phase 2: denoising-state and clearance fine-tuning

Nominal examples use denoising times `0.1`, `0.3`, and `0.5`. Their safety
target uses the same 20-action future target as Phase 1. Their clearance target
is the minimum signed robot-to-active-obstacle distance over the next 20
actions. Counterfactual branches use hard collision safety and the minimum
clearance over their five simulated actions. Clearance is normalized by the
collection cap, 0.03 m.

For each example:

```text
L_safety = BCEWithLogits(safety_logit, y)
           + 0.25 * Brier(sigmoid(safety_logit), y)

L_clearance = SmoothL1(predicted_clearance, measured_clearance / 0.03)
```

For a same-state symmetric `(-,+)` branch pair, define the online guidance
score and measured normalized clearance difference:

```text
score = logsigmoid(safety_logit) + 0.5 * predicted_clearance
delta_score = score_plus - score_minus
delta_c = (clearance_plus - clearance_minus) / 0.03

L_pair = softplus(-sign(delta_c) * delta_score)
         + SmoothL1(predicted_delta_clearance, delta_c)
```

The Phase-2 objective is

```text
L_phase2 = L_safety + L_clearance + L_pair
```

Only 368/4,716 pairs have at least 0.5 mm clearance separation. Each pair
minibatch therefore draws 50% from these informative pairs and 50% from all
pairs. This keeps flat local examples as regularization while giving every
update a useful directional signal on average.

Both phases use AdamW (`lr=1e-4`, weight decay `1e-4`), batch size 256, global
gradient-norm clipping at 1.0, and seed 7. Adam moments are reset at the phase
boundary. Phase 1 runs at most 10 epochs; Phase 2 runs at most 50 epochs with
patience 8. The selected checkpoint minimizes complete-rollout-grouped Phase-2
validation total loss.

## Validation status

Training validation is not a safety certification. After training,
`evaluate_time_conditioned_gradient.py` evaluates the held-out symmetric pairs.
It requires the rollout-clustered 95% lower confidence bound of direction
accuracy to exceed 0.5 and the lower confidence bound of selected-minus-rejected
clearance to exceed zero. A failed gate leaves the checkpoint as a research
artifact and must not be described as safety-validated.

## Reproduction

```bash
ALLOW_SOURCE_ONLY_GROUPS=1 \
BOOTSTRAP_ROOT=/home/lythk/safe-flow-matching/training_dataset/pi05_hidden_chunks \
TRACE_ROOT=/home/lythk/safe-flow-matching/training_dataset/regeneration_backups/pi05_denoising_value_v1_pre_regeneration_39396 \
OUTPUT_DIR=/home/lythk/safe-flow-matching/Safety-value-function/time_conditioned_clearance_phase1_updated_phase2_backup39396_v1 \
INFORMATIVE_PAIR_SAMPLING_FRACTION=0.5 \
bash /home/lythk/safe-flow-matching/scripts/run_time_conditioned_value_training_v1.sh
```
