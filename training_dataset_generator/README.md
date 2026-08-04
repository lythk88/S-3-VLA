# SafeLIBERO training dataset generator

This isolated generator uses the existing SafeLIBERO BDDL tasks but does not
load their fixed test initialization states. Each rollout instead retains the
scene sampled by the environment reset for its assigned seed.

The default seed range is inclusive: 100 through 179.

## Output

The default dataset path is:

```text
/home/lythk/safe-flow-matching/training_dataset
```

Each seed has its own directory, preventing rollouts from overwriting one
another.

## Run

Start the pi0.5 policy server on port 8001, then run:

```bash
cd /home/lythk/safe-flow-matching
python training_dataset_generator/generate_training.py
```

For a small smoke test:

```bash
python training_dataset_generator/generate_training.py \
  --seed-start 100 \
  --seed-end 100 \
  --suite safelibero_spatial \
  --level I \
  --task-index 0
```
