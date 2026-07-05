# Hidden-State MLP Classifier

This folder trains a binary MLP classifier from pi0.5 SafeLIBERO hidden-state `.npz` files.

Input files are read from:

`/home/namn1/vlsa-aegis/results/pi05_no_safety_with_hidden_full`

Each rollout is labeled from `collision`: `collision=True` is `unsafe`, and `collision=False` is `safe`.

The trainer converts each variable-length hidden-state array shaped `(T, 10, 1024)` into one fixed vector:

`mean_all_tokens || std_all_tokens || max_all_tokens || last_chunk_mean`

Training outputs are written under:

`/home/namn1/vlsa-aegis/MLP-classifier/runs/`

Main command:

```bash
/home/namn1/vlsa-aegis/main/.venv/bin/python /home/namn1/vlsa-aegis/MLP-classifier/train_hidden_state_mlp.py
```

Prediction command:

```bash
/home/namn1/vlsa-aegis/main/.venv/bin/python /home/namn1/vlsa-aegis/MLP-classifier/predict_hidden_state.py \
  --run-dir /path/to/run_dir \
  --npz /path/to/file_last_layer_hidden_states.npz
```
