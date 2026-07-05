#!/usr/bin/env python3
"""Predict safe/unsafe from one hidden-state .npz file."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from train_hidden_state_mlp import HiddenStateMLP, feature_from_hidden_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--npz", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location=device)
    normalizer = np.load(args.run_dir / "normalizer.npz")
    model = HiddenStateMLP(
        int(checkpoint["input_dim"]),
        tuple(checkpoint["hidden_dims"]),
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(args.npz) as data:
        feature = feature_from_hidden_states(data["last_layer_hidden_states"])
    feature = (feature - normalizer["mean"]) / normalizer["std"]

    with torch.no_grad():
        logits = model(torch.from_numpy(feature).float().unsqueeze(0).to(device))
        unsafe_prob = torch.sigmoid(logits).item()

    result = {
        "npz": str(args.npz),
        "safe_prob": 1.0 - unsafe_prob,
        "unsafe_prob": unsafe_prob,
        "pred_label": "unsafe" if unsafe_prob >= float(checkpoint["threshold"]) else "safe",
        "threshold": float(checkpoint["threshold"]),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
