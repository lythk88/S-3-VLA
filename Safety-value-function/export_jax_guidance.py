#!/usr/bin/env python3
"""Export the PyTorch safety-value checkpoint as a framework-neutral NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location="cpu")
    normalizer = np.load(args.run_dir / "normalizer.npz")
    arrays = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in checkpoint["model_state_dict"].items()
    }
    arrays["feature_mean"] = normalizer["mean"].astype(np.float32)
    arrays["feature_std"] = np.where(
        normalizer["std"] < 1e-6,
        1.0,
        normalizer["std"],
    ).astype(np.float32)
    output = args.run_dir / "jax_guidance_model.npz"
    np.savez_compressed(output, **arrays)
    manifest_path = args.run_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("artifacts", {})[output.name] = {
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {output}")


if __name__ == "__main__":
    main()
