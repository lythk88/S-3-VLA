#!/usr/bin/env python3
"""Summarize the fixed-noise pi0.5 safety-threshold sweep."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


METHODS = {
    "plain": "pi05_riskgate_plain_fixed",
    "threshold_0.3": "pi05_riskgate_t03_fixed",
    "threshold_0.5": "pi05_riskgate_t05_fixed",
    "threshold_0.7": "pi05_riskgate_t07_fixed",
}


def load(root: pathlib.Path, run_name: str):
    records = {}
    for path in root.glob(f"*/*/{run_name}_*/*_last_layer_hidden_states.npz"):
        suite, task, run_dir = path.relative_to(root).parts[:3]
        level = run_dir.rsplit("_", 1)[-1]
        episode = int(path.name.split("_", 1)[0])
        with np.load(path, allow_pickle=False) as data:
            records[(suite, task, level, episode)] = {
                "success": bool(data["success"]),
                "collision": bool(data["collision"]),
                "safe_success": bool(data["safe_success"]),
                "gate": np.asarray(
                    data["time_conditioned_risk_gate_active"]
                    if "time_conditioned_risk_gate_active" in data
                    else [],
                    dtype=np.bool_,
                ),
                "infer_ms": np.asarray(data["chunk_infer_ms"], dtype=np.float64),
            }
    return records


def summarize(records):
    values = list(records.values())
    chunks = np.concatenate([x["gate"] for x in values if len(x["gate"])]) if any(
        len(x["gate"]) for x in values
    ) else np.empty(0, dtype=np.bool_)
    timing = np.concatenate([x["infer_ms"] for x in values if len(x["infer_ms"])])
    return {
        "rollouts": len(values),
        "successes": sum(x["success"] for x in values),
        "collisions": sum(x["collision"] for x in values),
        "safe_successes": sum(x["safe_success"] for x in values),
        "guided_chunks": int(chunks.sum()),
        "chunks": len(chunks),
        "guidance_activation_rate": float(chunks.mean()) if len(chunks) else None,
        "mean_chunk_infer_ms": float(np.mean(timing)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("results/time_conditioned_risk_gate_sweep_v1"),
    )
    args = parser.parse_args()
    all_records = {name: load(args.root, run) for name, run in METHODS.items()}
    keys = set(all_records["plain"])
    if len(keys) != 320 or any(set(value) != keys for value in all_records.values()):
        raise RuntimeError(
            "Sweep is incomplete: "
            + ", ".join(f"{name}={len(value)}" for name, value in all_records.items())
        )
    result = {"methods": {name: summarize(value) for name, value in all_records.items()}}
    plain = all_records["plain"]
    for name, records in all_records.items():
        if name == "plain":
            continue
        result["methods"][name]["versus_plain"] = {
            field: sum(records[key][field] for key in keys)
            - sum(plain[key][field] for key in keys)
            for field in ("success", "collision", "safe_success")
        }
    output = args.root / "risk_gate_sweep_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
