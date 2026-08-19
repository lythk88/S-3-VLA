#!/usr/bin/env python3
"""Compare stronger and multi-time guidance with the fixed-noise t=0.1 baseline."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_time_conditioned_risk_gate_sweep import load, summarize


ROOT = pathlib.Path("results/time_conditioned_strength_timing_sweep_v1")
BASE_ROOT = pathlib.Path("results/time_conditioned_risk_gate_sweep_v1")
METHODS = {
    "current_s0.35_t0.1": (BASE_ROOT, "pi05_riskgate_t07_fixed"),
    "strong_single_s0.7_t0.1": (ROOT, "pi05_riskgate_t07_strong_single_fixed"),
    "multi_s0.35_t0.3_0.1": (ROOT, "pi05_riskgate_t07_multi_fixed"),
    "strong_multi_s0.5_t0.5_0.3_0.1": (
        ROOT,
        "pi05_riskgate_t07_strong_multi_fixed",
    ),
}


def main() -> None:
    records = {name: load(root, run) for name, (root, run) in METHODS.items()}
    keys = set(records["current_s0.35_t0.1"])
    if len(keys) != 320 or any(set(method) != keys for method in records.values()):
        raise RuntimeError(
            "Sweep is incomplete: "
            + ", ".join(f"{name}={len(method)}" for name, method in records.items())
        )
    result = {"methods": {name: summarize(method) for name, method in records.items()}}
    baseline = records["current_s0.35_t0.1"]
    for name, method in records.items():
        values = result["methods"][name]
        values["success_rate"] = values["successes"] / values["rollouts"]
        values["collision_rate"] = values["collisions"] / values["rollouts"]
        values["safe_success_rate"] = values["safe_successes"] / values["rollouts"]
        if method is not baseline:
            values["versus_current"] = {
                field: sum(method[key][field] for key in keys)
                - sum(baseline[key][field] for key in keys)
                for field in ("success", "collision", "safe_success")
            }
    output = ROOT / "strength_timing_sweep_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
