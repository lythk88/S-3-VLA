"""Controlled ten-way two-phase value/guidance experiment definitions."""

from __future__ import annotations

import argparse
import json


VARIANTS = {
    "01_balanced": {
        "description": "Balanced safety, clearance, pair-rank, and pair-delta supervision.",
        "training_args": [],
        "guidance": {
            "times": [0.3], "scale": 0.5, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": False, "clearance_score_weight": 0.5,
        },
    },
    "02_safety_only": {
        "description": "Phase-2 collision-safety classification without clearance or pair losses.",
        "training_args": ["--clearance-weight", "0", "--pair-weight", "0"],
        "guidance": {
            "times": [0.3], "scale": 0.35, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.0,
        },
    },
    "03_clearance_only": {
        "description": "Phase-2 pointwise clearance regression with the Phase-1 safety prior frozen only by initialization.",
        "training_args": ["--phase2-safety-weight", "0", "--pair-weight", "0"],
        "guidance": {
            "times": [0.3], "scale": 0.35, "normalization": "task-step-rms",
            "geometry": "orthogonal", "integration": "state", "translation_only": True,
            "value_backtracking": False, "clearance_score_weight": 1.0,
        },
    },
    "04_pair_rank_only": {
        "description": "Same-state pairwise clearance ranking without pointwise Phase-2 losses.",
        "training_args": [
            "--phase2-safety-weight", "0", "--clearance-weight", "0",
            "--pair-difference-weight", "0", "--pair-weight", "2",
            "--informative-pair-sampling-fraction", "1",
            "--minimum-pair-clearance-difference", "0.00005",
        ],
        "guidance": {
            "times": [0.1], "scale": 0.25, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.5,
        },
    },
    "05_pair_delta_only": {
        "description": "Same-state pairwise clearance-difference regression without rank or pointwise Phase-2 losses.",
        "training_args": [
            "--phase2-safety-weight", "0", "--clearance-weight", "0",
            "--pair-rank-weight", "0", "--pair-weight", "2",
            "--informative-pair-sampling-fraction", "0",
            "--minimum-pair-clearance-difference", "0",
        ],
        "guidance": {
            "times": [0.1], "scale": 0.25, "normalization": "task-step-rms",
            "geometry": "orthogonal", "integration": "state", "translation_only": True,
            "value_backtracking": False, "clearance_score_weight": 1.0,
        },
    },
    "06_bilinear": {
        "description": "Hidden-conditioned bilinear action head for an explicitly structured action gradient.",
        "training_args": ["--bilinear-action-head", "--pair-weight", "2"],
        "guidance": {
            "times": [0.3], "scale": 0.35, "normalization": "task-step-rms",
            "geometry": "task-compatible", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.5,
        },
    },
    "07_local_gradient": {
        "description": "Local finite-difference training at one-quarter perturbation radius with dense weak-pair sampling.",
        "training_args": [
            "--pair-local-fraction", "0.25", "--pair-weight", "2",
            "--informative-pair-sampling-fraction", "1",
            "--minimum-pair-clearance-difference", "0.00005",
            "--finetune-learning-rate", "0.00005",
        ],
        "guidance": {
            "times": [0.1], "scale": 0.25, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.5,
        },
    },
    "08_late_time": {
        "description": "Late-denoising specialist trained only at t=0.1 plus branch states.",
        "training_args": ["--trace-times", "0.1", "--pair-weight", "2"],
        "guidance": {
            "times": [0.1], "scale": 0.35, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.5,
        },
    },
    "09_dense_time": {
        "description": "All-ten-step denoising supervision with three sequential state injections.",
        "training_args": ["--trace-times", "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1"],
        "guidance": {
            "times": [0.5, 0.3, 0.1], "scale": 0.15,
            "normalization": "task-step-rms", "geometry": "task-compatible",
            "integration": "state", "translation_only": True,
            "value_backtracking": False, "clearance_score_weight": 0.5,
        },
    },
    "10_wide_deep": {
        "description": "Higher-capacity 384-wide, three-layer, six-head token transformer.",
        "training_args": ["--width", "384", "--layers", "3", "--heads", "6", "--pair-weight", "2"],
        "guidance": {
            "times": [0.3], "scale": 0.5, "normalization": "task-step-rms",
            "geometry": "direct", "integration": "state", "translation_only": True,
            "value_backtracking": True, "clearance_score_weight": 0.5,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--training-args", action="store_true")
    parser.add_argument("--gate-clearance-weight", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(VARIANTS))
        return
    if args.variant is None:
        print(json.dumps(VARIANTS, indent=2, sort_keys=True))
        return
    variant = VARIANTS[args.variant]
    if args.training_args:
        # Do not emit a blank line for the balanced variant: Bash mapfile
        # would turn it into one empty command-line argument.
        print("\n".join(variant["training_args"]), end="")
    elif args.gate_clearance_weight:
        print(variant["guidance"]["clearance_score_weight"])
    else:
        print(json.dumps(variant, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
