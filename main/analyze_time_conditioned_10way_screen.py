"""Join ten-way training gates with the paired fixed-noise simulator screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "time_conditioned_method_screen",
    HERE / "analyze_time_conditioned_method_screen.py",
)
screen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(screen)


def _load_catalog(path: pathlib.Path) -> dict:
    catalog_spec = importlib.util.spec_from_file_location("tenway_catalog", path)
    catalog = importlib.util.module_from_spec(catalog_spec)
    assert catalog_spec.loader is not None
    catalog_spec.loader.exec_module(catalog)
    return catalog.VARIANTS


def analyze(args) -> dict:
    training_root = pathlib.Path(args.training_root)
    results_root = pathlib.Path(args.results_root)
    variants = _load_catalog(pathlib.Path(args.catalog))
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    baseline, baseline_manifests = screen._load(
        results_root, args.baseline_run_name, episodes
    )
    methods = []
    for variant, definition in variants.items():
        run_dir = training_root / variant
        run_name = f"pi05_tc10_{variant}_screen"
        guided, manifests = screen._load(results_root, run_name, episodes)
        if set(guided) != set(baseline):
            raise RuntimeError(f"Paired rollout keys differ for {variant}")
        gate = json.loads((run_dir / "gradient_gate_metrics.json").read_text())
        training = json.loads((run_dir / "training_manifest.json").read_text())
        approach = json.loads((run_dir / "approach.json").read_text())
        paired = {}
        for metric in ("success", "collision", "safe_success"):
            delta = np.asarray(
                [
                    float(guided[key][metric]) - float(baseline[key][metric])
                    for key in sorted(baseline)
                ]
            )
            beneficial = -delta if metric == "collision" else delta
            paired[metric] = {
                "guided_minus_plain": float(np.mean(delta)),
                "improved_pairs": int(np.sum(beneficial > 0)),
                "worsened_pairs": int(np.sum(beneficial < 0)),
                "unchanged_pairs": int(np.sum(beneficial == 0)),
            }
        methods.append(
            {
                "variant": variant,
                "description": definition["description"],
                "guidance": approach["guidance"],
                "run_name": run_name,
                "offline_gate": gate,
                "finetune_best_validation": training["metrics"][
                    "finetune_validation"
                ]["best_validation"],
                "summary": screen._summary(guided),
                "paired": paired,
                "manifests": manifests,
            }
        )

    def rank_key(item):
        values = item["summary"]
        return (
            values["safe_success_count"],
            values["success_count"],
            -values["collision_count"],
            item["offline_gate"]["gradient_direction_accuracy_cluster_ci95"][0],
            -values["mean_chunk_infer_ms"],
        )

    methods.sort(key=rank_key, reverse=True)
    for rank, item in enumerate(methods, 1):
        item["empirical_rank"] = rank
    eligible = sorted(
        [item for item in methods if item["offline_gate"]["pass"]],
        key=rank_key,
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "scope": "development screen; final confirmation uses disjoint held-out episodes",
        "pairing": {
            "suite": "safelibero_spatial",
            "episodes": episodes,
            "rollouts_per_method": len(baseline),
            "fixed_flow_noise": True,
            "task_level_strata": 8,
        },
        "selection_protocol": {
            "gate": "Only offline-gradient-gate-passing variants are eligible for confirmation.",
            "empirical_ranking": (
                "safe successes, successes, fewer collisions, offline accuracy CI lower bound, "
                "then lower mean chunk inference"
            ),
            "confirmation_variants": [item["variant"] for item in eligible],
        },
        "baseline": {
            "run_name": args.baseline_run_name,
            "summary": screen._summary(baseline),
            "manifests": baseline_manifests,
        },
        "methods": methods,
    }
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    base = report["baseline"]["summary"]
    lines = [
        "# Two-phase time-conditioned value: ten-way development screen",
        "",
        (
            f"Fixed-noise paired screen on episodes {episodes}: {len(baseline)} Spatial "
            "SafeLIBERO rollouts per method. Episodes 0-7 were used for training and are excluded."
        ),
        "",
        "Only variants passing the rollout-clustered offline gradient gate are promoted to the larger confirmation.",
        "",
        "| Rank | Variant | Offline gate | Success | Collision | Safe success | Chunk inference |",
        "|---:|---|:---:|---:|---:|---:|---:|",
        (
            f"| plain | pi0.5 | — | {base['success_count']}/{base['rollouts']} | "
            f"{base['collision_count']}/{base['rollouts']} | "
            f"{base['safe_success_count']}/{base['rollouts']} | "
            f"{base['mean_chunk_infer_ms']:.1f} ms |"
        ),
    ]
    for item in methods:
        values = item["summary"]
        lines.append(
            f"| {item['empirical_rank']} | {item['variant']} | "
            f"{'pass' if item['offline_gate']['pass'] else 'fail'} | "
            f"{values['success_count']}/{values['rollouts']} | "
            f"{values['collision_count']}/{values['rollouts']} | "
            f"{values['safe_success_count']}/{values['rollouts']} | "
            f"{values['mean_chunk_infer_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "Confirmation variants: "
            + (", ".join(item["variant"] for item in eligible) or "none"),
        ]
    )
    pathlib.Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--episodes", default="8,9")
    parser.add_argument("--baseline-run-name", default="pi05_plain_tc10_screen")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
