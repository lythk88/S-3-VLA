"""Select the best gate-passing variant on the disjoint four-suite confirmation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "time_conditioned_heldout", HERE / "analyze_time_conditioned_heldout.py"
)
heldout = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(heldout)

SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)


def _combine(root: pathlib.Path, run_name: str, episodes: list[int]):
    combined = {}
    per_suite = {}
    manifests = {}
    for suite in SUITES:
        records, suite_manifests = heldout._load(root / suite, run_name, episodes)
        per_suite[suite] = heldout._summary(records)
        manifests[suite] = suite_manifests
        combined.update({(suite, *key): value for key, value in records.items()})
    return combined, per_suite, manifests


def _paired(first, second, *, seed: int, replicates: int):
    if set(first) != set(second):
        raise RuntimeError("Paired confirmation keys differ")
    keys = sorted(first)
    clusters = np.asarray([key[-1] for key in keys])
    result = {}
    for metric in ("success", "collision", "safe_success"):
        delta = np.asarray(
            [float(second[key][metric]) - float(first[key][metric]) for key in keys]
        )
        beneficial = -delta if metric == "collision" else delta
        result[metric] = {
            "second_minus_first": float(np.mean(delta)),
            "episode_cluster_ci95": heldout._cluster_ci(
                delta, clusters, seed, replicates
            ),
            "second_improved_pairs": int(np.sum(beneficial > 0)),
            "second_worsened_pairs": int(np.sum(beneficial < 0)),
            "unchanged_pairs": int(np.sum(beneficial == 0)),
        }
    return result


def analyze(args) -> dict:
    root = pathlib.Path(args.results_root)
    training_root = pathlib.Path(args.training_root)
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    variants = [value for value in args.variants.split(",") if value]
    if len(episodes) != 18 or any(value < 10 for value in episodes):
        raise RuntimeError("Expected the 18 confirmation episodes disjoint from 0-9")
    if len(variants) < 2:
        raise RuntimeError("Confirmation must compare at least two finalists")

    plain, plain_suites, plain_manifests = _combine(
        root, args.baseline_run_name, episodes
    )
    methods = []
    records_by_variant = {}
    for variant in variants:
        gate = json.loads(
            (training_root / variant / "gradient_gate_metrics.json").read_text()
        )
        if not gate["pass"]:
            raise RuntimeError(f"Confirmation variant {variant} did not pass offline gate")
        run_name = f"pi05_tc10_{variant}_confirm"
        records, suites, manifests = _combine(root, run_name, episodes)
        records_by_variant[variant] = records
        methods.append(
            {
                "variant": variant,
                "run_name": run_name,
                "offline_gate": gate,
                "summary": heldout._summary(records),
                "per_suite": suites,
                "paired_vs_plain": _paired(
                    plain,
                    records,
                    seed=args.seed,
                    replicates=args.bootstrap_replicates,
                ),
                "manifests": manifests,
            }
        )

    methods.sort(
        key=lambda item: (
            item["summary"]["safe_success_count"],
            item["summary"]["success_count"],
            -item["summary"]["collision_count"],
            -item["summary"]["mean_chunk_infer_ms"],
        ),
        reverse=True,
    )
    winner = methods[0]
    runner_up = methods[1]
    finalist_comparison = _paired(
        records_by_variant[runner_up["variant"]],
        records_by_variant[winner["variant"]],
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    report = {
        "schema_version": 1,
        "scope": "four-suite, fixed-noise, training-and-screen-disjoint confirmation",
        "selection_rule": "safe successes, then successes, then fewer collisions, then lower latency",
        "pairing": {
            "suites": list(SUITES),
            "episodes": episodes,
            "rollouts_per_method": len(plain),
            "fixed_flow_noise": True,
            "task_level_strata": 32,
        },
        "plain": {
            "run_name": args.baseline_run_name,
            "summary": heldout._summary(plain),
            "per_suite": plain_suites,
            "manifests": plain_manifests,
        },
        "methods": methods,
        "selected_variant": winner["variant"],
        "selected_vs_runner_up": {
            "runner_up": runner_up["variant"],
            "paired": finalist_comparison,
        },
    }
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    plain_summary = report["plain"]["summary"]
    lines = [
        "# Time-conditioned safety-value confirmation",
        "",
        (
            f"Four SafeLIBERO suites, both safety levels, four tasks per suite, and "
            f"{len(episodes)} disjoint episodes: {len(plain)} fixed-noise rollouts per method."
        ),
        "",
        "| Method | Success | Collision | Safe success | Mean chunk inference |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Plain pi0.5 | {plain_summary['success_count']}/{len(plain)} "
            f"({plain_summary['success_rate']:.1%}) | "
            f"{plain_summary['collision_count']}/{len(plain)} "
            f"({plain_summary['collision_rate']:.1%}) | "
            f"{plain_summary['safe_success_count']}/{len(plain)} "
            f"({plain_summary['safe_success_rate']:.1%}) | "
            f"{plain_summary['mean_chunk_infer_ms']:.1f} ms |"
        ),
    ]
    for item in methods:
        values = item["summary"]
        lines.append(
            f"| {item['variant']} | {values['success_count']}/{len(plain)} "
            f"({values['success_rate']:.1%}) | {values['collision_count']}/{len(plain)} "
            f"({values['collision_rate']:.1%}) | "
            f"{values['safe_success_count']}/{len(plain)} "
            f"({values['safe_success_rate']:.1%}) | "
            f"{values['mean_chunk_infer_ms']:.1f} ms |"
        )
    lines.extend(["", f"Selected variant: **{winner['variant']}**.", ""])
    for item in methods:
        lines.append(f"## {item['variant']} versus plain pi0.5")
        lines.append("")
        for metric, values in item["paired_vs_plain"].items():
            ci = values["episode_cluster_ci95"]
            lines.append(
                f"- {metric}: {values['second_minus_first']:+.3f} "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            )
        lines.append("")
    pathlib.Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--variants", default="01_balanced,08_late_time")
    parser.add_argument(
        "--episodes", default="10,11,12,13,15,16,17,18,19,21,22,23,24,25,26,27,28,29"
    )
    parser.add_argument("--baseline-run-name", default="pi05_plain_tc10_confirm")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
