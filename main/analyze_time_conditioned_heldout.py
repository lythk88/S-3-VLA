"""Exact paired report for plain and time-conditioned SafeLIBERO rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: pathlib.Path, run_name: str, episodes: list[int]):
    records = {}
    run_dirs = sorted(root.glob(f"*/{run_name}_I")) + sorted(
        root.glob(f"*/{run_name}_II")
    )
    if len(run_dirs) != 8:
        raise RuntimeError(f"Expected 8 {run_name} task/level directories, found {len(run_dirs)}")
    manifests = {}
    for directory in run_dirs:
        level = directory.name.rsplit("_", 1)[-1]
        task = directory.parent.name
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        manifests[f"{task}/{level}"] = _sha256(manifest)
        for episode in episodes:
            matches = list(directory.glob(f"{episode}_*_last_layer_hidden_states.npz"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one rollout for {task}/{level}/{episode}, found {len(matches)}"
                )
            with np.load(matches[0], allow_pickle=False) as archive:
                def mean_or_nan(name):
                    if name not in archive:
                        return float("nan")
                    values = np.asarray(archive[name], dtype=np.float64)
                    return float(np.nanmean(values)) if values.size else float("nan")

                records[(task, level, episode)] = {
                    "success": bool(archive["success"]),
                    "collision": bool(archive["collision"]),
                    "safe_success": bool(archive["safe_success"]),
                    "infer_ms": float(np.nanmean(archive["chunk_infer_ms"])),
                    "score_change": (
                        mean_or_nan("time_conditioned_scores_after")
                        - mean_or_nan("time_conditioned_scores")
                    ),
                    "normalized_clearance_change": (
                        mean_or_nan("time_conditioned_clearance_predictions_after")
                        - mean_or_nan("time_conditioned_clearance_predictions")
                    ),
                    "correction_rms": mean_or_nan(
                        "time_conditioned_perturbation_rms"
                    ),
                    "correction_to_step": mean_or_nan(
                        "time_conditioned_correction_to_step"
                    ),
                }
    return records, manifests


def _cluster_ci(differences, clusters, seed=7, replicates=10000):
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(clusters == cluster) for cluster in sampled]
        )
        estimates.append(float(np.mean(differences[indices])))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _summary(records):
    count = len(records)
    def finite_mean(field):
        values = np.asarray(
            [value[field] for value in records.values()], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else None

    return {
        "rollouts": count,
        "success_count": sum(value["success"] for value in records.values()),
        "success_rate": float(np.mean([value["success"] for value in records.values()])),
        "collision_count": sum(value["collision"] for value in records.values()),
        "collision_rate": float(np.mean([value["collision"] for value in records.values()])),
        "safe_success_count": sum(value["safe_success"] for value in records.values()),
        "safe_success_rate": float(
            np.mean([value["safe_success"] for value in records.values()])
        ),
        "mean_chunk_infer_ms": float(
            np.mean([value["infer_ms"] for value in records.values()])
        ),
        "mean_score_change": finite_mean("score_change"),
        "mean_normalized_clearance_change": finite_mean(
            "normalized_clearance_change"
        ),
        "mean_correction_rms": finite_mean("correction_rms"),
        "mean_correction_to_nominal_step": finite_mean("correction_to_step"),
    }


def analyze(args):
    split_path = pathlib.Path(args.split_manifest)
    split = json.loads(split_path.read_text())
    episodes = [int(value) for value in split["selected_episodes"]]
    baseline, baseline_manifests = _load(
        pathlib.Path(args.baseline_root), args.baseline_run_name, episodes
    )
    guided, guided_manifests = _load(
        pathlib.Path(args.guided_root), args.guided_run_name, episodes
    )
    if set(baseline) != set(guided):
        raise RuntimeError("Plain/guided paired keys differ")
    keys = sorted(baseline)
    clusters = np.asarray([key[2] for key in keys])
    paired = {}
    for metric in ("success", "collision", "safe_success"):
        delta = np.asarray(
            [float(guided[key][metric]) - float(baseline[key][metric]) for key in keys]
        )
        paired[metric] = {
            "guided_minus_plain": float(np.mean(delta)),
            "episode_cluster_ci95": _cluster_ci(
                delta, clusters, args.seed, args.bootstrap_replicates
            ),
            "improved_pairs": int(
                np.sum(delta < 0) if metric == "collision" else np.sum(delta > 0)
            ),
            "worsened_pairs": int(
                np.sum(delta > 0) if metric == "collision" else np.sum(delta < 0)
            ),
            "unchanged_pairs": int(np.sum(delta == 0)),
        }
    inference_ratio = np.asarray(
        [guided[key]["infer_ms"] / baseline[key]["infer_ms"] for key in keys]
    )
    per_stratum = {}
    for task, level in sorted({(key[0], key[1]) for key in keys}):
        stratum_keys = [key for key in keys if key[:2] == (task, level)]
        per_stratum[f"{task}/{level}"] = {
            "plain": _summary({key: baseline[key] for key in stratum_keys}),
            "guided": _summary({key: guided[key] for key in stratum_keys}),
        }
    report = {
        "schema_version": 1,
        "scope": (
            "diagnostic; value model failed the offline simulated-clearance gate"
            if "diag" in args.guided_run_name
            else "benchmark-eligible only if the referenced value manifest passed all gates"
        ),
        "pairing": {
            "suite": args.suite,
            "episodes": episodes,
            "rollouts_per_method": len(keys),
            "task_level_strata": 8,
            "fixed_flow_noise": True,
            "split_manifest": str(split_path.resolve()),
            "split_manifest_sha256": _sha256(split_path),
        },
        "plain": _summary(baseline),
        "guided": _summary(guided),
        "paired": paired,
        "guided_to_plain_chunk_inference_ratio": float(np.mean(inference_ratio)),
        "per_stratum": per_stratum,
        "manifest_sha256": {
            "plain": baseline_manifests,
            "guided": guided_manifests,
        },
    }
    output_json = pathlib.Path(args.output_json)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plain, guided_summary = report["plain"], report["guided"]
    lines = [
        f"# Time-conditioned {args.suite.replace('safelibero_', '', 1).title()} SafeLIBERO held-out result",
        "",
        (
            "**Diagnostic only:** the referenced value model failed its offline simulated-clearance gate."
            if "diag" in args.guided_run_name
            else "Gate eligibility must be verified from the referenced value-model manifest."
        ),
        "",
        f"Paired test: {len(keys)} rollouts per method, 20 training-disjoint initial states per task/level.",
        "",
        "| Method | Success | Collision | Safe success | Mean chunk inference |",
        "|---|---:|---:|---:|---:|",
        f"| Plain pi0.5 | {plain['success_count']}/{len(keys)} ({plain['success_rate']:.1%}) | "
        f"{plain['collision_count']}/{len(keys)} ({plain['collision_rate']:.1%}) | "
        f"{plain['safe_success_count']}/{len(keys)} ({plain['safe_success_rate']:.1%}) | "
        f"{plain['mean_chunk_infer_ms']:.1f} ms |",
        f"| Time-conditioned guided pi0.5 | {guided_summary['success_count']}/{len(keys)} "
        f"({guided_summary['success_rate']:.1%}) | {guided_summary['collision_count']}/{len(keys)} "
        f"({guided_summary['collision_rate']:.1%}) | {guided_summary['safe_success_count']}/{len(keys)} "
        f"({guided_summary['safe_success_rate']:.1%}) | {guided_summary['mean_chunk_infer_ms']:.1f} ms |",
        "",
        "Paired guided-minus-plain effects (episode-clustered 95% bootstrap CI):",
        "",
    ]
    for metric, values in paired.items():
        lines.append(
            f"- {metric}: {values['guided_minus_plain']:+.3f} "
            f"[{values['episode_cluster_ci95'][0]:+.3f}, {values['episode_cluster_ci95'][1]:+.3f}]"
        )
    lines.extend(
        [
            "",
            f"Guided mean value-score change per chunk: {guided_summary['mean_score_change']:+.4f}.",
            f"Guided mean correction RMS: {guided_summary['mean_correction_rms']:.4f}; "
            f"mean correction/nominal-step ratio: {guided_summary['mean_correction_to_nominal_step']:.3f}.",
            f"Guided/plain mean chunk-inference ratio: {report['guided_to_plain_chunk_inference_ratio']:.1f}x.",
            "",
            "## Per task and safety level",
            "",
            "| Task | Level | Plain success | Guided success | Plain collision | Guided collision | Plain safe success | Guided safe success |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stratum, summaries in per_stratum.items():
        task, level = stratum.rsplit("/", 1)
        plain_stratum = summaries["plain"]
        guided_stratum = summaries["guided"]
        count = plain_stratum["rollouts"]
        lines.append(
            f"| {task.replace('_', ' ')} | {level} | "
            f"{plain_stratum['success_count']}/{count} | "
            f"{guided_stratum['success_count']}/{count} | "
            f"{plain_stratum['collision_count']}/{count} | "
            f"{guided_stratum['collision_count']}/{count} | "
            f"{plain_stratum['safe_success_count']}/{count} | "
            f"{guided_stratum['safe_success_count']}/{count} |"
        )
    output_md = pathlib.Path(args.output_markdown)
    output_md.write_text("\n".join(lines) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--guided-root", required=True)
    parser.add_argument("--suite", default="safelibero_spatial")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--baseline-run-name", default="pi05_plain_heldout20")
    parser.add_argument("--guided-run-name", default="pi05_time_value_guided_heldout20")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
