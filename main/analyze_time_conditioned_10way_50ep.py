"""Compare guided 50-episode finalists with the existing plain pi0.5 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np


SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)
LEVELS = ("I", "II")
EPISODES = tuple(range(50))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_or_nan(archive, name: str) -> float:
    if name not in archive:
        return float("nan")
    values = np.asarray(archive[name], dtype=np.float64)
    return float(np.nanmean(values)) if values.size else float("nan")


def _read_rollout(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "success": bool(archive["success"]),
            "collision": bool(archive["collision"]),
            "safe_success": bool(archive["safe_success"]),
            "infer_ms": _mean_or_nan(archive, "chunk_infer_ms"),
            "score_change": (
                _mean_or_nan(archive, "time_conditioned_scores_after")
                - _mean_or_nan(archive, "time_conditioned_scores")
            ),
            "normalized_clearance_change": (
                _mean_or_nan(
                    archive, "time_conditioned_clearance_predictions_after"
                )
                - _mean_or_nan(archive, "time_conditioned_clearance_predictions")
            ),
            "correction_rms": _mean_or_nan(
                archive, "time_conditioned_perturbation_rms"
            ),
            "correction_to_step": _mean_or_nan(
                archive, "time_conditioned_correction_to_step"
            ),
        }


def _episode_from_path(path: pathlib.Path) -> int:
    try:
        return int(path.name.split("_", 1)[0])
    except ValueError as error:
        raise RuntimeError(f"Cannot parse episode from {path}") from error


def _load_plain(root: pathlib.Path):
    audit_path = root / "audit_summary.json"
    audit = json.loads(audit_path.read_text())
    task_suite = {item["task"]: item["suite"] for item in audit["strata"]}
    records = {}
    manifests = {}
    run_dirs = sorted((root / "rollouts").glob("*/pi05_original_50ep_I"))
    run_dirs += sorted((root / "rollouts").glob("*/pi05_original_50ep_II"))
    if len(run_dirs) != 32:
        raise RuntimeError(f"Expected 32 plain task/level directories, found {len(run_dirs)}")
    for directory in run_dirs:
        task = directory.parent.name
        suite = task_suite[task]
        level = directory.name.rsplit("_", 1)[-1]
        manifest = directory / "manifest.json"
        manifests[f"{suite}/{task}/{level}"] = _sha256(manifest)
        for path in sorted(directory.glob("*_last_layer_hidden_states.npz")):
            episode = _episode_from_path(path)
            key = (suite, task, level, episode)
            if key in records:
                raise RuntimeError(f"Duplicate plain key {key}")
            records[key] = _read_rollout(path)
    return records, manifests, _sha256(audit_path), audit


def _load_guided(root: pathlib.Path, run_name: str):
    records = {}
    manifests = {}
    for suite in SUITES:
        run_dirs = sorted((root / suite).glob(f"*/{run_name}_I"))
        run_dirs += sorted((root / suite).glob(f"*/{run_name}_II"))
        if len(run_dirs) != 8:
            raise RuntimeError(
                f"Expected 8 {suite}/{run_name} task/level directories, "
                f"found {len(run_dirs)}"
            )
        for directory in run_dirs:
            task = directory.parent.name
            level = directory.name.rsplit("_", 1)[-1]
            manifest = directory / "manifest.json"
            manifests[f"{suite}/{task}/{level}"] = _sha256(manifest)
            for episode in EPISODES:
                matches = list(
                    directory.glob(f"{episode}_*_last_layer_hidden_states.npz")
                )
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one guided rollout for "
                        f"{suite}/{task}/{level}/{episode}, found {len(matches)}"
                    )
                key = (suite, task, level, episode)
                records[key] = _read_rollout(matches[0])
    if len(records) != 1600:
        raise RuntimeError(f"Expected 1,600 guided rollouts, found {len(records)}")
    return records, manifests


def _finite_mean(records: dict, field: str):
    values = np.asarray([value[field] for value in records.values()], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def _summary(records: dict) -> dict:
    count = len(records)
    if not count:
        raise RuntimeError("Cannot summarize an empty rollout set")
    return {
        "rollouts": count,
        "success_count": int(sum(value["success"] for value in records.values())),
        "success_rate": float(np.mean([value["success"] for value in records.values()])),
        "collision_count": int(sum(value["collision"] for value in records.values())),
        "collision_rate": float(
            np.mean([value["collision"] for value in records.values()])
        ),
        "safe_success_count": int(
            sum(value["safe_success"] for value in records.values())
        ),
        "safe_success_rate": float(
            np.mean([value["safe_success"] for value in records.values()])
        ),
        "mean_chunk_infer_ms": _finite_mean(records, "infer_ms"),
        "mean_score_change": _finite_mean(records, "score_change"),
        "mean_normalized_clearance_change": _finite_mean(
            records, "normalized_clearance_change"
        ),
        "mean_correction_rms": _finite_mean(records, "correction_rms"),
        "mean_correction_to_nominal_step": _finite_mean(
            records, "correction_to_step"
        ),
    }


def _cluster_ci(differences, clusters, *, seed: int, replicates: int):
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    by_cluster = [differences[clusters == value] for value in unique]
    for index in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = np.mean(np.concatenate([by_cluster[i] for i in sampled]))
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _matched_effect(plain: dict, guided: dict, *, seed: int, replicates: int):
    keys = sorted(set(plain) & set(guided))
    clusters = np.asarray([key[-1] for key in keys])
    metrics = {}
    for metric in ("success", "collision", "safe_success"):
        delta = np.asarray(
            [float(guided[key][metric]) - float(plain[key][metric]) for key in keys]
        )
        beneficial = -delta if metric == "collision" else delta
        metrics[metric] = {
            "guided_minus_plain": float(np.mean(delta)),
            "episode_cluster_ci95": _cluster_ci(
                delta, clusters, seed=seed, replicates=replicates
            ),
            "guided_improved_keys": int(np.sum(beneficial > 0)),
            "guided_worsened_keys": int(np.sum(beneficial < 0)),
            "unchanged_keys": int(np.sum(beneficial == 0)),
        }
    return keys, metrics


def analyze(args) -> dict:
    baseline_root = pathlib.Path(args.baseline_root)
    guided_root = pathlib.Path(args.guided_root)
    training_root = pathlib.Path(args.training_root)
    variants = [value for value in args.variants.split(",") if value]
    plain, plain_manifests, audit_sha, audit = _load_plain(baseline_root)
    if len(plain) != 1594:
        raise RuntimeError(f"Expected the audited 1,594 plain rollouts, found {len(plain)}")

    methods = []
    all_records = {}
    for variant in variants:
        gate_path = training_root / variant / "gradient_gate_metrics.json"
        gate = json.loads(gate_path.read_text())
        if not gate["pass"]:
            raise RuntimeError(f"Finalist {variant} did not pass the offline gate")
        run_name = f"pi05_tc10_{variant}_confirm"
        records, manifests = _load_guided(guided_root, run_name)
        keys, effects = _matched_effect(
            plain,
            records,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        )
        matched_records = {key: records[key] for key in keys}
        methods.append(
            {
                "variant": variant,
                "run_name": run_name,
                "offline_gate": gate,
                "full_1600_summary": _summary(records),
                "matched_1594_summary": _summary(matched_records),
                "matched_vs_plain": effects,
                "manifest_sha256": manifests,
            }
        )
        all_records[variant] = records

    methods.sort(
        key=lambda item: (
            item["matched_1594_summary"]["safe_success_count"],
            item["matched_1594_summary"]["success_count"],
            -item["matched_1594_summary"]["collision_count"],
            -item["matched_1594_summary"]["mean_chunk_infer_ms"],
        ),
        reverse=True,
    )
    winner = methods[0]
    runner_up = methods[1]
    finalist_keys, finalist_effects = _matched_effect(
        all_records[runner_up["variant"]],
        all_records[winner["variant"]],
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    all_expected = {
        (item["suite"], item["task"], item["level"], episode)
        for item in audit["strata"]
        for episode in EPISODES
    }
    if len(all_expected) != 1600:
        raise RuntimeError(
            f"Expected 1,600 keys from the plain audit strata, found {len(all_expected)}"
        )
    missing_plain = sorted(all_expected - set(plain))
    report = {
        "schema_version": 1,
        "scope": "guided-only 50-episode four-suite comparison to existing plain pi0.5 corpus",
        "selection_rule": "matched safe successes, then successes, then fewer collisions, then lower latency",
        "comparison_design": {
            "fixed_flow_noise": False,
            "matched_environment_keys": len(plain),
            "guided_rollouts_per_method": 1600,
            "task_level_strata": 32,
            "episodes": list(EPISODES),
            "note": (
                "Task/level/episode keys are matched, but policy flow noise was not fixed; "
                "effects are matched-key rather than deterministic paired-trajectory estimates."
            ),
        },
        "plain": {
            "summary": _summary(plain),
            "missing_keys": [list(key) for key in missing_plain],
            "audit_complete": audit["complete"],
            "audit_sha256": audit_sha,
            "manifest_sha256": plain_manifests,
        },
        "methods": methods,
        "selected_variant": winner["variant"],
        "selected_vs_runner_up": {
            "runner_up": runner_up["variant"],
            "keys": len(finalist_keys),
            "effects": finalist_effects,
        },
    }
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    count = len(plain)
    plain_summary = report["plain"]["summary"]
    lines = [
        "# Time-conditioned safety-value 50-episode result",
        "",
        (
            "Four SafeLIBERO suites, two safety levels, four tasks per suite, "
            "and episodes 0-49. Guidance results contain 1,600 rollouts per "
            "method; comparison uses the 1,594 keys present in the supplied "
            "plain pi0.5 corpus."
        ),
        "",
        "| Method | Success | Collision | Safe success | Mean chunk inference |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Plain pi0.5 | {plain_summary['success_count']}/{count} "
            f"({plain_summary['success_rate']:.1%}) | "
            f"{plain_summary['collision_count']}/{count} "
            f"({plain_summary['collision_rate']:.1%}) | "
            f"{plain_summary['safe_success_count']}/{count} "
            f"({plain_summary['safe_success_rate']:.1%}) | "
            f"{plain_summary['mean_chunk_infer_ms']:.1f} ms |"
        ),
    ]
    for item in methods:
        values = item["matched_1594_summary"]
        lines.append(
            f"| {item['variant']} | {values['success_count']}/{count} "
            f"({values['success_rate']:.1%}) | "
            f"{values['collision_count']}/{count} ({values['collision_rate']:.1%}) | "
            f"{values['safe_success_count']}/{count} "
            f"({values['safe_success_rate']:.1%}) | "
            f"{values['mean_chunk_infer_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            f"Selected variant: **{winner['variant']}**.",
            "",
            (
                "The original corpus has six missing rollouts and did not use "
                "fixed flow noise. Confidence intervals below cluster by episode; "
                "they do not imply deterministic trajectory pairing."
            ),
            "",
        ]
    )
    for item in methods:
        lines.extend([f"## {item['variant']} versus plain pi0.5", ""])
        for metric, values in item["matched_vs_plain"].items():
            ci = values["episode_cluster_ci95"]
            lines.append(
                f"- {metric}: {values['guided_minus_plain']:+.3f} "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            )
        lines.append("")
    pathlib.Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--guided-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--variants", default="01_balanced,08_late_time")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
