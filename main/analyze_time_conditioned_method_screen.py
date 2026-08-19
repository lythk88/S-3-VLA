"""Compare exact paired diagnostic guidance methods against plain pi0.5."""

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


def _mean_or_nan(archive, name: str) -> float:
    if name not in archive:
        return float("nan")
    values = np.asarray(archive[name], dtype=np.float64)
    return float(np.nanmean(values)) if values.size else float("nan")


def _load(root: pathlib.Path, run_name: str, episodes: list[int]):
    run_dirs = sorted(root.glob(f"*/{run_name}_I")) + sorted(
        root.glob(f"*/{run_name}_II")
    )
    if len(run_dirs) != 8:
        raise RuntimeError(
            f"Expected 8 exact {run_name!r} task/level directories, found {len(run_dirs)}"
        )
    records = {}
    manifests = {}
    for directory in run_dirs:
        task = directory.parent.name
        level = directory.name.rsplit("_", 1)[-1]
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_name") != run_name:
            raise RuntimeError(
                f"Run-name mismatch in {manifest_path}: {manifest.get('run_name')!r}"
            )
        manifests[f"{task}/{level}"] = {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "identity_sha256": manifest.get("identity_sha256"),
        }
        for episode in episodes:
            matches = list(directory.glob(f"{episode}_*_last_layer_hidden_states.npz"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one artifact for {task}/{level}/{episode}, found {len(matches)}"
                )
            with np.load(matches[0], allow_pickle=False) as archive:
                records[(task, level, episode)] = {
                    "success": bool(archive["success"]),
                    "collision": bool(archive["collision"]),
                    "safe_success": bool(archive["safe_success"]),
                    "infer_ms": _mean_or_nan(archive, "chunk_infer_ms"),
                    "score_before": _mean_or_nan(archive, "time_conditioned_scores"),
                    "score_after": _mean_or_nan(
                        archive, "time_conditioned_scores_after"
                    ),
                    "clearance_before": _mean_or_nan(
                        archive, "time_conditioned_clearance_predictions"
                    ),
                    "clearance_after": _mean_or_nan(
                        archive, "time_conditioned_clearance_predictions_after"
                    ),
                    "correction_rms": _mean_or_nan(
                        archive, "time_conditioned_perturbation_rms"
                    ),
                    "correction_to_step": _mean_or_nan(
                        archive, "time_conditioned_correction_to_step"
                    ),
                    "accepted_factor": _mean_or_nan(
                        archive, "time_conditioned_accepted_factors"
                    ),
                }
    return records, manifests


def _nanmean(records, field: str) -> float | None:
    values = np.asarray([value[field] for value in records.values()], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def _summary(records):
    count = len(records)
    return {
        "rollouts": count,
        "success_count": int(sum(item["success"] for item in records.values())),
        "collision_count": int(sum(item["collision"] for item in records.values())),
        "safe_success_count": int(
            sum(item["safe_success"] for item in records.values())
        ),
        "success_rate": float(np.mean([item["success"] for item in records.values()])),
        "collision_rate": float(
            np.mean([item["collision"] for item in records.values()])
        ),
        "safe_success_rate": float(
            np.mean([item["safe_success"] for item in records.values()])
        ),
        "mean_chunk_infer_ms": _nanmean(records, "infer_ms"),
        "mean_score_change": (
            None
            if _nanmean(records, "score_after") is None
            else _nanmean(records, "score_after") - _nanmean(records, "score_before")
        ),
        "mean_normalized_clearance_change": (
            None
            if _nanmean(records, "clearance_after") is None
            else _nanmean(records, "clearance_after")
            - _nanmean(records, "clearance_before")
        ),
        "mean_correction_rms": _nanmean(records, "correction_rms"),
        "mean_correction_to_nominal_step": _nanmean(
            records, "correction_to_step"
        ),
        "mean_accepted_backtracking_factor": _nanmean(
            records, "accepted_factor"
        ),
    }


def analyze(args):
    root = pathlib.Path(args.results_root)
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    methods = [value for value in args.method_run_names.split(",") if value]
    baseline, baseline_manifests = _load(root, args.baseline_run_name, episodes)
    baseline_summary = _summary(baseline)
    reports = []
    for run_name in methods:
        guided, manifests = _load(root, run_name, episodes)
        if set(guided) != set(baseline):
            raise RuntimeError(f"Paired keys differ for {run_name}")
        summary = _summary(guided)
        deltas = {}
        for metric in ("success", "collision", "safe_success"):
            values = np.asarray(
                [float(guided[key][metric]) - float(baseline[key][metric]) for key in baseline]
            )
            deltas[metric] = {
                "guided_minus_plain": float(np.mean(values)),
                "improved_pairs": int(np.sum(values > 0)),
                "worsened_pairs": int(np.sum(values < 0)),
                "unchanged_pairs": int(np.sum(values == 0)),
            }
        reports.append(
            {
                "run_name": run_name,
                "summary": summary,
                "paired": deltas,
                "manifests": manifests,
            }
        )
    reports.sort(
        key=lambda item: (
            item["summary"]["safe_success_count"],
            item["summary"]["success_count"],
            -item["summary"]["collision_count"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "scope": "diagnostic development screen; not a safety-validated test result",
        "pairing": {
            "suite": "safelibero_spatial",
            "episodes": episodes,
            "rollouts_per_method": len(baseline),
            "fixed_flow_noise": True,
            "task_level_strata": 8,
        },
        "ranking_rule": "safe successes, then successes, then fewer collisions",
        "baseline": {
            "run_name": args.baseline_run_name,
            "summary": baseline_summary,
            "manifests": baseline_manifests,
        },
        "methods": reports,
    }
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Time-conditioned gradient integration diagnostic screen",
        "",
        "This is a fixed-noise development screen using an offline-gradient-gate-failed value model. It is not a safety-validated benchmark result.",
        "",
        f"Episodes: {episodes}; {len(baseline)} paired rollouts per method across 8 task/level strata.",
        "",
        "| Rank | Method | Success | Collision | Safe success | Score change | Correction / nominal step | Chunk inference |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_rows = [
        {
            "run_name": args.baseline_run_name,
            "summary": baseline_summary,
        }
    ] + reports
    for rank, item in enumerate(all_rows):
        values = item["summary"]
        score_change = values["mean_score_change"]
        ratio = values["mean_correction_to_nominal_step"]
        infer = values["mean_chunk_infer_ms"]
        lines.append(
            f"| {'plain' if rank == 0 else rank} | {item['run_name']} | "
            f"{values['success_count']}/{values['rollouts']} | "
            f"{values['collision_count']}/{values['rollouts']} | "
            f"{values['safe_success_count']}/{values['rollouts']} | "
            f"{'—' if score_change is None else f'{score_change:+.4f}'} | "
            f"{'—' if ratio is None else f'{ratio:.3f}'} | "
            f"{'—' if infer is None else f'{infer:.1f} ms'} |"
        )
    pathlib.Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--baseline-run-name", default="pi05_tcdiag_plain_dev2")
    parser.add_argument("--method-run-names", required=True)
    parser.add_argument("--episodes", default="0,1")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
