"""Audit the completed two-phase, ten-way time-conditioned guidance study."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


EXPECTED_VARIANTS = (
    "01_balanced",
    "02_safety_only",
    "03_clearance_only",
    "04_pair_rank_only",
    "05_pair_delta_only",
    "06_bilinear",
    "07_local_gradient",
    "08_late_time",
    "09_dense_time",
    "10_wide_deep",
)
FINALISTS = ("01_balanced", "08_late_time")
EXPECTED_BOOTSTRAP_SHA256 = (
    "fdff4830060c3471c500cff6e99f9b0476c14fed93484c130a0ee798aa16c314"
)
EXPECTED_TRACE_SHA256 = (
    "30e24e469a15b66c38f02220cc063636aea6d0568a59890cec638584f16d783d"
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _record(checks: list[dict], name: str, passed: bool, evidence) -> None:
    checks.append({"name": name, "pass": bool(passed), "evidence": evidence})


def _rollout_keys(root: pathlib.Path, variant: str, suffix: str) -> set[tuple]:
    keys = set()
    pattern = f"*/*/pi05_tc10_{variant}_confirm_*/*{suffix}"
    for path in root.glob(pattern):
        suite, task, run_name = path.parts[-4:-1]
        level = run_name.rsplit("_", 1)[-1]
        episode = int(path.name.split("_", 1)[0])
        key = (suite, task, level, episode)
        if key in keys:
            raise RuntimeError(f"Duplicate {suffix} rollout key {key}")
        keys.add(key)
    return keys


def audit(args) -> dict:
    bootstrap_root = pathlib.Path(args.bootstrap_root)
    trace_root = pathlib.Path(args.trace_root)
    training_root = pathlib.Path(args.training_root)
    guided_root = pathlib.Path(args.guided_root)
    analysis_path = pathlib.Path(args.analysis_json)
    checks: list[dict] = []

    bootstrap_npz = sorted(bootstrap_root.rglob("*.npz"))
    bootstrap_mp4 = sorted(bootstrap_root.rglob("*.mp4"))
    trace_npz = sorted(trace_root.rglob("*.npz"))
    trace_mp4 = sorted(trace_root.rglob("*.mp4"))
    trace_audit = _json(trace_root / "audit_report.json")
    _record(
        checks,
        "phase1_corpus_complete",
        len(bootstrap_npz) == 248 and len(bootstrap_mp4) == 248,
        {"npz": len(bootstrap_npz), "mp4": len(bootstrap_mp4)},
    )
    _record(
        checks,
        "phase2_corpus_complete",
        len(trace_npz) == 248
        and len(trace_mp4) == 248
        and trace_audit["status"] == "passed"
        and not trace_audit["errors"]
        and not trace_audit["missing_groups"]
        and not trace_audit["extra_groups"],
        {
            "npz": len(trace_npz),
            "mp4": len(trace_mp4),
            "audit_status": trace_audit["status"],
            "counts": trace_audit["counts"],
        },
    )
    _record(
        checks,
        "dataset_hashes_match",
        trace_audit["source_dataset_sha256"] == EXPECTED_BOOTSTRAP_SHA256
        and trace_audit["dataset_sha256"] == EXPECTED_TRACE_SHA256,
        {
            "phase1": trace_audit["source_dataset_sha256"],
            "phase2": trace_audit["dataset_sha256"],
        },
    )

    variant_dirs = tuple(path.name for path in sorted(training_root.glob("[0-9][0-9]_*")))
    _record(
        checks,
        "ten_expected_variants_present",
        variant_dirs == EXPECTED_VARIANTS,
        list(variant_dirs),
    )
    approach_identities = set()
    checkpoint_hashes = set()
    train_splits = []
    validation_splits = []
    offline_passes = []
    variant_evidence = {}
    artifact_hashes_valid = True
    two_phase_valid = True
    provenance_valid = True
    for variant in EXPECTED_VARIANTS:
        run_dir = training_root / variant
        approach = _json(run_dir / "approach.json")
        manifest = _json(run_dir / "training_manifest.json")
        gate = _json(run_dir / "gradient_gate_metrics.json")
        checkpoint = run_dir / "best_model.pt"
        checkpoint_sha = _sha256(checkpoint)
        gate_sha = _sha256(run_dir / "gradient_gate_metrics.json")
        artifact_hashes_valid &= (
            manifest["artifacts"]["best_model.pt"]["sha256"] == checkpoint_sha
            and manifest["artifacts"]["gradient_gate_metrics.json"]["sha256"]
            == gate_sha
        )
        checkpoint_hashes.add(checkpoint_sha)
        approach_identities.add(
            json.dumps(
                {
                    "training_args": approach["training_args"],
                    "guidance": approach["guidance"],
                },
                sort_keys=True,
            )
        )
        train = tuple(manifest["split"]["train_groups"])
        validation = tuple(manifest["split"]["validation_groups"])
        train_splits.append(train)
        validation_splits.append(validation)
        two_phase_valid &= (
            manifest["counts"]["bootstrap_train"] > 0
            and manifest["counts"]["trace_train"] > 0
            and "pretrain_validation" in manifest["metrics"]
            and "finetune_validation" in manifest["metrics"]
        )
        provenance = manifest["data_provenance"]
        provenance_valid &= (
            provenance["bootstrap_dataset_sha256"] == EXPECTED_BOOTSTRAP_SHA256
            and provenance["trace_dataset_sha256"] == EXPECTED_TRACE_SHA256
        )
        if gate["pass"]:
            offline_passes.append(variant)
        variant_evidence[variant] = {
            "description": approach["description"],
            "checkpoint_sha256": checkpoint_sha,
            "offline_gate_pass": gate["pass"],
            "offline_gradient_direction_accuracy": gate[
                "gradient_direction_accuracy"
            ],
            "status": manifest["status"],
        }
    _record(
        checks,
        "all_variants_trained_in_two_phases",
        two_phase_valid,
        {"variants": len(EXPECTED_VARIANTS)},
    )
    _record(
        checks,
        "all_training_artifact_hashes_valid",
        artifact_hashes_valid,
        {"unique_checkpoint_hashes": len(checkpoint_hashes)},
    )
    _record(
        checks,
        "all_training_provenance_matches_datasets",
        provenance_valid,
        {
            "phase1_sha256": EXPECTED_BOOTSTRAP_SHA256,
            "phase2_sha256": EXPECTED_TRACE_SHA256,
        },
    )
    _record(
        checks,
        "ten_materially_distinct_approaches",
        len(approach_identities) == 10 and len(checkpoint_hashes) == 10,
        {
            "unique_training_and_guidance_configs": len(approach_identities),
            "unique_checkpoints": len(checkpoint_hashes),
        },
    )
    common_train = set(train_splits[0])
    common_validation = set(validation_splits[0])
    split_valid = (
        all(split_value == train_splits[0] for split_value in train_splits)
        and all(split_value == validation_splits[0] for split_value in validation_splits)
        and len(common_train) == 186
        and len(common_validation) == 62
        and not (common_train & common_validation)
    )
    _record(
        checks,
        "group_split_is_shared_and_disjoint",
        split_valid,
        {
            "training_groups": len(common_train),
            "validation_groups": len(common_validation),
            "overlap": len(common_train & common_validation),
        },
    )
    _record(
        checks,
        "offline_gate_promoted_exactly_two_finalists",
        tuple(offline_passes) == FINALISTS,
        offline_passes,
    )

    guided_evidence = {}
    guided_valid = True
    for variant in FINALISTS:
        npz_keys = _rollout_keys(guided_root, variant, "_last_layer_hidden_states.npz")
        mp4_keys = _rollout_keys(guided_root, variant, ".mp4")
        groups = {(suite, task, level) for suite, task, level, _ in npz_keys}
        episodes_valid = all(
            {episode for s, t, l, episode in npz_keys if (s, t, l) == group}
            == set(range(50))
            for group in groups
        )
        valid = (
            len(npz_keys) == 1600
            and len(mp4_keys) == 1600
            and npz_keys == mp4_keys
            and len(groups) == 32
            and episodes_valid
        )
        guided_valid &= valid
        guided_evidence[variant] = {
            "npz": len(npz_keys),
            "mp4": len(mp4_keys),
            "task_level_groups": len(groups),
            "episodes_0_through_49": episodes_valid,
        }
    error_outputs = sorted(
        str(path.relative_to(guided_root))
        for path in guided_root.rglob("*")
        if path.is_file() and ("error" in path.name.lower() or "skip" in path.name.lower())
    )
    _record(
        checks,
        "guided_confirmation_complete",
        guided_valid and not error_outputs,
        {"finalists": guided_evidence, "error_or_skip_outputs": error_outputs},
    )

    analysis = _json(analysis_path)
    _record(
        checks,
        "plain_comparison_is_audited_and_matched",
        analysis["comparison_design"]["matched_environment_keys"] == 1594
        and len(analysis["plain"]["missing_keys"]) == 6
        and len(analysis["methods"]) == 2,
        {
            "matched_environment_keys": analysis["comparison_design"][
                "matched_environment_keys"
            ],
            "missing_plain_keys": analysis["plain"]["missing_keys"],
            "fixed_flow_noise": analysis["comparison_design"]["fixed_flow_noise"],
        },
    )
    winner = analysis["selected_variant"]
    live_path = training_root / winner / "live_gradient_gate_metrics.json"
    live = _json(live_path)
    winner_manifest = _json(training_root / winner / "training_manifest.json")
    live_artifact = winner_manifest["artifacts"][live_path.name]
    live_valid = (
        winner == "08_late_time"
        and live["pass"]
        and winner_manifest["status"] == "live_gradient_gate_passed"
        and live_artifact["sha256"] == _sha256(live_path)
        and live["rollout_clusters_reached"] == live["rollouts_requested"]
        and not live["missing_target_steps"]
    )
    _record(
        checks,
        "selected_winner_passed_live_gate",
        live_valid,
        {
            "winner": winner,
            "status": winner_manifest["status"],
            "rollout_clusters": live["rollout_clusters_reached"],
            "probe_contexts": live["probe_contexts"],
            "informative_probe_contexts": live["informative_probe_contexts"],
            "gradient_direction_accuracy": live["gradient_direction_accuracy"],
            "gradient_direction_accuracy_cluster_ci95": live[
                "gradient_direction_accuracy_cluster_ci95"
            ],
            "plus_minus_clearance_gain_m": live["plus_minus_clearance_gain_m"],
            "plus_minus_clearance_gain_cluster_ci95": live[
                "plus_minus_clearance_gain_cluster_ci95"
            ],
        },
    )

    report = {
        "schema_version": 1,
        "pass": all(check["pass"] for check in checks),
        "selected_variant": winner,
        "checks": checks,
        "variants": variant_evidence,
        "confirmation_analysis_path": str(analysis_path.resolve()),
        "live_gate_path": str(live_path.resolve()),
    }
    output_path = pathlib.Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-root", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--guided-root", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)
