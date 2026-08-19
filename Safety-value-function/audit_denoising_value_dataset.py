"""Validate alignment, provenance, and label coverage of denoising-value data."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import cv2
import numpy as np


REQUIRED_KEYS = {
    "chunk_start_steps",
    "denoising_hidden_states",
    "denoising_noisy_actions",
    "denoising_times",
    "denoising_task_flows",
    "denoising_active_mask",
    "physical_action_chunks",
    "nominal_clearance",
    "nominal_collision",
    "branch_chunk_id",
    "branch_time",
    "branch_direction_id",
    "branch_sign",
    "branch_noisy_actions",
    "branch_hidden_states",
    "branch_clearance",
    "branch_collision",
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root)
    return (
        f"{relative.parts[0]}/"
        f"{relative.parts[1].rsplit('_', 1)[-1]}/"
        f"{path.name.split('_', 1)[0]}"
    )


def _source_inventory(root: pathlib.Path) -> tuple[set[str], str, int, int]:
    groups = set()
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.npz"))
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root)
        groups.add(
            f"{relative.parts[-3]}/"
            f"{relative.parts[-2].rsplit('_', 1)[-1]}/"
            f"{path.name.split('_', 1)[0]}"
        )
        file_digest = _sha256(path)
        digest.update(str(relative).encode())
        digest.update(file_digest.encode())
        total_bytes += path.stat().st_size
    return groups, digest.hexdigest(), len(files), total_bytes


def audit(
    trace_root: pathlib.Path,
    source_root: pathlib.Path,
    *,
    allow_source_only_groups: bool = False,
) -> dict:
    files = sorted(trace_root.rglob("*_denoising_value.npz"))
    digest = hashlib.sha256()
    groups = set()
    counts = {
        "files": 0,
        "videos": 0,
        "chunks": 0,
        "trace_states": 0,
        "branches": 0,
        "pairs": 0,
    }
    clearance_values, branch_clearance_values = [], []
    collision_values, branch_collision_values = [], []
    errors = []
    for path in files:
        groups.add(_group(path, trace_root))
        file_digest = _sha256(path)
        digest.update(str(path.relative_to(trace_root)).encode())
        digest.update(file_digest.encode())
        try:
            video_path = path.with_suffix(".mp4")
            if not video_path.is_file():
                raise ValueError(f"missing companion video {video_path.name}")
            with np.load(path, allow_pickle=False) as archive:
                missing = REQUIRED_KEYS - set(archive.files)
                if missing:
                    raise ValueError(f"missing keys {sorted(missing)}")
                hidden = np.asarray(archive["denoising_hidden_states"])
                noisy = np.asarray(archive["denoising_noisy_actions"])
                times = np.asarray(archive["denoising_times"])
                flows = np.asarray(archive["denoising_task_flows"])
                active = np.asarray(archive["denoising_active_mask"])
                chunks = len(archive["chunk_start_steps"])
                expected_hidden = (chunks, 10, 10, 1024)
                expected_action = (chunks, 10, 10, 32)
                if hidden.shape != expected_hidden:
                    raise ValueError(f"hidden shape {hidden.shape} != {expected_hidden}")
                if noisy.shape != expected_action or flows.shape != expected_action:
                    raise ValueError(
                        f"noisy/flow shape {noisy.shape}/{flows.shape} != {expected_action}"
                    )
                if times.shape != (chunks, 10) or active.shape != (chunks, 10):
                    raise ValueError(f"time/active shape {times.shape}/{active.shape}")
                numeric = (hidden, noisy, times, flows, archive["nominal_clearance"], archive["branch_hidden_states"], archive["branch_noisy_actions"], archive["branch_clearance"])
                if not all(np.isfinite(value).all() for value in numeric):
                    raise ValueError("non-finite values")
                branch_count = len(archive["branch_time"])
                if np.asarray(archive["branch_hidden_states"]).shape != (branch_count, 10, 1024):
                    raise ValueError("branch hidden alignment")
                if np.asarray(archive["branch_noisy_actions"]).shape != (branch_count, 10, 32):
                    raise ValueError("branch action alignment")
                keys = {}
                for index in range(branch_count):
                    key = (
                        int(archive["branch_chunk_id"][index]),
                        round(float(archive["branch_time"][index]), 5),
                        int(archive["branch_direction_id"][index]),
                    )
                    keys.setdefault(key, set()).add(int(archive["branch_sign"][index]))
                complete_pairs = sum(signs == {-1, 1} for signs in keys.values())
                if complete_pairs * 2 != branch_count:
                    raise ValueError("incomplete +/- perturbation pairs")
                capture = cv2.VideoCapture(str(video_path))
                try:
                    if not capture.isOpened():
                        raise ValueError("companion video is unreadable")
                    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise ValueError("companion video has no readable frame")
                finally:
                    capture.release()
                nominal_steps = len(archive["nominal_clearance"])
                if video_frames != nominal_steps:
                    raise ValueError(
                        f"video frame count {video_frames} != nominal steps {nominal_steps}"
                    )
                if (video_width, video_height) != (1024, 1024):
                    raise ValueError(
                        f"video resolution {video_width}x{video_height} != 1024x1024"
                    )
                if abs(video_fps - 30.0) > 1e-3:
                    raise ValueError(f"video fps {video_fps} != 30")
                counts["files"] += 1
                counts["videos"] += 1
                counts["chunks"] += chunks
                counts["trace_states"] += chunks * 10
                counts["branches"] += branch_count
                counts["pairs"] += complete_pairs
                clearance_values.append(np.asarray(archive["nominal_clearance"], dtype=np.float32))
                branch_clearance_values.append(np.asarray(archive["branch_clearance"], dtype=np.float32).reshape(-1))
                collision_values.append(np.asarray(archive["nominal_collision"], dtype=np.bool_))
                branch_collision_values.append(np.asarray(archive["branch_collision"], dtype=np.bool_))
                digest.update(str(video_path.relative_to(trace_root)).encode())
                digest.update(_sha256(video_path).encode())
        except Exception as error:
            errors.append(f"{path.relative_to(trace_root)}: {error}")

    source_groups, source_digest, source_files, source_bytes = _source_inventory(
        source_root
    )
    missing_groups = sorted(source_groups - groups)
    extra_groups = sorted(groups - source_groups)
    nominal_clearance = np.concatenate(clearance_values) if clearance_values else np.empty(0)
    branch_clearance = np.concatenate(branch_clearance_values) if branch_clearance_values else np.empty(0)
    nominal_collision = np.concatenate(collision_values) if collision_values else np.empty(0, dtype=bool)
    branch_collision = np.concatenate(branch_collision_values) if branch_collision_values else np.empty(0, dtype=bool)
    passed = (
        not errors
        and not extra_groups
        and (not missing_groups or allow_source_only_groups)
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "trace_root": str(trace_root.resolve()),
        "source_root": str(source_root.resolve()),
        "dataset_sha256": digest.hexdigest(),
        "source_dataset_sha256": source_digest,
        "source_files": source_files,
        "source_bytes": source_bytes,
        "source_only_groups_allowed": allow_source_only_groups,
        "counts": counts,
        "source_groups": len(source_groups),
        "collected_groups": len(groups),
        "missing_groups": missing_groups,
        "extra_groups": extra_groups,
        "errors": errors,
        "labels": {
            "nominal_clearance_min_m": float(nominal_clearance.min()) if len(nominal_clearance) else None,
            "nominal_clearance_mean_m": float(nominal_clearance.mean()) if len(nominal_clearance) else None,
            "nominal_collision_fraction": float(nominal_collision.mean()) if len(nominal_collision) else None,
            "branch_clearance_min_m": float(branch_clearance.min()) if len(branch_clearance) else None,
            "branch_clearance_mean_m": float(branch_clearance.mean()) if len(branch_clearance) else None,
            "branch_collision_fraction": float(branch_collision.mean()) if len(branch_collision) else None,
        },
    }
    (trace_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--allow-source-only-groups",
        action="store_true",
        help=(
            "Allow bootstrap rollouts without Phase-2 traces while still rejecting "
            "trace groups that have no bootstrap source."
        ),
    )
    args = parser.parse_args()
    report = audit(
        pathlib.Path(args.trace_root),
        pathlib.Path(args.source_root),
        allow_source_only_groups=args.allow_source_only_groups,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
