"""Immutable provenance manifests for SafeLIBERO evaluation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALUE_ARTIFACTS = (
    "best_model.pt",
    "training_manifest.json",
    "gradient_gate_metrics.json",
    "live_gradient_gate_metrics.json",
    "normalizer.npz",
    "jax_guidance_model.npz",
)
SOURCE_PATHS = (
    "main/main_aegis.py",
    "main/run_manifest.py",
    "main/analyze_spatial_flow_guidance.py",
    "main/analyze_time_conditioned_heldout.py",
    "main/analyze_time_conditioned_method_screen.py",
    "main/select_spatial_heldout_episodes.py",
    "openpi/src/openpi/models/pi0.py",
    "openpi/src/openpi/models/safety_value.py",
    "openpi/src/openpi/policies/policy.py",
    "openpi/src/openpi/policies/time_conditioned_guidance_policy.py",
    "openpi/src/openpi/serving/websocket_policy_server.py",
    "openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    "scripts/run_spatial_flow_guidance_pilot.sh",
    "scripts/run_spatial_flow_guidance_v2_20ep.sh",
    "scripts/run_pi05_value_guided_no_shield_20ep.sh",
    "scripts/run_pi05_spatial_20ep.sh",
    "scripts/run_pi05_spatial_heldout20.sh",
    "scripts/serve_time_conditioned_guidance_policy.py",
    "scripts/run_time_conditioned_guided_spatial_heldout20.sh",
    "scripts/run_time_conditioned_method_screen.sh",
    "scripts/run_time_conditioned_trust_heldout20_diagnostic.sh",
    "scripts/run_time_conditioned_10way_screen_variant.sh",
    "scripts/run_time_conditioned_10way_screen_array.sh",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        # Preserve the leading XY-status space on the first porcelain line.
        # ``strip()`` removed that space and therefore the first character of
        # the first changed path when paths were later sliced at offset three.
        status = subprocess.check_output(
            (
                "git", "status", "--porcelain", "--untracked-files=all",
                "--", *SOURCE_PATHS,
            ),
            cwd=ROOT,
            text=True,
        ).rstrip()
        # Evaluation provenance needs the code in scope, not a multi-gigabyte
        # binary patch for unrelated deleted checkpoints elsewhere in the tree.
        diff = subprocess.check_output(
            ("git", "diff", "--no-ext-diff", "HEAD", "--", *SOURCE_PATHS),
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_scope": list(SOURCE_PATHS),
        "changed_paths": [line[3:] for line in status.splitlines()],
    }


def _source_hashes() -> dict[str, str | None]:
    return {
        relative: _sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        for relative in SOURCE_PATHS
    }


def _value_hashes(value_run_dir: pathlib.Path | None) -> dict[str, Any] | None:
    if value_run_dir is None:
        return None
    return {
        "directory": str(value_run_dir.resolve()),
        "artifacts": {
            name: {
                "bytes": (value_run_dir / name).stat().st_size,
                "sha256": _sha256(value_run_dir / name),
            }
            for name in VALUE_ARTIFACTS
            if (value_run_dir / name).is_file()
        },
    }


def _checkpoint_inventory(checkpoint_dir: pathlib.Path | None) -> dict[str, Any] | None:
    """Fingerprint the large pi0.5 checkpoint without rereading several GiB per run."""
    if checkpoint_dir is None:
        return None
    entries = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append(
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "directory": str(checkpoint_dir.resolve()),
        "inventory_sha256": _canonical_sha256(entries),
        "files": len(entries),
        "bytes": sum(item["bytes"] for item in entries),
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_run_manifest(
    out_dir: pathlib.Path,
    *,
    run_name: str,
    task_description: str,
    safety_level: str,
    configuration: dict[str, Any],
    value_run_dir: pathlib.Path | None,
    checkpoint_dir: pathlib.Path | None,
) -> pathlib.Path:
    translation_only = bool(configuration.get("flow_guidance_translation_only"))
    time_conditioned = bool(configuration.get("use_time_conditioned_guidance"))
    flow_guided = bool(configuration.get("use_flow_guidance"))
    if time_conditioned:
        configured_times = configuration.get("time_conditioned_guidance_times")
        if not configured_times:
            configured_times = str(configuration.get("time_conditioned_guidance_time"))
        residual_normalization = {
            "identifier": "time_conditioned_noisy_action_gradient_integration_v2",
            "gradient_convention": (
                "partial derivative of log(sigmoid(safety_logit)) + clearance_score_weight*"
                "normalized_clearance with respect to noisy_action_t; hidden_t stop-gradient"
            ),
            "gradient_dimensions": (
                "10x3 XYZ translation channels"
                if configuration.get("time_conditioned_guidance_translation_only")
                else "10x32 padded action channels"
            ),
            "normalization": configuration.get(
                "time_conditioned_guidance_normalization"
            ),
            "geometry": configuration.get("time_conditioned_guidance_geometry"),
            "integration": configuration.get(
                "time_conditioned_guidance_integration"
            ),
            "denoising_times": configured_times,
            "scale": configuration.get("time_conditioned_guidance_scale"),
            "fixed_context_value_backtracking": configuration.get(
                "time_conditioned_value_backtracking"
            ),
            "value_gradient_device": configuration.get(
                "time_conditioned_value_device"
            ),
            "determinism_convention": (
                "torch.manual_seed(0); torch.use_deterministic_algorithms(True); "
                "CPU value gradients for promoted held-out run"
                if configuration.get("time_conditioned_value_device") == "cpu"
                else "server setting recorded in server log; CUDA diagnostic screen"
            ),
        }
    elif flow_guided:
        residual_normalization = {
            "identifier": (
                "active_xyz_gradient_rms__global_task_rms__global_reported_ratio_v1"
                if translation_only
                else "global_gradient_rms__global_task_rms__global_reported_ratio_v1"
            ),
            "gradient_rms_dimensions": "XYZ translation only" if translation_only else "all 32 action dimensions",
            "task_flow_rms_dimensions": "all 32 action dimensions",
            "reported_ratio_dimensions": "all 32 action dimensions",
            "schedule": "quadratic (1-t)^2 when t <= start_time; zero otherwise",
        }
    else:
        residual_normalization = {
            "identifier": "none_plain_pi05",
            "description": "No value residual or noisy-action perturbation is applied.",
        }
    core = {
        "schema_version": 1,
        "run_name": run_name,
        "task_description": task_description,
        "safety_level": safety_level,
        "configuration": configuration,
        "residual_normalization": residual_normalization,
        "git": _git_metadata(),
        "source_sha256": _source_hashes(),
        "value_model": _value_hashes(value_run_dir),
        "policy_checkpoint": _checkpoint_inventory(checkpoint_dir),
    }
    identity_sha256 = _canonical_sha256(core)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(
                f"refusing to mix configurations in {out_dir}: manifest identity differs"
            )
        return manifest_path
    manifest = {
        **core,
        "identity_sha256": identity_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "host": platform.node(),
            "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {
                name: _version(name)
                for name in ("jax", "jaxlib", "numpy", "torch", "mujoco")
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    return manifest_path
