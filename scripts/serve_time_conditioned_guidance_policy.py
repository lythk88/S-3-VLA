"""Serve pi0.5 with a gated time-conditioned noisy-action value guide."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import logging
import pathlib
import socket
import sys

import torch
import tyro

from openpi.models import pi0_denoising_trace
from openpi.policies import policy_config
from openpi.policies import time_conditioned_guidance_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class Args:
    port: int = 8007
    policy_config: str = "pi05_libero"
    checkpoint_dir: str = "/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
    value_run_dir: str = str(ROOT / "Safety-value-function/time_conditioned_clearance_v1")
    torch_device: str = "cpu"
    deterministic_value: bool = True
    allow_offline_only: bool = False
    allow_failed_offline_gate: bool = False


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_value_model(run_dir: pathlib.Path, args: Args):
    manifest = json.loads((run_dir / "training_manifest.json").read_text())
    permitted = {"live_gradient_gate_passed"}
    if args.allow_offline_only:
        permitted.add("offline_gradient_gate_passed_live_gate_pending")
    if args.allow_failed_offline_gate:
        permitted.add("offline_gradient_gate_failed")
    if manifest.get("status") not in permitted:
        raise RuntimeError(
            f"Value model status {manifest.get('status')!r} is not permitted; "
            f"expected one of {sorted(permitted)}"
        )
    if manifest.get("status") == "offline_gradient_gate_failed":
        gate_path = run_dir / "gradient_gate_metrics.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        logging.warning(
            "DIAGNOSTIC ONLY: loading an offline-gradient-gate-failed value model "
            "(direction_accuracy=%s, clearance_ci95=%s)",
            gate.get("gradient_direction_accuracy"),
            gate.get("selected_minus_rejected_clearance_cluster_ci95"),
        )
    checkpoint_path = run_dir / "best_model.pt"
    expected = manifest["artifacts"]["best_model.pt"]["sha256"]
    actual = _sha256(checkpoint_path)
    if actual != expected:
        raise RuntimeError(f"Value model checksum mismatch: {actual} != {expected}")
    model_path = ROOT / "Safety-value-function/time_conditioned_value_model.py"
    spec = importlib.util.spec_from_file_location(
        "time_conditioned_value_model", model_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["time_conditioned_value_model"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = module.TimeConditionedSafetyValue(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main(args: Args) -> None:
    imported_policy = pathlib.Path(policy_config.__file__).resolve()
    expected_source = (ROOT / "openpi/src").resolve()
    if expected_source not in imported_policy.parents:
        raise RuntimeError(
            f"Refusing non-repository OpenPI import {imported_policy}; "
            f"expected it below {expected_source}. Set PYTHONPATH explicitly."
        )
    logging.info("Using repository-local OpenPI from %s", imported_policy)
    if args.deterministic_value:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        logging.info(
            "Deterministic PyTorch value evaluation enabled on %s", args.torch_device
        )
    pi0_denoising_trace.install()
    base = policy_config.create_trained_policy(
        training_config.get_config(args.policy_config), args.checkpoint_dir
    )
    value_model = _load_value_model(pathlib.Path(args.value_run_dir), args)
    policy = time_conditioned_guidance_policy.TimeConditionedGuidancePolicy(
        base, value_model, torch_device=args.torch_device
    )
    logging.info(
        "Creating time-conditioned guidance server on %s:%d", socket.gethostname(), args.port
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
