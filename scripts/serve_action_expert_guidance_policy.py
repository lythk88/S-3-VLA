"""Serve pi0.5 with the draft success-and-safety action expert."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import pathlib
import socket

import torch
import tyro

from openpi.models import pi0_denoising_trace
from openpi.policies import action_expert_guidance_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class Args:
    port: int = 8008
    policy_config: str = "pi05_libero"
    checkpoint_dir: str = str(
        pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"
    )
    critic_run_dir: str = str(ROOT / "Safety-value-function/success_critic_v1")
    torch_device: str = "cpu"


def _load_critic(run_dir: pathlib.Path):
    manifest = json.loads((run_dir / "training_manifest.json").read_text())
    model_path = ROOT / "Safety-value-function/success_critic_model.py"
    spec = importlib.util.spec_from_file_location("success_critic_model", model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model = module.SuccessCritic(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info("Loaded success critic: validation=%s", manifest["validation"])
    return model


def main(args: Args) -> None:
    pi0_denoising_trace.install()
    base = policy_config.create_trained_policy(
        training_config.get_config(args.policy_config), args.checkpoint_dir
    )
    policy = action_expert_guidance_policy.ActionExpertGuidancePolicy(
        base,
        _load_critic(pathlib.Path(args.critic_run_dir)),
        torch_device=args.torch_device,
    )
    logging.info(
        "Creating action-expert server on %s:%d", socket.gethostname(), args.port
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
