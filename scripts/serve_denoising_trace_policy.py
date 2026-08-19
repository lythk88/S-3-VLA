"""Serve a pi0.5 checkpoint with the isolated denoising-trace protocol."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.models import pi0_denoising_trace
from openpi.policies import denoising_trace_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    port: int = 8006
    policy_config: str = "pi05_libero"
    checkpoint_dir: str = "/home/lythk/.cache/openpi/openpi-assets/checkpoints/pi05_libero"


def main(args: Args) -> None:
    pi0_denoising_trace.install()
    base = policy_config.create_trained_policy(
        training_config.get_config(args.policy_config), args.checkpoint_dir
    )
    policy = denoising_trace_policy.DenoisingTracePolicy(base)
    hostname = socket.gethostname()
    logging.info("Creating denoising-trace server on %s:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

