from __future__ import annotations

import copy

import run_manifest


def _core() -> dict:
    source_hashes = {path: f"hash:{path}" for path in run_manifest.METHOD_IDENTITY_SOURCE_PATHS}
    source_hashes["openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py"] = "transport-v1"
    return {
        "schema_version": 1,
        "run_name": "test-run",
        "task_description": "test task",
        "safety_level": "I",
        "configuration": {"task_index": 0, "episode_index": [0], "safe_distance": 0.01},
        "residual_normalization": {"identifier": "test"},
        "source_sha256": source_hashes,
        "value_model": {"checkpoint": "critic"},
        "policy_checkpoint": {"checkpoint": "policy"},
    }


def test_method_identity_ignores_transport_client_changes() -> None:
    original = _core()
    transport_changed = copy.deepcopy(original)
    transport_changed["source_sha256"][
        "openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py"
    ] = "transport-v2"

    assert run_manifest._method_identity(original) == run_manifest._method_identity(transport_changed)


def test_method_identity_includes_action_expert_changes() -> None:
    original = _core()
    policy_changed = copy.deepcopy(original)
    policy_changed["source_sha256"]["openpi/src/openpi/policies/action_expert_qp.py"] = "changed"

    assert run_manifest._method_identity(original) != run_manifest._method_identity(policy_changed)
