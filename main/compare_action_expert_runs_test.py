import json

import compare_action_expert_runs as comparison


def _record(*, episode, success, collision, steps=10):
    return {
        "task_suite": "safelibero_goal",
        "safety_level": "I",
        "task_index": 0,
        "episode_index": episode,
        "success": success,
        "collision": collision,
        "paper_protocol_collision": collision,
        "episode_steps": steps,
    }


def test_paired_report_tracks_directional_changes():
    old = [
        _record(episode=0, success=False, collision=False),
        _record(episode=1, success=True, collision=True),
    ]
    new = [
        _record(episode=0, success=True, collision=False),
        _record(episode=1, success=True, collision=False),
    ]

    report = comparison._paired_report(old, new)

    assert report["task_success_gained"] == 1
    assert report["task_success_lost"] == 0
    assert report["paper_safety_gained"] == 1
    assert report["paper_safety_lost"] == 0
    assert report["delta_new_minus_old"]["SSR"] == 1.0


def test_load_ignores_quarantined_results(tmp_path):
    active = tmp_path / "run" / "0"
    active.mkdir(parents=True)
    (active / "action_expert_safety_summary.json").write_text(
        json.dumps(_record(episode=0, success=True, collision=False))
    )
    quarantined = tmp_path / "run" / ".aborted_episode"
    quarantined.mkdir()
    (quarantined / "action_expert_safety_summary.json").write_text(
        json.dumps(_record(episode=1, success=False, collision=True))
    )

    records = comparison._load(tmp_path)

    assert list(records) == [("safelibero_goal", "I", 0, 0)]
