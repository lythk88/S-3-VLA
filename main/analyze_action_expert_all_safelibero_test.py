import json

import pytest

import analyze_action_expert_all_safelibero as analyzer


def _summary(episode_index: int) -> dict:
    return {
        "task_suite": "safelibero_spatial",
        "task_index": 0,
        "episode_index": episode_index,
        "safety_level": "I",
        "success": True,
        "collision": False,
        "paper_protocol_collision": False,
        "episode_steps": 10,
    }


def test_invalid_initial_state_counts_as_processed_attempt(tmp_path, monkeypatch):
    run = tmp_path / "safelibero_spatial" / "task" / "run_I"
    episode = run / "0"
    episode.mkdir(parents=True)
    (episode / "action_expert_safety_summary.json").write_text(json.dumps(_summary(0)))
    (run / "1_skipped_no_active_obstacle.txt").write_text(
        "No active obstacle found in the workspace.\n"
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--root", str(tmp_path), "--output", str(output), "--expected-episodes", "2"],
    )

    analyzer.main()

    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert report["attempted_episodes"] == 2
    assert report["evaluable_episodes"] == 1
    assert report["excluded_invalid_initial_states"] == 1
    assert report["overall"]["episodes"] == 1


def test_summary_supersedes_stale_skip_marker(tmp_path, monkeypatch):
    run = tmp_path / "safelibero_spatial" / "task" / "run_I"
    episode = run / "0"
    episode.mkdir(parents=True)
    (episode / "action_expert_safety_summary.json").write_text(json.dumps(_summary(0)))
    (run / "0_skipped_no_active_obstacle.txt").write_text("stale marker\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--root", str(tmp_path), "--output", str(output), "--expected-episodes", "1"],
    )

    analyzer.main()

    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert report["attempted_episodes"] == 1
    assert report["excluded_invalid_initial_states"] == 0


def test_missing_attempt_still_fails_completeness_check(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--root", str(tmp_path), "--output", str(output), "--expected-episodes", "1"],
    )

    with pytest.raises(SystemExit, match="Incomplete benchmark"):
        analyzer.main()


def test_quarantined_artifacts_are_ignored(tmp_path, monkeypatch):
    run = tmp_path / "safelibero_spatial" / "task" / "run_I"
    episode = run / "0"
    episode.mkdir(parents=True)
    (episode / "action_expert_safety_summary.json").write_text(json.dumps(_summary(0)))
    quarantined = run / ".aborted_episode"
    quarantined.mkdir()
    (quarantined / "action_expert_safety_summary.json").write_text(json.dumps(_summary(1)))
    (quarantined / "2_skipped_no_active_obstacle.txt").write_text("quarantined\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--root", str(tmp_path), "--output", str(output), "--expected-episodes", "1"],
    )

    analyzer.main()

    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert report["attempted_episodes"] == 1
    assert report["by_level"]["I"]["episodes"] == 1
    assert report["failure_modes"]["safe_success"]["episodes"] == 1
