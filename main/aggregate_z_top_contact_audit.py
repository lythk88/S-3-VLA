"""Aggregate diagnostic gripper contacts for the two zero-Z-top groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def _metrics(rows: list[dict]) -> dict:
    episodes = len(rows)

    def count(key: str) -> int:
        return sum(bool(row.get(key, False)) for row in rows)

    action_steps = sum(int(row.get("episode_steps", 0)) for row in rows)
    contact_steps = sum(int(row.get("gripper_obstacle_contact_steps", 0)) for row in rows)
    unmodeled_steps = sum(
        int(row.get("gripper_unmodeled_upper_contact_steps", 0)) for row in rows
    )
    top_band_steps = sum(
        int(row.get("gripper_actual_top_band_contact_steps", 0)) for row in rows
    )
    contact_displacement_steps = sum(
        int(row.get("gripper_contact_with_displacement_steps", 0)) for row in rows
    )
    displaced = [row for row in rows if row.get("obstacle_displaced", False)]
    return {
        "episodes": episodes,
        "successes": count("success"),
        "safe_successes": count("safe_success"),
        "displaced_episodes": count("obstacle_displaced"),
        "gripper_contact_episodes": count("gripper_obstacle_contact"),
        "unmodeled_upper_contact_episodes": count("gripper_unmodeled_upper_contact"),
        "actual_top_band_contact_episodes": count("gripper_actual_top_band_contact"),
        "contact_with_displacement_episodes": count("gripper_contact_with_displacement"),
        "displaced_with_unmodeled_upper_contact_episodes": sum(
            bool(row.get("gripper_unmodeled_upper_contact", False))
            for row in displaced
        ),
        "action_steps": action_steps,
        "gripper_contact_steps": contact_steps,
        "unmodeled_upper_contact_steps": unmodeled_steps,
        "actual_top_band_contact_steps": top_band_steps,
        "contact_with_displacement_steps": contact_displacement_steps,
        "gripper_contact_episode_rate": count("gripper_obstacle_contact") / max(episodes, 1),
        "unmodeled_upper_contact_episode_rate": count("gripper_unmodeled_upper_contact") / max(episodes, 1),
        "actual_top_band_contact_episode_rate": count("gripper_actual_top_band_contact") / max(episodes, 1),
        "displacement_episode_rate": count("obstacle_displaced") / max(episodes, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("action_expert_safety_summary.json")):
        row = json.loads(path.read_text())
        if not row.get("gripper_contact_diagnostic", {}).get("enabled", False):
            continue
        row["path"] = str(path)
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_suite"], row["safety_level"], int(row["task_index"]))].append(row)

    by_task = []
    for (suite, level, task_index), group in sorted(groups.items()):
        by_task.append(
            {
                "task_suite": suite,
                "safety_level": level,
                "task_index": task_index,
                **_metrics(group),
            }
        )
    summary = {"overall": _metrics(rows), "by_task": by_task, "rows": rows}
    (args.root / "z_top_contact_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if by_task:
        with (args.root / "z_top_contact_audit_by_task.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(by_task[0]))
            writer.writeheader()
            writer.writerows(by_task)
    print(json.dumps(summary["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
