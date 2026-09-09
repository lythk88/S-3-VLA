"""Aggregate all-suite Action Expert safety summaries against published SafeLIBERO results."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np


PUBLIC_RESULTS = {
    "VLSA/AEGIS": {"CAR": 0.7785, "TSR": 0.6813, "ETS": 262.30},
    "Constrained flow matching": {"CAR": 0.8281, "TSR": 0.8162, "ETS": 299.97},
}


def _metrics(records: list[dict]) -> dict:
    if not records:
        return {"episodes": 0, "CAR": None, "strict_CAR": None, "TSR": None, "SSR": None, "ETS": None}
    paper_safe = np.asarray([not record["paper_protocol_collision"] for record in records], dtype=np.bool_)
    strict_safe = np.asarray([not record["collision"] for record in records], dtype=np.bool_)
    success = np.asarray([record["success"] for record in records], dtype=np.bool_)
    steps = np.asarray([record["episode_steps"] for record in records], dtype=np.float64)
    return {
        "episodes": len(records),
        "CAR": float(np.mean(paper_safe)),
        "strict_CAR": float(np.mean(strict_safe)),
        "TSR": float(np.mean(success)),
        "SSR": float(np.mean(success & paper_safe)),
        "strict_SSR": float(np.mean(success & strict_safe)),
        "ETS": float(np.mean(steps)),
    }


def _is_quarantined(path: pathlib.Path) -> bool:
    return any(part.startswith(".aborted_") for part in path.parts)


def _failure_modes(records: list[dict]) -> dict:
    modes = {
        "safe_success": 0,
        "unsafe_success": 0,
        "safe_failure": 0,
        "collision_failure": 0,
    }
    for record in records:
        success = bool(record["success"])
        collision = bool(record["paper_protocol_collision"])
        if success and not collision:
            modes["safe_success"] += 1
        elif success and collision:
            modes["unsafe_success"] += 1
        elif not success and not collision:
            modes["safe_failure"] += 1
        else:
            modes["collision_failure"] += 1
    total = len(records)
    return {
        name: {"episodes": count, "rate": None if not total else count / total}
        for name, count in modes.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=1600)
    args = parser.parse_args()

    records = []
    malformed = []
    evaluated_episode_keys = set()
    for path in sorted(args.root.rglob("action_expert_safety_summary.json")):
        if _is_quarantined(path):
            continue
        try:
            record = json.loads(path.read_text())
            required = {
                "task_suite",
                "task_index",
                "episode_index",
                "safety_level",
                "success",
                "collision",
                "paper_protocol_collision",
                "episode_steps",
            }
            missing = required - record.keys()
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            record["path"] = str(path)
            records.append(record)
            evaluated_episode_keys.add((path.parent.parent.resolve(), int(record["episode_index"])))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            malformed.append({"path": str(path), "error": str(exc)})

    invalid_initial_states = []
    invalid_episode_keys = set()
    for path in sorted(args.root.rglob("*_skipped_no_active_obstacle.txt")):
        if _is_quarantined(path):
            continue
        try:
            episode_index = int(path.name.split("_", 1)[0])
            key = (path.parent.resolve(), episode_index)
            if key in evaluated_episode_keys:
                # A later successful rerun supersedes an old skip marker.
                continue
            invalid_episode_keys.add(key)
            invalid_initial_states.append(
                {
                    "path": str(path),
                    "episode_index": episode_index,
                    "reason": path.read_text().splitlines()[0],
                }
            )
        except (OSError, ValueError, IndexError) as exc:
            malformed.append({"path": str(path), "error": str(exc)})

    attempted_episodes = len(evaluated_episode_keys | invalid_episode_keys)

    groups = defaultdict(list)
    for record in records:
        groups[(record["task_suite"], record["safety_level"], int(record["task_index"]))].append(record)

    overall = _metrics(records)
    report = {
        "root": str(args.root.resolve()),
        "expected_episodes": args.expected_episodes,
        "complete": attempted_episodes == args.expected_episodes and not malformed,
        "attempted_episodes": attempted_episodes,
        "evaluable_episodes": len(records),
        "excluded_invalid_initial_states": len(invalid_initial_states),
        "invalid_initial_states": invalid_initial_states,
        "overall": overall,
        "by_suite": {
            suite: _metrics([record for record in records if record["task_suite"] == suite])
            for suite in sorted({record["task_suite"] for record in records})
        },
        "by_level": {
            level: _metrics([record for record in records if record["safety_level"] == level])
            for level in sorted({record["safety_level"] for record in records})
        },
        "failure_modes": _failure_modes(records),
        "by_suite_level_task": {
            f"{suite}/{level}/task{task}": _metrics(items)
            for (suite, level, task), items in sorted(groups.items())
        },
        "public_results": PUBLIC_RESULTS,
        "delta_vs_public": {
            name: {
                "CAR": None if overall["CAR"] is None else overall["CAR"] - values["CAR"],
                "TSR": None if overall["TSR"] is None else overall["TSR"] - values["TSR"],
                "ETS": None if overall["ETS"] is None else overall["ETS"] - values["ETS"],
            }
            for name, values in PUBLIC_RESULTS.items()
        },
        "malformed": malformed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(
            f"Incomplete benchmark: found {attempted_episodes} processed attempts "
            f"({len(records)} evaluable, {len(invalid_initial_states)} invalid initial states), "
            f"expected {args.expected_episodes}; malformed={len(malformed)}"
        )


if __name__ == "__main__":
    main()
