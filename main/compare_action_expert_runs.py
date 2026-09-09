"""Paired comparison of two Action Expert SafeLIBERO result trees."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

from analyze_action_expert_all_safelibero import _is_quarantined
from analyze_action_expert_all_safelibero import _metrics


REQUIRED = {
    "task_suite",
    "task_index",
    "episode_index",
    "safety_level",
    "success",
    "collision",
    "paper_protocol_collision",
    "episode_steps",
}


def _key(record: dict) -> tuple[str, str, int, int]:
    return (
        str(record["task_suite"]),
        str(record["safety_level"]),
        int(record["task_index"]),
        int(record["episode_index"]),
    )


def _load(root: pathlib.Path) -> dict[tuple[str, str, int, int], dict]:
    records = {}
    for path in sorted(root.rglob("action_expert_safety_summary.json")):
        if _is_quarantined(path):
            continue
        record = json.loads(path.read_text())
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        key = _key(record)
        if key in records:
            raise ValueError(f"duplicate episode key {key}: {path}")
        records[key] = record
    return records


def _outcome(record: dict) -> str:
    success = bool(record["success"])
    collision = bool(record["paper_protocol_collision"])
    if success and not collision:
        return "safe_success"
    if success:
        return "unsafe_success"
    if collision:
        return "collision_failure"
    return "safe_failure"


def _delta(old: dict, new: dict) -> dict:
    return {
        name: None if old[name] is None or new[name] is None else new[name] - old[name]
        for name in ("CAR", "strict_CAR", "TSR", "SSR", "strict_SSR", "ETS")
    }


def _paired_report(old_records: list[dict], new_records: list[dict]) -> dict:
    transitions = Counter(
        f"{_outcome(old)}->{_outcome(new)}"
        for old, new in zip(old_records, new_records, strict=True)
    )
    return {
        "episodes": len(old_records),
        "old": _metrics(old_records),
        "new": _metrics(new_records),
        "delta_new_minus_old": _delta(_metrics(old_records), _metrics(new_records)),
        "task_success_gained": sum(
            not bool(old["success"]) and bool(new["success"])
            for old, new in zip(old_records, new_records, strict=True)
        ),
        "task_success_lost": sum(
            bool(old["success"]) and not bool(new["success"])
            for old, new in zip(old_records, new_records, strict=True)
        ),
        "paper_safety_gained": sum(
            bool(old["paper_protocol_collision"]) and not bool(new["paper_protocol_collision"])
            for old, new in zip(old_records, new_records, strict=True)
        ),
        "paper_safety_lost": sum(
            not bool(old["paper_protocol_collision"]) and bool(new["paper_protocol_collision"])
            for old, new in zip(old_records, new_records, strict=True)
        ),
        "outcome_transitions": dict(sorted(transitions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=pathlib.Path, required=True)
    parser.add_argument("--new-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=319)
    args = parser.parse_args()

    old = _load(args.old_root)
    new = _load(args.new_root)
    common = sorted(old.keys() & new.keys())
    old_only = sorted(old.keys() - new.keys())
    new_only = sorted(new.keys() - old.keys())

    def select(keys):
        return [old[key] for key in keys], [new[key] for key in keys]

    old_common, new_common = select(common)
    report = {
        "complete": len(common) == args.expected_pairs and not old_only and not new_only,
        "expected_pairs": args.expected_pairs,
        "paired_episodes": len(common),
        "old_only": old_only,
        "new_only": new_only,
        "overall": _paired_report(old_common, new_common),
        "by_level": {},
        "by_suite": {},
    }
    for level in sorted({key[1] for key in common}):
        keys = [key for key in common if key[1] == level]
        old_group, new_group = select(keys)
        report["by_level"][level] = _paired_report(old_group, new_group)
    for suite in sorted({key[0] for key in common}):
        keys = [key for key in common if key[0] == suite]
        old_group, new_group = select(keys)
        report["by_suite"][suite] = _paired_report(old_group, new_group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(
            f"Incomplete paired comparison: paired={len(common)}, expected={args.expected_pairs}, "
            f"old_only={len(old_only)}, new_only={len(new_only)}"
        )


if __name__ == "__main__":
    main()
