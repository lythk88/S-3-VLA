#!/usr/bin/env python3
"""Flatten the pi0.5 / VLSA rollout videos into one folder and tally outcomes.

main_aegis.py writes to <videos>/<suite>/<task language>/<method>_<level>/
<episode>_<success|failure>_<safe|unsafe>.mp4. This copies every rollout into a
single directory named by suite, benchmark task index, method and outcome.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safelibero"))

SUITES = (
    "safelibero_spatial",
    "safelibero_goal",
    "safelibero_object",
    "safelibero_long",
)
METHODS = ("pi05", "vlsa", "guided_late_default", "guided_riskgate_sm")


def task_folder_names(suite: str, safety_level: str) -> dict[str, int]:
    """Map the underscored task language main_aegis uses to its task index."""
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite](safety_level=safety_level)
    return {
        task_suite.get_task(i).language.replace(" ", "_"): i
        for i in range(task_suite.n_tasks)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=pathlib.Path,
        default=ROOT / "results/chocolate_pudding_all_suites_eval",
    )
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--out-dir-name", default="all_videos")
    args = parser.parse_args()

    videos_root = args.eval_root / "videos"
    out_dir = args.eval_root / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []
    for suite in SUITES:
        suite_root = videos_root / suite
        if not suite_root.is_dir():
            missing.append(f"{suite}: no rollouts")
            continue
        index_of = task_folder_names(suite, args.safety_level)
        for task_folder, task_index in sorted(index_of.items(), key=lambda kv: kv[1]):
            for method in METHODS:
                method_dir = suite_root / task_folder / f"{method}_{args.safety_level}"
                clips = sorted(method_dir.glob("*.mp4")) if method_dir.is_dir() else []
                if not clips:
                    missing.append(f"{suite} task {task_index} {method}")
                    continue
                for clip in clips:
                    # "<episode>_<success|failure>_<safe|unsafe>"
                    episode, success, safety = clip.stem.split("_", 2)
                    name = (
                        f"{suite.replace('safelibero_', '')}_task{task_index}"
                        f"_{method}_{success}_{safety}.mp4"
                    )
                    shutil.copy2(clip, out_dir / name)
                    records.append(
                        {
                            "suite": suite,
                            "task_index": task_index,
                            "task": task_folder.replace("_", " "),
                            "method": method,
                            "episode": int(episode),
                            "success": success == "success",
                            "collision": safety != "safe",
                            "video": str(out_dir / name),
                            "source": str(clip),
                        }
                    )

    (args.eval_root / "video_index.json").write_text(
        json.dumps(
            {"safety_level": args.safety_level, "rollouts": records}, indent=2
        )
        + "\n"
    )

    tally = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, success, collision, safe-success
    for r in records:
        row = tally[(r["suite"], r["method"])]
        row[0] += 1
        row[1] += r["success"]
        row[2] += r["collision"]
        row[3] += r["success"] and not r["collision"]

    print(f"{'suite':10s} {'method':6s} {'n':>2s} {'success':>8s} {'collision':>10s} {'safe-success':>13s}")
    for suite in SUITES:
        for method in METHODS:
            if (suite, method) not in tally:
                continue
            n, ok, col, safe = tally[(suite, method)]
            print(
                f"{suite.replace('safelibero_',''):10s} {method:6s} {n:2d} "
                f"{ok:3d}/{n:<4d} {col:5d}/{n:<4d} {safe:6d}/{n:<6d}"
            )
    totals = collections.defaultdict(lambda: [0, 0, 0, 0])
    for (suite, method), row in tally.items():
        for i in range(4):
            totals[method][i] += row[i]
    print()
    for method in METHODS:
        n, ok, col, safe = totals[method]
        if n:
            print(
                f"TOTAL      {method:6s} {n:2d} {ok:3d}/{n:<4d} {col:5d}/{n:<4d} {safe:6d}/{n:<6d}"
            )
    print(f"\n{len(records)} videos -> {out_dir}")
    if missing:
        print("missing rollouts:")
        for item in missing:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
