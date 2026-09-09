"""Evaluate local Qwen3-VL obstacle naming against SafeLIBERO ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from main_aegis import LIBERO_DUMMY_ACTION, _obstacle_prompt_from_instance
from qwen3_vl_obstacle_selector import Qwen3VLObstacleSelector


DEFAULT_SUITES = (
    "safelibero_object",
    "safelibero_spatial",
    "safelibero_goal",
    "safelibero_long",
)
DEFAULT_LEVELS = ("I", "II")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/dev/shm/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--task-indices", nargs="+", type=int, default=list(range(4)))
    parser.add_argument("--episode-indices", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-all-images", action="store_true")
    return parser.parse_args()


def active_obstacles(env, obs) -> list[str]:
    names = [
        name.replace("_joint0", "")
        for name in env.sim.model.joint_names
        if "obstacle" in name
    ]
    return [
        name
        for name in names
        if (
            np.asarray(obs[f"{name}_pos"])[2] > -0.05
            and -0.5 < np.asarray(obs[f"{name}_pos"])[0] < 0.5
            and -0.5 < np.asarray(obs[f"{name}_pos"])[1] < 0.5
        )
    ]


def case_key(suite: str, level: str, task_index: int, episode_index: int) -> str:
    return f"{suite}/{level}/{task_index}/{episode_index}"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_summary(rows: list[dict], output_dir: Path) -> None:
    scored = [row for row in rows if row.get("status") == "scored"]
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_class: dict[str, list[dict]] = defaultdict(list)
    confusion = Counter()
    for row in scored:
        by_stratum[(row["suite"], row["level"])].append(row)
        by_class[row["ground_truth"]].append(row)
        confusion[(row["ground_truth"], row.get("prediction"))] += 1

    def metrics(items: list[dict]) -> dict:
        count = len(items)
        correct = sum(bool(row["correct"]) for row in items)
        strict = sum(bool(row["strict_text_match"]) for row in items)
        parsed = sum(row.get("prediction") is not None for row in items)
        return {
            "cases": count,
            "correct": correct,
            "semantic_accuracy": correct / count if count else None,
            "strict_text_matches": strict,
            "strict_text_accuracy": strict / count if count else None,
            "parsed": parsed,
            "parse_rate": parsed / count if count else None,
            "mean_latency_s": (
                sum(float(row["latency_s"]) for row in items) / count if count else None
            ),
        }

    summary = {
        "overall": metrics(scored),
        "by_suite_level": {
            f"{suite}/{level}": metrics(items)
            for (suite, level), items in sorted(by_stratum.items())
        },
        "by_ground_truth": {
            label: metrics(items) for label, items in sorted(by_class.items())
        },
        "confusion": [
            {"ground_truth": truth, "prediction": prediction, "count": count}
            for (truth, prediction), count in sorted(
                confusion.items(),
                key=lambda item: (item[0][0], str(item[0][1])),
            )
        ],
        "non_scored": [row for row in rows if row.get("status") != "scored"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    table_path = output_dir / "summary_by_suite_level.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "suite_level",
                "cases",
                "correct",
                "semantic_accuracy",
                "strict_text_matches",
                "strict_text_accuracy",
                "parsed",
                "parse_rate",
                "mean_latency_s",
            ),
        )
        writer.writeheader()
        for stratum, values in summary["by_suite_level"].items():
            writer.writerow({"suite_level": stratum, **values})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    rows = load_rows(predictions_path)
    completed = {row["key"] for row in rows}

    selector = Qwen3VLObstacleSelector(
        args.model,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    benchmark_dict = benchmark.get_benchmark_dict()
    completed_this_run = 0

    with predictions_path.open("a", buffering=1) as output:
        for suite_name in args.suites:
            for level in args.levels:
                suite = benchmark_dict[suite_name](safety_level=level)
                for task_index in args.task_indices:
                    task = suite.get_task(task_index)
                    initial_states = suite.get_task_init_states(task_index)
                    bddl_path = (
                        Path(get_libero_path("bddl_files"))
                        / task.problem_folder
                        / task.bddl_file
                    )
                    env = OffScreenRenderEnv(
                        bddl_file_name=bddl_path,
                        camera_heights=args.resolution,
                        camera_widths=args.resolution,
                        camera_depths=True,
                    )
                    env.seed(args.seed)
                    try:
                        for episode_index in args.episode_indices:
                            key = case_key(suite_name, level, task_index, episode_index)
                            if key in completed:
                                continue
                            if args.max_cases is not None and completed_this_run >= args.max_cases:
                                write_summary(rows, args.output_dir)
                                return
                            row = {
                                "key": key,
                                "suite": suite_name,
                                "level": level,
                                "task_index": task_index,
                                "task": task.language,
                                "episode_index": episode_index,
                                "model": args.model,
                                "device": args.device,
                            }
                            try:
                                env.reset()
                                obs = env.set_init_state(initial_states[episode_index])
                                for _ in range(args.settle_steps):
                                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                                active = active_obstacles(env, obs)
                                if len(active) != 1:
                                    row.update(
                                        status="invalid_ground_truth",
                                        active_obstacles=active,
                                    )
                                else:
                                    truth_instance = active[0]
                                    truth = _obstacle_prompt_from_instance(truth_instance)
                                    image_array = np.ascontiguousarray(
                                        obs["agentview_image"][::-1, ::-1]
                                    )
                                    image = Image.fromarray(image_array.astype(np.uint8))
                                    prediction = selector.predict(
                                        image,
                                        task.language,
                                        suite_name,
                                    )
                                    normalized_raw = " ".join(
                                        prediction.raw_answer.lower().split()
                                    ).strip(" .")
                                    strict_match = normalized_raw == truth
                                    correct = prediction.canonical_answer == truth
                                    image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
                                    row.update(
                                        status="scored",
                                        ground_truth_instance=truth_instance,
                                        ground_truth=truth,
                                        raw_answer=prediction.raw_answer,
                                        prediction=prediction.canonical_answer,
                                        strict_text_match=strict_match,
                                        correct=correct,
                                        latency_s=prediction.latency_s,
                                        prompt=prediction.prompt,
                                        image_sha256=image_hash,
                                    )
                                    if args.save_all_images or not correct:
                                        group = "images" if args.save_all_images else "errors"
                                        image_dir = args.output_dir / group / suite_name / level
                                        image_dir.mkdir(parents=True, exist_ok=True)
                                        image.save(
                                            image_dir
                                            / f"task{task_index}_episode{episode_index}_{truth_instance}.jpg",
                                            quality=95,
                                        )
                            except Exception as exc:
                                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                            output.write(json.dumps(row) + "\n")
                            rows.append(row)
                            completed.add(key)
                            completed_this_run += 1
                            write_summary(rows, args.output_dir)
                            print(
                                json.dumps(
                                    {
                                        "completed": len(rows),
                                        "key": key,
                                        "truth": row.get("ground_truth"),
                                        "answer": row.get("raw_answer"),
                                        "correct": row.get("correct"),
                                        "status": row["status"],
                                    }
                                ),
                                flush=True,
                            )
                    finally:
                        env.close()
    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
