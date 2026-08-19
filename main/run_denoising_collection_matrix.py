"""Recollect the exact rollout groups present in pi05_hidden_chunks."""

from __future__ import annotations

import argparse
import pathlib

from libero.libero import benchmark

from collect_denoising_value_data import Args
from collect_denoising_value_data import collect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    values = parser.parse_args()
    if values.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if not 0 <= values.worker_index < values.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    source_root = pathlib.Path(values.source_root)

    observed: dict[tuple[str, str], list[int]] = {}
    for path in source_root.rglob("*.npz"):
        relative = path.relative_to(source_root)
        if len(relative.parts) < 3:
            raise RuntimeError(f"Unexpected source rollout path: {relative}")
        task_segment = relative.parts[-3]
        level = relative.parts[-2].rsplit("_", 1)[-1]
        episode = int(path.name.split("_", 1)[0])
        observed.setdefault((task_segment, level), []).append(episode)

    mapping: dict[tuple[str, str], tuple[str, int]] = {}
    for suite_name in (
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    ):
        for level in ("I", "II"):
            suite = benchmark.get_benchmark_dict()[suite_name](safety_level=level)
            for task_id in range(suite.n_tasks):
                segment = suite.get_task(task_id).language.replace(" ", "_")
                mapping[(segment, level)] = (suite_name, task_id)

    missing = sorted(set(observed) - set(mapping))
    if missing:
        raise RuntimeError(f"Could not map source rollout strata: {missing}")
    selected = [
        item
        for index, item in enumerate(sorted(observed.items()))
        if index % values.num_workers == values.worker_index
    ]
    for (task_segment, level), episodes in selected:
        suite_name, task_id = mapping[(task_segment, level)]
        collect(
            Args(
                host=values.host,
                port=values.port,
                task_suite_name=suite_name,
                safety_level=level,
                task_index=[task_id],
                episode_index=sorted(set(episodes)),
                output_root=values.output_root,
                resume_existing=True,
            )
        )


if __name__ == "__main__":
    main()
