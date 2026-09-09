"""Merge sharded SafeLIBERO obstacle-coverage audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_obstacle_coverage import _aggregate, _write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, default=320)
    args = parser.parse_args()

    rows_by_key = {}
    shard_paths = sorted(args.root.glob("shard_*/coverage_rows.jsonl"))
    if not shard_paths:
        parser.error(f"no shard rows found under {args.root}")
    for path in shard_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["suite"],
                row["level"],
                row["task_index"],
                row["episode_index"],
            )
            if key in rows_by_key:
                raise RuntimeError(f"duplicate case {key} in {path}")
            rows_by_key[key] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    (args.root / "coverage_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    _write_csv(args.root / "coverage_rows.csv", rows)
    summary = {
        "completed_cases": len(rows),
        "expected_cases": args.expected,
        "complete": len(rows) == args.expected,
        "shards": [str(path.parent) for path in shard_paths],
        "coverage_definition": {
            "axis": "estimated primitive AABB interval contains the complete MuJoCo collision-geom AABB interval",
            "full_geometry": "all vertices of all group-0 MuJoCo collision boxes lie inside the fitted convex primitive",
            "raw": "best of OBB/cylinder/capsule with 5 mm fit padding",
            "executed": "world-axis-aligned OBB with 10 mm side/bottom padding and 20 mm top padding",
        },
        "by_task_level": _aggregate(rows, ("suite", "level", "task_index", "task")),
        "by_suite_level": _aggregate(rows, ("suite", "level")),
        "by_level": _aggregate(rows, ("level",)),
        "overall": _aggregate(rows, tuple()),
    }
    (args.root / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.root / "coverage_by_task_level.csv", summary["by_task_level"])
    _write_csv(args.root / "coverage_by_suite_level.csv", summary["by_suite_level"])
    _write_csv(args.root / "coverage_by_level.csv", summary["by_level"])
    report = [
        "# SafeLIBERO obstacle geometry coverage (320 initial states)",
        "",
        "Coverage uses all cases as the denominator. A perception failure therefore counts as non-coverage.",
        "The executed geometry is the current world-axis-aligned OBB with 1 cm side/bottom padding and 2 cm top padding.",
        "",
        "## By suite and level",
        "",
        "| Suite | Level | Cases | Perception | X | Y | Z bottom | Z top | Z both | XYZ / full geometry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_suite_level"]:
        report.append(
            "| {suite} | {level} | {cases} | {perception:.1%} | {x:.1%} | {y:.1%} | {z_bottom:.1%} | {z_top:.1%} | {z:.1%} | {xyz:.1%} |".format(
                suite=item["suite"].removeprefix("safelibero_"),
                level=item["level"],
                cases=item["cases"],
                perception=item["perception_success_rate"],
                x=item["executed_x_cover_rate_all"],
                y=item["executed_y_cover_rate_all"],
                z_bottom=item["executed_z_lower_cover_rate_all"],
                z_top=item["executed_z_upper_cover_rate_all"],
                z=item["executed_z_cover_rate_all"],
                xyz=item["executed_xyz_cover_rate_all"],
            )
        )
    report.extend(
        [
            "",
            "## By task and level",
            "",
            "| Suite | Level | Task | Perception | Raw selected | X | Y | Z bottom | Z top | Z both | XYZ / full geometry |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["by_task_level"]:
        kinds = ", ".join(
            f"{kind}={count}" for kind, count in sorted(item["raw_kind_counts"].items())
        )
        report.append(
            "| {suite} | {level} | {task} | {perception:.0%} | {kinds} | {x:.0%} | {y:.0%} | {z_bottom:.0%} | {z_top:.0%} | {z:.0%} | {xyz:.0%} |".format(
                suite=item["suite"].removeprefix("safelibero_"),
                level=item["level"],
                task=item["task"].replace("|", "\\|"),
                perception=item["perception_success_rate"],
                kinds=kinds,
                x=item["executed_x_cover_rate_all"],
                y=item["executed_y_cover_rate_all"],
                z_bottom=item["executed_z_lower_cover_rate_all"],
                z_top=item["executed_z_upper_cover_rate_all"],
                z=item["executed_z_cover_rate_all"],
                xyz=item["executed_xyz_cover_rate_all"],
            )
        )
    report.extend(
        [
            "",
            "## Raw selector versus executed geometry",
            "",
            "| Geometry | X | Y | Z | XYZ AABB | Full collision geometry |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    overall = summary["overall"][0]
    report.append(
        "| Raw best primitive | {x:.1%} | {y:.1%} | {z:.1%} | {xyz:.1%} | {full:.1%} |".format(
            x=overall["raw_x_cover_rate_all"],
            y=overall["raw_y_cover_rate_all"],
            z=overall["raw_z_cover_rate_all"],
            xyz=overall["raw_xyz_cover_rate_all"],
            full=overall["raw_full_geometry_cover_rate_all"],
        )
    )
    report.append(
        "| Executed axis-aligned OBB | {x:.1%} | {y:.1%} | {z:.1%} | {xyz:.1%} | {full:.1%} |".format(
            x=overall["executed_x_cover_rate_all"],
            y=overall["executed_y_cover_rate_all"],
            z=overall["executed_z_cover_rate_all"],
            xyz=overall["executed_xyz_cover_rate_all"],
            full=overall["executed_full_geometry_cover_rate_all"],
        )
    )
    report.extend(
        [
            "",
            "Raw primitive counts among successful perception cases: "
            + ", ".join(
                f"{kind}={count}"
                for kind, count in sorted(overall["raw_kind_counts"].items())
            )
            + ".",
            "",
        ]
    )
    (args.root / "coverage_report.md").write_text("\n".join(report))
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "completed_cases": len(rows),
                "expected_cases": args.expected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
