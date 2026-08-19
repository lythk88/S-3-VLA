"""Aggregate completed per-suite time-conditioned SafeLIBERO reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction(summary: dict, field: str) -> str:
    count = int(summary[f"{field}_count"])
    total = int(summary["rollouts"])
    return f"{count}/{total} ({count / total:.1%})"


def aggregate(args) -> dict:
    paths = [pathlib.Path(value) for value in args.input_json]
    reports = [json.loads(path.read_text()) for path in paths]
    suites = {}
    for path, report in zip(paths, reports):
        suite = report["pairing"]["suite"]
        if suite in suites:
            raise RuntimeError(f"duplicate suite report: {suite}")
        if report["plain"]["rollouts"] != 160 or report["guided"]["rollouts"] != 160:
            raise RuntimeError(f"{suite} is not a complete 160-rollout paired report")
        suites[suite] = report

    expected = {"safelibero_object", "safelibero_goal", "safelibero_long"}
    if set(suites) != expected:
        raise RuntimeError(f"expected {sorted(expected)}, found {sorted(suites)}")

    combined = {}
    for method in ("plain", "guided"):
        total = sum(int(report[method]["rollouts"]) for report in reports)
        combined[method] = {
            "rollouts": total,
            "success_count": sum(int(report[method]["success_count"]) for report in reports),
            "collision_count": sum(int(report[method]["collision_count"]) for report in reports),
            "safe_success_count": sum(
                int(report[method]["safe_success_count"]) for report in reports
            ),
        }
        for field in ("success", "collision", "safe_success"):
            combined[method][f"{field}_rate"] = (
                combined[method][f"{field}_count"] / total
            )

    output = {
        "schema_version": 1,
        "scope": "diagnostic; value model failed the offline simulated-clearance gate",
        "suites": suites,
        "combined": combined,
        "source_sha256": {str(path.resolve()): _sha256(path) for path in paths},
    }
    output_json = pathlib.Path(args.output_json)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Time-conditioned guidance on remaining SafeLIBERO suites",
        "",
        "**Diagnostic only:** the referenced value model failed its offline simulated-clearance gate.",
        "",
        "Each suite contains 160 rollouts per method: four tasks, two safety levels, and 20 training-disjoint initial states.",
        "",
        "| Suite | Method | Success | Collision | Safe success |",
        "|---|---|---:|---:|---:|",
    ]
    for suite in sorted(suites):
        label = suite.replace("safelibero_", "", 1).title()
        report = suites[suite]
        for method, method_label in (("plain", "Plain pi0.5"), ("guided", "Trust-guided pi0.5")):
            summary = report[method]
            lines.append(
                f"| {label} | {method_label} | {_fraction(summary, 'success')} | "
                f"{_fraction(summary, 'collision')} | {_fraction(summary, 'safe_success')} |"
            )

    lines.extend(
        [
            "| **All remaining** | **Plain pi0.5** | "
            f"**{_fraction(combined['plain'], 'success')}** | "
            f"**{_fraction(combined['plain'], 'collision')}** | "
            f"**{_fraction(combined['plain'], 'safe_success')}** |",
            "| **All remaining** | **Trust-guided pi0.5** | "
            f"**{_fraction(combined['guided'], 'success')}** | "
            f"**{_fraction(combined['guided'], 'collision')}** | "
            f"**{_fraction(combined['guided'], 'safe_success')}** |",
            "",
            "Per-task and paired confidence-interval details are in `OBJECT_RESULTS.md`, `GOAL_RESULTS.md`, and `LONG_RESULTS.md` in this directory.",
        ]
    )
    pathlib.Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(aggregate(parse_args())["combined"], indent=2, sort_keys=True))
