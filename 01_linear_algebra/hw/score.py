"""Local scoring — run with: uv run python score.py [group]

Shows your current score grouped by section. A section earns its full
points only when ALL tests in that section pass.

Examples:
    uv run python score.py             # score everything
    uv run python score.py scratch     # only Part 1 (from scratch)
    uv run python score.py pytorch     # only Part 2 (PyTorch)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Points per test class. Every test in a class must pass to earn credit.
POINTS: dict[str, int] = {
    # Part 1: From Scratch (27 pts)
    "TestVectorAdd": 2,
    "TestScalarMultiply": 2,
    "TestDotProduct": 3,
    "TestVectorMagnitude": 2,
    "TestNormalizeVector": 3,
    "TestMatrixAdd": 3,
    "TestMatrixVectorMultiply": 4,
    "TestMatrixMultiply": 5,
    "TestMatrixTranspose": 3,
    # Part 2: PyTorch Tensors (27 pts)
    "TestRowNormalize": 4,
    "TestColumnMeans": 3,
    "TestTensorDotProduct": 2,
    "TestTensorMagnitude": 2,
    "TestTensorNormalize": 2,
    "TestTensorMatmul": 2,
    "TestTensorTranspose": 1,
    "TestCosineSimilarity": 5,
    "TestGramMatrix": 3,
    "TestSolveLinearSystem": 3,
}

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Part 1: From Scratch",
        [
            "TestVectorAdd",
            "TestScalarMultiply",
            "TestDotProduct",
            "TestVectorMagnitude",
            "TestNormalizeVector",
            "TestMatrixAdd",
            "TestMatrixVectorMultiply",
            "TestMatrixMultiply",
            "TestMatrixTranspose",
        ],
    ),
    (
        "Part 2: PyTorch Tensors",
        [
            "TestRowNormalize",
            "TestColumnMeans",
            "TestTensorDotProduct",
            "TestTensorMagnitude",
            "TestTensorNormalize",
            "TestTensorMatmul",
            "TestTensorTranspose",
            "TestCosineSimilarity",
            "TestGramMatrix",
            "TestSolveLinearSystem",
        ],
    ),
]

REPORT_FILE = Path(".report.json")


def run_pytest(marker_filter: str | None = None) -> None:
    cmd = [
        "uv",
        "run",
        "pytest",
        "--json-report",
        f"--json-report-file={REPORT_FILE}",
        "-q",
        "--tb=no",
    ]
    if marker_filter:
        cmd += ["-m", marker_filter]
    subprocess.run(cmd, capture_output=True)


def load_report() -> dict:
    return json.loads(REPORT_FILE.read_text())


def parse_class_name(nodeid: str) -> str | None:
    parts = nodeid.split("::")
    return parts[1] if len(parts) >= 2 else None


def collect_results(report: dict) -> dict[str, dict]:
    """Group test outcomes by class name."""
    by_class: dict[str, dict] = {}
    for test in report.get("tests", []):
        cls = parse_class_name(test["nodeid"])
        if cls is None or cls not in POINTS:
            continue
        if cls not in by_class:
            by_class[cls] = {"passed": 0, "total": 0, "failures": []}
        by_class[cls]["total"] += 1
        if test["outcome"] == "passed":
            by_class[cls]["passed"] += 1
        else:
            method = test["nodeid"].split("::")[-1]
            by_class[cls]["failures"].append(method)
    return by_class


def print_score(by_class: dict[str, dict]) -> None:
    total_earned = 0
    total_possible = 0

    print()
    for section_name, classes in SECTIONS:
        sec_earned = 0
        sec_possible = sum(POINTS[c] for c in classes if c in POINTS)
        lines: list[str] = []

        for cls in classes:
            pts = POINTS[cls]
            if cls not in by_class:
                lines.append(f"  [ -- ] {cls}: not run")
                continue

            info = by_class[cls]
            all_pass = info["passed"] == info["total"]
            earned = pts if all_pass else 0
            sec_earned += earned

            tag = "PASS" if all_pass else "FAIL"
            detail = (
                f"  [{tag}] {cls}: {earned}/{pts}"
                if all_pass
                else f"  [{tag}] {cls}: 0/{pts}  ({info['passed']}/{info['total']} tests passed)"
            )
            lines.append(detail)
            for f in info["failures"]:
                lines.append(f"         x {f}")

        total_earned += sec_earned
        total_possible += sec_possible

        print(f"{section_name}  [{sec_earned}/{sec_possible}]")
        for line in lines:
            print(line)
        print()

    print("=" * 50)
    print(f"  TOTAL SCORE (autograded):  {total_earned} / {total_possible}")
    print("  Note: Math exercises are graded separately.")
    print()


def main() -> None:
    marker = sys.argv[1] if len(sys.argv) > 1 else None
    run_pytest(marker)

    if not REPORT_FILE.exists():
        print("Error: could not generate test report.")
        print("Make sure pytest-json-report is installed: uv sync")
        sys.exit(1)

    report = load_report()
    by_class = collect_results(report)
    print_score(by_class)


if __name__ == "__main__":
    main()
