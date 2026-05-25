"""Score calculator for HW04.

Run this script to see your current score before submitting.

Usage:
    uv run python score.py                        # full report
    uv run python score.py unit                   # unit tests only
    uv run python score.py integration            # integration tests only
    uv run python score.py analysis               # analysis tests only
    uv run python score.py --gradescope <path>    # write Gradescope results.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# (test_class_name, marker, points, label)
SECTIONS: list[tuple[str, str, int, str]] = [
    # Unit - 35 pts
    ("TestDropoutTrainingMode", "unit", 10, "Dropout (training mode)"),
    ("TestDropoutEvalMode", "unit", 5, "Dropout (eval mode)"),
    ("TestDropoutScaling", "unit", 5, "Dropout (inverted-dropout scaling)"),
    ("TestMLPArchitecture", "unit", 10, "MLP architecture (layer composition)"),
    ("TestMLPForwardShapes", "unit", 5, "MLP forward (output shapes)"),
    # Integration - 30 pts
    ("TestTrainOneEpoch", "integration", 10, "train_one_epoch"),
    ("TestValidate", "integration", 10, "validate"),
    ("TestTrainOrchestration", "integration", 10, "train (full orchestration + early stop)"),
    # Analysis - 35 pts
    ("TestLoadFashionMNIST", "analysis", 15, "load_fashion_mnist"),
    ("TestBuildModel", "analysis", 20, "build_model (all 4 configs)"),
]

WRITEUP_POINTS = 0  # writeup graded separately by the teacher
TOTAL_POINTS = sum(pts for _, _, pts, _ in SECTIONS) + WRITEUP_POINTS

_SECTION_HEADERS: dict[str, str] = {
    "unit": "Unit tests (35 pts)",
    "integration": "Integration tests (30 pts)",
    "analysis": "Analysis tests (35 pts)",
}


def _run_class(class_name: str, tb: str = "no") -> tuple[int, int, str]:
    """Run tests for one class. Return (passed, failed, captured_output)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"--tb={tb}",
        "-q",
        "--no-header",
        "-k",
        class_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            parts = line.split()
            p, f = 0, 0
            for i, part in enumerate(parts):
                if part in ("passed", "passed,"):
                    try:
                        p = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                if part in ("failed", "failed,", "error", "error,"):
                    try:
                        f += int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
            return p, f, output
    if "no tests ran" in result.stdout or "ERROR" in result.stdout:
        return 0, 1, output
    return 0, 0, output


def score_by_section(marker_filter: str | None = None) -> int:
    """Run all test classes and accumulate points (all-or-nothing per class)."""
    earned = 0
    current_section = ""

    print()
    print(f"{'Test class':<48} {'Status':<14} {'Points':>7}")
    print("-" * 73)

    for class_name, marker, pts, label in SECTIONS:
        if marker_filter and marker != marker_filter:
            continue

        section_label = _SECTION_HEADERS.get(marker, marker)
        if section_label != current_section:
            current_section = section_label
            print(f"\n{section_label}")

        passed, failed, _ = _run_class(class_name)
        if failed == 0 and passed > 0:
            status = "PASS"
            class_pts = pts
        else:
            status = (
                f"FAIL ({passed}/{passed + failed})" if passed + failed > 0 else "FAIL (not run)"
            )
            class_pts = 0

        earned += class_pts
        print(f"  {label:<46} {status:<14} {class_pts:>3}/{pts}")

    print()
    print("-" * 73)
    if marker_filter:
        section_pts = sum(pts for _, m, pts, _ in SECTIONS if m == marker_filter)
        print(f"{'Section score':<48} {earned:>3}/{section_pts}")
    else:
        autograded_total = sum(pts for _, _, pts, _ in SECTIONS)
        print(f"{'Autograded score':<48} {earned:>3}/{autograded_total}")
        print(f"{'Total':<48} {earned:>3}/{TOTAL_POINTS}")
    print()
    return earned


def score_gradescope(output_path: str) -> None:
    """Run all tests and write Gradescope results.json to output_path."""
    tests: list[dict] = []
    total = 0

    for class_name, _marker, max_pts, label in SECTIONS:
        passed, failed, captured = _run_class(class_name, tb="short")
        all_pass = failed == 0 and passed > 0
        pts = max_pts if all_pass else 0
        total += pts

        if all_pass:
            detail = f"All {passed} test(s) passed."
        elif passed + failed == 0:
            detail = (
                "No tests ran. Make sure the function is implemented "
                "and does not raise NotImplementedError."
            )
        else:
            detail = f"{passed}/{passed + failed} test(s) passed.\n\n" + captured[:1200]

        tests.append(
            {
                "score": pts,
                "max_score": max_pts,
                "name": label,
                "output": detail,
                "visibility": "visible",
            }
        )

    autograded_total = sum(pts for _, _, pts, _ in SECTIONS)
    results = {
        "score": total,
        "output": f"Autograded score: {total}/{autograded_total}",
        "tests": tests,
    }
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"Wrote {output_path}  ({total}/{autograded_total})")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--gradescope":
        score_gradescope(sys.argv[2])
    else:
        marker_filter = sys.argv[1] if len(sys.argv) > 1 else None
        score_by_section(marker_filter)
