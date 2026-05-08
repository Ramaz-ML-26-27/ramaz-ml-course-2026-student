"""Score calculator for HW03.

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
    # Unit — 40 pts
    ("TestSigmoid", "unit", 5, "sigmoid"),
    ("TestLinearInit", "unit", 3, "LinearRegression.__init__"),
    ("TestLinearForward", "unit", 4, "LinearRegression.forward"),
    ("TestLinearLoss", "unit", 4, "LinearRegression.loss"),
    ("TestLinearGradient", "unit", 6, "LinearRegression.gradient"),
    ("TestLogisticInit", "unit", 3, "LogisticRegression.__init__"),
    ("TestLogisticForward", "unit", 4, "LogisticRegression.forward"),
    ("TestLogisticLoss", "unit", 5, "LogisticRegression.loss"),
    ("TestLogisticGradient", "unit", 6, "LogisticRegression.gradient"),
    # Integration — 30 pts
    ("TestLinearFit", "integration", 10, "LinearRegression.fit"),
    ("TestLinearPredict", "integration", 3, "LinearRegression.predict"),
    ("TestLogisticFit", "integration", 10, "LogisticRegression.fit"),
    ("TestLogisticPredict", "integration", 7, "LogisticRegression.predict / predict_proba"),
    # Analysis — 20 pts
    ("TestLoadRegressionData", "analysis", 5, "load_regression_data"),
    ("TestTrainLinearModel", "analysis", 5, "train_linear_model"),
    ("TestLoadClassificationData", "analysis", 5, "load_classification_data"),
    ("TestTrainLogisticModel", "analysis", 5, "train_logistic_model"),
]

WRITEUP_POINTS = 10
TOTAL_POINTS = sum(pts for _, _, pts, _ in SECTIONS) + WRITEUP_POINTS

_SECTION_HEADERS: dict[str, str] = {
    "unit": "Unit tests (40 pts)",
    "integration": "Integration tests (30 pts)",
    "analysis": "Analysis tests (20 pts)",
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
    print(f"{'Test class':<40} {'Status':<14} {'Points':>7}")
    print("-" * 65)

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
        print(f"  {label:<38} {status:<14} {class_pts:>3}/{pts}")

    print()
    print("-" * 65)
    if marker_filter:
        section_pts = sum(pts for _, m, pts, _ in SECTIONS if m == marker_filter)
        print(f"{'Section score':<40} {earned:>3}/{section_pts}")
    else:
        autograded_total = sum(pts for _, _, pts, _ in SECTIONS)
        print(f"{'Autograded score':<40} {earned:>3}/{autograded_total}")
        print(f"{'Writeup (manual)':<40} {'?':>3}/{WRITEUP_POINTS}")
        print(f"{'Total (est.)':<40} {earned:>3}/{TOTAL_POINTS}")
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
                "No tests ran. Make sure the method is implemented "
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
        "output": (
            f"Autograded score: {total}/{autograded_total}  "
            f"(writeup graded separately, +{WRITEUP_POINTS} pts max)"
        ),
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
