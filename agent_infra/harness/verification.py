from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


Runner = Callable[[list[str], Path], tuple[bool, str]]


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    command: list[str]


@dataclass(frozen=True)
class VerificationPlan:
    project_type: str
    checks: list[VerificationCheck]


@dataclass(frozen=True)
class VerificationResult:
    name: str
    passed: bool
    command: str
    output: str


def detect_verification_plan(root: Path) -> VerificationPlan:
    if (root / "pyproject.toml").exists():
        checks = [
            VerificationCheck(name="build", command=[sys.executable, "-m", "py_compile", *python_sources(root)]),
            VerificationCheck(name="lint", command=detect_lint_command(root)),
        ]
        if (root / "tests").exists():
            checks.append(
                VerificationCheck(name="tests", command=[sys.executable, "-m", "unittest", "discover", "-s", "tests"])
            )
        return VerificationPlan(project_type="python", checks=checks)

    return VerificationPlan(
        project_type="generic",
        checks=[VerificationCheck(name="files", command=[sys.executable, "-c", "print('files-ok')"])],
    )


def run_verification(root: Path, runner: Runner | None = None) -> tuple[str, list[VerificationResult]]:
    plan = detect_verification_plan(root)
    execute = runner or _subprocess_runner
    results: list[VerificationResult] = []
    for check in plan.checks:
        passed, output = execute(check.command, root)
        results.append(
            VerificationResult(
                name=check.name,
                passed=passed,
                command=" ".join(check.command),
                output=output.strip(),
            )
        )
    return plan.project_type, results


def report_to_json(project_type: str, report: list[VerificationResult]) -> str:
    payload = {
        "project_type": project_type,
        "overall": "ready" if all(item.passed for item in report) else "not_ready",
        "checks": [asdict(item) for item in report],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def summarize_report(report: list[VerificationResult]) -> str:
    lines = ["VERIFICATION REPORT", "==================", ""]
    for item in report:
        label = item.name.capitalize()
        status = "PASS" if item.passed else "FAIL"
        lines.append(f"{label}: {status}")
        if item.output:
            lines.append(item.output)
        lines.append("")
    overall = "READY" if all(item.passed for item in report) else "NOT READY"
    lines.append(f"Overall: {overall}")
    return "\n".join(lines).strip()


def python_sources(root: Path) -> list[str]:
    sources = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        sources.append(str(rel))
    return sorted(sources) or ["."]


def detect_lint_command(root: Path) -> list[str]:
    if shutil.which("ruff"):
        return ["ruff", "check", "."]
    return [sys.executable, "-c", "print('ruff not installed; lint skipped')"]


def _subprocess_runner(command: list[str], cwd: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, proc.stdout
