from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_CATEGORY_SCORE = 10
RUBRIC_VERSION = "jarvis.audit.v1"


@dataclass(frozen=True)
class AuditCategorySpec:
    category: str
    title: str
    required_path: str
    action: str
    evidence_globs: tuple[str, ...] = ()
    operational_globs: tuple[str, ...] = ()


CATEGORY_SPECS = (
    AuditCategorySpec(
        category="tool_coverage",
        title="Tool Coverage",
        required_path="agent_infra/core/tools.py",
        action="Add tool execution surface.",
        evidence_globs=("tests/*tools*.py",),
        operational_globs=("agent_infra/execution/*.py",),
    ),
    AuditCategorySpec(
        category="quality_gates",
        title="Quality Gates",
        required_path="agent_infra/harness/verification.py",
        action="Add unified verification loop.",
        evidence_globs=("tests/*verif*.py", "tests/test_*.py"),
        operational_globs=("agent_infra/cli.py",),
    ),
    AuditCategorySpec(
        category="memory_persistence",
        title="Memory Persistence",
        required_path="agent_infra/memory/episodic.py",
        action="Persist session events.",
        evidence_globs=("tests/*memory*.py",),
        operational_globs=("agent_infra/capture/*.py", "agent_infra/capture/adapters/*.py"),
    ),
    AuditCategorySpec(
        category="security_guardrails",
        title="Security Guardrails",
        required_path="agent_infra/patterns/guardrails.py",
        action="Add input/output guardrails.",
        evidence_globs=("tests/*guard*.py",),
        operational_globs=("agent_infra/patterns/*.py",),
    ),
    AuditCategorySpec(
        category="session_status",
        title="Session Status",
        required_path="agent_infra/harness/status.py",
        action="Add portable status payload.",
        evidence_globs=("tests/*status*.py",),
        operational_globs=("agent_infra/monitor/*.py", "agent_infra/cli.py"),
    ),
    AuditCategorySpec(
        category="continuous_learning",
        title="Continuous Learning",
        required_path="agent_infra/intelligence/learning.py",
        action="Add workflow distillation.",
        evidence_globs=("tests/*learning*.py",),
        operational_globs=("agent_infra/memory/procedural.py", "agent_infra/intelligence/*.py"),
    ),
    AuditCategorySpec(
        category="adapter_registry",
        title="Adapter Registry",
        required_path="agent_infra/harness/adapters.py",
        action="Add harness adapter registry.",
        evidence_globs=("tests/*adapter*.py",),
        operational_globs=("agent_infra/pi_bridge.py", "agent_infra/cli.py"),
    ),
    AuditCategorySpec(
        category="media_intake",
        title="Media Intake",
        required_path="agent_infra/capture/media.py",
        action="Add structured media intake.",
        evidence_globs=("tests/*media*.py",),
        operational_globs=("agent_infra/capture/adapters/files.py", "agent_infra/config/*.py"),
    ),
)


def run_harness_audit(root: Path) -> dict:
    categories = [_score_category(root, spec) for spec in CATEGORY_SPECS]
    overall = sum(item["score"] for item in categories)
    max_score = sum(item["max_score"] for item in categories)
    top_actions = [
        {"category": item["category"], "path": item["path"], "action": item["action"]}
        for item in sorted(categories, key=lambda category: category["score"])
        if item["score"] < item["max_score"]
    ][:3]
    return {
        "rubric_version": RUBRIC_VERSION,
        "overall_score": overall,
        "max_score": max_score,
        "categories": categories,
        "top_actions": top_actions,
    }


def _score_category(root: Path, spec: AuditCategorySpec) -> dict:
    required_path = root / spec.required_path
    evidence_hits = _match_any(root, spec.evidence_globs)
    operational_hits = _match_any(root, spec.operational_globs)

    dimensions = {
        "presence": 3 if required_path.exists() else 0,
        "depth": _score_depth(required_path),
        "evidence": min(3, len(evidence_hits)),
        "operational": min(2, len(operational_hits)),
    }
    score = sum(dimensions.values())
    evidence = [str(required_path)] if required_path.exists() else []
    evidence.extend(str(path) for path in evidence_hits[:3])
    evidence.extend(str(path) for path in operational_hits[:2])

    return {
        "category": spec.category,
        "title": spec.title,
        "path": str(required_path),
        "action": spec.action,
        "score": score,
        "max_score": MAX_CATEGORY_SCORE,
        "passed": score == MAX_CATEGORY_SCORE,
        "dimensions": dimensions,
        "evidence": evidence,
    }


def _score_depth(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 1
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 80:
        return 2
    if len(lines) >= 20:
        return 1
    return 0


def _match_any(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique
