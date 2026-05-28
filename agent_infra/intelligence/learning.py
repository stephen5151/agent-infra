from __future__ import annotations

from collections import Counter

from agent_infra.memory.procedural import Skill, get_procedural_memory
from agent_infra.memory.episodic import get_episodic_memory


def derive_skills_from_events(events: list[dict], min_repetitions: int = 2) -> list[Skill]:
    commands = []
    for event in events:
        if event.get("source") != "cli":
            continue
        content = str(event.get("content", "")).strip()
        if not content:
            continue
        commands.append((content.split()[0], event))

    counts = Counter(base for base, _ in commands)
    skills: list[Skill] = []
    for base, count in counts.items():
        if count < min_repetitions:
            continue
        source_events = [item["id"] for cmd, item in commands if cmd == base and item.get("id")]
        skills.append(
            Skill(
                name=f"cli_{base}_workflow",
                description=f"从近期事件中提炼出的 {base} CLI 工作流",
                trigger_patterns=[base],
                source_events=source_events,
                category="continuous-learning",
                steps=[
                    {"action": "inspect_context", "description": f"确认为什么要运行 {base}"},
                    {"action": "execute", "description": f"执行 {base} 并记录结果"},
                    {"action": "verify", "description": "用 verify 闭环确认结果有效"},
                ],
            )
        )
    return skills


def learn_recent_workflows(limit: int = 50, min_repetitions: int = 2) -> list[Skill]:
    episodic = get_episodic_memory()
    procedural = get_procedural_memory()
    recent = episodic.recent(limit=limit)
    skills = derive_skills_from_events(recent, min_repetitions=min_repetitions)
    for skill in skills:
        procedural.save_skill(skill)
    return skills
