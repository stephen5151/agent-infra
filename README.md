# Agent Infra — Jarvis

[中文版](README.zh.md)

Personal AI Agent infrastructure built on LangGraph + Claude. Jarvis runs locally, captures context from your environment, maintains memory, and proactively surfaces insights.

## Architecture

```
agent_infra/
├── capture/          # Input adapters (clipboard, files, Claude Code hooks, CLI)
├── core/             # LLM routing, state management, tools
├── memory/           # Episodic, semantic, procedural, working memory
├── intelligence/     # Context augmentation, daily digest, LLM router
├── patterns/         # Agent patterns (chaining, RAG, multi-agent, HITL, planning...)
├── execution/        # Sandbox, filesystem, code tools
├── healing/          # Circuit breaker, self-monitoring, auto-restart
├── automation/       # Self-maintenance routines
├── monitor/          # Dashboard, task registry
└── config/           # Settings, soul.toml (persona & routing config)
```

## Key Features

- **Zero-cost LLM routing** — Claude Code CLI (Pro/Max subscription) → Ollama (local) → Claude API fallback
- **Multi-layer memory** — episodic (SQLite), semantic (Obsidian vault + DuckDB RAG), procedural, working memory
- **Passive capture** — clipboard, file watcher, Claude Code hooks, CLI history
- **Proactive augmentation** — desktop notifications with ranked insights after each captured event
- **Self-healing** — circuit breaker, health checks, auto-restart
- **Self-maintenance** — nightly code review via LLM, auto-lint fix, report to Obsidian
- **Daily digest** — morning summary written to Obsidian at 08:00

## Quick Start

```bash
# Install
pip install -e ".[full]"

# Configure
cp .env.example .env
# Edit agent_infra/config/soul.toml to set your Obsidian vault path

# Run
jarvis start
```

## Optional Pi.dev CLI Input

Jarvis also exposes a Pi-friendly CLI bridge. It keeps the existing `ask` and
`chat` commands unchanged, while giving Pi.dev a simple command it can call.

```bash
# Pass a prompt as arguments
jarvis pi "帮我总结今天的记忆"

# Or pipe input from another CLI
echo "根据最近的笔记给我三个行动建议" | jarvis pi

# Machine-readable output for automation
jarvis pi --json --thread pi "分析这段输入"

# Use the lighter ask path instead of the full agent graph
jarvis pi --mode ask "查一下我的知识库里有没有相关内容"
```

## Requirements

- Python 3.11+
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`) — for zero-cost LLM calls
- Ollama (optional, for local model fallback): `brew install ollama && ollama pull hermes3`
- Anthropic API key (optional, ultimate fallback)

## Configuration

All persona, memory, capture, and routing settings live in [`agent_infra/config/soul.toml`](agent_infra/config/soul.toml).
