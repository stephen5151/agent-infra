# Agent Infra — Jarvis

[English](README.md)

Agent Infra 是一套运行在本地的个人 AI Agent 基础设施。Jarvis 基于 LangGraph 和 Claude 构建，可以从你的工作环境中捕获上下文，维护长期记忆，并在合适的时候主动给出提醒和洞察。

## 架构

```text
agent_infra/
├── capture/          # 输入适配器：剪贴板、文件、Claude Code hooks、CLI
├── core/             # LLM 路由、状态管理、工具
├── memory/           # 情节记忆、语义记忆、程序记忆、工作记忆
├── intelligence/     # 上下文增强、日报、LLM 路由
├── patterns/         # Agent 模式：链式处理、RAG、多 Agent、HITL、规划等
├── execution/        # 沙箱、文件系统、代码工具
├── healing/          # 断路器、自我监控、自动重启
├── automation/       # 自维护任务
├── monitor/          # 仪表盘、任务注册
└── config/           # 设置、soul.toml（人格与路由配置）
```

## 核心能力

- **低成本 LLM 路由**：优先使用 Claude Code CLI（Pro/Max 订阅），再回退到本地 Ollama，最后才使用 Claude API。
- **多层记忆**：支持情节记忆（SQLite）、语义记忆（Obsidian vault + DuckDB RAG）、程序记忆和工作记忆。
- **被动捕获**：可以从剪贴板、文件变更、Claude Code hooks 和 CLI 历史中捕获上下文。
- **主动反哺**：每次捕获到事件后，根据重要性给出桌面通知和相关洞察。
- **自我修复**：包含断路器、健康检查和自动重启机制。
- **自我维护**：夜间自动做代码检查、自动修复 lint 问题，并把报告写入 Obsidian。
- **每日摘要**：每天早上 08:00 将摘要写入 Obsidian。

## 快速开始

```bash
# 安装
pip install -e ".[full]"

# 配置
cp .env.example .env
# 编辑 agent_infra/config/soul.toml，设置你的 Obsidian vault 路径

# 启动
jarvis start
```

## 可选的 Pi.dev CLI 输入

Jarvis 提供了一个适合 Pi.dev 调用的 CLI 入口。它不会改变已有的 `ask` 和 `chat` 命令，只是额外提供一个简单、可组合的输入方式。

```bash
# 直接传入问题
jarvis pi "帮我总结今天的记忆"

# 从其他 CLI 管道传入
echo "根据最近的笔记给我三个行动建议" | jarvis pi

# 输出 JSON，方便自动化工具读取
jarvis pi --json --thread pi "分析这段输入"

# 使用更轻量的问答路径，而不是完整 Agent 图
jarvis pi --mode ask "查一下我的知识库里有没有相关内容"
```

## 环境要求

- Python 3.11+
- Claude Code CLI：`npm install -g @anthropic-ai/claude-code`，用于通过订阅调用 Claude。
- Ollama（可选，用于本地模型回退）：`brew install ollama && ollama pull hermes3`
- Anthropic API key（可选，作为最后回退）

## 配置

人格、记忆、捕获和路由相关设置都在 [`agent_infra/config/soul.toml`](agent_infra/config/soul.toml) 中。
