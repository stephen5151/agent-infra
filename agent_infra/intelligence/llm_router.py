"""
LLM 路由器 — 混合调用策略
============================
优先级（从便宜到贵）：

  1. Claude Code CLI (claude -p)  ← 你的 Claude Pro/Max 订阅，零额外成本
  2. Codex CLI (codex)            ← 你的 Codex Pro 订阅
  3. Ollama 本地                  ← 完全免费，隐私安全，但能力有限
  4. Claude API                   ← 付费 API，作为最终保障

路由决策树：
  隐私敏感内容                  → Ollama（不出网）
  简单任务 & Ollama 可用        → Ollama
  复杂任务 & claude CLI 可用    → Claude CLI（订阅）
  claude CLI 不可用 & codex 可用→ Codex CLI（订阅）
  以上都失败                    → Claude API（付费兜底）

零成本方案：claude CLI + Ollama 即可覆盖 90% 场景。
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from agent_infra.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── 简单任务判定 ──────────────────────────────────────────────────────────────
_SIMPLE_KEYWORDS = [
    "总结", "摘要", "summary", "翻译", "translate",
    "格式化", "format", "分类", "classify",
    "是否", "判断", "yes or no", "true or false",
]

def _is_simple_task(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _SIMPLE_KEYWORDS) and len(text) < 500


# ── 驱动优先级 ────────────────────────────────────────────────────────────────

class Driver:
    """LLM 驱动基类。"""
    name: str = "base"

    async def generate(self, human: str, system: str = "", **kwargs) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return False


class ClaudeCLIDriver(Driver):
    """
    Claude Code CLI 驱动。
    使用你的 Claude Pro/Max 订阅，无需 API Key。

    要求: `claude` 命令可用（已安装 Claude Code）
    支持 --model 切换不同版本
    """
    name = "claude_cli"

    def available(self) -> bool:
        return shutil.which("claude") is not None

    async def generate(
        self,
        human: str,
        system: str = "",
        model: str | None = None,
        timeout: int = 120,
        **kwargs,
    ) -> str:
        """
        调用 claude -p "<prompt>" 获取回复。
        system prompt 拼接在 human 消息之前。
        """
        full_prompt = human
        if system:
            full_prompt = f"[System]\n{system}\n\n[User]\n{human}"

        cmd = ["claude", "-p", full_prompt]
        if model:
            cmd += ["--model", model]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"claude CLI error (rc={proc.returncode}): {err[:200]}")

            response = stdout.decode("utf-8", errors="ignore").strip()
            logger.debug(f"claude CLI responded ({len(response)} chars)")
            return response
        except asyncio.TimeoutError:
            raise RuntimeError(f"claude CLI timeout after {timeout}s")


class CodexCLIDriver(Driver):
    """
    Codex CLI 驱动（OpenAI Codex Pro 订阅）。
    要求: `codex` 命令可用
    """
    name = "codex_cli"

    def available(self) -> bool:
        return shutil.which("codex") is not None

    async def generate(
        self,
        human: str,
        system: str = "",
        timeout: int = 120,
        **kwargs,
    ) -> str:
        full_prompt = human
        if system:
            full_prompt = f"{system}\n\n{human}"

        cmd = ["codex", full_prompt]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"codex CLI error: {err[:200]}")
            return stdout.decode("utf-8", errors="ignore").strip()
        except asyncio.TimeoutError:
            raise RuntimeError(f"codex CLI timeout after {timeout}s")


class OllamaDriver(Driver):
    """Ollama 本地驱动 — 免费、私密。"""
    name = "ollama"

    def __init__(self, url: str, model: str) -> None:
        self._url = url
        self._model = model

    def available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._url}/api/tags", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        human: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        import httpx
        model = model or self._model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": human})
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self._url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        logger.debug(f"Ollama [{model}] responded ({len(content)} chars)")
        return content


class ClaudeAPIDriver(Driver):
    """Claude API 驱动 — 付费兜底，最强推理能力。"""
    name = "claude_api"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def available(self) -> bool:
        return bool(self._api_key)

    async def generate(
        self,
        human: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        model = model or self._model
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": human}],
        }
        if system:
            create_kwargs["system"] = system
        response = await client.messages.create(**create_kwargs)
        content = response.content[0].text if response.content else ""
        logger.debug(f"Claude API [{model}] responded ({len(content)} chars)")
        return content


# ── 路由器主类 ────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    智能 LLM 路由，优先使用订阅资源（零成本），API 兜底。

    用法:
        router = LLMRouter()
        resp = await router.agenerate("解释一下 asyncio")

    零成本配置:
        - 安装 Claude Code: npm install -g @anthropic-ai/claude-code
        - 安装 Ollama: brew install ollama && ollama pull hermes3
        - 不需要任何 API Key！
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.cfg = cfg.llm
        self.privacy = cfg.privacy

        # 按优先级排列的驱动链
        self._subscription_drivers: list[Driver] = [
            ClaudeCLIDriver(),
            CodexCLIDriver(),
        ]
        self._local_driver = OllamaDriver(
            url=self.cfg.local_url,
            model=self.cfg.local_model,
        )
        self._api_driver = ClaudeAPIDriver(
            api_key=self.cfg.anthropic_api_key,
            model=self.cfg.primary_model,
        )

    # ── 主接口 ────────────────────────────────────────────────────────────────

    async def agenerate(
        self,
        human: str,
        system: str = "",
        simple: bool | None = None,
        force_local: bool = False,
        force_subscription: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        异步路由生成。优先级: 订阅CLI > 本地Ollama > Claude API。

        Args:
            human: 用户消息
            system: 系统提示
            simple: True=强制简单路径, False=强制复杂路径, None=自动
            force_local: 强制本地（隐私场景）
            force_subscription: 强制使用订阅CLI（跳过本地）
        """
        # 1. 隐私内容 → 只走本地
        if not force_subscription and (
            force_local or self.privacy.is_sensitive(human)
        ):
            return await self._call_with_fallback(
                self._local_driver, human, system, **kwargs
            )

        # 2. 简单任务 → 本地优先
        if not force_subscription:
            is_simple = simple if simple is not None else _is_simple_task(human)
            if is_simple and self.cfg.local_for_simple and self._local_driver.available():
                try:
                    return await self._local_driver.generate(human, system, **kwargs)
                except Exception as e:
                    logger.warning(f"Ollama failed for simple task: {e}")

        # 3. 复杂任务 → 订阅 CLI（优先，零成本）
        for driver in self._subscription_drivers:
            if driver.available():
                try:
                    return await driver.generate(human, system, **kwargs)
                except Exception as e:
                    logger.warning(f"{driver.name} failed: {e}, trying next...")

        # 4. 兜底 → Ollama（如果可用）
        if self._local_driver.available():
            try:
                return await self._local_driver.generate(human, system, **kwargs)
            except Exception as e:
                logger.warning(f"Ollama fallback failed: {e}")

        # 5. 最终兜底 → Claude API（付费）
        if self._api_driver.available():
            return await self._api_driver.generate(human, system, **kwargs)

        raise RuntimeError(
            "所有 LLM 驱动均不可用。\n"
            "请至少满足一项：\n"
            "  • 安装 Claude Code: npm install -g @anthropic-ai/claude-code\n"
            "  • 启动 Ollama: ollama serve\n"
            "  • 设置 ANTHROPIC_API_KEY 环境变量"
        )

    async def _call_with_fallback(
        self,
        primary: Driver,
        human: str,
        system: str,
        **kwargs,
    ) -> str:
        """调用主驱动，失败时尝试其他可用驱动。"""
        try:
            return await primary.generate(human, system, **kwargs)
        except Exception as e:
            logger.warning(f"{primary.name} failed: {e}")
            # 尝试其他驱动
            for driver in self._subscription_drivers + [self._api_driver]:
                if driver is not primary and driver.available():
                    try:
                        return await driver.generate(human, system, **kwargs)
                    except Exception:
                        continue
            raise RuntimeError("所有驱动均失败")

    def generate(self, human: str, system: str = "", **kwargs) -> str:
        """同步版本。"""
        return asyncio.run(self.agenerate(human=human, system=system, **kwargs))

    # ── 状态 ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        subscription_status = {}
        for d in self._subscription_drivers:
            subscription_status[d.name] = d.available()

        return {
            # 订阅资源（零成本）
            "claude_cli": ClaudeCLIDriver().available(),
            "codex_cli": CodexCLIDriver().available(),
            # 本地
            "ollama": self._local_driver.available(),
            "ollama_model": self.cfg.local_model,
            # API（付费）
            "claude_api": self._api_driver.available(),
            "claude_api_model": self.cfg.primary_model,
            # 路由策略
            "fallback_to_local": self.cfg.fallback_to_local,
            "local_for_simple": self.cfg.local_for_simple,
        }

    def cost_summary(self) -> str:
        """返回成本分析摘要。"""
        lines = ["[LLM 路由成本分析]"]
        claude_cli = ClaudeCLIDriver().available()
        codex_cli = CodexCLIDriver().available()
        ollama = self._local_driver.available()
        api = self._api_driver.available()

        if claude_cli:
            lines.append("  ✓ Claude Code CLI — 使用你的 Claude Pro/Max 订阅，零 API 成本")
        else:
            lines.append("  ✗ Claude Code CLI — 未安装 (npm install -g @anthropic-ai/claude-code)")

        if codex_cli:
            lines.append("  ✓ Codex CLI — 使用你的订阅，零 API 成本")
        else:
            lines.append("  ✗ Codex CLI — 未安装")

        if ollama:
            lines.append(f"  ✓ Ollama ({self.cfg.local_model}) — 本地运行，完全免费")
        else:
            lines.append("  ✗ Ollama — 未运行 (ollama serve)")

        if api:
            lines.append(f"  ✓ Claude API ({self.cfg.primary_model}) — 付费兜底")
        else:
            lines.append("  ✗ Claude API — 无 API Key（可选）")

        if not claude_cli and not codex_cli and not ollama and not api:
            lines.append("\n  ⚠ 警告：没有可用的 LLM！")
        elif claude_cli or codex_cli or ollama:
            lines.append("\n  → 当前方案：零 API 成本运行")

        return "\n".join(lines)
