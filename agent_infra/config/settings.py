"""
配置系统 — pydantic-settings + TOML
=====================================
所有配置从 soul.toml 读取，类型安全，支持环境变量覆盖。

优先级: 环境变量 > soul.toml > 默认值
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib                  # Python 3.11+
except ImportError:
    import tomli as tomllib         # type: ignore

from pydantic import BaseModel, Field, field_validator


# ── TOML 数据模型 ─────────────────────────────────────────────────────────────

class IdentityPersona(BaseModel):
    style: str = "concise"
    tone: str = "direct"
    proactive: bool = True


class Identity(BaseModel):
    name: str = "Jarvis"
    owner: str = "User"
    language: str = "zh"
    timezone: str = "Asia/Shanghai"
    persona: IdentityPersona = Field(default_factory=IdentityPersona)


class LLMConfig(BaseModel):
    primary: str = "claude"
    primary_model: str = "claude-sonnet-4-6"
    local: str = "ollama"
    local_model: str = "hermes3"
    local_url: str = "http://localhost:11434"
    fallback_to_local: bool = True
    local_for_simple: bool = True

    @property
    def anthropic_api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def ollama_available(self) -> bool:
        """检查 Ollama 是否在运行。"""
        import httpx
        try:
            r = httpx.get(f"{self.local_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False


class MemoryConfig(BaseModel):
    episodic_db: str = "~/.jarvis/episodic.db"
    obsidian_vault: str = "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
    obsidian_inbox: str = "Jarvis Inbox"
    obsidian_index: str = "~/.jarvis/semantic.db"
    working_memory_ttl_hours: int = 24
    episodic_retention_days: int = 365

    @property
    def episodic_db_path(self) -> Path:
        return Path(self.episodic_db).expanduser()

    @property
    def vault_path(self) -> Path:
        return Path(self.obsidian_vault).expanduser()

    # aliases used by semantic.py
    @property
    def obsidian_vault_path(self) -> Path:
        return self.vault_path

    @property
    def obsidian_inbox_path(self) -> Path:
        return self.vault_path / self.obsidian_inbox

    @property
    def index_path(self) -> Path:
        return Path(self.obsidian_index).expanduser()


class CaptureConfig(BaseModel):
    clipboard_enabled: bool = True
    clipboard_min_length: int = 50
    file_watch_enabled: bool = True
    file_watch_paths: list[str] = ["~/Desktop", "~/Documents"]
    file_watch_extensions: list[str] = [".py", ".js", ".ts", ".md", ".txt"]
    claude_code_enabled: bool = True
    cli_enabled: bool = True
    cli_ignore_commands: list[str] = ["ls", "pwd", "cd", "clear", "history"]
    min_capture_interval_s: int = 2


class AugmentationConfig(BaseModel):
    mode: str = "mixed"
    instant_delay_s: int = 5
    notify_desktop: bool = True
    notify_voice: bool = False
    importance_threshold: float = 0.6
    max_insights_per_event: int = 3


class DigestConfig(BaseModel):
    enabled: bool = True
    schedule: str = "08:00"
    write_to_obsidian: bool = True
    obsidian_folder: str = "Jarvis Inbox/Daily Digest"
    notify_voice: bool = False


class HealingConfig(BaseModel):
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout_s: int = 60
    health_check_interval_s: int = 30
    auto_restart: bool = True


class MaintenanceConfig(BaseModel):
    enabled: bool = True
    schedule: str = "03:00"
    auto_fix: bool = True
    llm_review: bool = True
    write_to_obsidian: bool = True
    notify_if_errors: bool = True


class PrivacyConfig(BaseModel):
    local_only_patterns: list[str] = ["password", "密码", "token", "secret"]
    capture_blacklist_apps: list[str] = ["1Password", "Keychain"]

    def is_sensitive(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p.lower() in text_lower for p in self.local_only_patterns)


class KnowledgeConfig(BaseModel):
    enabled: bool = True
    fetch_schedule: str = "07:00"
    feed_size: int = 7
    exploration_ratio: float = 0.2
    hn_min_score: int = 100
    arxiv_categories: list[str] = ["cs.AI", "cs.LG", "cs.SE", "cs.CR"]
    write_to_obsidian: bool = True


class Settings(BaseModel):
    """完整配置，从 soul.toml 加载。"""
    identity: Identity = Field(default_factory=Identity)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    healing: HealingConfig = Field(default_factory=HealingConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)

    def ensure_dirs(self) -> None:
        """确保所有数据目录存在。"""
        self.memory.episodic_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory.index_path.parent.mkdir(parents=True, exist_ok=True)


# ── 加载函数 ──────────────────────────────────────────────────────────────────

_DEFAULT_SOUL = Path(__file__).parent / "soul.toml"


@lru_cache(maxsize=1)
def get_settings(soul_path: str | None = None) -> Settings:
    """
    加载配置（单例，缓存）。

    Args:
        soul_path: soul.toml 路径，None 时使用默认路径

    环境变量覆盖示例:
        ANTHROPIC_API_KEY=sk-ant-...  (始终从环境变量读取)
        JARVIS_LLM__PRIMARY=ollama    (覆盖 [llm] primary 字段)
    """
    path = Path(soul_path) if soul_path else _DEFAULT_SOUL
    if not path.exists():
        # 没有配置文件时用全部默认值
        settings = Settings()
    else:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        settings = Settings(**data)

    settings.ensure_dirs()
    return settings


def reload_settings(soul_path: str | None = None) -> Settings:
    """强制重新加载配置（修改 soul.toml 后调用）。"""
    get_settings.cache_clear()
    return get_settings(soul_path)
