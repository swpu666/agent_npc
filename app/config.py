"""应用配置：全部来自环境变量或 .env，密钥不落代码。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = BASE_DIR / "app" / "data" / "knowledge" / "kb_v1.json"
KNOWLEDGE_META_FILE = BASE_DIR / "app" / "data" / "knowledge" / "kb_meta.json"
STATIC_DIR = BASE_DIR / "app" / "static"
EVAL_DIR = BASE_DIR / "eval"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 模型 ----------
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 1
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512

    # ---------- 上下文与检索 ----------
    history_max_turns: int = 6
    history_max_chars: int = 4000
    retrieval_top_k: int = 4
    retrieval_min_score: float = 3.0
    game_version: str = "S47"

    # ---------- 接口 ----------
    max_message_chars: int = 1000
    app_port: int = 8000

    # ---------- 使用记录（含 IP 与对话内容，详见 app/observability.py 的隐私说明）----------
    log_enabled: bool = True
    log_dir: str = "logs"
    # 单条消息/回答落盘的最大字符数，超出截断（避免日志无限膨胀）
    log_max_chars: int = 2000

    # ---------- 开关 ----------
    mock_llm: bool = False

    @property
    def has_api_key(self) -> bool:
        key = (self.llm_api_key or "").strip()
        return bool(key) and not key.startswith("sk-xxxx")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """测试用：清空配置缓存。"""
    get_settings.cache_clear()
