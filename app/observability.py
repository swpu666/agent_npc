"""使用记录（可观测性）：把每次问答的请求来源与对话内容落盘，并支持查看。

记录内容（按需求"记录详细的使用记录、IP、对话内容"）：
  时间 / request_id / 客户端 IP / X-Forwarded-For / User-Agent / session_id /
  玩家问题原文 / 携带的历史轮数 / 识别到的意图与路由状态 / 引用到的知识条目 /
  分环节耗时 / 回答原文 / 错误信息。

存储：`logs/requests-YYYY-MM-DD.jsonl`，一行一条，追加写入（本地文件，无数据库）。

隐私说明（重要）：
  记录中**包含客户端 IP 与完整对话内容**，属于个人信息。当前实现定位为本地调试与效果复盘，
  默认只写在本机、`logs/` 已被 .gitignore 忽略。若要用于线上环境，必须先做
  脱敏（IP 截断/哈希）、取得用户告知同意、并设置保留期与访问控制——
  详见 README「使用记录与隐私」。可以通过 `LOG_ENABLED=0` 完全关闭。
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from app.config import BASE_DIR, Settings, get_settings

_LOCK = threading.Lock()


@dataclass
class RequestRecord:
    request_id: str
    ip: str
    message: str
    answer: str = ""
    user_agent: str = ""
    forwarded_for: str = ""
    session_id: str = ""
    history_turns: int = 0
    intent: str = ""
    route_state: str | None = None
    intent_source: str = ""
    citations: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    error: str | None = None
    mock: bool = False
    ts: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.ts or datetime.now().astimezone().isoformat(timespec="seconds"),
            "request_id": self.request_id,
            "ip": self.ip,
            "forwarded_for": self.forwarded_for or None,
            "user_agent": self.user_agent,
            "session_id": self.session_id or None,
            "message": self.message,
            "message_chars": len(self.message),
            "history_turns": self.history_turns,
            "intent": self.intent,
            "route_state": self.route_state,
            "intent_source": self.intent_source,
            "citations": self.citations,
            "timings": self.timings,
            "answer": self.answer,
            "answer_chars": len(self.answer),
            "error": self.error,
            "mock": self.mock,
        }


class RequestLogger:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dir = Path(self.settings.log_dir)
        if not self.dir.is_absolute():
            self.dir = BASE_DIR / self.dir

    # ------------------------------------------------------------------ 写入
    def _file_for(self, day: str) -> Path:
        return self.dir / f"requests-{day}.jsonl"

    def _truncate(self, text: str) -> str:
        limit = self.settings.log_max_chars
        text = text or ""
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + f"…（已截断，原文 {len(text)} 字）"

    def log(self, record: RequestRecord) -> None:
        if not self.settings.log_enabled:
            return
        record.message = self._truncate(record.message)
        record.answer = self._truncate(record.answer)
        record.ts = record.ts or datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self._file_for(datetime.now().strftime("%Y-%m-%d"))
            line = json.dumps(record.to_dict(), ensure_ascii=False)
            with _LOCK:
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(line + "\n")
        except OSError:
            # 记录失败绝不能影响正常问答
            pass

    # ------------------------------------------------------------------ 读取
    def _iter_lines(self, days: int) -> Iterator[dict]:
        for offset in range(days):
            day = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
            path = self._file_for(day)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def recent(self, limit: int = 50, days: int = 7, session_id: str | None = None) -> list[dict]:
        """返回最近 limit 条记录（最新的在前）。

        `session_id` 传入时只返回该会话的记录——这是"玩家只看自己记录"的隔离依据。
        注意：这种按 `session_id` 过滤**不是**安全边界（前端可以伪造 session_id），
        它只是产品层面的"默认只看自己"。真正的安全隔离需要登录与鉴权，当前 Demo 没有账号体系。
        """
        records = list(self._iter_lines(days))
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        records.reverse()
        return records[: max(1, min(limit, 500))]

    def stats(self, days: int = 7, session_id: str | None = None) -> dict:
        records = list(self._iter_lines(days))
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        if not records:
            return {"total": 0, "days": days, "by_intent": {}, "by_day": {}, "unique_ips": 0}
        return {
            "total": len(records),
            "days": days,
            "by_intent": dict(Counter(r.get("intent", "?") for r in records)),
            "by_route_state": dict(Counter(str(r.get("route_state")) for r in records)),
            "by_day": dict(Counter(str(r.get("ts", ""))[:10] for r in records)),
            "unique_ips": len({r.get("ip") for r in records if r.get("ip")}),
            "errors": sum(1 for r in records if r.get("error")),
            "mock": sum(1 for r in records if r.get("mock")),
            "avg_total_ms": round(
                sum((r.get("timings") or {}).get("total_ms", 0) for r in records) / len(records), 1
            ),
        }


_logger: RequestLogger | None = None


def get_request_logger(settings: Settings | None = None) -> RequestLogger:
    global _logger
    if settings is not None:
        return RequestLogger(settings)
    if _logger is None:
        _logger = RequestLogger()
    return _logger
