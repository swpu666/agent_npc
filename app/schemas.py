"""接口层的请求 / 响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]
IntentName = Literal["knowledge_qa", "situational_advice", "chitchat", "out_of_scope"]
RouteState = Literal[
    "need_more_info",
    "kb_miss",
    "kb_gap",
    "kb_general",
    "unclear_input",
    "conversation_recall",
]


class ChatMessage(BaseModel):
    # role 刻意声明为宽松的 str 而不是 Literal：前端历史是不可信输入，
    # 非法 role 应当被静默丢弃，而不是让整个请求以 422 失败。
    role: str = "user"
    content: str = Field(default="", max_length=4000)


class ChatRequest(BaseModel):
    message: str = ""
    history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = None
    # 调试开关：返回检索候选与得分明细（不影响回答逻辑）
    debug: bool = False


class Citation(BaseModel):
    id: str
    title: str
    url: str
    collected_at: str
    version: str
    score: float
    topic: str = ""
    # 来源强度：direct = 该页直接写明本条内容；topic = 覆盖该主题但非逐句对应
    support: str = "direct"
    support_note: str = ""


class Timings(BaseModel):
    """分环节耗时（毫秒）。

    本地环节（意图路由、知识检索）实测不足 1ms，若统一取整会全部显示为 0ms、
    无法体现"本地环节几乎不耗时间"这一事实，因此这两项保留两位小数；
    total_ms 与 llm_ms 量级在秒级，取整更易读。
    """

    total_ms: int
    intent_ms: float
    retrieval_ms: float
    llm_ms: int


class GuardInfo(BaseModel):
    injection_suspected: bool = False
    matched_patterns: list[str] = Field(default_factory=list)
    sanitized_urls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    request_id: str
    # 长期记忆状态摘要（提问次数、关注主题、本次是否注入了画像）。
    # 暴露出来是为了让"系统记住了什么"对玩家可见，而不是后台悄悄累积。
    memory: dict | None = None
    answer: str
    intent: IntentName
    route_state: RouteState | None = None
    citations: list[Citation] = Field(default_factory=list)
    timings: Timings
    model: str
    mock: bool = False
    retryable: bool = False
    guard: GuardInfo = Field(default_factory=GuardInfo)
    debug: dict | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    retryable: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str
    mock: bool
    has_api_key: bool
    game_version: str
