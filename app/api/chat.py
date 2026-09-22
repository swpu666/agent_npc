"""对话接口与辅助接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.agent.llm import LLMError
from app.agent.pipeline import get_agent
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()

MAX_HISTORY_MESSAGES = 40


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        mock=settings.mock_llm,
        has_api_key=settings.has_api_key,
        game_version=settings.game_version,
    )


@router.get("/kb/stats")
def kb_stats() -> dict:
    agent = get_agent()
    stats = agent.kb.stats()
    stats["mock"] = get_settings().mock_llm
    return stats


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """玩家消息 → 意图 → 检索 → 回答。

    声明为同步函数，由 FastAPI 放入线程池执行，避免阻塞事件循环。
    """
    settings = get_settings()
    message = (payload.message or "").strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_input", "message": "请输入你的问题后再发送。", "retryable": False},
        )
    if len(message) > settings.max_message_chars:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "too_long",
                "message": f"问题太长了，请控制在 {settings.max_message_chars} 字以内（当前 {len(message)} 字）。",
                "retryable": False,
            },
        )

    history = [
        {"role": m.role, "content": m.content.strip()}
        for m in payload.history[-MAX_HISTORY_MESSAGES:]
        if m.role in ("user", "assistant") and m.content and m.content.strip()
    ]

    try:
        result = get_agent().answer(message, history)
    except LLMError as exc:
        status = 504 if exc.kind == "timeout" else 502
        return JSONResponse(
            status_code=status,
            content={
                "error": exc.kind,
                "message": str(exc),
                "retryable": exc.retryable,
            },
        )
    except Exception as exc:  # noqa: BLE001 —— 兜底，避免把堆栈暴露给前端
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": f"服务内部错误：{type(exc).__name__}", "retryable": True},
        )

    return ChatResponse(
        request_id=result.request_id,
        answer=result.answer,
        intent=result.intent,
        route_state=result.route_state,
        citations=result.citations,
        timings=result.timings,
        model=result.model,
        mock=result.mock,
        retryable=result.retryable,
        guard=result.guard,
        debug=result.debug if payload.debug else None,
    )
