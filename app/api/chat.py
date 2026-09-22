"""对话接口与辅助接口。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.agent.llm import LLMError
from app.agent.normalize import has_meaningful_text
from app.agent.pipeline import get_agent
from app.config import get_settings
from app.observability import RequestRecord, get_request_logger
from app.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()

MAX_HISTORY_MESSAGES = 40


def _client_ip(request: Request | None) -> str:
    """取客户端 IP。

    优先使用 X-Forwarded-For 的第一段（部署在反向代理后时才有意义），
    否则用直连地址。**注意**：X-Forwarded-For 可被客户端伪造，仅用于本地调试与复盘，
    不作为任何安全判断依据。
    """
    if request is None:
        return ""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


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
def chat(payload: ChatRequest, request: Request):
    """玩家消息 → 意图 → 检索 → 回答。

    声明为同步函数，由 FastAPI 放入线程池执行，避免阻塞事件循环。
    """
    settings = get_settings()
    logger = get_request_logger(settings)
    message = (payload.message or "").strip()
    ip = _client_ip(request)
    user_agent = request.headers.get("user-agent", "")
    forwarded_for = request.headers.get("x-forwarded-for", "")

    def write_log(**kwargs) -> None:
        logger.log(
            RequestRecord(
                request_id=kwargs.pop("request_id", "-"),
                ip=ip,
                user_agent=user_agent,
                forwarded_for=forwarded_for,
                session_id=payload.session_id or "",
                message=message,
                **kwargs,
            )
        )

    if not message:
        write_log(intent="(invalid)", error="empty_input")
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_input", "message": "请输入你的问题后再发送。", "retryable": False},
        )
    if not has_meaningful_text(message):
        # 纯符号输入（"。。。。"）在归一化后是空串，规则层毫无信号，
        # 之前会被上下文继承硬猜成"上一轮的延续"去作答。这里明确拦下。
        write_log(intent="(invalid)", error="unrecognized_input")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unrecognized_input",
                "message": "没看懂这条消息，请用文字描述你的问题（例如「红BUFF有什么用」）。",
                "retryable": False,
            },
        )
    if len(message) > settings.max_message_chars:
        write_log(intent="(invalid)", error="too_long")
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
    history_turns = sum(1 for m in history if m["role"] == "user")

    try:
        result = get_agent().answer(message, history)
    except LLMError as exc:
        status = 504 if exc.kind == "timeout" else 502
        write_log(intent="(llm_error)", error=f"{exc.kind}: {exc}", history_turns=history_turns)
        return JSONResponse(
            status_code=status,
            content={
                "error": exc.kind,
                "message": str(exc),
                "retryable": exc.retryable,
            },
        )
    except Exception as exc:  # noqa: BLE001 —— 兜底，避免把堆栈暴露给前端
        write_log(intent="(internal_error)", error=f"{type(exc).__name__}: {exc}", history_turns=history_turns)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": f"服务内部错误：{type(exc).__name__}", "retryable": True},
        )

    write_log(
        request_id=result.request_id,
        history_turns=history_turns,
        intent=result.intent,
        route_state=result.route_state,
        intent_source=str(result.intent_detail.get("source", "")),
        citations=[c["id"] for c in result.citations],
        timings=result.timings,
        answer=result.answer,
        mock=result.mock,
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


@router.get("/logs")
def logs(limit: int = Query(default=50, ge=1, le=500), days: int = Query(default=7, ge=1, le=90)) -> dict:
    """查看最近的使用记录（最新的在前）。数据来自本地 jsonl 文件。"""
    settings = get_settings()
    logger = get_request_logger(settings)
    return {
        "enabled": settings.log_enabled,
        "log_dir": str(logger.dir),
        "privacy_note": "记录含客户端 IP 与完整对话内容，仅用于本地调试与效果复盘；上线前须脱敏、告知并设置保留期。",
        "records": logger.recent(limit=limit, days=days),
    }


@router.get("/logs/stats")
def logs_stats(days: int = Query(default=7, ge=1, le=90)) -> dict:
    return get_request_logger(get_settings()).stats(days=days)
