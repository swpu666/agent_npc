"""对话接口与辅助接口。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.agent.llm import LLMError
from app.agent.pipeline import get_agent
from app.config import REPORTS_DIR, get_settings
from app.memory import aggregate_knowledge_gaps, get_memory_store
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
    # 说明：纯符号 / 纯数字（"。。。。"、"123456"）**不再返回 400**。
    # 它们不是"请求格式错误"，而是"我没听懂"——属于一次正常的对话轮次，
    # 由 Agent 走 `unclear_input` 模板给出拟人化回应（见 templates.unclear_input_answer）。
    # 400 只保留给真正无法处理的请求：空输入、超长输入。
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
        result = get_agent().answer(message, history, session_id=payload.session_id)
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
        memory=result.memory,
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


# --------------------------------------------------------------- 长期记忆
@router.get("/memory/{session_id}")
def get_memory(session_id: str) -> dict:
    """查看某个会话的画像：系统到底记住了什么，玩家自己也能看。"""
    settings = get_settings()
    store = get_memory_store(settings)
    profile = store.load(session_id)
    if profile is None:
        return {
            "enabled": settings.memory_enabled,
            "session_id": session_id,
            "turns": 0,
            "summary": ["这个会话还没有记录（至少问 3 次后才会用于调整回答）"],
            "profile": None,
        }
    return {
        "enabled": settings.memory_enabled,
        "session_id": session_id,
        "turns": profile.turns,
        "summary": profile.summary_lines(),
        "profile": profile.to_dict(),
        "note": "画像只统计「问了什么」，不推断「喜欢玩什么」；这些统计不会在其他会话之间共享。",
    }


@router.delete("/memory/{session_id}")
def clear_memory(session_id: str) -> dict:
    """清除该会话的记忆（隐私要求：记忆必须可删除）。"""
    ok = get_memory_store(get_settings()).clear(session_id)
    return {"cleared": ok, "session_id": session_id}


@router.get("/knowledge-gaps")
def knowledge_gaps(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """跨会话聚合"没答上来"的问题——用真实使用数据驱动知识库迭代。"""
    gaps = aggregate_knowledge_gaps(get_memory_store(get_settings()), limit=limit)
    return {
        "count": len(gaps),
        "gaps": gaps,
        "note": "来自各会话中「知识库未收录 / 覆盖不足」的问题，按出现次数排序；"
        "反复出现的就是知识库最该补的内容。",
    }


_REPORT_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _report_files(suffix: str = ".json") -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(
        (p for p in REPORTS_DIR.glob(f"*{suffix}") if _REPORT_NAME_RE.match(p.stem)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _safe_report_path(name: str, suffix: str) -> Path | None:
    """把报告名解析成路径，并确保它**没有跑出 reports 目录**。

    报告名来自 URL 参数，属于不可信输入：只允许 `[A-Za-z0-9_-]`，
    再校验解析后的真实路径仍在 REPORTS_DIR 之内（防目录穿越）。
    """
    stem = name[:-len(suffix)] if name.endswith(suffix) else name
    if not _REPORT_NAME_RE.match(stem):
        return None
    candidate = (REPORTS_DIR / f"{stem}{suffix}").resolve()
    try:
        candidate.relative_to(REPORTS_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


@router.get("/reports")
def list_reports(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """列出可查看的评测报告（最新的在前）。"""
    items = []
    for path in _report_files(".json")[:limit]:
        summary: dict = {}
        config: dict = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary") or {}
            config = payload.get("config") or {}
        except (OSError, json.JSONDecodeError):
            pass
        metrics = summary.get("metrics") or {}
        items.append(
            {
                "name": path.stem,
                "tag": config.get("tag"),
                "run_at": config.get("run_at"),
                "model": config.get("model"),
                "mock": config.get("mock", False),
                "judge_enabled": config.get("judge_enabled", True),
                "cases": len(payload.get("records") or []) if summary else 0,
                "intent_accuracy": (metrics.get("意图准确率") or {}).get("value"),
                "answer_pass_rate": (metrics.get("回答通过率") or {}).get("value"),
                "has_html": path.with_suffix(".html").is_file(),
            }
        )
    return {"count": len(items), "items": items}


@router.get("/reports/{name}/html", response_class=HTMLResponse)
def report_html(name: str) -> HTMLResponse:
    """返回某次评测报告的 HTML（页面内直接查看，无需下载文件）。

    报告可能是旧版本生成的（没有 .html），此时**用当前代码即时渲染**——
    评估报告的结构会随版本演进，重渲染能保证旧数据也用最新的样式呈现。
    """
    json_path = _safe_report_path(name, ".json")
    if json_path is None:
        raise HTTPException(status_code=404, detail={"error": "report_not_found", "message": f"找不到报告：{name}"})
    html_path = json_path.with_suffix(".html")
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail={"error": "report_broken", "message": "报告文件无法解析"})
    from eval.report_html import render_report_html

    return HTMLResponse(content=render_report_html(payload, title=f"离线评测报告 · {json_path.stem}"))
