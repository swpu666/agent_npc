"""FastAPI 应用装配入口。

启动：
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if __package__ in (None, ""):  # 支持 python app/main.py 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import __version__
from app.api.chat import router as chat_router
from app.config import REPORTS_DIR, STATIC_DIR, get_settings

app = FastAPI(
    title="王者荣耀训练向导 Agent",
    description="游戏问答 Agent 最小可用 Demo：意图路由 + 本地知识检索 + 模型生成",
    version=__version__,
)

app.include_router(chat_router, prefix="/api")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/report", include_in_schema=False)
def report_page():
    """最新一次评测报告的 HTML 页面。

    没跑过评测时给出明确指引而不是 404——评审第一次进来很可能还没跑过。
    渲染逻辑与 `/api/reports/{name}/html` 共用，保证同一种表现。
    """
    from fastapi.responses import HTMLResponse

    from app.api.chat import _report_files, _safe_report_path

    files = _report_files(".json")
    if not files:
        return HTMLResponse(
            content=(
                "<meta charset='utf-8'>"
                "<div style='font-family:system-ui;padding:40px;line-height:1.9;max-width:640px'>"
                "<h2>还没有评测报告</h2>"
                "<p>先在项目根目录执行一次离线评测：</p>"
                "<pre style='background:#f4f4f6;padding:12px;border-radius:8px'>"
                "python -m eval.run_eval --tag demo</pre>"
                "<p>跑完会产出 .json / .md / .html 三份报告，刷新本页即可查看。</p>"
                "<p style='color:#888'>报告目录：" + str(REPORTS_DIR) + "</p>"
                "</div>"
            )
        )
    latest = files[0].stem
    json_path = _safe_report_path(latest, ".json")
    html_path = json_path.with_suffix(".html") if json_path else None
    if html_path and html_path.is_file():
        return FileResponse(html_path)
    from eval.report_html import render_report_html

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return HTMLResponse(content=render_report_html(payload, title=f"离线评测报告 · {latest}"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn

    settings = get_settings()
    if settings.mock_llm:
        print("[MOCK] MOCK_LLM=1：当前为 Mock 模式，不会调用真实模型，结果仅供链路验证。")
    elif not settings.has_api_key:
        print("[WARN] 未检测到有效 LLM_API_KEY，/api/chat 将返回 502。请复制 .env.example 为 .env 并填写密钥。")
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.app_port, reload=False)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
