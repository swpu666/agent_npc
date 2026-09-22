"""FastAPI 应用装配入口。

启动：
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if __package__ in (None, ""):  # 支持 python app/main.py 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import __version__
from app.api.chat import router as chat_router
from app.config import STATIC_DIR, get_settings

app = FastAPI(
    title="王者荣耀训练向导 Agent",
    description="游戏问答 Agent 最小可用 Demo：意图路由 + 本地知识检索 + 模型生成",
    version=__version__,
)

app.include_router(chat_router, prefix="/api")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
