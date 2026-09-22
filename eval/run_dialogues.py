"""对话脚本运行器：回放一组多样化的测试对话，留存完整记录并输出可读转录。

与 `run_eval.py` 的分工：
- `run_eval.py` 跑的是**固定测试集**，带预期意图 / 要点覆盖 / 自动判分（用于算指标）；
- 本脚本跑的是**开放式多轮对话**，不做判分，重点是把「真实提问长什么样、系统实际答成什么样」
  原样留存下来，供人工复盘与回归对照（用于看行为、找问题）。

用法：
    python -m eval.run_dialogues                                  # 默认走 HTTP，顺带写入使用记录
    python -m eval.run_dialogues --mode core --tag core
    python -m eval.run_dialogues --only D05,D09
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import LLMError  # noqa: E402
from app.agent.pipeline import Agent  # noqa: E402
from app.config import EVAL_DIR, Settings, get_settings  # noqa: E402

DIALOGUE_DIR = EVAL_DIR / "dialogues"
DEFAULT_CASES = DIALOGUE_DIR / "cases.jsonl"
DEFAULT_OUT = DIALOGUE_DIR / "records"


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


class HttpChat:
    """通过真实接口调用，顺带把记录写进后端使用日志。

    **必须复用同一个 Client（keep-alive）**：实测本机每新建一次连接要固定多花约 320ms，
    这笔开销来自连接建立，不是应用处理。早期版本用模块级 `httpx.post`（每次新建连接），
    结果把 320ms 记进了"响应耗时"，让脚本测出的延迟比真实值高出一大截。
    浏览器默认复用连接，因此这笔开销对真实用户并不存在——纯属测量偏差。
    """

    name = "http"

    def __init__(self, settings: Settings, base_url: str):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=settings.llm_timeout_s + 20)

    def close(self) -> None:
        self._client.close()

    def chat(self, message: str, history: list[dict]) -> tuple[dict | None, str | None]:
        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={"message": message, "history": history},
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"http_error: {type(exc).__name__}: {exc}"
        if resp.status_code >= 400:
            return None, f"http_{resp.status_code}: {resp.text[:200]}"
        return resp.json(), None


class CoreChat:
    """直接调用 Agent 核心链路（不经过 HTTP，也不会写入使用记录）。"""

    name = "core"

    def __init__(self, settings: Settings):
        self.agent = Agent(settings=settings)

    def chat(self, message: str, history: list[dict]) -> tuple[dict | None, str | None]:
        try:
            result = self.agent.answer(message, history)
        except LLMError as exc:
            return None, f"{exc.kind}: {exc}"
        return {
            "answer": result.answer,
            "intent": result.intent,
            "route_state": result.route_state,
            "citations": result.citations,
            "timings": result.timings,
            "model": result.model,
            "mock": result.mock,
            "guard": result.guard,
        }, None


def run_case(case: dict, runner) -> list[dict]:
    turns: list[dict] = []
    history: list[dict] = []
    for index, message in enumerate(case["turns"], 1):
        started = time.perf_counter()
        response, error = runner.chat(message, history)
        wall_ms = int((time.perf_counter() - started) * 1000)
        record = {
            "case_id": case["id"],
            "case_title": case["title"],
            "focus": case["focus"],
            "tags": case.get("tags", []),
            "turn": index,
            "total_turns": len(case["turns"]),
            "user": message,
            "error": error,
            "wall_ms": wall_ms,
        }
        if response:
            record.update(
                {
                    "answer": response.get("answer", ""),
                    "intent": response.get("intent"),
                    "route_state": response.get("route_state"),
                    "citations": [c.get("id") for c in (response.get("citations") or [])],
                    "citation_support": [c.get("support") for c in (response.get("citations") or [])],
                    "timings": response.get("timings"),
                    "guard": response.get("guard"),
                    "mock": response.get("mock", False),
                }
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": str(response.get("answer", ""))})
        turns.append(record)
    return turns


def render_markdown(records: list[dict], config: dict) -> str:
    lines = [
        "# 多样化测试对话记录",
        "",
        f"- 运行时间：{config['run_at']}",
        f"- 调用方式：{config['mode']}｜被测模型：{config['model']}"
        + ("（**MOCK 结果，不代表真实质量**）" if config["mock"] else ""),
        f"- 对话数：{config['dialogue_count']}｜轮次总数：{config['turn_count']}",
        f"- 脚本：`eval/dialogues/cases.jsonl`（case 设计说明见 `eval/dialogues/README.md`）",
        "",
        "> 本文件是**行为记录**，不做自动判分。判分指标由 `python -m eval.run_eval` 产出。",
        "",
    ]
    current = None
    for r in records:
        if r["case_id"] != current:
            current = r["case_id"]
            lines += [
                "",
                f"## {r['case_id']}　{r['case_title']}",
                "",
                f"**考察点**：{r['focus']}　｜　**标签**：{', '.join(r['tags'])}",
                "",
            ]
        lines.append(f"**第 {r['turn']}/{r['total_turns']} 轮**")
        lines.append("")
        lines.append(f"- 玩家：{r['user']}")
        if r["error"]:
            lines.append(f"- 系统：❌ 调用失败 —— {r['error']}")
            lines.append("")
            continue
        timings = r.get("timings") or {}
        route = r.get("route_state") or "-"
        cites = "、".join(r.get("citations") or []) or "无"
        lines.append(
            f"- 意图：`{r.get('intent')}`　路由：`{route}`　耗时：{timings.get('total_ms', 0)}ms"
            f"（意图 {timings.get('intent_ms', 0)} / 检索 {timings.get('retrieval_ms', 0)}"
            f" / 生成 {timings.get('llm_ms', 0)}）"
        )
        lines.append(f"- 引用条目：{cites}")
        lines.append(f"- 小玖：{str(r.get('answer', '')).strip()}")
        lines.append("")
    return "\n".join(lines)


def write_reports(records: list[dict], config: dict, out_dir: Path, tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"dialogues_{tag}_{stamp}"
    jsonl_path = base.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    md_path = base.with_suffix(".md")
    md_path.write_text(render_markdown(records, config), encoding="utf-8")
    return jsonl_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="回放多样化测试对话并留存记录")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--mode", choices=["http", "core"], default="http")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--only", default=None, help="只跑指定对话，逗号分隔，例如 D05,D09")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(Path(args.cases))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    tag = args.tag or ("smoke" if settings.mock_llm else "run")

    print("=" * 78)
    print(f"对话回放开始｜{len(cases)} 段对话｜方式：{args.mode}｜模型：{settings.llm_model}")
    if settings.mock_llm:
        print("[MOCK] 当前为 Mock 模式，结果只验证链路，不代表真实回答质量")
    print("=" * 78)

    runner = HttpChat(settings, args.base_url) if args.mode == "http" else CoreChat(settings)

    all_records: list[dict] = []
    try:
        for case in cases:
            print(f"\n── {case['id']}　{case['title']}")
            for record in run_case(case, runner):
                all_records.append(record)
                flag = "ERR " if record["error"] else "    "
                answer = str(record.get("answer", "")).replace("\n", " ")
                print(f"  {flag}[{record['turn']}/{record['total_turns']}] 玩家：{record['user'][:34]}")
                if record["error"]:
                    print(f"        系统：失败 {record['error']}")
                else:
                    timings = record.get("timings") or {}
                    print(
                        f"        意图={record.get('intent')} 路由={record.get('route_state') or '-'} "
                        f"应用内={timings.get('total_ms', 0)}ms｜{answer[:56]}…"
                    )
    finally:
        if hasattr(runner, "close"):
            runner.close()

    config = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "model": settings.llm_model,
        "mock": settings.mock_llm,
        "dialogue_count": len(cases),
        "turn_count": len(all_records),
    }
    jsonl_path, md_path = write_reports(all_records, config, Path(args.out), tag)

    failures = [r for r in all_records if r["error"]]
    print("-" * 78)
    print(f"共 {len(all_records)} 轮，失败 {len(failures)} 轮")
    print(f"记录：{jsonl_path}")
    print(f"转录：{md_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
