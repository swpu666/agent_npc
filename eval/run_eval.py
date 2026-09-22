"""离线评测入口。

用法：
    python -m eval.run_eval                                  # 默认核心链路 + 默认测试集
    python -m eval.run_eval --mode http --base-url http://127.0.0.1:8000
    python -m eval.run_eval --no-judge --limit 4
    python -m eval.run_eval --only T06,T09,T13
    python -m eval.run_eval --tag smoke

产出三份报告（同一份数据，三种用途）：
    eval/reports/report_<tag>_<时间戳>.json   # 机器可读，供二次分析
    eval/reports/report_<tag>_<时间戳>.md     # 便于阅读，随仓库提交
    eval/reports/report_<tag>_<时间戳>.html   # 自包含单文件，供逐条抽查；也可由 GET /report 在网页查看

说明：
- "离线"指基于固定数据集批量评测，不依赖真实在线玩家流量；允许调用在线模型接口。
- 被测 Agent 只接收 turns（玩家消息），测试集中的 expect_* / must_cover / must_not
  等评测字段不会传入 Agent，两者在数据结构上物理分离。
- 评审模型耗时的统计口径与被测 Agent 的响应耗时分开。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import LLMClient, LLMError  # noqa: E402
from app.agent.pipeline import Agent  # noqa: E402
from app.config import EVAL_DIR, KNOWLEDGE_FILE, Settings, get_settings  # noqa: E402
from eval.judge import JudgeOutcome, llm_judge, rule_check  # noqa: E402
from eval.metrics import summarize  # noqa: E402
from eval.report_html import render_report_html  # noqa: E402

DEFAULT_DATASET = EVAL_DIR / "dataset" / "testset.jsonl"
DEFAULT_OUT = EVAL_DIR / "reports"


# --------------------------------------------------------------------- 数据集
def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def kb_id_set() -> set[str]:
    data = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    return {e["id"] for e in data.get("entries", [])}


# --------------------------------------------------------------------- 调用被测
def result_to_dict(result) -> dict:
    return {
        "answer": result.answer,
        "intent": result.intent,
        "route_state": result.route_state,
        "citations": result.citations,
        "timings": result.timings,
        "model": result.model,
        "mock": result.mock,
        "guard": result.guard,
    }


class CoreRunner:
    """直接调用 Agent 核心链路（与网页共用同一套 pipeline）。"""

    name = "core"

    def __init__(self, settings: Settings):
        self.agent = Agent(settings=settings)
        self.llm = self.agent.llm

    def chat(self, message: str, history: list[dict]) -> tuple[dict | None, str | None]:
        try:
            return result_to_dict(self.agent.answer(message, history)), None
        except LLMError as exc:
            return None, f"{exc.kind}: {exc}"


class HttpRunner:
    """通过 HTTP 调用真实的 /api/chat，验证网页使用的同一条接口链路。

    复用同一个 Client（keep-alive）：本机每新建一次连接会固定多花约 320ms 的连接建立开销，
    用模块级 `httpx.post`（每次新建连接）会把这笔开销算进"响应耗时"，
    使 http 模式测出的延迟虚高。浏览器默认复用连接，因此那笔开销对真实用户并不存在。
    """

    name = "http"

    def __init__(self, settings: Settings, base_url: str):
        import httpx

        self._client = httpx.Client(timeout=settings.llm_timeout_s + 10)
        self.base_url = base_url.rstrip("/")
        self.llm = LLMClient(settings)

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
            body = resp.text[:200]
            return None, f"http_{resp.status_code}: {body}"
        return resp.json(), None


# --------------------------------------------------------------------- 单案例
def run_case(case: dict, runner, kb_ids: set[str], judge_llm: LLMClient | None, use_judge: bool) -> dict:
    record: dict = {
        "id": case["id"],
        "category": case.get("category"),
        "coverage": case.get("coverage", []),
        "turns": case["turns"],
        "expect_intent": case["expect_intent"],
        "expect_route_state": case.get("expect_route_state"),
        "expect_kb_ids": case.get("expect_kb_ids", []),
        "needs_kb": bool(case.get("needs_kb")),
        "expect_behavior": case.get("expect_behavior", ""),
        "notes": case.get("notes", ""),
    }

    history: list[dict] = []
    response: dict | None = None
    error: str | None = None
    started = time.perf_counter()
    for turn in case["turns"]:
        response, error = runner.chat(turn, history)
        if error:
            break
        history.append({"role": "user", "content": turn})
        history.append({"role": "assistant", "content": str(response.get("answer", ""))})
    wall_ms = int((time.perf_counter() - started) * 1000)

    record["success"] = error is None and response is not None
    record["error"] = error
    if response:
        record["intent"] = response.get("intent")
        record["route_state"] = response.get("route_state")
        record["answer"] = response.get("answer", "")
        record["citations"] = response.get("citations", [])
        record["model"] = response.get("model")
        record["mock"] = response.get("mock", False)
        record["guard"] = response.get("guard", {})
        timings = response.get("timings") or {}
        record["total_ms"] = timings.get("total_ms", wall_ms)
        record["intent_ms"] = timings.get("intent_ms", 0)
        record["retrieval_ms"] = timings.get("retrieval_ms", 0)
        record["llm_ms"] = timings.get("llm_ms", 0)
    else:
        record["total_ms"] = wall_ms
        record["intent_ms"] = record["retrieval_ms"] = record["llm_ms"] = 0

    # ---- 阶段一：规则断言（确定性硬约束，失败即短路，不再调用评审模型）
    rule = rule_check(case, response, kb_ids, error=error)
    record["rule_pass"] = rule.passed
    record["rule_checks"] = rule.checks
    record["rule_failures"] = rule.failures
    record["intent_ok"] = bool(rule.checks.get("intent"))
    record["kb_hit"] = bool(rule.checks.get("kb_hit")) if case.get("needs_kb") else None

    if not rule.passed or not use_judge or judge_llm is None:
        record["judged"] = False
        record["judge_verdict"] = None
        record["judge_ms"] = 0
        if not rule.passed:
            record["passed"] = False
            record["fail_stage"] = "rule"
            record["fail_reason"] = "；".join(rule.failures)
        elif use_judge:
            record["passed"] = False
            record["fail_stage"] = "judge"
            record["fail_reason"] = "评审未执行"
        else:
            # 未启用评审时，结论退化为"仅规则断言通过"，报告中会明确标注
            record["passed"] = True
            record["fail_stage"] = None
            record["fail_reason"] = ""
        return record

    # ---- 阶段二：LLM 评审
    outcome: JudgeOutcome = llm_judge(judge_llm, case, case["turns"], str(record.get("answer", "")))
    record["judged"] = True
    record["judge_verdict"] = outcome.verdict
    record["judge_self_verdict"] = outcome.judge_self_verdict
    record["judge_coverage"] = outcome.coverage_ratio
    record["judge_points"] = outcome.points
    record["judge_forbidden"] = outcome.forbidden_hits
    record["judge_reason"] = outcome.reason
    record["judge_error"] = outcome.error
    record["judge_ms"] = outcome.judge_ms
    # 证据可复核性：评审判为"已覆盖"的要点中，有多少条给出的原文片段确实来自回答
    checked = [p for p in outcome.points if p.get("covered")]
    record["evidence_checked"] = len(checked)
    record["evidence_verified"] = len([p for p in checked if p.get("evidence_verified")])
    record["passed"] = outcome.verdict == "pass"
    record["fail_stage"] = None if record["passed"] else "judge"
    record["fail_reason"] = "" if record["passed"] else (outcome.reason or outcome.error or "要点覆盖不足")
    return record


# --------------------------------------------------------------------- 报告
def write_reports(
    records: list[dict], summary: dict, config: dict, out_dir: Path, tag: str
) -> tuple[Path, Path, Path]:
    """同时写出 JSON（机器可读）、Markdown（便于阅读）与 HTML（便于逐条抽查）。

    HTML 是自包含单文件（内联 CSS/JS），评审不用装环境就能打开，
    也可以由服务端 `GET /report` 直接在网页里查看。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"report_{tag}_{stamp}"

    payload = {"config": config, "summary": summary, "records": records}
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = base.with_suffix(".md")
    md_path.write_text(render_markdown(records, summary, config), encoding="utf-8")

    html_path = base.with_suffix(".html")
    html_path.write_text(
        render_report_html(payload, title=f"离线评测报告 · report_{tag}_{stamp}"), encoding="utf-8"
    )
    return json_path, md_path, html_path


def render_markdown(records: list[dict], summary: dict, config: dict) -> str:
    lines: list[str] = []
    lines.append("# 离线评测报告")
    lines.append("")
    lines.append(f"- 运行时间：{config['run_at']}")
    lines.append(f"- 调用方式：{config['mode']}（{'MOCK 模式，未调用真实模型' if config['mock'] else '真实模型调用'}）")
    lines.append(f"- 被测模型：{config['model']}｜评审模型：{config.get('judge_model') or '未启用'}")
    lines.append(f"- 测试集：{config['dataset']}（{summary.get('total_cases', 0)} 个案例）")
    lines.append(f"- 游戏版本：{config['game_version']}｜知识库条目：{config['kb_entries']} 条")
    lines.append(f"- 评审阶段：{'已启用' if config.get('judge_enabled') else '未启用（--no-judge，回答通过率退化为规则断言通过率）'}")
    lines.append("")
    if not config.get("judge_enabled"):
        lines.append("> 注意：本次运行未启用 LLM 评审，因此「回答通过率」等同「规则断言通过率」，")
        lines.append("> 只反映确定性硬约束，不代表回答内容质量。")
        lines.append("")
    lines.append("## 一、汇总指标")
    lines.append("")
    lines.append("| 指标 | 结果 | 分子/分母 | 适用案例 |")
    lines.append("| --- | --- | --- | --- |")
    for name, m in summary.get("metrics", {}).items():
        lines.append(f"| {name} | {m['value']:.2%} | {m['numerator']}/{m['denominator']} | {m['applicable']} |")
    lines.append("")
    lines.append("## 二、耗时（仅统计成功案例，单位为毫秒）")
    lines.append("")
    lines.append("| 环节 | 样本数 | 均值 | P50 | P90 | 最大 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in [
        ("total", "总耗时（全部成功案例）"),
        ("total_with_llm", "└ 其中：调用模型的案例"),
        ("total_template_only", "└ 其中：走确定性模板的案例（0 次模型调用）"),
        ("intent", "意图路由（本地）"),
        ("retrieval", "知识检索（本地）"),
        ("llm", "模型生成（含网络）"),
        ("judge", "评审模型（不计入 Agent 耗时）"),
    ]:
        lat = summary.get("latency_ms", {}).get(key, {})
        lines.append(
            f"| {label} | {lat.get('count', 0)} | {lat.get('mean', 0)} | {lat.get('p50', 0)} | "
            f"{lat.get('p90', 0)} | {lat.get('max', 0)} |"
        )
    share = summary.get("latency_ms", {}).get("model_call_share")
    if share is not None:
        lines.append("")
        lines.append(f"实际发起模型调用的案例占比：{share:.0%}（其余走确定性模板，响应接近 0ms）。")
    lines.append("")
    lines.append("## 三、逐条结果")
    lines.append("")
    lines.append("| 案例 | 期望意图 | 实际意图 | 路由状态 | 引用条目 | 规则 | 评审 | 总耗时 | 结论 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        cits = ",".join(c.get("id", "") for c in (r.get("citations") or [])) or "-"
        verdict = r.get("judge_verdict") or ("-" if r.get("judged") is False else "-")
        result = "通过" if r.get("passed") else f"未通过({r.get('fail_stage')})"
        lines.append(
            f"| {r['id']} | {r['expect_intent']} | {r.get('intent', '-')} | {r.get('route_state') or '-'} | "
            f"{cits} | {'通过' if r.get('rule_pass') else '失败'} | {verdict} | {r.get('total_ms', 0)}ms | {result} |"
        )
    lines.append("")

    failed = [r for r in records if not r.get("passed")]
    lines.append("## 四、未通过案例明细（供人工抽查）")
    lines.append("")
    if not failed:
        lines.append("本次运行全部通过。")
    for r in failed:
        lines.append(f"### {r['id']}")
        lines.append("")
        lines.append(f"- 多轮输入：{' ｜ '.join(r['turns'])}")
        lines.append(f"- 期望行为：{r['expect_behavior']}")
        lines.append(f"- 失败阶段：{r.get('fail_stage')}")
        for f in r.get("rule_failures") or []:
            lines.append(f"- 规则断言未通过：{f}")
        if r.get("judge_reason"):
            lines.append(f"- 评审说明：{r['judge_reason']}")
        for p in r.get("judge_points") or []:
            mark = "已覆盖" if p.get("covered") else "未覆盖"
            lines.append(f"  - [{mark}] {p['point']}｜证据：{p.get('evidence') or '（无）'}")
        for h in r.get("judge_forbidden") or []:
            lines.append(f"  - [出现禁止内容] {h.get('item')}｜证据：{h.get('evidence')}")
        lines.append(f"- 实际回答：{str(r.get('answer', ''))[:400]}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="离线评测：意图 / 检索 / 回答质量 / 耗时")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--mode", choices=["core", "http"], default="core")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tag", default=None, help="报告文件名标签，Mock 冒烟建议用 smoke")
    parser.add_argument("--no-judge", action="store_true", help="只跑规则断言，不调用评审模型")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="只跑指定案例，逗号分隔，例如 T06,T09")
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    dataset_path = Path(args.dataset)
    cases = load_dataset(dataset_path)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    tag = args.tag or ("smoke" if settings.mock_llm else "run")

    print("=" * 78)
    print("离线评测开始")
    print(f"  测试集      : {dataset_path}（{len(cases)} 个案例）")
    print(f"  调用方式    : {args.mode}")
    print(f"  被测模型    : {settings.llm_model}")
    print(f"  评审        : {'关闭' if args.no_judge else '开启（与生成分开计时）'}")
    if settings.mock_llm:
        print("  [MOCK] 当前为 Mock 模式：结果只用于验证链路，不能替代真实模型的质量评测！")
    elif not settings.has_api_key:
        print("  [WARN] 未配置 LLM_API_KEY，调用会失败；请先配置 .env")
    print("=" * 78)

    if args.mode == "core":
        runner = CoreRunner(settings)
    else:
        runner = HttpRunner(settings, args.base_url)

    judge_llm = None if args.no_judge else runner.llm
    kb_ids = kb_id_set()

    records: list[dict] = []
    for case in cases:
        record = run_case(case, runner, kb_ids, judge_llm, use_judge=not args.no_judge)
        records.append(record)
        flag = "PASS" if record.get("passed") else "FAIL"
        detail = ""
        if not record.get("passed"):
            detail = " | " + (record.get("fail_reason") or "；".join(record.get("rule_failures") or []))[:70]
        print(
            f"[{flag}] {record['id']:<4} 意图={record.get('intent', '-'):<18} "
            f"路由={str(record.get('route_state') or '-'):<14} "
            f"耗时={record.get('total_ms', 0):>6}ms{detail}"
        )
        if args.show_answers:
            print(f"       回答：{str(record.get('answer', ''))[:180].replace(chr(10), ' ')}")

    summary = summarize(records)
    config = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "dataset": str(dataset_path),
        "model": settings.llm_model,
        "judge_model": (judge_llm.model if judge_llm else None),
        "judge_enabled": not args.no_judge,
        "mock": settings.mock_llm,
        "game_version": settings.game_version,
        "kb_entries": len(kb_ids),
        "retrieval_min_score": settings.retrieval_min_score,
        "retrieval_top_k": settings.retrieval_top_k,
    }
    json_path, md_path, html_path = write_reports(records, summary, config, Path(args.out), tag)

    print("-" * 78)
    for name, m in summary.get("metrics", {}).items():
        print(f"  {name:<12}: {m['value']:>7.2%}  ({m['numerator']}/{m['denominator']})")
    lat = summary.get("latency_ms", {})
    print(
        f"  总耗时      : 均值 {lat.get('total', {}).get('mean', 0)}ms "
        f"P50 {lat.get('total', {}).get('p50', 0)}ms P90 {lat.get('total', {}).get('p90', 0)}ms"
    )
    print(
        f"  模型生成    : 均值 {lat.get('llm', {}).get('mean', 0)}ms ｜ "
        f"本地意图+检索: 均值 "
        f"{(lat.get('intent', {}).get('mean', 0) + lat.get('retrieval', {}).get('mean', 0)):.1f}ms"
    )
    print(f"  评审耗时    : 均值 {lat.get('judge', {}).get('mean', 0)}ms（单列，不计入上面）")
    print("-" * 78)
    print(f"报告已写入：{json_path}")
    print(f"           {md_path}")
    print(f"           {html_path}  ← 自包含单文件，可直接双击打开；也可访问 GET /report 查看")
    if settings.mock_llm:
        print("[MOCK] 该报告为 Mock 结果，仅验证链路，不可用于质量结论。")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
