"""把评测报告渲染成**自包含的 HTML**（无外部依赖，可离线双击打开）。

为什么要有 HTML 而不只有 JSON/Markdown：
- Markdown 适合读，不适合"抽查"。评测报告的价值在于**逐条核对**——
  实际回答、评审给出的原文证据、判决理由，三者要能一眼对照。
- HTML 可以做成自包含单文件（内联 CSS/JS），评审不用装任何环境就能看，
  也能直接挂到服务上（`GET /report`）在网页里查看。

设计上与 `run_eval.py` 解耦：本模块只接收已聚合好的报告 dict，返回 HTML 字符串，
因此可以单测（见 tests/test_eval_judge.py）。
"""

from __future__ import annotations

import html
import json
from datetime import datetime

_CSS = """
:root{--bg:#0b1020;--panel:#141c34;--line:#26314f;--text:#e8edf9;--dim:#98a4c4;
--cyan:#56d7e7;--green:#4ade80;--red:#ff7b7b;--amber:#f0c46a;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0b1020,#0a0e1c 60%);color:var(--text);
font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.7}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:21px;margin:0 0 6px}
h2{font-size:16px;margin:30px 0 12px;padding-left:9px;border-left:3px solid var(--cyan)}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:18px;word-break:break-all}
.warn{padding:10px 13px;border-radius:11px;font-size:12.5px;margin:12px 0;
border:1px dashed rgba(240,196,106,.5);background:rgba(240,196,106,.08);color:var(--amber)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:10px;margin:16px 0}
.card{padding:13px 15px;border-radius:13px;border:1px solid var(--line);background:var(--panel)}
.card .k{font-size:12px;color:var(--dim)}
.card .v{font-size:23px;font-weight:600;margin-top:3px}
.card .n{font-size:11px;color:var(--dim);margin-top:2px}
.ok{color:var(--green)}.bad{color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:500;background:rgba(255,255,255,.02)}
.item{border:1px solid var(--line);border-radius:13px;background:var(--panel);padding:13px 15px;margin:10px 0}
.item.fail{border-color:rgba(255,123,123,.45)}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:7px}
.id{font-family:ui-monospace,Consolas,monospace;font-weight:600}
.tag{font-size:11px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.tag.pass{color:var(--green);border-color:rgba(74,222,128,.45);background:rgba(74,222,128,.08)}
.tag.fail{color:var(--red);border-color:rgba(255,123,123,.45);background:rgba(255,123,123,.1)}
.q{color:var(--dim);font-size:12.5px;margin:4px 0}
.a{white-space:pre-wrap;word-break:break-word;background:rgba(255,255,255,.03);
border-radius:10px;padding:9px 11px;margin:7px 0}
.pt{font-size:12.5px;margin:5px 0;padding-left:11px;border-left:2px solid var(--line)}
.pt.no{border-color:rgba(255,123,123,.5)}
.pt .ev{color:var(--dim);font-size:11.5px;display:block;margin-top:2px;word-break:break-word}
.fail-why{color:var(--red);font-size:12.5px;margin-top:6px}
.filters{display:flex;gap:8px;margin:14px 0}
.filters button{font:inherit;font-size:12.5px;padding:5px 13px;border-radius:999px;cursor:pointer;
border:1px solid var(--line);background:transparent;color:var(--dim)}
.filters button.on{color:var(--cyan);border-color:rgba(86,215,231,.5);background:rgba(86,215,231,.08)}
code{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--cyan)}
"""

_JS = """
function filterItems(kind, btn){
  document.querySelectorAll('.filters button').forEach(function(b){b.classList.remove('on');});
  btn.classList.add('on');
  document.querySelectorAll('.item').forEach(function(el){
    var isFail = el.classList.contains('fail');
    el.style.display = (kind==='all' || (kind==='fail')===isFail) ? '' : 'none';
  });
}
"""


def _e(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _rate_class(value: float) -> str:
    return "ok" if value >= 1.0 else ("bad" if value < 0.9 else "")


def _metric_cards(summary: dict) -> str:
    metrics = summary.get("metrics") or {}
    cards = []
    order = [
        "意图准确率",
        "知识检索命中率",
        "回答通过率",
        "规则断言通过率",
        "评审通过率",
        "评审证据可复核率",
        "调用成功率",
    ]
    for name in order:
        item = metrics.get(name)
        if not item:
            continue
        value = item.get("value", 0)
        cards.append(
            '<div class="card">'
            f'<div class="k">{_e(name)}</div>'
            f'<div class="v {_rate_class(value)}">{value:.2%}</div>'
            f'<div class="n">{_e(item.get("numerator"))} / {_e(item.get("denominator"))}</div>'
            "</div>"
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _latency_table(summary: dict) -> str:
    lat = (summary.get("latency_ms") or {})
    rows = [
        ("total", "总耗时（全部成功案例）"),
        ("total_with_llm", "└ 其中：调用模型的案例"),
        ("total_template_only", "└ 其中：走确定性模板的案例"),
        ("intent", "意图路由（本地）"),
        ("retrieval", "知识检索（本地）"),
        ("llm", "模型生成（含网络）"),
        ("judge", "评审模型（单独统计）"),
    ]
    body = []
    for key, label in rows:
        item = lat.get(key) or {}
        if not item:
            continue
        body.append(
            f"<tr><td>{_e(label)}</td><td>{_e(item.get('count'))}</td><td>{_e(item.get('mean'))}</td>"
            f"<td>{_e(item.get('p50'))}</td><td>{_e(item.get('p90'))}</td><td>{_e(item.get('max'))}</td></tr>"
        )
    share = lat.get("model_call_share")
    note = (
        f'<div class="sub">实际发起模型调用的案例占比：{share:.0%}'
        "（其余走确定性模板，应用内耗时接近 0ms）。</div>"
        if share is not None
        else ""
    )
    return (
        "<h2>耗时分层</h2>"
        "<table><tr><th>环节</th><th>样本</th><th>均值</th><th>P50</th><th>P90</th><th>最大</th></tr>"
        + "".join(body)
        + "</table>"
        + note
    )


def _record_item(rec: dict) -> str:
    passed = bool(rec.get("passed"))
    cls = "item" if passed else "item fail"
    turns = rec.get("turns") or []
    questions = "　→　".join(str(t) for t in turns)

    tags = [
        f'<span class="tag {"pass" if passed else "fail"}">{"通过" if passed else "未通过"}</span>',
        f'<span class="tag">意图 {_e(rec.get("intent"))}</span>',
    ]
    if rec.get("expect_intent") and rec["expect_intent"] != rec.get("intent"):
        tags.append(f'<span class="tag fail">期望 {_e(rec["expect_intent"])}</span>')
    if rec.get("route_state"):
        tags.append(f'<span class="tag">路由 {_e(rec["route_state"])}</span>')
    timings = rec.get("timings") or {}
    if timings:
        tags.append(
            f'<span class="tag">{_e(timings.get("total_ms"))}ms'
            f'（意图 {_e(timings.get("intent_ms"))} / 检索 {_e(timings.get("retrieval_ms"))}'
            f' / 生成 {_e(timings.get("llm_ms"))}）</span>'
        )
    cites = [c.get("id") for c in (rec.get("citations") or [])]
    if cites:
        tags.append(f'<span class="tag">引用 {"、".join(_e(c) for c in cites)}</span>')

    parts = [
        f'<div class="{cls}">',
        f'<div class="head"><span class="id">{_e(rec.get("id"))}</span>{"".join(tags)}</div>',
        f'<div class="q">玩家：{_e(questions)}</div>',
        f'<div class="a">{_e(rec.get("answer"))}</div>',
    ]

    points = rec.get("judge_points") or []
    if points:
        parts.append('<div class="sub">评审逐要点判定：</div>')
        for pt in points:
            ok = bool(pt.get("covered"))
            verified = pt.get("evidence_verified")
            mark = "已覆盖" if ok else "未覆盖"
            if ok and verified is False:
                mark += "（证据无法复核）"
            parts.append(
                f'<div class="pt {"no" if not ok else ""}">'
                f'{_e(mark)}：{_e(pt.get("point"))}'
                f'<span class="ev">证据：{_e(pt.get("evidence")) or "（未给出）"}</span>'
                "</div>"
            )

    if rec.get("judge_forbidden"):
        parts.append('<div class="sub">命中「不应出现内容」：</div>')
        for item in rec["judge_forbidden"]:
            parts.append(f'<div class="pt no">{_e(item)}</div>')

    if not passed:
        why = rec.get("fail_reason") or rec.get("judge_reason") or ""
        parts.append(f'<div class="fail-why">未通过原因：{_e(why)}</div>')
    if rec.get("error"):
        parts.append(f'<div class="fail-why">调用错误：{_e(rec["error"])}</div>')

    parts.append("</div>")
    return "".join(parts)


def render_report_html(report: dict, *, title: str = "离线评测报告") -> str:
    summary = report.get("summary") or {}
    config = report.get("config") or {}
    records = report.get("records") or []

    meta_bits = [
        f"运行时间 {config.get('run_at', '-')}",
        f"被测模型 {config.get('model', '-')}",
        f"评审模型 {config.get('judge_model') or '未启用'}",
        f"游戏版本 {config.get('game_version', '-')}",
        f"知识库 {config.get('kb_entries', '-')} 条",
        f"调用模式 {config.get('mode', '-')}",
        f"标签 {config.get('tag', '-')}",
        f"生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    warns = []
    if config.get("mock"):
        warns.append(
            "本次运行使用 Mock 模型，回答内容为固定假回复，**不能用于质量结论**，只验证链路是否跑通。"
        )
    if config.get("judge_enabled") is False:
        warns.append(
            "本次运行未启用 LLM 评审，「回答通过率」等同「规则断言通过率」，只反映确定性硬约束。"
        )
    warn_html = "".join(f'<div class="warn">{_e(w)}</div>' for w in warns)

    fail_count = sum(1 for r in records if not r.get("passed"))
    items = "".join(_record_item(r) for r in records)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_e(title)}</h1>
  <div class="sub">{_e(" ｜ ".join(meta_bits))}</div>
  {warn_html}
  {_metric_cards(summary)}
  {_latency_table(summary)}

  <h2>逐条结果（{len(records)} 例，未通过 {fail_count} 例）</h2>
  <div class="filters">
    <button class="on" onclick="filterItems('all', this)">全部</button>
    <button onclick="filterItems('fail', this)">只看未通过</button>
  </div>
  {items}
</div>
<script>{_JS}</script>
</body>
</html>"""


def render_report_file(json_path, html_path=None) -> str:
    """读取报告 JSON 并写出同名 HTML，返回 HTML 路径。"""
    from pathlib import Path

    src = Path(json_path)
    report = json.loads(src.read_text(encoding="utf-8"))
    dst = Path(html_path) if html_path else src.with_suffix(".html")
    dst.write_text(render_report_html(report, title=f"离线评测报告 · {src.stem}"), encoding="utf-8")
    return str(dst)
