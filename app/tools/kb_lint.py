"""知识库自检：字段完整性、id 唯一性、主题合法性、版本标注、来源可达性（可选）。

用法：
    python -m app.tools.kb_lint
    python -m app.tools.kb_lint --check-urls      # 联网校验来源链接可达性
    python -m app.tools.kb_lint --write-meta      # 校验通过后刷新 kb_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# 刻意不依赖 app.config / pydantic：知识库自检应当零依赖即可运行
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_FILE = BASE_DIR / "app" / "data" / "knowledge" / "kb_v1.json"
KNOWLEDGE_META_FILE = BASE_DIR / "app" / "data" / "knowledge" / "kb_meta.json"

REQUIRED_FIELDS = ["id", "topic", "subtopic", "fact_key", "title", "content", "version", "source", "confidence"]
SOURCE_FIELDS = ["title", "url", "collected_at"]
TOPICS = ["对局目标", "地图机制", "位置职责", "基础操作", "资源与经济"]
CONFIDENCE = ["high", "medium", "low"]

# 内容长度上限：与设计文档 §3.1「一条知识聚焦一个事实点」一致
MAX_CONTENT_CHARS = 260
MIN_KEYWORDS = 3
MAX_KEYWORDS = 8

# 带单位的具体数值（"90秒""2级""10分钟"），这类内容最可能随版本变化
NUMERIC_CLAIM_RE = re.compile(r"[0-9一二三四五六七八九十百千两]+\s*(秒|分钟|小时|级|金币|点券|％|%|次|层|座|件|成)")


def load_kb(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def lint(kb: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    entries = kb.get("entries", [])

    if not entries:
        errors.append("知识库为空：entries 缺失或为 []")

    seen_ids: set[str] = set()
    fact_keys: dict[str, list[str]] = {}

    for idx, e in enumerate(entries):
        tag = e.get("id") or f"#{idx}"

        for f in REQUIRED_FIELDS:
            if f not in e or e[f] in (None, "", []):
                errors.append(f"[{tag}] 缺少必填字段 {f}")

        eid = e.get("id", "")
        if eid in seen_ids:
            errors.append(f"[{tag}] id 重复")
        seen_ids.add(eid)

        if e.get("topic") not in TOPICS:
            errors.append(f"[{tag}] topic 非法：{e.get('topic')}（允许：{'/'.join(TOPICS)}）")

        if e.get("confidence") not in CONFIDENCE:
            errors.append(f"[{tag}] confidence 非法：{e.get('confidence')}")

        content = e.get("content", "")
        if len(content) > MAX_CONTENT_CHARS:
            warnings.append(f"[{tag}] 正文 {len(content)} 字，超过建议上限 {MAX_CONTENT_CHARS}，建议拆分为多条知识")

        kws = e.get("keywords", [])
        if not isinstance(kws, list) or not (MIN_KEYWORDS <= len(kws) <= MAX_KEYWORDS):
            warnings.append(f"[{tag}] keywords 数量 {len(kws) if isinstance(kws, list) else 'N/A'}，建议 {MIN_KEYWORDS}-{MAX_KEYWORDS} 个")

        src = e.get("source", {})
        for f in SOURCE_FIELDS:
            if not src.get(f):
                errors.append(f"[{tag}] source.{f} 缺失")
        url = str(src.get("url", ""))
        if url and not url.startswith("http"):
            errors.append(f"[{tag}] source.url 不是公开可访问的 http(s) 地址：{url}")

        # 数值类内容必须标注版本，避免出现"看起来是硬规则其实随版本变化"的断言
        if NUMERIC_CLAIM_RE.search(content) and e.get("version") in (None, "", "全版本"):
            if e.get("confidence") == "high":
                warnings.append(f"[{tag}] 正文含具体数值但标注为「全版本/high」，请确认该数值确实不随版本变化")

        fk = e.get("fact_key")
        if fk:
            fact_keys.setdefault(fk, []).append(eid)

    for fk, ids in fact_keys.items():
        if len(ids) > 1:
            warnings.append(f"fact_key「{fk}」存在 {len(ids)} 条：{ids}（属于版本冲突，检索按版本折扣处理，请确认是有意保留）")

    return errors, warnings


def check_urls(entries: list[dict], timeout: float = 8.0) -> list[str]:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return ["未安装 httpx，跳过链接校验"]

    notes: list[str] = []
    seen: dict[str, bool] = {}
    headers = {"User-Agent": "Mozilla/5.0 (kb-lint)"}
    for e in entries:
        url = str(e.get("source", {}).get("url", ""))
        if not url:
            continue
        if url not in seen:
            try:
                resp = httpx.get(url, timeout=timeout, headers=headers, follow_redirects=True)
                seen[url] = resp.status_code < 400
                notes.append(f"{'OK ' if seen[url] else 'FAIL'} {resp.status_code} {url}")
            except Exception as exc:  # noqa: BLE001
                seen[url] = False
                notes.append(f"FAIL {type(exc).__name__} {url}")
        if not seen[url]:
            notes.append(f"  → 被 {e.get('id')} 引用")
    return notes


def write_meta(kb: dict, path: Path) -> None:
    entries = kb["entries"]
    meta = {
        "kb_version": kb.get("kb_version"),
        "game": kb.get("game"),
        "entry_count": len(entries),
        "topic_distribution": dict(Counter(e["topic"] for e in entries)),
        "confidence_distribution": dict(Counter(e["confidence"] for e in entries)),
        "linted_at": date.today().isoformat(),
        "source_domains": sorted({str(e["source"]["url"]).split("/")[2] for e in entries}),
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库自检")
    parser.add_argument("--check-urls", action="store_true", help="联网校验来源链接可达性")
    parser.add_argument("--write-meta", action="store_true", help="刷新 kb_meta.json")
    parser.add_argument("--file", default=str(KNOWLEDGE_FILE))
    args = parser.parse_args()

    kb = load_kb(Path(args.file))
    errors, warnings = lint(kb)

    print(f"知识库文件：{args.file}")
    print(f"条目数：{len(kb.get('entries', []))}")
    print(f"主题分布：{Counter(e['topic'] for e in kb.get('entries', []))}")

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")

    if args.check_urls:
        print("--- 来源链接可达性 ---")
        for line in check_urls(kb.get("entries", [])):
            print(line)

    if errors:
        print(f"\n结论：校验失败（{len(errors)} 个错误，{len(warnings)} 个警告）")
        return 1

    print(f"\n结论：校验通过（{len(warnings)} 个警告）")
    if args.write_meta:
        write_meta(kb, KNOWLEDGE_META_FILE)
        print(f"已刷新 {KNOWLEDGE_META_FILE}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
