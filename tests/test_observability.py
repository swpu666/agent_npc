"""使用记录（可观测性）单测：落盘内容、读取、隐私开关、失败不影响主链路。"""

from __future__ import annotations

import json

from app.config import Settings
from app.observability import RequestLogger, RequestRecord


def make_logger(tmp_path, **overrides) -> RequestLogger:
    base = dict(_env_file=None, llm_api_key="", log_enabled=True, log_dir=str(tmp_path), log_max_chars=200)
    base.update(overrides)
    return RequestLogger(Settings(**base))


def test_record_contains_ip_and_conversation(tmp_path) -> None:
    logger = make_logger(tmp_path)
    logger.log(
        RequestRecord(
            request_id="r-1",
            ip="203.0.113.7",
            user_agent="pytest/1.0",
            message="补刀是什么意思？",
            answer="补刀就是由你自己打出最后一击。",
            intent="knowledge_qa",
            route_state=None,
            intent_source="rule",
            citations=["OPS-FARM-001"],
            timings={"total_ms": 1200, "intent_ms": 0.3, "retrieval_ms": 0.4, "llm_ms": 1199},
        )
    )
    records = logger.recent()
    assert len(records) == 1
    rec = records[0]
    assert rec["ip"] == "203.0.113.7"
    assert rec["message"] == "补刀是什么意思？"
    assert rec["answer"].startswith("补刀就是")
    assert rec["citations"] == ["OPS-FARM-001"]
    assert rec["timings"]["llm_ms"] == 1199
    assert rec["ts"]


def test_long_text_is_truncated(tmp_path) -> None:
    logger = make_logger(tmp_path, log_max_chars=50)
    logger.log(RequestRecord(request_id="r-1", ip="127.0.0.1", message="水" * 200))
    rec = logger.recent()[0]
    assert len(rec["message"]) < 200
    assert "已截断" in rec["message"]


def test_logging_can_be_disabled(tmp_path) -> None:
    logger = make_logger(tmp_path, log_enabled=False)
    logger.log(RequestRecord(request_id="r-1", ip="127.0.0.1", message="你好"))
    assert logger.recent() == []
    assert not list(tmp_path.glob("*.jsonl"))


def test_recent_returns_newest_first(tmp_path) -> None:
    logger = make_logger(tmp_path)
    for i in range(3):
        logger.log(RequestRecord(request_id=f"r-{i}", ip="127.0.0.1", message=f"问题{i}"))
    assert [r["message"] for r in logger.recent()] == ["问题2", "问题1", "问题0"]


def test_stats_aggregate(tmp_path) -> None:
    logger = make_logger(tmp_path)
    logger.log(RequestRecord(request_id="r-1", ip="1.1.1.1", message="a", intent="knowledge_qa"))
    logger.log(RequestRecord(request_id="r-2", ip="1.1.1.1", message="b", intent="chitchat"))
    logger.log(
        RequestRecord(request_id="r-3", ip="2.2.2.2", message="c", intent="out_of_scope", error="too_long")
    )
    stats = logger.stats()
    assert stats["total"] == 3
    assert stats["by_intent"]["knowledge_qa"] == 1
    assert stats["unique_ips"] == 2
    assert stats["errors"] == 1


def test_broken_log_dir_does_not_raise(tmp_path) -> None:
    """记录失败绝不能让正常问答挂掉。"""
    bad = tmp_path / "a_file_not_dir"
    bad.write_text("x", encoding="utf-8")
    logger = RequestLogger(Settings(_env_file=None, log_enabled=True, log_dir=str(bad / "logs")))
    logger.log(RequestRecord(request_id="r-1", ip="127.0.0.1", message="你好"))  # 不应抛异常
    assert logger.recent() == []


def test_jsonl_is_one_line_per_record(tmp_path) -> None:
    logger = make_logger(tmp_path)
    logger.log(RequestRecord(request_id="r-1", ip="127.0.0.1", message="第一行\n第二行"))
    files = list(tmp_path.glob("requests-*.jsonl"))
    assert len(files) == 1
    lines = [ln for ln in files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, "回答里带换行也不能把记录写成多行"
    assert json.loads(lines[0])["message"] == "第一行\n第二行"
