"""知识检索模块单测：同义词扩展、版本折扣、阈值未命中、追问改写。"""

from __future__ import annotations

import json

import pytest

from app.agent.retrieval import (
    KnowledgeBase,
    augment_query,
    load_knowledge_base,
    previous_user_message,
)
from app.config import KNOWLEDGE_FILE

CURRENT_VERSION = "S47"


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return load_knowledge_base(str(KNOWLEDGE_FILE), CURRENT_VERSION)


def _top_ids(kb: KnowledgeBase, query: str, top_k: int = 3) -> list[str]:
    docs, _hit = kb.search(query, top_k=top_k, min_score=0.0, explain=False)
    return [d.id for d in docs]


# --------------------------------------------------------------- 基础召回
@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("防御塔需要按什么顺序推？", "OBJ-TOWER-002"),
        ("怎么才能赢下一局？", "OBJ-WIN-001"),
        ("打野的职责是什么？", "ROLE-JUNGLE-001"),
        ("补刀是什么意思？", "OPS-FARM-001"),
        ("红BUFF有什么作用？", "MAP-BUFF-001"),
        ("蓝buff有什么用", "MAP-BUFF-002"),
    ],
)
def test_top_hit(kb: KnowledgeBase, query: str, expected_id: str) -> None:
    assert expected_id in _top_ids(kb, query)


# --------------------------------------------------------------- 同义词/简称
def test_synonym_expansion_dalong_to_overlord(kb: KnowledgeBase) -> None:
    """俗称「大龙」应命中「主宰」。"""
    docs, hit = kb.search("大龙打完之后有什么用？", top_k=3, min_score=3.0)
    assert hit
    assert docs[0].id == "MAP-OVERLORD-001"
    assert "MAP-OVERLORD-001" in _top_ids(kb, "主宰有什么用")


def test_synonym_expansion_abbreviation(kb: KnowledgeBase) -> None:
    """英文缩写「adc / BUFF」等写法也应能召回。"""
    assert "ROLE-ADC-001" in _top_ids(kb, "adc该干什么")


# --------------------------------------------------------------- 未命中
@pytest.mark.parametrize(
    "query",
    [
        "巅峰赛的积分是怎么计算的？",
        "王者荣耀的赛季什么时候结束？",
        "英雄的皮肤怎么获得？",
    ],
)
def test_miss_below_threshold(kb: KnowledgeBase, query: str) -> None:
    docs, hit = kb.search(query, top_k=4, min_score=3.0)
    assert not hit
    assert docs == [], "未命中时不应返回任何条目，避免被当作来源展示"


# --------------------------------------------------------------- 版本处理
def test_version_mismatch_applies_discount() -> None:
    entries = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))["entries"]
    old_kb = KnowledgeBase(entries, current_version="S30")
    new_kb = KnowledgeBase(entries, current_version="S47")

    # 人为构造一条只在 S36+ 生效的知识
    entry = dict(entries[0])
    entry.update({"id": "TMP-001", "fact_key": "tmp", "version": "S36+", "keywords": ["临时词条"], "title": "临时词条"})
    doc_old, _ = KnowledgeBase([entry], current_version="S30").search("临时词条", min_score=0.0, explain=False)
    doc_new, _ = KnowledgeBase([entry], current_version="S40").search("临时词条", min_score=0.0, explain=False)
    assert doc_old[0].version_factor == pytest.approx(0.6)
    assert doc_new[0].version_factor == pytest.approx(1.0)
    # 与版本无关的知识不受影响
    docs, hit = old_kb.search("补刀是什么意思？", min_score=3.0)
    assert hit and new_kb.search("补刀是什么意思？", min_score=3.0)[1]


def test_all_version_factor(kb: KnowledgeBase) -> None:
    docs, _ = kb.search("补刀是什么意思？", min_score=0.0, explain=False)
    assert docs[0].version_factor == pytest.approx(1.0)
    assert docs[0].version == "全版本"


# --------------------------------------------------------------- 查询改写
def test_previous_user_message_picks_latest_user_turn() -> None:
    history = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "回答"},
        {"role": "user", "content": "第二个问题"},
        {"role": "assistant", "content": "回答"},
    ]
    assert previous_user_message(history) == "第二个问题"
    assert previous_user_message(None) is None


def test_augment_query_prepends_previous_question() -> None:
    assert augment_query("那这个呢？", "打野的职责是什么？") == "打野的职责是什么？ 那这个呢？"
    assert augment_query("那这个呢？", None) == "那这个呢？"


def test_topic_boost_applied() -> None:
    kb = load_knowledge_base(str(KNOWLEDGE_FILE), CURRENT_VERSION)
    plain, _ = kb.search("那这个位置前期该干嘛？", top_k=3, min_score=0.0, explain=False)
    boosted, _ = kb.search(
        "那这个位置前期该干嘛？", top_k=3, min_score=0.0, boost_topic="位置职责", explain=False
    )
    plain_map = {d.id: d.score for d in plain}
    boosted_map = {d.id: d.score for d in boosted}
    shared = set(plain_map) & set(boosted_map)
    assert shared
    for doc_id in shared:
        if [d for d in boosted if d.id == doc_id][0].topic == "位置职责":
            assert boosted_map[doc_id] > plain_map[doc_id]


# --------------------------------------------------------------- 知识库自检
def test_kb_lint_passes() -> None:
    from app.tools.kb_lint import lint, load_kb

    errors, _warnings = lint(load_kb(KNOWLEDGE_FILE))
    assert errors == []


def test_kb_covers_at_least_three_topics() -> None:
    from app.tools.kb_lint import load_kb

    kb_data = load_kb(KNOWLEDGE_FILE)
    topics = {e["topic"] for e in kb_data["entries"]}
    assert len(topics) >= 3
    assert len(kb_data["entries"]) >= 12


def test_every_source_is_verified_and_labelled() -> None:
    """回归：早期版本 31 条全部指向同一个 404 链接，而且没有来源强度标注。
    现在每条都必须写明校验日期与强度，topic 档还必须说明与正文的差距。"""
    from app.tools.kb_lint import load_kb

    entries = load_kb(KNOWLEDGE_FILE)["entries"]
    for e in entries:
        src = e["source"]
        assert src["url"].startswith("http"), f"{e['id']} 来源不是 http(s) 地址"
        assert src.get("verified_at"), f"{e['id']} 缺少链接校验日期"
        assert src.get("support") in ("direct", "topic"), f"{e['id']} 来源强度非法"
        if src["support"] == "topic":
            assert src.get("support_note"), f"{e['id']} 标注为主题来源却没有说明差距"


def test_kb_lint_rejects_invalid_support_level() -> None:
    from app.tools.kb_lint import lint, load_kb

    kb_data = load_kb(KNOWLEDGE_FILE)
    kb_data["entries"][0]["source"]["support"] = "bogus"
    errors, _warnings = lint(kb_data)
    assert any("source.support 非法" in e for e in errors)
