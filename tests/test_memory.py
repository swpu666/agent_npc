"""长期记忆单测：画像累积、注入门槛、会话隔离、清除、知识缺口聚合、路径安全。"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.memory import (
    MemoryStore,
    UserProfile,
    aggregate_knowledge_gaps,
    render_profile_note,
    valid_session_id,
)


def make_store(tmp_path, **overrides) -> MemoryStore:
    base = dict(_env_file=None, llm_api_key="", memory_enabled=True, memory_dir=str(tmp_path))
    base.update(overrides)
    return MemoryStore(Settings(**base))


def record(store: MemoryStore, session_id: str, message: str, **kwargs) -> UserProfile | None:
    payload = dict(intent="knowledge_qa", route_state=None, doc_topics=["地图机制"])
    payload.update(kwargs)
    return store.update(session_id, message=message, **payload)


# ------------------------------------------------------------------ 会话标识
@pytest.mark.parametrize("bad", ["", None, "../etc/passwd", "a/b", "a" * 65, "s id", "会话"])
def test_invalid_session_id_is_rejected(bad) -> None:
    assert valid_session_id(bad) is None


def test_valid_session_id_accepted() -> None:
    assert valid_session_id("sabc-123_x") == "sabc-123_x"


def test_session_id_traversal_cannot_escape_dir(tmp_path) -> None:
    store = make_store(tmp_path)
    assert store._path("../../evil") is None
    assert store.load("../../evil") is None


# ------------------------------------------------------------------ 累积
def test_profile_accumulates_observable_facts(tmp_path) -> None:
    store = make_store(tmp_path)
    record(
        store, "s1", "红BUFF有什么用？",
        doc_topics=["地图机制"], conditions={"position": ["打野"], "hero": ["孙悟空"]},
    )
    profile = record(
        store, "s1", "暴君多久刷新？",
        doc_topics=["地图机制"], conditions={"position": ["打野"], "hero": []},
    )
    assert profile is not None
    assert profile.turns == 2
    assert profile.topics["地图机制"] == 2
    assert profile.positions["打野"] == 2
    assert profile.heroes["孙悟空"] == 1
    assert profile.recent_questions == ["红BUFF有什么用？", "暴君多久刷新？"]


def test_positions_are_normalised(tmp_path) -> None:
    """「射手」与「下路」应合并成同一个位置，否则画像会被同一件事拆成好几份。"""
    store = make_store(tmp_path)
    record(store, "s1", "a", conditions={"position": ["射手"], "hero": []})
    profile = record(store, "s1", "b", conditions={"position": ["下路"], "hero": []})
    assert dict(profile.positions) == {"发育路": 2}


def test_sessions_are_isolated(tmp_path) -> None:
    store = make_store(tmp_path)
    record(store, "s1", "红BUFF有什么用？")
    record(store, "s2", "暴君多久刷新？")
    assert store.load("s1").turns == 1
    assert store.load("s2").turns == 1
    assert store.load("s1").recent_questions == ["红BUFF有什么用？"]


def test_memory_can_be_cleared(tmp_path) -> None:
    store = make_store(tmp_path)
    record(store, "s1", "红BUFF有什么用？")
    assert store.clear("s1") is True
    assert store.load("s1") is None
    assert store.clear("s1") is False


def test_memory_disabled_writes_nothing(tmp_path) -> None:
    store = make_store(tmp_path, memory_enabled=False)
    assert record(store, "s1", "红BUFF有什么用？") is None
    assert not list(tmp_path.glob("*.json"))


def test_broken_memory_dir_does_not_raise(tmp_path) -> None:
    """记忆写入失败绝不能让正常问答挂掉。"""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    store = MemoryStore(Settings(_env_file=None, memory_enabled=True, memory_dir=str(blocker / "m")))
    record(store, "s1", "红BUFF有什么用？")  # 不应抛异常
    assert store.load("s1") is None


# ------------------------------------------------------------------ 注入
def test_profile_note_needs_enough_turns(tmp_path) -> None:
    """样本太小就不注入：问两次就讲"你常问什么"反而像在盯着玩家。"""
    store = make_store(tmp_path)
    profile = None
    for i in range(2):
        profile = record(store, "s1", f"问题{i}")
    assert render_profile_note(profile) == ""


def test_profile_note_forbids_inferring_preferences(tmp_path) -> None:
    """画像注入了统计，同时必须明确禁止模型把它当事实复述或据此推断喜好。"""
    store = make_store(tmp_path)
    profile = None
    for i in range(4):
        profile = record(store, "s1", f"问题{i}", conditions={"position": ["打野"], "hero": []})
    note = render_profile_note(profile)
    assert "常问方向" in note
    assert "不是他本人说的" in note
    assert "不要据此推断他喜欢玩什么位置或英雄" in note


def test_profile_note_empty_for_unknown_session() -> None:
    assert render_profile_note(None) == ""


# ------------------------------------------------------------------ 知识缺口
def test_unresolved_questions_are_recorded(tmp_path) -> None:
    store = make_store(tmp_path)
    record(store, "s1", "巅峰赛积分怎么算？", route_state="kb_miss")
    record(store, "s1", "孙悟空是什么", route_state="kb_gap")
    profile = store.load("s1")
    assert "巅峰赛积分怎么算？" in profile.unresolved
    assert "孙悟空是什么" in profile.unresolved


def test_answered_question_is_not_a_gap(tmp_path) -> None:
    """答上来的问题不该进缺口清单，否则清单会被噪声淹没。"""
    store = make_store(tmp_path)
    record(
        store, "s1", "红BUFF有什么用？",
        route_state=None, doc_topics=["地图机制"], answer_had_kb=True,
    )
    profile = store.load("s1")
    assert profile.unresolved == []


def test_knowledge_gaps_aggregate_across_sessions(tmp_path) -> None:
    store = make_store(tmp_path)
    for sid in ("s1", "s2", "s3"):
        record(store, sid, "巅峰赛积分怎么算？", route_state="kb_miss")
    record(store, "s1", "孙悟空是什么", route_state="kb_gap")
    gaps = aggregate_knowledge_gaps(store)
    assert gaps[0] == {"question": "巅峰赛积分怎么算？", "count": 3}
    assert {g["question"] for g in gaps} == {"巅峰赛积分怎么算？", "孙悟空是什么"}


# ------------------------------------------------------------------ 序列化
def test_profile_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)
    record(store, "s1", "红BUFF有什么用？", conditions={"position": ["打野"], "hero": ["孙悟空"]})
    raw = json.loads((tmp_path / "s1.json").read_text(encoding="utf-8"))
    assert raw["turns"] == 1
    restored = UserProfile.from_dict(raw)
    assert restored.topics["地图机制"] == 1
    assert restored.positions["打野"] == 1


def test_corrupted_profile_file_is_ignored(tmp_path) -> None:
    (tmp_path / "s1.json").write_text("{ broken json", encoding="utf-8")
    store = make_store(tmp_path)
    assert store.load("s1") is None
