"""意图识别与路由模块（独立可测试）。

设计（对应设计文档 §4）：
- 规则为主路径：纯内存正则/词表打分，0 次模型调用，通常 < 1ms。
- 模型为兜底：仅当规则完全没有信号，或第一名与第二名分差 <= 1 时，调用一次受限分类。
- 最终兜底：仍不确定则判为 knowledge_qa（本 Agent 场景限定在单游戏向导内，
  "当作知识问题去检索、检索不到就诚实说未收录"是风险最低的行为）。

本模块不发起任何网络请求：模型兜底通过 `llm_classifier` 依赖注入，
测试中传入 stub 即可离线断言。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from app.agent.normalize import (
    detect_injection,
    extract_conditions,
    has_topic_term,
    normalize,
)
from app.agent.synonyms import (
    CHITCHAT_PATTERNS,
    OUT_OF_SCOPE_RULES,
    QUESTION_PATTERNS,
    SITUATIONAL_CONTEXT,
    SITUATIONAL_SUBJECT,
)

INTENT_KNOWLEDGE = "knowledge_qa"
INTENT_SITUATIONAL = "situational_advice"
INTENT_CHITCHAT = "chitchat"
INTENT_OUT_OF_SCOPE = "out_of_scope"

INTENTS = (INTENT_KNOWLEDGE, INTENT_SITUATIONAL, INTENT_CHITCHAT, INTENT_OUT_OF_SCOPE)

ROUTE_NEED_MORE_INFO = "need_more_info"
ROUTE_KB_MISS = "kb_miss"

# 多意图优先级（分数相同时按此顺序裁决）
PRIORITY = [INTENT_OUT_OF_SCOPE, INTENT_SITUATIONAL, INTENT_KNOWLEDGE, INTENT_CHITCHAT]

# 规则打分权重：只表达相对强弱，绝对值不参与对外解释。
# 注意 SCORE_SITUATIONAL 与 SCORE_KNOWLEDGE_FULL 之间刻意留出 > AMBIGUOUS_GAP 的间距：
# 带第一人称的情景问句几乎必然同时命中游戏主题词，若两者只差 1 就会被判为"不确定"，
# 从而每次都多花一次模型调用——这正是需要避免的无必要串行开销。
SCORE_OOS = 12
SCORE_SITUATIONAL = 8
SCORE_KNOWLEDGE_FULL = 5      # 疑问特征 + 主题词
SCORE_KNOWLEDGE_TOPIC = 3     # 只有主题词
SCORE_KNOWLEDGE_QUESTION = 2  # 只有疑问特征
SCORE_CHITCHAT = 4

_SITU_SUBJECT_RE = re.compile(SITUATIONAL_SUBJECT)
_SITU_CONTEXT_RE = re.compile(SITUATIONAL_CONTEXT)
_OOS_RES = [(reason, re.compile(p)) for reason, p in OUT_OF_SCOPE_RULES]

# "分差过小"的判定阈值
AMBIGUOUS_GAP = 1

INTENT_LABELS_ZH = {
    INTENT_KNOWLEDGE: "游戏知识问答",
    INTENT_SITUATIONAL: "游戏情境建议",
    INTENT_CHITCHAT: "日常闲聊",
    INTENT_OUT_OF_SCOPE: "能力范围外请求",
}


@dataclass
class IntentResult:
    intent: str
    scores: dict[str, int] = field(default_factory=dict)
    matched: dict[str, list[str]] = field(default_factory=dict)
    source: str = "rule"                       # rule | llm | fallback
    route_state: str | None = None
    conditions: dict[str, list[str]] = field(default_factory=dict)
    injection: list[str] = field(default_factory=list)
    oos_reason: str | None = None

    @property
    def is_injection_suspected(self) -> bool:
        return bool(self.injection)

    def to_debug(self) -> dict:
        return {
            "intent": self.intent,
            "scores": self.scores,
            "matched": self.matched,
            "source": self.source,
            "route_state": self.route_state,
            "conditions": self.conditions,
            "injection_suspected": self.is_injection_suspected,
            "oos_reason": self.oos_reason,
        }


def _history_user_texts(history: Iterable[dict] | None) -> list[str]:
    if not history:
        return []
    texts = []
    for m in history:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            texts.append(m["content"])
    return texts


def _match_oos(norm_text: str) -> tuple[str | None, list[str]]:
    for reason, pattern in _OOS_RES:
        if pattern.search(norm_text):
            return reason, [pattern.pattern]
    return None, []


def _rule_scores(norm_text: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}

    oos_reason, oos_hits = _match_oos(norm_text)
    if oos_hits:
        matched["out_of_scope"] = oos_hits

    situ_hits: list[str] = []
    if _SITU_SUBJECT_RE.search(norm_text):
        situ_hits.append("第一人称/战局主体")
    if _SITU_CONTEXT_RE.search(norm_text):
        situ_hits.append("局势/行动描述")
    situ = len(situ_hits) == 2
    if situ:
        matched["situational_advice"] = situ_hits

    q_hits = [w for w in QUESTION_PATTERNS if w in norm_text]
    topic = has_topic_term(norm_text)
    chat_hits = [w for w in CHITCHAT_PATTERNS if w in norm_text]

    if q_hits:
        matched["knowledge_question"] = q_hits
    if topic:
        matched.setdefault("knowledge_topic", []).append("命中强游戏主题词")
    if chat_hits:
        matched["chitchat"] = chat_hits

    if q_hits and topic:
        knowledge_score = SCORE_KNOWLEDGE_FULL
    elif topic:
        knowledge_score = SCORE_KNOWLEDGE_TOPIC
    elif q_hits:
        knowledge_score = SCORE_KNOWLEDGE_QUESTION
    else:
        knowledge_score = 0

    scores = {
        INTENT_OUT_OF_SCOPE: SCORE_OOS if oos_hits else 0,
        INTENT_SITUATIONAL: SCORE_SITUATIONAL if situ else 0,
        INTENT_KNOWLEDGE: knowledge_score,
        INTENT_CHITCHAT: SCORE_CHITCHAT if chat_hits else 0,
    }
    if oos_reason:
        matched["oos_reason"] = [oos_reason]
    return scores, matched


def _pick(scores: dict[str, int]) -> tuple[str | None, bool]:
    """返回 (最高分意图, 是否不确定)。"""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], PRIORITY.index(kv[0])))
    top_intent, top_score = ordered[0]
    if top_score <= 0:
        return None, True
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if second_score > 0 and (top_score - second_score) <= AMBIGUOUS_GAP:
        return top_intent, True
    return top_intent, False


def classify(
    text: str,
    history: list[dict] | None = None,
    llm_classifier: Callable[[str, list[dict] | None], str | None] | None = None,
    inherit_intent: str | None = None,
) -> IntentResult:
    """对单条用户消息做意图分类。

    Args:
        text: 当前用户消息。
        history: 最近若干轮对话（[{role, content}]），仅用于补充情境条件判定。
        llm_classifier: 规则不确定时调用的兜底分类器，返回四类之一或 None。
        inherit_intent: 省略式追问时的继承意图（由调用方基于上一轮计算）。
    """
    norm_text = normalize(text)
    injection = detect_injection(text)
    scores, matched = _rule_scores(norm_text)
    conditions = extract_conditions(norm_text)
    oos_reason = (matched.get("oos_reason") or [None])[0]

    # 安全网 1：注入式指令且完全不含游戏主题词 → 属于"不支持的请求"，
    # 直接归类为能力范围外，不交给模型自由发挥。
    if injection and not has_topic_term(norm_text) and not oos_reason:
        return IntentResult(
            intent=INTENT_OUT_OF_SCOPE,
            scores=scores,
            matched=matched,
            source="rule",
            route_state=None,
            conditions=conditions,
            injection=injection,
            oos_reason="unsupported_instruction",
        )

    intent, uncertain = _pick(scores)
    source = "rule"

    # 省略式追问（"那这个呢"）：单独无法判定，继承上一轮意图
    if uncertain and inherit_intent and inherit_intent in INTENTS:
        intent, source, uncertain = inherit_intent, "inherit", False

    if uncertain:
        llm_intent = None
        if llm_classifier is not None:
            try:
                llm_intent = llm_classifier(text, history)
            except Exception:  # noqa: BLE001 —— 兜底分类失败不应影响主链路
                llm_intent = None
        if llm_intent in INTENTS:
            intent, source = llm_intent, "llm"
        else:
            intent, source = INTENT_KNOWLEDGE, "fallback"

    # 情境问题：当前消息与历史里都没有位置/阶段/英雄条件 → 需要追问补充
    route_state: str | None = None
    if intent == INTENT_SITUATIONAL:
        merged = dict(conditions)
        for prev in _history_user_texts(history)[-4:]:
            prev_cond = extract_conditions(prev)
            for key in merged:
                merged[key] = sorted(set(merged[key]) | set(prev_cond[key]))
        conditions = merged
        if not (merged["position"] or merged["phase"] or merged["hero"]):
            route_state = ROUTE_NEED_MORE_INFO

    return IntentResult(
        intent=intent,
        scores=scores,
        matched=matched,
        source=source,
        route_state=route_state,
        conditions=conditions,
        injection=injection,
        oos_reason=oos_reason,
    )


def intent_label_zh(intent: str) -> str:
    return INTENT_LABELS_ZH.get(intent, intent)
