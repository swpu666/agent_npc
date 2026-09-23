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
    detect_anaphora,
    detect_injection,
    extract_conditions,
    has_topic_term,
    is_conversation_recall,
    is_meaningless_input,
    is_topic_followup,
    normalize,
)
from app.agent.synonyms import (
    CHITCHAT_PATTERNS,
    CONDITION_REPLY_PATTERNS,
    HERO_HINTS,
    KB_GAP_RULES,
    OUT_OF_SCOPE_RULES,
    QUESTION_PATTERNS,
    SITUATIONAL_CONTEXT,
    SITUATIONAL_SUBJECT,
    VERSION_SENSITIVE_RULES,
)
from app.agent.templates import is_need_more_info_reply

INTENT_KNOWLEDGE = "knowledge_qa"
INTENT_SITUATIONAL = "situational_advice"
INTENT_CHITCHAT = "chitchat"
INTENT_OUT_OF_SCOPE = "out_of_scope"

INTENTS = (INTENT_KNOWLEDGE, INTENT_SITUATIONAL, INTENT_CHITCHAT, INTENT_OUT_OF_SCOPE)

ROUTE_NEED_MORE_INFO = "need_more_info"
ROUTE_KB_MISS = "kb_miss"
ROUTE_KB_GAP = "kb_gap"
# 没听懂玩家在说什么（纯数字 / 纯符号 / 未知字母串）：承认没看懂并拟人化引导，
# 既不能当成"越界请求"，也不该消耗一次模型调用。
ROUTE_UNCLEAR_INPUT = "unclear_input"
# 问的是"对话本身"（"我前面问了什么""刚才说到哪了"）：答案在 short-term history 里，
# 走模板直接复述，既不检索也不调模型。
ROUTE_CONVERSATION_RECALL = "conversation_recall"
# 知识库没收录，但属于**不随版本变动的稳定常识**：允许用通用理解作答、
# 不展示来源。与 kb_miss 的分工见 synonyms.VERSION_SENSITIVE_RULES 的注释。
ROUTE_KB_GENERAL = "kb_general"

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
_KB_GAP_RES = [(reason, re.compile(p)) for reason, p in KB_GAP_RULES]
_VERSION_SENSITIVE_RES = [(reason, re.compile(p)) for reason, p in VERSION_SENSITIVE_RULES]
_CONDITION_REPLY_RES = [re.compile(p) for p in CONDITION_REPLY_PATTERNS]

# 情境建议的关键条件：位置与阶段缺一不可（知道位置才能谈打法，知道阶段才能谈节奏）。
# 英雄是加分项——缺英雄不影响给出可执行的建议，不该因此把玩家拦下来。
CONDITION_KEYS = ("position", "phase", "hero")
ESSENTIAL_CONDITIONS = ("position", "phase")

# 只用长度 >= 2 的英雄名做实体匹配：单字英雄名（镜、曜、铠、澜…）在普通文本里
# 误命中概率太高，宁可不识别也不要错判。
HERO_NAME_POOL = sorted({h for h in HERO_HINTS if len(h) >= 2}, key=len, reverse=True)

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
    source: str = "rule"                       # rule | inherit | llm | fallback
    route_state: str | None = None
    conditions: dict[str, list[str]] = field(default_factory=dict)
    missing_conditions: list[str] = field(default_factory=list)
    # 这轮对话此前是否已经追问过补充条件。路由层知道"哪些条件已经问过、哪些已经答过"，
    # 模型不知道——所以要把这件事显式传给生成阶段，否则模型会自己再问一遍。
    already_asked_conditions: bool = False
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
            "missing_conditions": self.missing_conditions,
            "already_asked_conditions": self.already_asked_conditions,
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


def is_condition_reply(raw_text: str, norm_text: str, history: list[dict] | None) -> bool:
    """是否是在**回答上一轮的追问**（补充条件）。

    这类短句既没有疑问词也没有主题词（"我玩的是悟空"），规则层原本完全没信号，
    只能交给模型兜底分类——实测不但判成了"闲聊"，还让意图环节多花 0.8~1.2 秒。

    四个必要条件（缺一不可）：
    1. 必须有历史上下文——孤立的"我玩的是悟空"没有归属，不该擅自定性；
    2. 必须是短句——补充条件通常就几个字；
    3. 原文不能带问号——**这个信号比词表可靠**："我玩打野，前期该做什么？"同样以"我玩"开头，
       但它是个真问题（这条是被单测抓出来的）；
    4. 不能命中疑问特征词——问号之外的双保险。
    """
    if not history or not norm_text or len(norm_text) > 20:
        return False
    if any(ch in raw_text for ch in "？?"):
        return False
    if any(word in norm_text for word in QUESTION_PATTERNS):
        return False
    return any(r.search(norm_text) for r in _CONDITION_REPLY_RES)


def is_kb_gap_question(text: str) -> str | None:
    """问题是否落在知识库明确不覆盖的维度上（英雄级信息 / 动态数据）。

    返回缺口原因标签，未命中返回 None。判定只看问题本身，与检索结果无关：
    这类问题即使"碰巧"检索到一条半相关条目（例如"辅助"命中了"游走位职责"），
    也不该把那条件目当成它的依据。
    """
    t = normalize(text)
    if not t:
        return None
    for reason, pattern in _KB_GAP_RES:
        if pattern.search(t):
            return reason
    for hero in HERO_NAME_POOL:
        if normalize(hero) in t:
            return "hero_entity"
    return None


def is_version_sensitive_question(text: str) -> str | None:
    """问题是否落在"会随版本变动、不能靠通用理解补"的范围内。

    返回敏感原因标签（refresh_time / number_value / price / scoring / version_data），
    未命中返回 None。

    它决定"未命中"之后是走通用理解（`kb_general`）还是如实说没有（`kb_miss`）：
    实测把两者混为一谈的代价是——「王者荣耀一共有几种召唤师技能」这类完全答得上来的
    常识问题，也会被一句"知识库没有收录"打发掉，具体数值类的边界反而没守住。
    判定只看问题本身，不看检索结果；只判"是不是数值/算法类"，不判"知识库里有没有"。
    """
    t = normalize(text)
    if not t:
        return None
    for reason, pattern in _VERSION_SENSITIVE_RES:
        if pattern.search(t):
            return reason
    return None


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
    prev_route_state: str | None = None,
) -> IntentResult:
    """对单条用户消息做意图分类。

    Args:
        text: 当前用户消息。
        history: 最近若干轮对话（[{role, content}]），用于补充条件判定、
            以及判断"是否已经追问过"。
        llm_classifier: 规则不确定时调用的兜底分类器，返回四类之一或 None。
        inherit_intent: 省略式追问时的继承意图（由调用方基于上一轮计算）。
        prev_route_state: 上一轮的路由状态。上一轮是 need_more_info 时，
            本轮直接按情境建议处理——这是在补条件，不该再花一次模型调用去猜。
    """
    norm_text = normalize(text)
    injection = detect_injection(text)
    scores, matched = _rule_scores(norm_text)
    conditions = extract_conditions(norm_text)
    oos_reason = (matched.get("oos_reason") or [None])[0]

    # 短路 0：完全没有可理解内容的输入（纯数字 / 纯符号 / 未知字母串）。
    # 放在最前面，因为它既不该被当成"越界请求"，也不该消耗一次模型调用——
    # 实测「123456」会走 LLM 兜底、花掉 1.3 秒、被判成 out_of_scope，
    # 最后回一句冷冰冰的"这个请求超出我的能力范围了"。
    if is_meaningless_input(text):
        return IntentResult(
            intent=INTENT_CHITCHAT,
            scores=scores,
            matched={**matched, "unclear_input": ["没有可识别的内容"]},
            source="rule",
            route_state=ROUTE_UNCLEAR_INPUT,
            conditions=conditions,
            injection=injection,
        )

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

    # 短路 1：问的是"对话本身"（"我前面问了什么"）。
    # 放在注入安全网**之后**：夹带"忽略以上规则"的输入仍然按越界处理，不能被它绕开。
    #
    # 不放行到下面的"继承上一轮意图"：实测「你是谁 → 韩信是什么 → 我前面问了什么」
    # 里，规则零信号 → 继承到 knowledge_qa → 拼接上一轮问题检索 → 未命中 →
    # 回一句"知识库里没有收录"。历史就在手里却答"未收录"，是答非所问。
    if is_conversation_recall(text):
        return IntentResult(
            intent=INTENT_CHITCHAT,
            scores=scores,
            matched={**matched, "conversation_recall": ["询问之前问过什么"]},
            source="rule",
            route_state=ROUTE_CONVERSATION_RECALL,
            conditions=conditions,
            injection=injection,
        )

    # 规则补充：在补上一轮追问的条件（"我玩的是悟空"），这种短句规则层原本毫无信号
    if scores[INTENT_SITUATIONAL] == 0 and is_condition_reply(text, norm_text, history):
        scores[INTENT_SITUATIONAL] = SCORE_SITUATIONAL
        matched["condition_reply"] = ["在补充上一轮追问的条件"]

    intent, uncertain = _pick(scores)
    source = "rule"

    if uncertain:
        # 优先级 1：上一轮刚问过"补充条件"，本轮几乎必然是同一情境对话的延续。
        # 直接继承，**不调用模型**——早期版本这里会走 LLM 兜底，
        # 实测不但把它判成了"闲聊"，还让意图环节多花 0.8~1.2 秒。
        if prev_route_state == ROUTE_NEED_MORE_INFO:
            intent, source, uncertain = INTENT_SITUATIONAL, "inherit", False
        # 优先级 2：省略式追问（"那这个呢"），继承上一轮意图——**但必须带承接信号**。
        #
        # 早期版本只要存在上一轮意图就无条件继承，于是「你有病」这种和上文毫不相干的
        # 零信号短句也被判成知识问答：接着被"通用理解"接住，模型顺着历史讲了一整段
        # 上一个英雄（实测反馈「我骂一句，它接着聊猴子」）。省略式追问的特征就是有指代
        # 或"X 呢"结构，没有这个信号就不该替玩家认定他在追问。
        elif (
            inherit_intent
            and inherit_intent in INTENTS
            and (detect_anaphora(text) or is_topic_followup(text))
        ):
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

    # 路由状态：情境问题缺条件 → 追问；知识问题落在知识库覆盖缺口 → kb_gap
    route_state: str | None = None
    missing_conditions: list[str] = []
    already_asked_conditions = False
    if intent == INTENT_KNOWLEDGE:
        gap_reason = is_kb_gap_question(text)
        # 继承上一轮的覆盖缺口：连续追问英雄时（"孙悟空是什么" → "那伽罗呢"），
        # 第二个英雄可能不在词表里，若只靠词表判定就会降级成"未收录"直接拒答，
        # 而玩家问的明明是同一类问题。
        if (
            not gap_reason
            and prev_route_state == ROUTE_KB_GAP
            and (detect_anaphora(text) or is_topic_followup(text))
        ):
            gap_reason = "inherited_hero_question"
        if gap_reason:
            route_state = ROUTE_KB_GAP
            matched["kb_gap_reason"] = [gap_reason]
    elif intent == INTENT_SITUATIONAL:
        merged = dict(conditions)
        for prev in _history_user_texts(history)[-4:]:
            prev_cond = extract_conditions(prev)
            for key in merged:
                merged[key] = sorted(set(merged[key]) | set(prev_cond[key]))
        conditions = merged
        missing_conditions = [k for k in CONDITION_KEYS if not merged[k]]

        # 追问只在"关键条件（位置、阶段）都不具备"时触发一次。
        # 已经追问过的对话不再机械重复同一组问题——玩家会在这种重复里感到被当成没说过话。
        already_asked_conditions = any(
            is_need_more_info_reply(str(m.get("content", "")))
            for m in (history or [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
        if all(not merged[k] for k in ESSENTIAL_CONDITIONS) and not already_asked_conditions:
            route_state = ROUTE_NEED_MORE_INFO
            matched["need_more_info_missing"] = missing_conditions

    return IntentResult(
        intent=intent,
        scores=scores,
        matched=matched,
        source=source,
        route_state=route_state,
        conditions=conditions,
        missing_conditions=missing_conditions,
        already_asked_conditions=already_asked_conditions,
        injection=injection,
        oos_reason=oos_reason,
    )


def intent_label_zh(intent: str) -> str:
    return INTENT_LABELS_ZH.get(intent, intent)
